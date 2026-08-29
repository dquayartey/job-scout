import os
import json
import time
import logging
import boto3
import requests
from groq import Groq
from groq import (
    RateLimitError,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---- Config from environment (set by SAM/Terraform) ----
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
DDB_TABLE = os.environ["DDB_TABLE"]
S3_BUCKET = os.environ["S3_BUCKET"]
SES_SENDER = os.environ["SES_SENDER"]
SES_RECIPIENT = os.environ["SES_RECIPIENT"]
MATCH_THRESHOLD = int(os.environ.get("MATCH_THRESHOLD", "7"))
SEEN_TTL_DAYS = int(os.environ.get("SEEN_TTL_DAYS", "30"))
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
CV_LINK_EXPIRY_S = int(os.environ.get("CV_LINK_EXPIRY_S", str(7 * 24 * 3600)))  # 7 days

# Job search config
JOB_QUERIES = [q.strip() for q in os.environ.get("JOB_QUERIES", "cloud engineer,devops engineer").split(",") if q.strip()]
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "100"))  # Remotive per-query limit
ARBEITNOW_MAX_PAGES = int(os.environ.get("ARBEITNOW_MAX_PAGES", "3"))

# Retries for transient Groq errors (rate limit / connection / timeout),
# on top of whatever the SDK does internally
GROQ_MAX_ATTEMPTS = int(os.environ.get("GROQ_MAX_ATTEMPTS", "3"))
GROQ_RETRY_BACKOFF_S = int(os.environ.get("GROQ_RETRY_BACKOFF_S", "5"))

USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; JobScoutBot/1.0)"}
REMOTE_LOCATION_MARKERS = ("worldwide", "anywhere", "global")

# Used only to tag seniority for the LLM's benefit — NOT to filter jobs out.
SENIOR_TITLE_MARKERS = ("senior", "sr.", "sr ", "lead", "staff", "principal", "manager", "director", "head of")
JUNIOR_TITLE_MARKERS = ("junior", "jr.", "jr ", "entry", "entry-level", "graduate", "associate")

TIER_LABELS = {
    "no_auth_required": "Remote — worldwide, no work permit needed",
    "in_person_visa_sponsorship": "In-person — visa sponsorship available",
}

s3 = boto3.client("s3")
ddb = boto3.resource("dynamodb").Table(DDB_TABLE)
ses = boto3.client("ses")
groq_client = Groq(api_key=GROQ_API_KEY)


# ---- job source helpers ----

def _is_open_location(location_text):
    """True if a remote listing's location field states worldwide eligibility, or states nothing."""
    text = (location_text or "").strip().lower()
    if text == "":
        return True
    return any(marker in text for marker in REMOTE_LOCATION_MARKERS)


def _keyword_match(*fields):
    haystack = " ".join(f or "" for f in fields).lower()
    return any(query in haystack for query in JOB_QUERIES)


def _seniority_from_title(title):
    """Soft guess from title text — used as a fallback when a source has no real level field."""
    text = (title or "").strip().lower()
    if any(marker in text for marker in SENIOR_TITLE_MARKERS):
        return "senior"
    if any(marker in text for marker in JUNIOR_TITLE_MARKERS):
        return "junior"
    return "unspecified"


def _seniority_from_jobicy(job_level, title):
    """Jobicy provides a real jobLevel field — prefer it, fall back to title guess if missing.
    Observed real values: 'Senior', 'Junior', 'Entry Level', 'Officer/Mid-Level', etc.
    Matched case-insensitively/by substring since the exact vocabulary isn't officially documented.
    """
    text = (job_level or "").strip().lower()
    if not text:
        return _seniority_from_title(title)
    if "senior" in text or "lead" in text or "staff" in text or "principal" in text:
        return "senior"
    if "entry" in text or "junior" in text:
        return "junior"
    if "mid" in text:
        return "mid"
    return _seniority_from_title(title)


# ---- raw fetchers ----

def _get_remotive_jobs():
    all_jobs = []
    for query in JOB_QUERIES:
        resp = requests.get(
            "https://remotive.com/api/remote-jobs",
            params={"search": query, "limit": MAX_RESULTS},
            headers=USER_AGENT,
            timeout=20,
        )
        resp.raise_for_status()
        all_jobs.extend(resp.json().get("jobs", []))
    # Remotive's server-side "search" is loose (matches tags/category too),
    # so re-filter locally on title/description like the other sources.
    return [j for j in all_jobs if _keyword_match(j.get("title"), j.get("description"))]


def _get_remoteok_jobs():
    resp = requests.get("https://remoteok.com/api", headers=USER_AGENT, timeout=20)
    resp.raise_for_status()
    raw = resp.json()
    raw = raw[1:] if raw and "legal" in str(raw[0]).lower() else raw
    return [j for j in raw if _keyword_match(j.get("position"), j.get("description"))]


def _get_arbeitnow_jobs():
    seen_slugs = set()
    all_jobs = []
    for page in range(1, ARBEITNOW_MAX_PAGES + 1):
        resp = requests.get(
            "https://arbeitnow.com/api/job-board-api",
            params={"page": page},
            headers=USER_AGENT,
            timeout=20,
        )
        resp.raise_for_status()
        page_data = resp.json().get("data", [])
        if not page_data:
            break  # no more pages
        new_on_this_page = 0
        for job in page_data:
            slug = job.get("slug")
            if slug and slug not in seen_slugs:
                seen_slugs.add(slug)
                all_jobs.append(job)
                new_on_this_page += 1
        if new_on_this_page == 0:
            break  # page repeated the same content as before — past the real last page
    return [j for j in all_jobs if _keyword_match(j.get("title"), j.get("description"))]


def _get_jobicy_jobs():
    # Jobicy's public API caps "count" (no documented page/offset param),
    # so 50 is close to the practical ceiling for a single call.
    resp = requests.get(
        "https://jobicy.com/api/v2/remote-jobs",
        params={"count": 50},
        headers=USER_AGENT,
        timeout=20,
    )
    resp.raise_for_status()
    raw = resp.json().get("jobs", [])
    return [j for j in raw if _keyword_match(j.get("jobTitle"), j.get("jobExcerpt"))]


# ---- normalizers: map each source into one common schema ----

def normalize_remotive(job):
    return {
        "id": f"remotive:{job.get('id')}",
        "source": "remotive",
        "title": job.get("title"),
        "company": job.get("company_name"),
        "url": job.get("url"),
        "location_raw": job.get("candidate_required_location"),
        "remote": True,
        "open_worldwide": _is_open_location(job.get("candidate_required_location")),
        "visa_sponsorship": None,
        "seniority": _seniority_from_title(job.get("title")),
        "description": job.get("description"),
    }


def normalize_remoteok(job):
    return {
        "id": f"remoteok:{job.get('id')}",
        "source": "remoteok",
        "title": job.get("position"),
        "company": job.get("company"),
        "url": job.get("url") or job.get("apply_url"),
        "location_raw": job.get("location"),
        "remote": True,
        "open_worldwide": _is_open_location(job.get("location")),
        "visa_sponsorship": None,
        "seniority": _seniority_from_title(job.get("position")),
        "description": job.get("description"),
    }


def normalize_arbeitnow(job):
    return {
        "id": f"arbeitnow:{job.get('slug')}",
        "source": "arbeitnow",
        "title": job.get("title"),
        "company": job.get("company_name"),
        "url": job.get("url"),
        "location_raw": job.get("location"),
        "remote": bool(job.get("remote")),
        "open_worldwide": _is_open_location(job.get("location")),
        "visa_sponsorship": bool(job.get("visa_sponsorship")),
        "seniority": _seniority_from_title(job.get("title")),
        "description": job.get("description"),
    }


def normalize_jobicy(job):
    return {
        "id": f"jobicy:{job.get('id')}",
        "source": "jobicy",
        "title": job.get("jobTitle"),
        "company": job.get("companyName"),
        "url": job.get("url"),
        "location_raw": job.get("jobGeo"),
        "remote": True,
        "open_worldwide": _is_open_location(job.get("jobGeo")),
        "visa_sponsorship": None,
        "seniority": _seniority_from_jobicy(job.get("jobLevel"), job.get("jobTitle")),
        "description": job.get("jobExcerpt"),
    }


# ---- filter + tier + dedup ----
# Seniority is NOT filtered here — it's tagged and passed through so the LLM
# scoring step can weigh it against the candidate's actual CV, rather than a
# title-text guess silently dropping real listings.

def passes_filters(job):
    if job["remote"] and job["open_worldwide"]:
        return True
    if job["source"] == "arbeitnow" and not job["remote"] and job.get("visa_sponsorship"):
        return True
    return False


def match_tier(job):
    if job["remote"] and job["open_worldwide"]:
        return "no_auth_required"
    if job["source"] == "arbeitnow" and not job["remote"] and job.get("visa_sponsorship"):
        return "in_person_visa_sponsorship"
    return None


def dedup(jobs):
    seen = set()
    unique = []
    for job in jobs:
        key = (job["title"] or "").strip().lower(), (job["company"] or "").strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique


def fetch_jobs():
    """Pull remote/visa-eligible listings from four free job APIs, filtered and deduped."""
    raw = []
    raw += [normalize_remotive(j) for j in _get_remotive_jobs()]
    raw += [normalize_remoteok(j) for j in _get_remoteok_jobs()]
    raw += [normalize_arbeitnow(j) for j in _get_arbeitnow_jobs()]
    raw += [normalize_jobicy(j) for j in _get_jobicy_jobs()]

    filtered = [j for j in raw if passes_filters(j)]
    for j in filtered:
        j["match_tier"] = match_tier(j)

    return dedup(filtered)


def already_seen(job_id):
    item = ddb.get_item(Key={"job_id": job_id}).get("Item")
    return item is not None


def mark_seen(job_id):
    ddb.put_item(Item={
        "job_id": job_id,
        "seen_at": int(time.time()),
        "ttl": int(time.time()) + SEEN_TTL_DAYS * 86400,
    })


def get_master_cv():
    obj = s3.get_object(Bucket=S3_BUCKET, Key="cv/master_cv.txt")
    return obj["Body"].read().decode("utf-8")


def _chat_with_retry(messages, json_mode=False):
    """Call Groq with a small retry layer for transient errors.

    RateLimitError (429) and connection/timeout errors are retried with
    backoff. Anything else (bad request, auth) is not — raises immediately.
    """
    kwargs = {
        "model": GROQ_MODEL,
        "messages": messages,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(1, GROQ_MAX_ATTEMPTS + 1):
        try:
            return groq_client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            last_err = e
            logger.warning(f"Groq RateLimitError on attempt {attempt}/{GROQ_MAX_ATTEMPTS}: {e}")
            if attempt < GROQ_MAX_ATTEMPTS:
                time.sleep(GROQ_RETRY_BACKOFF_S * attempt)
        except (APIConnectionError, APITimeoutError) as e:
            last_err = e
            logger.warning(f"Groq connection/timeout error on attempt {attempt}/{GROQ_MAX_ATTEMPTS}: {e}")
            if attempt < GROQ_MAX_ATTEMPTS:
                time.sleep(GROQ_RETRY_BACKOFF_S * attempt)
        except APIStatusError as e:
            # Bad request, auth failure, etc. — not transient, don't retry
            logger.error(f"Groq APIStatusError (not retrying): {e}")
            raise
    raise last_err


def score_job(master_cv, job):
    """Score a job against the CV. Seniority is given as context, not a hard
    filter — the model decides whether a 'senior'-labeled posting is still
    worth pursuing given the candidate's actual experience."""
    prompt = f"""You are screening a job posting against a candidate's CV.

The job's stated seniority level is: {job['seniority']}. Titles like "Senior"
are used loosely across companies and don't always require senior-level
experience — judge actual fit from the description and required skills, not
just the title label. If the role clearly requires far more experience than
the CV shows, reflect that with a lower score rather than rejecting it outright.

Return ONLY valid JSON: {{"match_score": <1-10 integer>, "reasoning": "<one sentence>"}}

CV:
{master_cv}

Job title: {job['title']}
Company: {job['company']}
Work arrangement: {job['match_tier']}
Description: {(job.get('description') or '')[:2000]}
"""
    response = _chat_with_retry(
        messages=[{"role": "user", "content": prompt}],
        json_mode=True,
    )
    text = response.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"match_score": 0, "reasoning": "Could not parse model output"}


def tailor_cv(master_cv, job):
    sponsorship_note = ""
    if job["match_tier"] == "in_person_visa_sponsorship":
        sponsorship_note = (
            "\nThis is an in-person role where the employer offers visa sponsorship. "
            "Where natural, the CV may reflect openness to relocation, but do not "
            "invent statements about visa status or authorization."
        )

    prompt = f"""Rewrite the CV below to better match this specific job posting.
Keep it truthful — only reorder, re-emphasize, and reword existing experience.
Do not invent skills or experience that aren't in the original CV.
Return plain text only, no commentary.
{sponsorship_note}
Original CV:
{master_cv}

Job title: {job['title']}
Job description: {(job.get('description') or '')[:2000]}
"""
    response = _chat_with_retry(
        messages=[{"role": "user", "content": prompt}],
        json_mode=False,
    )
    return response.choices[0].message.content


def store_tailored_cv(job_id, content):
    # job_id contains a colon (e.g. "remotive:12345") — safe as an S3 key component
    key = f"tailored-cvs/{job_id.replace(':', '_')}.txt"
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=content.encode("utf-8"))
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=CV_LINK_EXPIRY_S,
    )
    return url


def send_digest(matches, failed_count):
    if not matches and not failed_count:
        return
    body_lines = []
    if matches:
        body_lines.append("New job matches found:\n")
        for m in matches:
            tier_label = TIER_LABELS.get(m["match_tier"], m["match_tier"])
            body_lines.append(
                f"- {m['title']} @ {m['company']} (score {m['score']}/10)\n"
                f"  {tier_label}\n"
                f"  {m['reasoning']}\n"
                f"  Listing: {m['url']}\n"
                f"  Tailored CV: {m['cv_key']}\n"
            )
    if failed_count:
        body_lines.append(
            f"\nNote: {failed_count} job(s) could not be scored due to a "
            f"temporary Groq API issue and will be retried on the next run.\n"
        )
    body = "\n".join(body_lines)
    ses.send_email(
        Source=SES_SENDER,
        Destination={"ToAddresses": [SES_RECIPIENT]},
        Message={
            "Subject": {"Data": f"Job Scout: {len(matches)} new match(es)"},
            "Body": {"Text": {"Data": body}},
        },
    )


def handler(event, context):
    master_cv = get_master_cv()
    jobs = fetch_jobs()
    matches = []
    failed_count = 0

    for job in jobs:
        job_id = job["id"]
        if already_seen(job_id):
            continue

        try:
            result = score_job(master_cv, job)
        except (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError) as e:
            # Don't mark_seen — leave it unscored so it's retried next run
            logger.error(f"Skipping job {job_id} after scoring failure: {e}")
            failed_count += 1
            continue

        mark_seen(job_id)  # mark regardless of score so we never re-score it

        if result.get("match_score", 0) >= MATCH_THRESHOLD:
            try:
                tailored = tailor_cv(master_cv, job)
                cv_key = store_tailored_cv(job_id, tailored)
                matches.append({
                    "title": job["title"],
                    "company": job["company"],
                    "score": result["match_score"],
                    "reasoning": result.get("reasoning", ""),
                    "url": job["url"],
                    "cv_key": cv_key,
                    "match_tier": job["match_tier"],
                })
            except (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError) as e:
                # Job was already scored and marked seen — a tailoring
                # failure just means no tailored CV/digest entry this run
                logger.error(f"CV tailoring failed for job {job_id}: {e}")
                failed_count += 1

    send_digest(matches, failed_count)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "jobs_checked": len(jobs),
            "new_matches": len(matches),
            "failed": failed_count,
        }),
    }