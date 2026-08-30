import os
import json
import re
import time
import logging
import boto3
import requests
from datetime import datetime, timezone
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError
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
MAX_SCORE_ATTEMPTS = int(os.environ.get("MAX_SCORE_ATTEMPTS", "3"))
NOTIFY_DAYS_REQUIRED = int(os.environ.get("NOTIFY_DAYS_REQUIRED", "3"))
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
CV_LINK_EXPIRY_S = int(os.environ.get("CV_LINK_EXPIRY_S", str(7 * 24 * 3600)))  # 7 days

CANDIDATE_LOCATION = os.environ.get("CANDIDATE_LOCATION", "Ghana")

JOB_QUERIES = [q.strip() for q in os.environ.get("JOB_QUERIES", "cloud engineer,devops engineer").split(",") if q.strip()]
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "100"))
ARBEITNOW_MAX_PAGES = int(os.environ.get("ARBEITNOW_MAX_PAGES", "3"))
HIMALAYAS_MAX_PAGES = int(os.environ.get("HIMALAYAS_MAX_PAGES", "3"))

DESCRIPTION_CHAR_LIMIT = int(os.environ.get("DESCRIPTION_CHAR_LIMIT", "4500"))

GROQ_MAX_ATTEMPTS = int(os.environ.get("GROQ_MAX_ATTEMPTS", "3"))
GROQ_RETRY_BACKOFF_S = int(os.environ.get("GROQ_RETRY_BACKOFF_S", "5"))

SCORE_BATCH_SIZE = int(os.environ.get("SCORE_BATCH_SIZE", "4"))

TAILOR_BATCH_SIZE = int(os.environ.get("TAILOR_BATCH_SIZE", "3"))

USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; JobScoutBot/1.0)"}
REMOTE_LOCATION_MARKERS = ("worldwide", "anywhere", "global")

SENIOR_TITLE_MARKERS = ("senior", "sr.", "sr ", "lead", "staff", "principal", "manager", "director", "head of")
JUNIOR_TITLE_MARKERS = ("junior", "jr.", "jr ", "entry", "entry-level", "graduate", "associate")

TIER_LABELS = {
    "no_auth_required": "Remote — worldwide, no work permit needed",
    "remote_location_uncertain": "Remote — location eligibility judged by model",
    "in_person_visa_sponsorship": "In-person — visa sponsorship available",
}

LOCK_JOB_ID = "__run_lock__"
LOCK_DURATION_S = 600

s3 = boto3.client("s3")
ddb = boto3.resource("dynamodb").Table(DDB_TABLE)
ses = boto3.client("ses")
groq_client = Groq(api_key=GROQ_API_KEY)


def acquire_lock():
    now = int(time.time())
    try:
        ddb.put_item(
            Item={"job_id": LOCK_JOB_ID, "locked_until": now + LOCK_DURATION_S, "ttl": now + LOCK_DURATION_S},
            ConditionExpression=Attr("job_id").not_exists() | Attr("locked_until").lt(now),
        )
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise


def release_lock():
    now = int(time.time())
    ddb.put_item(Item={"job_id": LOCK_JOB_ID, "locked_until": now, "ttl": now + LOCK_DURATION_S})


def _is_open_location(location_text):
    text = (location_text or "").strip().lower()
    if text == "":
        return True
    return any(marker in text for marker in REMOTE_LOCATION_MARKERS)


def _keyword_match(*fields):
    haystack = " ".join(f or "" for f in fields).lower()
    return any(query in haystack for query in JOB_QUERIES)


def _seniority_from_title(title):
    text = (title or "").strip().lower()
    if any(marker in text for marker in SENIOR_TITLE_MARKERS):
        return "senior"
    if any(marker in text for marker in JUNIOR_TITLE_MARKERS):
        return "junior"
    return "unspecified"


def _seniority_from_jobicy(job_level, title):
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


def _seniority_from_himalayas(seniority_list, title):
    mapping = {
        "entry-level": "junior",
        "mid-level": "mid",
        "senior": "senior",
        "manager": "senior",
        "director": "senior",
        "executive": "senior",
    }
    for level in (seniority_list or []):
        key = (level or "").strip().lower()
        if key in mapping:
            return mapping[key]
    return _seniority_from_title(title)


def _get_remotive_jobs():
    seen_ids = set()
    all_jobs = []
    for query in JOB_QUERIES:
        resp = requests.get(
            "https://remotive.com/api/remote-jobs",
            params={"search": query, "limit": MAX_RESULTS},
            headers=USER_AGENT,
            timeout=20,
        )
        resp.raise_for_status()
        for job in resp.json().get("jobs", []):
            job_id = job.get("id")
            if job_id and job_id not in seen_ids:
                seen_ids.add(job_id)
                all_jobs.append(job)
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
            break
        new_on_this_page = 0
        for job in page_data:
            slug = job.get("slug")
            if slug and slug not in seen_slugs:
                seen_slugs.add(slug)
                all_jobs.append(job)
                new_on_this_page += 1
        if new_on_this_page == 0:
            break
    return [j for j in all_jobs if _keyword_match(j.get("title"), j.get("description"))]


def _get_jobicy_jobs():
    seen_ids = set()
    all_jobs = []
    for query in JOB_QUERIES:
        resp = requests.get(
            "https://jobicy.com/api/v2/remote-jobs",
            params={"count": 50, "tag": query},
            headers=USER_AGENT,
            timeout=20,
        )
        resp.raise_for_status()
        for job in resp.json().get("jobs", []):
            job_id = job.get("id")
            if job_id and job_id not in seen_ids:
                seen_ids.add(job_id)
                all_jobs.append(job)
    return [j for j in all_jobs if _keyword_match(j.get("jobTitle"), j.get("jobExcerpt"))]


def _get_himalayas_jobs():
    seen_guids = set()
    all_jobs = []
    for query in JOB_QUERIES:
        for page in range(1, HIMALAYAS_MAX_PAGES + 1):
            resp = requests.get(
                "https://himalayas.app/jobs/api/search",
                params={"q": query, "worldwide": "true", "page": page},
                headers=USER_AGENT,
                timeout=20,
            )
            resp.raise_for_status()
            page_jobs = resp.json().get("jobs", [])
            if not page_jobs:
                break
            new_on_this_page = 0
            for job in page_jobs:
                guid = job.get("guid")
                if guid and guid not in seen_guids:
                    seen_guids.add(guid)
                    all_jobs.append(job)
                    new_on_this_page += 1
            if new_on_this_page == 0:
                break
    return [j for j in all_jobs if _keyword_match(j.get("title"), j.get("excerpt"))]


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


def normalize_himalayas(job):
    location_restrictions = job.get("locationRestrictions") or []
    return {
        "id": f"himalayas:{job.get('guid')}",
        "source": "himalayas",
        "title": job.get("title"),
        "company": job.get("companyName"),
        "url": job.get("applicationLink"),
        "location_raw": ", ".join(c.get("name", "") for c in location_restrictions) or "Worldwide",
        "remote": True,
        "open_worldwide": len(location_restrictions) == 0,
        "visa_sponsorship": None,
        "seniority": _seniority_from_himalayas(job.get("seniority"), job.get("title")),
        "description": job.get("description") or job.get("excerpt"),
    }


def passes_filters(job):
    if job["remote"]:
        return True
    if job["source"] == "arbeitnow" and not job["remote"] and job.get("visa_sponsorship"):
        return True
    return False


def match_tier(job):
    if job["remote"] and job["open_worldwide"]:
        return "no_auth_required"
    if job["remote"] and not job["open_worldwide"]:
        return "remote_location_uncertain"
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
    raw = []
    raw += [normalize_remotive(j) for j in _get_remotive_jobs()]
    raw += [normalize_remoteok(j) for j in _get_remoteok_jobs()]
    raw += [normalize_arbeitnow(j) for j in _get_arbeitnow_jobs()]
    raw += [normalize_jobicy(j) for j in _get_jobicy_jobs()]
    raw += [normalize_himalayas(j) for j in _get_himalayas_jobs()]

    filtered = [j for j in raw if passes_filters(j)]
    for j in filtered:
        j["match_tier"] = match_tier(j)

    return dedup(filtered)


def _get_seen_item(job_id):
    return ddb.get_item(Key={"job_id": job_id}).get("Item")


def _today_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def should_skip(job_id):
    item = _get_seen_item(job_id)
    if item is None:
        return False
    if item.get("matched"):
        return len(item.get("match_dates", [])) >= NOTIFY_DAYS_REQUIRED
    return item.get("attempts", 0) >= MAX_SCORE_ATTEMPTS


def record_attempt(job_id, matched, score=None, reasoning=None):
    item = _get_seen_item(job_id) or {}
    attempts = item.get("attempts", 0) + 1

    resolved_score = score if score is not None else item.get("last_score")
    resolved_reasoning = reasoning if reasoning is not None else item.get("last_reasoning")

    ddb.put_item(Item={
        "job_id": job_id,
        "attempts": attempts,
        "matched": matched or item.get("matched", False),
        "last_score": resolved_score if resolved_score is not None else "N/A",
        "last_reasoning": resolved_reasoning if resolved_reasoning is not None else "No reasoning recorded.",
        "match_dates": item.get("match_dates", []),
        "cv_key": item.get("cv_key"),
        "seen_at": int(time.time()),
        "ttl": int(time.time()) + SEEN_TTL_DAYS * 86400,
    })
    return attempts


def record_notify_day(job_id, cv_key=None):
    item = _get_seen_item(job_id) or {}
    match_dates = list(item.get("match_dates", []))
    today = _today_str()
    if today not in match_dates:
        match_dates.append(today)
    ddb.put_item(Item={
        "job_id": job_id,
        "attempts": item.get("attempts", 0),
        "matched": True,
        "last_score": item.get("last_score"),
        "last_reasoning": item.get("last_reasoning"),
        "match_dates": match_dates,
        "cv_key": cv_key if cv_key is not None else item.get("cv_key"),
        "seen_at": int(time.time()),
        "ttl": int(time.time()) + SEEN_TTL_DAYS * 86400,
    })
    return len(match_dates)


def get_master_cv():
    obj = s3.get_object(Bucket=S3_BUCKET, Key="cv/master_cv.txt")
    return obj["Body"].read().decode("utf-8")


def _chat_with_retry(messages, json_mode=False):
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
            logger.error(f"Groq APIStatusError (not retrying): {e}")
            raise
    raise last_err


def _location_instructions(job):
    if job["match_tier"] != "remote_location_uncertain":
        return ""
    return f"""
This job's listed location/region eligibility is: {job['location_raw']!r}.
The candidate is based in {CANDIDATE_LOCATION}. Judge whether this listed
restriction plausibly covers the candidate — for example, a restriction to
a specific country, named region, or set of countries that does not
include {CANDIDATE_LOCATION} or a broader region containing it (e.g.
worldwide, global, Africa) means the candidate is NOT eligible. A real
location/eligibility restriction that excludes the candidate is a HARD
DISQUALIFIER: cap match_score at 2 or below regardless of skill fit, and
say so plainly in the reasoning. Only score normally on skill fit if the
listed location is genuinely ambiguous about whether it includes the
candidate (e.g. it's unclear if "remote" here means payroll-region only
vs. a hard residency requirement) — in that case, note the ambiguity in
the reasoning rather than assuming disqualification.
"""


def _job_block(job, index):
    return f"""
--- Job {index} (id: {job['id']}) ---
Title: {job['title']}
Company: {job['company']}
Stated seniority: {job['seniority']} (titles like "Senior" are used loosely across companies and don't
always require senior-level experience — judge actual fit from the description and required skills,
not just the title label. If the role clearly requires far more experience than the CV shows, reflect
that with a lower score rather than rejecting it outright.)
Work arrangement: {job['match_tier']}
{_location_instructions(job)}
Description: {(job.get('description') or '')[:DESCRIPTION_CHAR_LIMIT]}
"""


def score_jobs_batch(master_cv, jobs_batch):
    jobs_text = "\n".join(_job_block(job, i) for i, job in enumerate(jobs_batch))
    prompt = f"""You are screening job postings against a candidate's CV.
For EACH job below, return a match_score (1-10) and a one-sentence reasoning,
following the seniority and location instructions given per job.

CV:
{master_cv}

Jobs:
{jobs_text}

Return ONLY valid JSON: {{"results": [{{"id": "<job id exactly as given>", "match_score": <1-10 integer>, "reasoning": "<one sentence>"}}, ...]}}
Include exactly one entry per job listed above, in any order.
"""
    response = _chat_with_retry(
        messages=[{"role": "user", "content": prompt}],
        json_mode=True,
    )
    text = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(text)
        results = {r["id"]: r for r in parsed.get("results", [])}
    except (json.JSONDecodeError, KeyError, TypeError):
        results = {}
    return {
        job["id"]: results.get(job["id"], {"match_score": 0, "reasoning": "Could not parse model output"})
        for job in jobs_batch
    }


def _tailor_job_block(job, index):
    sponsorship_note = ""
    if job["match_tier"] == "in_person_visa_sponsorship":
        sponsorship_note = (
            "This is an in-person role where the employer offers visa sponsorship. "
            "Where natural, the CV may reflect openness to relocation, but do not "
            "invent statements about visa status or authorization.\n"
        )
    return f"""
--- Application {index} (id: {job['id']}) ---
Job title: {job['title']}
Company: {job['company']}
{sponsorship_note}Job description: {(job.get('description') or '')[:DESCRIPTION_CHAR_LIMIT]}
"""


def tailor_cvs_batch(master_cv, jobs_batch):
    jobs_text = "\n".join(_tailor_job_block(job, i) for i, job in enumerate(jobs_batch))
    prompt = f"""Rewrite the CV below into a separate, tailored version for EACH job application listed below.
Keep every version truthful — only reorder, re-emphasize, and reword existing experience.
Do not invent skills or experience that aren't in the original CV.

Original CV:
{master_cv}

Applications:
{jobs_text}

For EACH application above, output its tailored CV wrapped EXACTLY like this, with no other commentary
before, between, or after the blocks:
===APPLICATION id=<id exactly as given above>===
<the full tailored CV as plain text>
===END===

Output one such block per application listed above, nothing else.
"""
    response = _chat_with_retry(
        messages=[{"role": "user", "content": prompt}],
        json_mode=False,
    )
    text = response.choices[0].message.content

    results = {}
    for m in re.finditer(
        r"===APPLICATION id=(?P<id>.*?)===\s*(?P<cv>.*?)\s*===END===",
        text,
        re.DOTALL,
    ):
        results[m.group("id").strip()] = m.group("cv").strip()
    return results


def _presigned_cv_url(key):
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=CV_LINK_EXPIRY_S,
    )


def store_tailored_cv(job_id, content):
    key = f"tailored-cvs/{job_id.replace(':', '_')}.txt"
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=content.encode("utf-8"))
    return key, _presigned_cv_url(key)


def send_digest(matches, failed_count):
    if not matches and not failed_count:
        return
    body_lines = []
    if matches:
        body_lines.append(f"Job matches ({NOTIFY_DAYS_REQUIRED}-day reminder cycle):\n")
        for m in matches:
            tier_label = TIER_LABELS.get(m["match_tier"], m["match_tier"])
            day_label = (
                "First alert" if m["notify_day"] == 1
                else f"Reminder {m['notify_day']}/{NOTIFY_DAYS_REQUIRED}"
            )
            body_lines.append(
                f"- {m['title']} @ {m['company']} (score {m['score']}/10) — {day_label}\n"
                f"  {tier_label}\n"
                f"  {m['reasoning']}\n"
                f"  Listing: {m['url']}\n"
                f"  Tailored CV: {m['cv_url']}\n"
            )
    if failed_count:
        body_lines.append(
            f"\nNote: {failed_count} job(s) could not be scored/tailored due to a "
            f"temporary Groq API issue and will be retried on the next run.\n"
        )
    body = "\n".join(body_lines)
    ses.send_email(
        Source=SES_SENDER,
        Destination={"ToAddresses": [SES_RECIPIENT]},
        Message={
            "Subject": {"Data": f"Job Scout: {len(matches)} match(es) today"},
            "Body": {"Text": {"Data": body}},
        },
    )


def handler(event, context):
    if not acquire_lock():
        logger.warning("Another invocation is already running — skipping this one.")
        return {"statusCode": 200, "body": json.dumps({"skipped": "lock_held"})}

    try:
        master_cv = get_master_cv()
        jobs = fetch_jobs()

        to_score = []
        needs_tailoring = []  # [(job, score, reasoning), ...]
        to_renotify = []  # [(job, item), ...]
        skipped_max_attempts = 0
        skipped_fully_seen = 0

        for job in jobs:
            item = _get_seen_item(job["id"])
            if item is None:
                to_score.append(job)
                continue
            if item.get("matched"):
                if len(item.get("match_dates", [])) >= NOTIFY_DAYS_REQUIRED:
                    skipped_fully_seen += 1
                    continue
                if not item.get("cv_key"):
                    needs_tailoring.append((job, item.get("last_score"), item.get("last_reasoning")))
                else:
                    to_renotify.append((job, item))
                continue
            if item.get("attempts", 0) >= MAX_SCORE_ATTEMPTS:
                skipped_max_attempts += 1
                continue
            to_score.append(job)

        matches = []
        failed_count = 0
        no_match_count = 0

        for i in range(0, len(to_score), SCORE_BATCH_SIZE):
            batch = to_score[i:i + SCORE_BATCH_SIZE]
            try:
                results = score_jobs_batch(master_cv, batch)
            except (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError) as e:
                logger.error(f"Skipping batch of {len(batch)} jobs after scoring failure: {e}")
                failed_count += len(batch)
                continue

            for job in batch:
                job_id = job["id"]
                result = results[job_id]
                score = result.get("match_score", 0)
                reasoning = result.get("reasoning", "")
                matched = score >= MATCH_THRESHOLD
                attempts = record_attempt(job_id, matched, score, reasoning)

                if matched:
                    logger.info(
                        f"MATCH job_id={job_id} title={job['title']!r} "
                        f"company={job['company']!r} score={score} attempt={attempts} "
                        f"reasoning={reasoning!r}"
                    )
                    needs_tailoring.append((job, score, reasoning))
                else:
                    no_match_count += 1
                    logger.info(
                        f"NO MATCH job_id={job_id} title={job['title']!r} "
                        f"company={job['company']!r} source={job['source']} "
                        f"seniority={job['seniority']} match_tier={job['match_tier']} "
                        f"score={score} threshold={MATCH_THRESHOLD} "
                        f"attempt={attempts}/{MAX_SCORE_ATTEMPTS} reasoning={reasoning!r}"
                    )

        for i in range(0, len(needs_tailoring), TAILOR_BATCH_SIZE):
            batch = needs_tailoring[i:i + TAILOR_BATCH_SIZE]
            jobs_only = [item[0] for item in batch]
            try:
                tailored_map = tailor_cvs_batch(master_cv, jobs_only)
            except (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError) as e:
                logger.error(f"CV tailoring failed for batch of {len(batch)} jobs: {e}")
                failed_count += len(batch)
                continue

            for job, score, reasoning in batch:
                tailored_text = tailored_map.get(job["id"])
                if not tailored_text:
                    logger.error(f"CV tailoring produced no output for job {job['id']}")
                    failed_count += 1
                    continue
                cv_key, cv_url = store_tailored_cv(job["id"], tailored_text)
                notify_day = record_notify_day(job["id"], cv_key)

                safe_score = score if score is not None else "N/A"
                safe_reasoning = reasoning if reasoning else "No reasoning recorded."

                matches.append({
                    "title": job["title"],
                    "company": job["company"],
                    "score": safe_score,
                    "reasoning": safe_reasoning,
                    "url": job["url"],
                    "cv_url": cv_url,
                    "match_tier": job["match_tier"],
                    "notify_day": notify_day,
                })

        for job, item in to_renotify:
            cv_url = _presigned_cv_url(item["cv_key"])
            notify_day = record_notify_day(job["id"], item["cv_key"])

            raw_score = item.get("last_score") if item.get("last_score") is not None else item.get("score")
            raw_reasoning = item.get("last_reasoning") or item.get("reasoning")

            safe_score = raw_score if raw_score is not None else "N/A"
            safe_reasoning = raw_reasoning if raw_reasoning else "No reasoning recorded."

            matches.append({
                "title": job["title"],
                "company": job["company"],
                "score": safe_score,
                "reasoning": safe_reasoning,
                "url": job["url"],
                "cv_url": cv_url,
                "match_tier": job["match_tier"],
                "notify_day": notify_day,
            })

        send_digest(matches, failed_count)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "jobs_checked": len(jobs),
                "new_matches": len(matches),
                "no_match": no_match_count,
                "skipped_max_attempts": skipped_max_attempts,
                "skipped_fully_seen": skipped_fully_seen,
                "failed": failed_count,
            }),
        }
    finally:
        release_lock()