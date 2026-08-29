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
# A matched job keeps appearing in the digest — reusing its existing tailored
# CV, no further Groq calls — until it's been included on this many distinct
# calendar dates, at which point should_skip() retires it for good.
NOTIFY_DAYS_REQUIRED = int(os.environ.get("NOTIFY_DAYS_REQUIRED", "3"))
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
CV_LINK_EXPIRY_S = int(os.environ.get("CV_LINK_EXPIRY_S", str(7 * 24 * 3600)))  # 7 days

# Candidate's actual location, used by score_jobs_batch() to judge whether a
# job's listed location/region restriction plausibly covers them, for
# jobs tagged "remote_location_uncertain" (see match_tier()).
CANDIDATE_LOCATION = os.environ.get("CANDIDATE_LOCATION", "Ghana")

# Job search config
JOB_QUERIES = [q.strip() for q in os.environ.get("JOB_QUERIES", "cloud engineer,devops engineer").split(",") if q.strip()]
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "100"))  # Remotive per-query limit
ARBEITNOW_MAX_PAGES = int(os.environ.get("ARBEITNOW_MAX_PAGES", "3"))
HIMALAYAS_MAX_PAGES = int(os.environ.get("HIMALAYAS_MAX_PAGES", "3"))

# Max chars of a job description sent to Groq for scoring/tailoring.
# Was 2000; raised to 4500 after test_sources.py's truncation diagnostic
# showed requirement/qualification content getting cut off for ~14% of a
# sample of Himalayas listings, with the worst case needing 4199 chars.
# Not a hard guarantee for all future postings — rerun the diagnostic
# periodically as real traffic comes in.
DESCRIPTION_CHAR_LIMIT = int(os.environ.get("DESCRIPTION_CHAR_LIMIT", "4500"))

# Retries for transient Groq errors (rate limit / connection / timeout),
# on top of whatever the SDK does internally
GROQ_MAX_ATTEMPTS = int(os.environ.get("GROQ_MAX_ATTEMPTS", "3"))
GROQ_RETRY_BACKOFF_S = int(os.environ.get("GROQ_RETRY_BACKOFF_S", "5"))

# How many jobs to score per Groq call. The CV is the single biggest
# recurring cost in a scoring prompt (it's resent in full every call) — 
# batching means it's sent once per SCORE_BATCH_SIZE jobs instead of once
# per job, which is the main lever for staying inside Groq's daily token
# budget. Keep this small enough that one batch's prompt (CV + N job
# descriptions) still fits comfortably under the account's per-minute
# token limit; tune down if a single batch call starts getting 429s on
# tokens-per-minute rather than tokens-per-day.
SCORE_BATCH_SIZE = int(os.environ.get("SCORE_BATCH_SIZE", "4"))

# Same batching principle applied to CV tailoring: the master CV is the
# shared input across every matched job in a run, so it's sent once per
# TAILOR_BATCH_SIZE matches instead of once per match. Output doesn't
# shrink this way (each match still needs its own full tailored CV), so
# the savings are smaller than on the scoring side, but still worth it
# whenever a run produces more than one match.
TAILOR_BATCH_SIZE = int(os.environ.get("TAILOR_BATCH_SIZE", "3"))

USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; JobScoutBot/1.0)"}
REMOTE_LOCATION_MARKERS = ("worldwide", "anywhere", "global")

# Used only to tag seniority for the LLM's benefit — NOT to filter jobs out.
SENIOR_TITLE_MARKERS = ("senior", "sr.", "sr ", "lead", "staff", "principal", "manager", "director", "head of")
JUNIOR_TITLE_MARKERS = ("junior", "jr.", "jr ", "entry", "entry-level", "graduate", "associate")

TIER_LABELS = {
    "no_auth_required": "Remote — worldwide, no work permit needed",
    "remote_location_uncertain": "Remote — location eligibility judged by model",
    "in_person_visa_sponsorship": "In-person — visa sponsorship available",
}

# ---- concurrency lock ----
# Multiple overlapping invocations (overlapping schedule ticks, Lambda's
# automatic async retry, a stray manual invoke) previously ran in parallel
# and each burned through the shared Groq daily token quota independently,
# which is what actually exhausted the account, not any one run alone.
# This is a hard guarantee at the code level regardless of what triggers
# the extra invocation.
LOCK_JOB_ID = "__run_lock__"
LOCK_DURATION_S = 600  # generous vs. observed ~2-5 min run time; lets a crashed run self-heal instead of deadlocking forever

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
    # Overwrite rather than delete: the Lambda's execution role only has
    # PutItem on this table (already needed for record_attempt), not
    # DeleteItem, and there's no reason to add a permission just for this
    # when a put with an already-expired locked_until does the same job.
    now = int(time.time())
    ddb.put_item(Item={"job_id": LOCK_JOB_ID, "locked_until": now, "ttl": now + LOCK_DURATION_S})


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


def _seniority_from_himalayas(seniority_list, title):
    """Himalayas provides a real structured `seniority` enum array
    (e.g. ['Senior'], ['Entry-level']) — prefer it, fall back to title guess if empty.
    Verified against himalayas.app/docs/data-dictionary: values are
    Entry-level, Mid-level, Senior, Manager, Director, Executive (no
    'Junior' value exists in their enum, unlike our internal taxonomy)."""
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


# ---- raw fetchers ----

def _get_remotive_jobs():
    # Queries overlap (e.g. a job can match both "cloud engineer" and
    # "devops engineer"), so dedup by id across queries the same way
    # Arbeitnow/Himalayas dedup across pages.
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
    # Remotive's server-side "search" is loose (matches tags/category too),
    # so re-filter locally on title/description like the other sources.
    return [j for j in all_jobs if _keyword_match(j.get("title"), j.get("description"))]


def _get_remoteok_jobs():
    # RemoteOK's public /api endpoint does not support server-side keyword
    # filtering — a ?tags= param was tested empirically and made no
    # difference to the response, despite third-party scrapers implying
    # otherwise. This just pulls the latest ~100 jobs across every industry
    # and re-filters locally; low yield is expected and not a bug.
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
    # tag= is a real, documented server-side filter (confirmed against
    # Jobicy's official OpenAPI spec at github.com/Jobicy/remote-jobs-api,
    # and verified live: appliedFilters echoed back tag=devops and returned
    # genuinely devops-relevant titles). Query it once per JOB_QUERIES term,
    # like the other multi-query sources, deduping by id.
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
    # tag= is a real server-side filter, but still re-check locally in case
    # its matching is loose, same caveat as Remotive/Himalayas server-side search.
    return [j for j in all_jobs if _keyword_match(j.get("jobTitle"), j.get("jobExcerpt"))]


def _get_himalayas_jobs():
    """Uses the real search endpoint (himalayas.app/jobs/api/search), which
    is page-based. Paginated across HIMALAYAS_MAX_PAGES per query — the
    search endpoint's per-page size isn't documented, so we can't assume one
    page covers everything (mirrors Arbeitnow's break-on-empty pattern).
    Re-applies local keyword matching same as the other sources, since q= is
    a free-text query and may match tags/categories loosely, same caveat as
    Remotive's server-side search."""
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
                break  # no more pages
            new_on_this_page = 0
            for job in page_jobs:
                guid = job.get("guid")
                if guid and guid not in seen_guids:
                    seen_guids.add(guid)
                    all_jobs.append(job)
                    new_on_this_page += 1
            if new_on_this_page == 0:
                break  # page repeated the same content as before — past the real last page
    return [j for j in all_jobs if _keyword_match(j.get("title"), j.get("excerpt"))]


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


def normalize_himalayas(job):
    location_restrictions = job.get("locationRestrictions") or []
    return {
        "id": f"himalayas:{job.get('guid')}",
        "source": "himalayas",
        "title": job.get("title"),
        "company": job.get("companyName"),
        "url": job.get("applicationLink"),
        "location_raw": ", ".join(c.get("name", "") for c in location_restrictions) or "Worldwide",
        "remote": True,  # Himalayas is a remote-only job board
        "open_worldwide": len(location_restrictions) == 0,  # per Himalayas' documented schema
        "visa_sponsorship": None,
        "seniority": _seniority_from_himalayas(job.get("seniority"), job.get("title")),
        # Full HTML description preferred over the short excerpt — the excerpt
        # alone starves the scoring/tailoring prompts of real content.
        "description": job.get("description") or job.get("excerpt"),
    }


# ---- filter + tier + dedup ----
# Seniority is NOT filtered here — it's tagged and passed through so the LLM
# scoring step can weigh it against the candidate's actual CV, rather than a
# title-text guess silently dropping real listings.
#
# Location eligibility works the same way as of this change: open_worldwide
# is only a confident "yes" when the location text uses an explicit
# worldwide/anywhere/global marker (or is empty). A real region list (e.g.
# "USA", "Bulgaria", "LATAM, Argentina, Brazil, Mexico") is genuinely
# ambiguous from keyword matching alone — it might quietly exclude the
# candidate, or the "remote" tag might just mean payroll/timezone alignment
# rather than a hard nationality/residency restriction. Rather than silently
# drop those jobs, any remote job now passes through with its location
# tagged as match_tier "remote_location_uncertain", so Groq can judge
# eligibility explicitly against the candidate's actual location in
# score_jobs_batch() below — treating a real, listed restriction that
# doesn't plausibly cover the candidate as a disqualifier, not just a
# skill-fit minus (unlike the softer seniority judgment).

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
    """Pull remote/visa-eligible listings from five free job APIs, filtered and deduped."""
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
    # Ghana and UTC are the same offset, so this lines up with the
    # candidate's local calendar day without any conversion.
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def should_skip(job_id):
    """A job is fully "seen" and done — never scored, tailored, or emailed
    about again — once either:
      - it has matched and been included in the digest on
        NOTIFY_DAYS_REQUIRED separate calendar dates, or
      - it has been scored MAX_SCORE_ATTEMPTS times without ever matching.
    LLM scoring has some run-to-run variance, so a single unlucky score
    shouldn't permanently bury a job that might score above threshold on a
    later attempt.

    Note: handler() re-implements this same partitioning inline (it also
    needs the "matched but not yet retired" case, which isn't a pure
    skip/don't-skip decision) — kept here too since it documents the
    retirement rule on its own and is a natural helper for anything else
    (tests, a future admin script) that just needs a yes/no per job.
    """
    item = _get_seen_item(job_id)
    if item is None:
        return False
    if item.get("matched"):
        return len(item.get("match_dates", [])) >= NOTIFY_DAYS_REQUIRED
    return item.get("attempts", 0) >= MAX_SCORE_ATTEMPTS


def record_attempt(job_id, matched, score=None, reasoning=None):
    """Increment this job's scoring-attempt count and record the outcome.
    Preserves match_dates/cv_key from any prior record — this only tracks
    scoring, not digest notifications (see record_notify_day for that).
    Returns the new attempt count (used for logging)."""
    item = _get_seen_item(job_id) or {}
    attempts = item.get("attempts", 0) + 1
    ddb.put_item(Item={
        "job_id": job_id,
        "attempts": attempts,
        "matched": matched or item.get("matched", False),
        "last_score": score if score is not None else item.get("last_score"),
        "last_reasoning": reasoning if reasoning is not None else item.get("last_reasoning"),
        "match_dates": item.get("match_dates", []),
        "cv_key": item.get("cv_key"),
        "seen_at": int(time.time()),
        "ttl": int(time.time()) + SEEN_TTL_DAYS * 86400,
    })
    return attempts


def record_notify_day(job_id, cv_key=None):
    """Record that this job was just included in the digest email. Adds
    today's date to match_dates if it isn't already there (so re-running
    the Lambda twice in one day doesn't double-count a notify day), and
    stores cv_key the first time it's tailored so later notify days can
    reuse the same tailored CV without calling Groq again.
    Returns the new count of distinct notify dates (used for logging)."""
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
    """Score several jobs against the CV in a single Groq call.

    The CV is the single biggest recurring cost in every scoring request —
    batching means it's only sent once per SCORE_BATCH_SIZE jobs instead of
    once per job, which is the main lever for staying inside Groq's daily
    token budget.

    Seniority is given as context, not a hard filter. Location is
    different: for jobs tagged "remote_location_uncertain", a restriction
    that clearly excludes the candidate is a hard disqualifier (must not
    score above threshold on skill fit alone) — see _location_instructions.

    Returns a dict of job_id -> {"match_score": int, "reasoning": str}. A
    job missing from the model's response (bad JSON, model dropped an
    entry, etc.) falls back to a 0 score rather than silently vanishing —
    same as the previous per-job parse-failure fallback.
    """
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
    """Tailor the CV for several matched jobs in a single Groq call.

    The master CV is sent once for the whole batch rather than once per
    job — same reasoning as score_jobs_batch. Each application still gets
    back its own full, separately-tailored CV; only the input side (the
    shared CV) is deduplicated, not the output.

    Uses a plain-text delimiter format rather than JSON mode: a tailored
    CV is long free-form text that can contain quotes and newlines the
    model may not escape reliably inside a JSON string, so a delimiter the
    model just has to reproduce verbatim is more robust here than nested
    JSON would be.

    Returns a dict of job_id -> tailored CV text. A job missing from the
    parsed output (delimiter mismatch, model dropped an entry) is simply
    absent from the returned dict — callers should treat that as a
    per-job tailoring failure, not raise for the whole batch.
    """
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
    # job_id contains a colon (e.g. "remotive:12345") — safe as an S3 key component
    key = f"tailored-cvs/{job_id.replace(':', '_')}.txt"
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=content.encode("utf-8"))
    # Returns both — callers that need to re-notify on a later day store the
    # key (via record_notify_day) and re-derive a fresh presigned URL from
    # it later, since ExpiresIn is shorter than a job could stay eligible for.
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

        # Partition every job currently returned by the sources into three
        # buckets: needs scoring (never seen, or seen but not yet matched
        # and under MAX_SCORE_ATTEMPTS), needs tailoring (matched — either
        # just now or on an earlier run — but has no stored cv_key yet,
        # whether because it's brand new or because a previous tailoring
        # attempt failed), or needs a plain re-notify (matched AND already
        # has a cv_key, so no Groq call is needed at all). Anything past
        # NOTIFY_DAYS_REQUIRED distinct notify dates is fully retired.
        to_score = []
        needs_tailoring = []  # [(job, score, reasoning), ...] — always retried until it succeeds
        to_renotify = []  # [(job, item), ...] — no Groq calls needed, reuses stored cv_key
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
                    # Matched (this run or a previous one) but never
                    # successfully tailored — e.g. a prior tailoring batch
                    # hit a Groq rate limit. Keep retrying every run until
                    # it succeeds, using the score/reasoning from whenever
                    # it originally matched.
                    needs_tailoring.append((job, item.get("last_score"), item.get("last_reasoning")))
                else:
                    # Still within the notify window — always re-notify on
                    # invocation, even if this Lambda already ran (and
                    # emailed about this job) earlier today.
                    # record_notify_day() dedupes by calendar date, so
                    # multiple same-day invocations don't advance the
                    # retirement count faster than once per day; they just
                    # mean the candidate gets emailed more than once that
                    # day, which is fine.
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
                # Don't record_attempt — leave these unscored so they're retried
                # next run. Transient failures shouldn't burn a job's limited attempts.
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

        # Tailor every matched-but-untailored job in its own batches —
        # covers both brand-new matches from this run and jobs matched on
        # an earlier run whose tailoring previously failed. A failure here
        # just leaves cv_key unset, so these same jobs land back in
        # needs_tailoring next run too (should_skip won't retire a job
        # with 0 match_dates, so nothing is lost, just delayed).
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
                matches.append({
                    "title": job["title"],
                    "company": job["company"],
                    "score": score,
                    "reasoning": reasoning,
                    "url": job["url"],
                    "cv_url": cv_url,
                    "match_tier": job["match_tier"],
                    "notify_day": notify_day,
                })

        # Re-notify jobs matched AND already tailored on an earlier run: no
        # Groq calls at all — reuse the CV already stored, just regenerate
        # a fresh presigned URL (ExpiresIn is shorter than a job could stay
        # eligible for) and log today as another notify date.
        for job, item in to_renotify:
            cv_url = _presigned_cv_url(item["cv_key"])
            notify_day = record_notify_day(job["id"], item["cv_key"])
            matches.append({
                "title": job["title"],
                "company": job["company"],
                "score": item.get("last_score"),
                "reasoning": item.get("last_reasoning"),
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