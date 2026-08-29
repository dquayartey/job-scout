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

# ---- Config from environment (set by SAM template) ----
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
ADZUNA_APP_ID = os.environ["ADZUNA_APP_ID"]
ADZUNA_APP_KEY = os.environ["ADZUNA_APP_KEY"]
DDB_TABLE = os.environ["DDB_TABLE"]
S3_BUCKET = os.environ["S3_BUCKET"]
SES_SENDER = os.environ["SES_SENDER"]
SES_RECIPIENT = os.environ["SES_RECIPIENT"]
JOB_QUERY = os.environ.get("JOB_QUERY", "cloud engineer")
JOB_COUNTRY = os.environ.get("JOB_COUNTRY", "gb")  # Adzuna country code
MATCH_THRESHOLD = int(os.environ.get("MATCH_THRESHOLD", "7"))
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "20"))
SEEN_TTL_DAYS = int(os.environ.get("SEEN_TTL_DAYS", "30"))
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Retries for transient errors (rate limit / connection / timeout),
# on top of whatever the SDK does internally
GROQ_MAX_ATTEMPTS = int(os.environ.get("GROQ_MAX_ATTEMPTS", "3"))
GROQ_RETRY_BACKOFF_S = int(os.environ.get("GROQ_RETRY_BACKOFF_S", "5"))

s3 = boto3.client("s3")
ddb = boto3.resource("dynamodb").Table(DDB_TABLE)
ses = boto3.client("ses")
groq_client = Groq(api_key=GROQ_API_KEY)


def fetch_jobs():
    """Pull live listings from Adzuna's free job-search API."""
    url = f"https://api.adzuna.com/v1/api/jobs/{JOB_COUNTRY}/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": MAX_RESULTS,
        "what": JOB_QUERY,
        "content-type": "application/json",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json().get("results", [])


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
    prompt = f"""You are screening a job posting against a candidate's CV.
Return ONLY valid JSON: {{"match_score": <1-10 integer>, "reasoning": "<one sentence>"}}

CV:
{master_cv}

Job title: {job.get('title')}
Company: {job.get('company', {}).get('display_name', 'Unknown')}
Description: {job.get('description', '')[:2000]}
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
    prompt = f"""Rewrite the CV below to better match this specific job posting.
Keep it truthful — only reorder, re-emphasize, and reword existing experience.
Do not invent skills or experience that aren't in the original CV.
Return plain text only, no commentary.

Original CV:
{master_cv}

Job title: {job.get('title')}
Job description: {job.get('description', '')[:2000]}
"""
    response = _chat_with_retry(
        messages=[{"role": "user", "content": prompt}],
        json_mode=False,
    )
    return response.choices[0].message.content


def store_tailored_cv(job_id, content):
    key = f"tailored-cvs/{job_id}.txt"
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=content.encode("utf-8"))
    return key


def send_digest(matches, failed_count):
    if not matches and not failed_count:
        return
    body_lines = []
    if matches:
        body_lines.append("New job matches found:\n")
        for m in matches:
            body_lines.append(
                f"- {m['title']} @ {m['company']} (score {m['score']}/10)\n"
                f"  {m['reasoning']}\n"
                f"  Listing: {m['url']}\n"
                f"  Tailored CV: s3://{S3_BUCKET}/{m['cv_key']}\n"
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
        job_id = str(job.get("id"))
        if not job_id or already_seen(job_id):
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
                    "title": job.get("title"),
                    "company": job.get("company", {}).get("display_name", "Unknown"),
                    "score": result["match_score"],
                    "reasoning": result.get("reasoning", ""),
                    "url": job.get("redirect_url", ""),
                    "cv_key": cv_key,
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