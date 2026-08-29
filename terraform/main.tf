data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------------------
# ECR - holds the Docker image built and pushed by the CI workflow
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "job_scout" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "job_scout" {
  repository = aws_ecr_repository.job_scout.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# DynamoDB - remembers which job IDs have already been processed
# ---------------------------------------------------------------------------
resource "aws_dynamodb_table" "seen_jobs" {
  name         = "${var.project_name}-seen-jobs"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "job_id"

  attribute {
    name = "job_id"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
}

# ---------------------------------------------------------------------------
# S3 - stores the master CV and every tailored CV generated
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "job_scout" {
  bucket = "${var.project_name}-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_public_access_block" "job_scout" {
  bucket                  = aws_s3_bucket.job_scout.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "job_scout" {
  bucket = aws_s3_bucket.job_scout.id
  versioning_configuration {
    status = "Enabled"
  }
}

# ---------------------------------------------------------------------------
# IAM - least-privilege execution role for the Lambda function
# ---------------------------------------------------------------------------
resource "aws_iam_role" "lambda_exec" {
  name = "${var.project_name}-lambda-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_app_permissions" {
  name = "${var.project_name}-app-permissions"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = ["${aws_s3_bucket.job_scout.arn}/*"]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
        ]
        Resource = [aws_dynamodb_table.seen_jobs.arn]
      },
      {
        Effect   = "Allow"
        Action   = ["ses:SendEmail"]
        Resource = "*"
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# Lambda - the container image built by CI, deployed here
# ---------------------------------------------------------------------------
resource "aws_lambda_function" "job_scout" {
  function_name = "${var.project_name}-auto"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.job_scout.repository_url}:${var.image_tag}"
  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout_s

  environment {
    variables = {
      GEMINI_API_KEY = var.gemini_api_key
      ADZUNA_APP_ID  = var.adzuna_app_id
      ADZUNA_APP_KEY = var.adzuna_app_key
      DDB_TABLE      = aws_dynamodb_table.seen_jobs.name
      S3_BUCKET      = aws_s3_bucket.job_scout.bucket
      SES_SENDER     = var.ses_sender
      SES_RECIPIENT  = var.ses_recipient
      JOB_QUERY      = var.job_query
      JOB_COUNTRY    = var.job_country
    }
  }
}

# ---------------------------------------------------------------------------
# EventBridge - daily schedule trigger
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_event_rule" "daily_schedule" {
  name                = "${var.project_name}-daily-schedule"
  schedule_expression = var.schedule_expression
}

resource "aws_cloudwatch_event_target" "job_scout" {
  rule = aws_cloudwatch_event_rule.daily_schedule.name
  arn  = aws_lambda_function.job_scout.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.job_scout.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.daily_schedule.arn
}
