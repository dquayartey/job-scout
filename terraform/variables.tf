variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefix used for naming resources"
  type        = string
  default     = "job-scout"
}

variable "image_tag" {
  description = "Tag of the Docker image in ECR to deploy (set by CI after build & push)"
  type        = string
  default     = "latest"
}

variable "gemini_api_key" {
  description = "Google AI Studio Gemini API key"
  type        = string
  sensitive   = true
}

variable "adzuna_app_id" {
  description = "Adzuna API app ID"
  type        = string
  sensitive   = true
}

variable "adzuna_app_key" {
  description = "Adzuna API app key"
  type        = string
  sensitive   = true
}

variable "ses_sender" {
  description = "Verified SES sender email address"
  type        = string
}

variable "ses_recipient" {
  description = "Verified SES recipient email address"
  type        = string
}

variable "job_query" {
  description = "Job search query, e.g. 'remote cloud engineer'"
  type        = string
  default     = "cloud engineer"
}

variable "job_country" {
  description = "Adzuna 2-letter country code"
  type        = string
  default     = "gb"
}

variable "schedule_expression" {
  description = "EventBridge schedule expression for how often Job Scout runs"
  type        = string
  default     = "cron(0 8 * * ? *)"
}

variable "lambda_memory_mb" {
  description = "Lambda memory allocation in MB"
  type        = number
  default     = 256
}

variable "lambda_timeout_s" {
  description = "Lambda timeout in seconds"
  type        = number
  default     = 300
}
