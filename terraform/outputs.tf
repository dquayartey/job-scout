output "ecr_repository_url" {
  description = "Push Docker images here"
  value       = aws_ecr_repository.job_scout.repository_url
}

output "lambda_function_name" {
  value = aws_lambda_function.job_scout.function_name
}

output "s3_bucket_name" {
  description = "Upload master_cv.txt to cv/master_cv.txt in this bucket"
  value       = aws_s3_bucket.job_scout.bucket
}

output "dynamodb_table_name" {
  value = aws_dynamodb_table.seen_jobs.name
}
