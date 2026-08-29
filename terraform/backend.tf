terraform {
  backend "s3" {
    bucket         = "job-scout-terraform-state-677276098035"
    key            = "job-scout/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "job-scout-terraform-locks"
    encrypt        = true
  }
}