terraform {
  required_version = ">= 1.7.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Local state by default so the project runs out of the box.
  # For real use, switch to a remote backend, e.g.:
  #
  # backend "s3" {
  #   bucket = "your-terraform-state-bucket"
  #   key    = "job-scout/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
}
