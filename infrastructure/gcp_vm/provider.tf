# Define the Google Cloud provider configuration

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}