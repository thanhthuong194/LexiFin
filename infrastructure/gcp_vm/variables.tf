variable "project_id" {
  description = "The unique ID of the GCP project"
  type        = string
}

variable "region" {
  description = "The default region for GCP resources"
  type        = string
  default     = "asia-southeast1" # Singapore region for low latency to VN
}

variable "zone" {
  description = "The default zone for GCP resources"
  type        = string
  default     = "asia-southeast1-a"
}

variable "machine_type" {
  description = "The machine  type for the Compute Engine instance"
  type        = string
  default     = "e2-standard-4" # 4 vCPUs, 16GB RAM for heavy data processing
}

variable "my_ip" {
  description = "Personal IP"
  type        = string
  default     = "0.0.0.0/0"
}