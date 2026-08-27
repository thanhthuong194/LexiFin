# Provision a Google Compute Engine instance

resource "google_compute_instance" "lexifin_vm" {
  name         = "lexifin-os-server"
  machine_type = var.machine_type

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50 # 50 GB storage to handle raw SEC data and PDF files
    }
  }

  network_interface {
    network = "default"
    access_config {} # Assign an ephemeral public address
  }

  # Attach network tags to map with the firewall rule defined in network.tf
  tags = ["lexifin-ports"]

  # Startup script to automatically install Docker and Docker Compose on boot_disk
  metadata_startup_script = <<-EOF
      #!/bin/bash
      sudo apt-get update
      sudo apt-get install -y git curl
    
      # Install Docker Engine
      curl -fsSL https://get.docker.com -o get-docker.sh
      sudo sh get-docker.sh
      sudo usermod -aG docker $USER
    EOF
}