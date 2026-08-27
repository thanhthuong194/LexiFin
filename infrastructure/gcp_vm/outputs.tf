# Output the public IP address of the instance

output "vm_external_ip" {
  description = "The public IP address of the VM for SSH and web interface access"
  value       = google_compute_instance.lexifin_vm.network_interface[0].access_config[0].nat_ip
}