# Create a firewall rule to allow necessary ports for LexiFin services

resource "google_compute_firewall" "lexifin_firewall" {
  name    = "allow-lexifin-ports"
  network = "default"

  allow {
    protocol = "tcp"
    # Port mapping:
    # 22: SSH access
    # 6333: Qdrant Vector Database HTTP API
    # 7474: Neo4j Graph Database Browser UI 
    # 7687: Neo4j Graph Database Bolt protocol
    # 8501: Streamlit Web UI
    ports = ["22", "6333", "7474", "7687", "8501"]
  }

  source_ranges = [var.my_ip]
  target_tags   = ["lexifin-ports"]
}