# Configuration Guide

## Overview
This folder contains configuration files for the BigQuery AI Clinical Trial Matching system.

## Files

### `default.config.json`
Default configuration template with placeholder values. Use this as a starting point.

### `user.config.json`
Your custom configuration. Copy from `default.config.json` and update with your values.

### `config_manager.py`
Python module that manages configuration loading and validation.

## Setup Instructions

### 1. Update Configuration Values

Replace the following placeholder values with your actual GCP settings:

```json
{
  "gcp": {
    "project_id": "YOUR_PROJECT_ID",  // Replace with your GCP project ID
    "service_account": "your-service-account@example.com"  // Your service account email
  },
  "api": {
    "cloud_run_url": "https://your-cloud-run-url.us-central1.run.app"  // Your Cloud Run URL
  }
}
```

### 2. Environment Variables (Alternative)

You can also use environment variables to override config values:

```bash
export GCP_PROJECT_ID="your-actual-project-id"
export GCP_DATASET_ID="clinical_trial_matching"
export GCP_REGION="us-central1"
```

### 3. MIMIC-IV Access

To use MIMIC-IV data, you need:
- PhysioNet credentialed access
- Link your Google account to PhysioNet
- Access to `physionet-data` project in BigQuery

## Important Notes

⚠️ **Security**: Never commit real credentials or project IDs to public repositories
⚠️ **Privacy**: The placeholder values are for demonstration only
✅ **Demo Mode**: The notebooks work with exported data without BigQuery access

## Configuration Sections

### GCP Settings
- `project_id`: Your Google Cloud project ID
- `region`: GCP region for resources
- `dataset_id`: BigQuery dataset name
- `service_account`: Service account for authentication

### BigQuery Settings
- `tables`: Table names for various data types
- `batch_size`: Processing batch size
- `timeout`: Query timeout in seconds

### ML Settings
- `embedding_model`: Model for generating embeddings
- `embedding_dim`: Dimension of embeddings (768)
- `gemini_model`: Gemini model for AI functions

### Vector Search
- `index_type`: Type of vector index (IVF/TREE_AH)
- `distance_metric`: Similarity metric (COSINE)

## For Judges

The submission includes:
1. **Exported data** in Google Drive (no BigQuery access needed)
2. **Auto-download notebook** that fetches data automatically
3. **Configuration templates** showing system architecture

You don't need to update these configs to review the submission!