# 📊 DATA ACCESS INSTRUCTIONS - BigQuery 2025 Competition

## Overview
This submission works for TWO types of users:
1. **Judges/Public** - Can run the demo notebook using pre-exported REAL data (no BigQuery access needed)
2. **MIMIC-IV Users** - Can run the full SQL pipeline with their own credentials

---

## 🎯 For Judges & Public Users (No BigQuery Access)

### Quick Start
1. **Download the complete dataset** (116 MB):
   - Google Drive: https://drive.google.com/drive/folders/1YCSzH2GA-GTf_x6JNOI4K4isayfZhUYK
   - Contains ALL 200,000 real matches + 15,000 real embeddings
   - Fully anonymized - No PHI or patient identifiers

2. **Run the notebook** (Choose one):
   - **Option A: Auto-Download Notebook** (Recommended for Judges)
     ```bash
     jupyter notebook demo_judge_complete.ipynb
     # This notebook auto-downloads data if not present
     ```
   - **Option B: Manual Setup**
     ```python
     # Set at the beginning of demo_bigquery_2025.ipynb
     USE_BIGQUERY = False  # Use exported data
     DATA_PATH = "./exported_data/"  # Local path after download
     ```

3. **What you'll see**:
   - All 200,000 real patient-trial matches
   - Real similarity scores and match quality
   - Real embeddings (768-dimensional vectors)
   - Performance metrics from actual runs
   - Explainability features showing WHY matches were made

### Dataset Contents

| File | Size | Rows | Description |
|------|------|------|-------------|
| `all_matches.csv` | 46 MB | 200,000 | Complete matching results with explanations |
| `all_patient_embeddings.parquet` | 46 MB | 10,000 | Patient profile embeddings (anonymized) |
| `all_trial_embeddings.parquet` | 24 MB | 5,000 | Clinical trial embeddings (public data) |
| `performance_metrics.json` | <1 MB | - | System performance statistics |
| `data_dictionary.json` | <1 MB | - | Column descriptions |

### Privacy & Safety
✅ **100% Safe** - No patient identifiers (no subject_id, hadm_id, names, dates)
✅ **Anonymous IDs** - MATCH_001, PATIENT_EMB_001 (not traceable)
✅ **Public Trials** - Real NCT IDs from ClinicalTrials.gov (public data)
✅ **Real Results** - Actual outputs from BigQuery pipeline

---

## 🔧 For Users with MIMIC-IV Access

### Prerequisites
1. **PhysioNet credentialed access to MIMIC-IV-3.1 in BigQuery**
   - Dataset: `physionet-data` project in BigQuery
   - Required tables:
     - `physionet-data.mimiciv_3_1_hosp.*` (hospital module)
     - `physionet-data.mimiciv_note.*` (clinical notes)
   - Sign up at: https://physionet.org/
   - Complete CITI training for access
   - Link your Google account to PhysioNet

2. **Google Cloud account with BigQuery enabled**
   - Must be the same account linked to PhysioNet

3. **Authentication**:
   ```bash
   gcloud auth application-default login
   # Use the email linked to your PhysioNet account
   ```

### Running Full Pipeline

1. **Clone the repository**:
   ```bash
   git clone [repository-url]
   cd SUBMISSION
   ```

2. **Update project ID in SQL files**:
   ```bash
   # Replace with your project ID
   sed -i 's/gen-lang-client-0017660547/YOUR_PROJECT_ID/g' sql_files/*.sql
   ```

3. **Execute SQL pipeline** (in order):
   ```bash
   # Run each SQL file in sequence
   for i in {01..10}; do
     bq query --use_legacy_sql=false < sql_files/${i}*.sql
   done
   ```

4. **Run notebook with live data**:
   ```python
   # Set in demo_bigquery_2025.ipynb
   USE_BIGQUERY = True  # Use live BigQuery
   PROJECT_ID = 'YOUR_PROJECT_ID'
   ```

### SQL Files Overview

1. `01_foundation_setup_complete.sql` - Create tables and import MIMIC data
2. `02_patient_profiling.sql` - Generate patient profiles
3. `03_vector_embeddings.sql` - Create embeddings (ML.GENERATE_EMBEDDING)
4. `04_vector_search_indexes.sql` - Build vector indexes (CREATE VECTOR INDEX)
5. `05_ai_functions.sql` - AI eligibility assessment (AI.GENERATE)
6. `06_matching_pipeline.sql` - Run semantic matching (VECTOR_SEARCH)
7. `07_ai_forecast.sql` - Predictive analytics
8. `08_applications.sql` - Real-world applications
9. `09_bigframes_integration.sql` - BigFrames setup
10. `10_validation_complete.sql` - Final validation

---

## 📈 Data Verification

### For Exported Data Users
```python
import pandas as pd
import json

# Load and verify data
matches = pd.read_csv('exported_data/all_matches.csv')
print(f"✅ Loaded {len(matches):,} real matches")
print(f"   Avg similarity: {matches['similarity_score'].mean():.3f}")
print(f"   Match distribution:")
print(matches['match_quality'].value_counts())

# Load performance metrics
with open('exported_data/performance_metrics.json', 'r') as f:
    metrics = json.load(f)
print(f"\n📊 Performance Metrics:")
print(f"   Total matches: {metrics['total_matches']:,}")
print(f"   Good matches: {metrics['good_matches']:,}")
```

### Expected Results
- **Total Matches**: 200,000
- **Average Similarity**: ~0.650
- **Match Distribution**:
  - WEAK_MATCH: ~105,770 (52.9%)
  - FAIR_MATCH: ~94,204 (47.1%)
  - GOOD_MATCH: ~26 (0.01%)

---

## 🔍 Understanding the Data

### Match Quality Explanations
Each match includes an explanation:
- **High semantic alignment** (>0.75): Strong conceptual match
- **Moderate alignment** (0.65-0.75): Relevant therapeutic area
- **Exploratory match** (<0.65): Potential cross-domain application

### Embeddings
- **Dimension**: 768 (using text-embedding-004 model)
- **Method**: Semantic encoding of clinical profiles
- **Distance**: Cosine similarity (higher = more similar)

### No Synthetic Data
⚠️ **Important**: This is REAL data from actual BigQuery runs
- Real similarity scores from vector search
- Real AI-generated explanations
- Real performance metrics
- Just anonymized for privacy

---

## 📞 Support

### Common Issues

| Issue | Solution |
|-------|----------|
| Can't download data | Check Google Drive link is public |
| Notebook won't run | Ensure USE_BIGQUERY = False for exported data |
| Missing columns | Use provided data_dictionary.json |
| BigQuery access denied | Run `gcloud auth application-default login` |

### File Structure
```
SUBMISSION/
├── exported_data/           # Complete dataset (114 MB)
│   ├── all_matches.csv
│   ├── all_patient_embeddings.parquet
│   ├── all_trial_embeddings.parquet
│   └── performance_metrics.json
├── sql_files/               # For users with MIMIC access
├── demo_bigquery_2025.ipynb # Works with both modes
└── export_all_data.py       # Script that generated the data
```

---

## ✅ Key Points

1. **Judges can verify everything** without BigQuery access
2. **All data is REAL** - no synthetic/sample data
3. **Complete dataset** - all 200K matches, not samples
4. **Privacy preserved** - no PHI or identifiers
5. **Reproducible** - users with access can regenerate

---

**Last Updated**: September 22, 2025
**Competition**: BigQuery 2025 Kaggle Hackathon
**Approach**: Semantic Detective 🕵️‍♀️