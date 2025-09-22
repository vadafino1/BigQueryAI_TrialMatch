# BigQuery 2025 Competition SQL Files - CONSOLIDATED

## ✅ Clean Structure (ONE File Per Step)

We've consolidated all redundant files. Execute these 10 files in order:

### 📋 Main Pipeline (Execute 1-10 in Order)

1. **`01_foundation_setup_complete.sql`** ⭐ THE ONLY FOUNDATION FILE
   - Dataset creation
   - Data import (Snapshot OR PhysioNet direct)
   - Clinical trials import
   - Temporal transformation to 2025
   - Patient status assessment
   - Vertex AI models

2. **`02_patient_profiling.sql`**
   - Comprehensive patient profiles
   - Clinical feature extraction

3. **`03_vector_embeddings.sql`**
   - Patient embeddings (768-dim)
   - Trial embeddings

4. **`04_vector_search_indexes.sql`**
   - TreeAH indexes
   - Optimized vector search

5. **`05_ai_functions.sql`**
   - AI.GENERATE functions
   - ML capabilities

6. **`06_matching_pipeline.sql`**
   - Patient-trial matching logic
   - Eligibility evaluation

7. **`07_ai_forecast.sql`**
   - Enrollment predictions
   - Timeline forecasting

8. **`08_applications.sql`**
   - Use case implementations

9. **`09_bigframes_integration.sql`**
   - Python DataFrame integration

10. **`10_validation_complete.sql`**
    - Final system validation

## 🚀 Quick Start

### Option 1: Python Wrapper (Recommended)
```bash
cd /workspaces/Kaggle_BigQuerry2025/SUBMISSION
python run_foundation_import.py
```

### Option 2: Direct SQL Execution
```bash
# Replace PROJECT_ID in the SQL file first
bq query --use_legacy_sql=false < sql_files/01_foundation_setup_complete.sql
```

## 📦 Data Import Options

The `01_foundation_setup_complete.sql` supports TWO paths:

### Path A: From Snapshot (5 minutes)
- Uses pre-processed snapshot data
- Set `USE_SNAPSHOT = TRUE` in the SQL file
- Fastest option if snapshot is available

### Path B: From PhysioNet (2-3 hours)
- Direct import from MIMIC-IV source data
- Requires PhysioNet credentials
- Set `USE_SNAPSHOT = FALSE` in the SQL file
- Complete fresh import

## ⚙️ Configuration

Edit these variables in `01_foundation_setup_complete.sql`:

```sql
DECLARE PROJECT_ID STRING DEFAULT 'your-project-id';
DECLARE DATASET_ID STRING DEFAULT 'clinical_trial_matching';
DECLARE USE_SNAPSHOT BOOL DEFAULT FALSE;  -- Set to TRUE if you have snapshot
```

## 📊 Expected Data Volumes

After successful import:
- **364,627** patients from MIMIC-IV
- **66,966** clinical trials from ClinicalTrials.gov
- **162,416+** Active_Ready patients
- **40M+** lab results with 2025 dates
- **16M+** medications with current status
- **2.3M+** radiology reports

## 🗂️ Archived Files

Old/redundant files have been moved to `archived/` folder:
- Multiple `01_foundation_*` versions (now consolidated)
- Temporary transformation files
- Test snippets

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| PhysioNet access denied | Ensure you have PhysioNet credentials and access |
| Snapshot dataset not found | Use Path B (direct PhysioNet import) |
| Dataset already exists | Script handles this automatically |
| Vertex AI connection errors | Models are optional, can skip Section 6 |
| PROJECT_ID not replaced | Replace all instances of PROJECT_ID and DATASET_ID |

## 🏃 Running the Complete Pipeline

```bash
# Step 1: Import foundation data
python run_foundation_import.py

# Step 2: Import clinical trials (if not done in step 1)
python python_files/import_clinical_trials_comprehensive.py

# Step 3: Run remaining SQL files (2-10)
for i in {2..10}; do
  file=$(printf "%02d" $i)
  echo "Running $file..."
  bq query --use_legacy_sql=false < sql_files/${file}_*.sql
done
```

## 📝 Notes

- All functionality has been consolidated into these 10 files
- The `01_foundation_setup_complete.sql` replaces ALL previous 01_* files
- Files are designed to be idempotent (safe to re-run)
- Each file validates its outputs at the end

---
**Last Updated**: September 2025
**Competition**: BigQuery 2025 Clinical Trial Matching