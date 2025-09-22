#!/usr/bin/env python3
"""
Complete Pipeline for BigQuery 2025 Competition
===============================================
This script runs the entire data import and processing pipeline:
1. Import MIMIC-IV data from PhysioNet
2. Apply temporal transformation to 2025
3. Import clinical trials from ClinicalTrials.gov
4. Generate embeddings (optional)
5. Create match scores (optional)

Usage: python complete_pipeline.py [--skip-embeddings] [--skip-matches]
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from google.cloud import bigquery

# Configuration
PROJECT_ID = "gen-lang-client-0017660547"
DATASET_ID = "clinical_trial_matching"
LOCATION = "US"

# Status file to track progress
STATUS_FILE = Path("pipeline_status.json")

def load_status() -> Dict:
    """Load pipeline status from file"""
    if STATUS_FILE.exists():
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    return {
        "dataset_created": False,
        "mimic_imported": False,
        "temporal_transformed": False,
        "trials_imported": False,
        "embeddings_generated": False,
        "matches_created": False,
        "last_updated": None
    }

def save_status(status: Dict):
    """Save pipeline status to file"""
    status["last_updated"] = datetime.now().isoformat()
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)

def run_command(cmd: str, description: str) -> bool:
    """Run a shell command and return success status"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout[:1000])  # Print first 1000 chars
        return True
    else:
        print(f"❌ {description} failed")
        print(f"Error: {result.stderr[:1000]}")
        return False

def check_table_exists(table_id: str) -> bool:
    """Check if a BigQuery table exists and has data"""
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{table_id}"

    try:
        table = client.get_table(table_ref)
        query = f"SELECT COUNT(*) as count FROM `{table_ref}` LIMIT 1"
        result = client.query(query).result()
        count = list(result)[0].count
        print(f"  Table {table_id}: {count:,} rows")
        return count > 0
    except Exception as e:
        print(f"  Table {table_id}: not found or empty")
        return False

def create_dataset():
    """Create BigQuery dataset if it doesn't exist"""
    client = bigquery.Client(project=PROJECT_ID)
    dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"

    try:
        client.get_dataset(dataset_ref)
        print(f"✅ Dataset {DATASET_ID} already exists")
        return True
    except:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = LOCATION
        dataset.description = "Clinical trial matching dataset for BigQuery 2025 competition"
        client.create_dataset(dataset)
        print(f"✅ Created dataset {DATASET_ID}")
        return True

def import_mimic_data():
    """Import MIMIC-IV data from PhysioNet"""
    # Check if data already exists
    if check_table_exists("discharge_summaries"):
        client = bigquery.Client(project=PROJECT_ID)
        query = f"SELECT COUNT(*) as count FROM `{PROJECT_ID}.{DATASET_ID}.discharge_summaries`"
        result = client.query(query).result()
        count = list(result)[0].count
        if count > 100000:  # If we have substantial data
            print(f"✅ MIMIC data already imported: {count:,} discharge summaries")
            return True

    print("Importing MIMIC data from PhysioNet...")

    # Create the corrected SQL with proper dataset names
    sql_queries = [
        # Discharge summaries
        f"""
        CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.discharge_summaries` AS
        WITH discharge_notes AS (
          SELECT
            CAST(n.subject_id AS STRING) AS patient_id,
            CAST(n.hadm_id AS STRING) AS hadm_id,
            n.note_id,
            n.charttime AS note_datetime,
            n.storetime AS storage_datetime,
            n.text AS discharge_text,
            LENGTH(n.text) AS text_length
          FROM `physionet-data.mimiciv_note.discharge` n
        ),
        diagnosis_data AS (
          SELECT
            CAST(d.subject_id AS STRING) AS patient_id,
            CAST(d.hadm_id AS STRING) AS hadm_id,
            STRING_AGG(
              CONCAT(d.icd_code, ': ', dd.long_title),
              '; ' ORDER BY d.seq_num LIMIT 10
            ) AS discharge_diagnosis
          FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
          LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses` dd
            ON d.icd_code = dd.icd_code AND d.icd_version = dd.icd_version
          GROUP BY d.subject_id, d.hadm_id
        )
        SELECT
          n.patient_id,
          n.hadm_id,
          n.note_id,
          n.note_datetime,
          n.storage_datetime,
          n.discharge_text,
          n.text_length,
          d.discharge_diagnosis,
          CASE WHEN d.discharge_diagnosis LIKE '%C%' OR LOWER(n.discharge_text) LIKE '%cancer%' THEN TRUE ELSE FALSE END AS mentions_cancer,
          CASE WHEN d.discharge_diagnosis LIKE '%E11%' OR LOWER(n.discharge_text) LIKE '%diabetes%' THEN TRUE ELSE FALSE END AS mentions_diabetes,
          CASE WHEN d.discharge_diagnosis LIKE '%I50%' OR LOWER(n.discharge_text) LIKE '%heart failure%' THEN TRUE ELSE FALSE END AS mentions_heart_failure,
          CURRENT_TIMESTAMP() AS imported_at
        FROM discharge_notes n
        LEFT JOIN diagnosis_data d
          ON n.patient_id = d.patient_id AND n.hadm_id = d.hadm_id
        WHERE n.discharge_text IS NOT NULL
        """,

        # Lab events
        f"""
        CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.lab_events` AS
        SELECT
          CAST(l.subject_id AS STRING) AS patient_id,
          CAST(l.hadm_id AS STRING) AS hadm_id,
          l.itemid,
          l.charttime,
          l.storetime,
          l.valuenum AS lab_value_numeric,
          l.valueuom AS lab_unit,
          l.ref_range_lower,
          l.ref_range_upper,
          l.flag AS abnormal_flag,
          d.label AS lab_test_name,
          CASE
            WHEN d.label LIKE '%hemoglobin%' THEN 'hemoglobin'
            WHEN d.label LIKE '%platelet%' THEN 'platelets'
            WHEN d.label LIKE '%creatinine%' THEN 'creatinine'
            WHEN d.label LIKE '%glucose%' THEN 'glucose'
            WHEN d.label LIKE '%white blood%' THEN 'wbc'
            ELSE 'other'
          END AS lab_test_category,
          CURRENT_TIMESTAMP() AS imported_at
        FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
        INNER JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` d
          ON l.itemid = d.itemid
        WHERE l.valuenum IS NOT NULL
          AND l.hadm_id IS NOT NULL
          AND d.label IN (
            'Hemoglobin', 'Hematocrit', 'Platelet Count', 'Creatinine',
            'Glucose', 'Sodium', 'Potassium', 'White Blood Cells',
            'Bilirubin, Total', 'Alanine Aminotransferase (ALT)',
            'Aspartate Aminotransferase (AST)', 'INR(PT)'
          )
        """,

        # Medications
        f"""
        CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.medications` AS
        SELECT
          CAST(p.subject_id AS STRING) AS patient_id,
          CAST(p.hadm_id AS STRING) AS hadm_id,
          p.starttime,
          p.stoptime,
          p.drug AS medication,
          p.dose_val_rx AS dose_value,
          p.dose_unit_rx AS dose_unit,
          p.route,
          CAST(p.doses_per_24_hrs AS STRING) AS frequency,
          CASE
            WHEN LOWER(p.drug) LIKE '%insulin%' OR LOWER(p.drug) LIKE '%metformin%' THEN 'diabetes'
            WHEN LOWER(p.drug) LIKE '%statin%' OR LOWER(p.drug) LIKE '%aspirin%' THEN 'cardiovascular'
            WHEN LOWER(p.drug) LIKE '%chemotherapy%' THEN 'oncology'
            ELSE 'other'
          END AS medication_category,
          CURRENT_TIMESTAMP() AS imported_at
        FROM `physionet-data.mimiciv_3_1_hosp.prescriptions` p
        WHERE p.hadm_id IS NOT NULL
        """,

        # Radiology reports
        f"""
        CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.radiology_reports` AS
        SELECT
          CAST(r.subject_id AS STRING) AS patient_id,
          CAST(r.hadm_id AS STRING) AS hadm_id,
          r.note_id,
          r.charttime AS study_datetime,
          r.text AS report_text,
          CASE
            WHEN UPPER(r.text) LIKE '%CT %' THEN 'CT'
            WHEN UPPER(r.text) LIKE '%MRI%' THEN 'MRI'
            WHEN UPPER(r.text) LIKE '%X-RAY%' THEN 'X-RAY'
            ELSE 'OTHER'
          END AS imaging_type,
          CASE WHEN UPPER(r.text) LIKE '%NORMAL%' THEN FALSE ELSE TRUE END AS has_findings,
          CASE WHEN UPPER(r.text) LIKE '%MASS%' OR UPPER(r.text) LIKE '%TUMOR%' THEN TRUE ELSE FALSE END AS mentions_mass,
          CURRENT_TIMESTAMP() AS imported_at
        FROM `physionet-data.mimiciv_note.radiology` r
        WHERE r.text IS NOT NULL AND r.hadm_id IS NOT NULL
        """
    ]

    # Execute each query
    client = bigquery.Client(project=PROJECT_ID)
    for i, query in enumerate(sql_queries, 1):
        table_name = ["discharge_summaries", "lab_events", "medications", "radiology_reports"][i-1]
        print(f"  Importing {table_name}...")
        try:
            job = client.query(query)
            job.result()  # Wait for completion
            print(f"    ✅ {table_name} imported")
        except Exception as e:
            print(f"    ❌ Failed to import {table_name}: {str(e)[:200]}")
            return False

    return True

def apply_temporal_transformation():
    """Apply temporal transformation to normalize dates to 2025"""
    # Check if already transformed
    if check_table_exists("discharge_summaries_2025"):
        print("✅ Temporal transformation already applied")
        return True

    # Run transformation script
    script_path = Path("python_files/temporal_transformation_2025.py")
    if script_path.exists():
        return run_command(f"python {script_path}", "Applying temporal transformation")
    else:
        # Use SQL transformation from the foundation script
        sql_file = Path("sql_files/01_foundation_setup_complete.sql")
        if sql_file.exists():
            with open(sql_file, 'r') as f:
                sql_content = f.read()

            # Replace placeholders
            sql_content = sql_content.replace("${PROJECT_ID}", PROJECT_ID)
            sql_content = sql_content.replace("${DATASET_ID}", DATASET_ID)

            # Extract temporal transformation section
            start_marker = "-- SECTION 4: TEMPORAL TRANSFORMATION TO 2025"
            end_marker = "-- SECTION 5:"

            start_idx = sql_content.find(start_marker)
            end_idx = sql_content.find(end_marker)

            if start_idx != -1 and end_idx != -1:
                transform_sql = sql_content[start_idx:end_idx]

                temp_sql = "/tmp/temporal_transform.sql"
                with open(temp_sql, 'w') as f:
                    f.write(transform_sql)

                cmd = f"bq query --use_legacy_sql=false --max_rows=0 < {temp_sql}"
                return run_command(cmd, "Applying temporal transformation")

    print("❌ Temporal transformation script not found")
    return False

def parse_age(age_str):
    """Parse age string to years"""
    if not age_str or age_str == "N/A":
        return None
    try:
        parts = age_str.split()
        if parts:
            age = int(parts[0])
            if "Month" in age_str:
                age = age // 12
            return age
    except:
        return None
    return None

def is_oncology_trial(conditions):
    """Check if trial is oncology-related"""
    oncology_keywords = ["cancer", "tumor", "carcinoma", "lymphoma", "leukemia", "sarcoma", "melanoma", "neoplasm"]
    conditions_text = " ".join(conditions).lower()
    return any(keyword in conditions_text for keyword in oncology_keywords)

def is_diabetes_trial(conditions):
    """Check if trial is diabetes-related"""
    diabetes_keywords = ["diabetes", "diabetic", "insulin", "glucose", "glycemic"]
    conditions_text = " ".join(conditions).lower()
    return any(keyword in conditions_text for keyword in diabetes_keywords)

def is_cardiac_trial(conditions):
    """Check if trial is cardiac-related"""
    cardiac_keywords = ["heart", "cardiac", "cardiovascular", "coronary", "myocardial", "atrial", "ventricular"]
    conditions_text = " ".join(conditions).lower()
    return any(keyword in conditions_text for keyword in cardiac_keywords)

def import_clinical_trials():
    """Import clinical trials from ClinicalTrials.gov"""
    # Check if already imported with substantial data
    if check_table_exists("trials_comprehensive"):
        client = bigquery.Client(project=PROJECT_ID)
        count_query = f"SELECT COUNT(*) as count FROM `{PROJECT_ID}.{DATASET_ID}.trials_comprehensive`"
        result = client.query(count_query).result()
        for row in result:
            if row.count > 1000:  # More than our simple 559 trials
                print("✅ Clinical trials already imported")
                return True

    print("🔄 Importing clinical trials from ClinicalTrials.gov...")

    import requests
    import time

    # Fetch all recruiting trials using ClinicalTrials.gov API
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "filter.overallStatus": "RECRUITING",
        "pageSize": 100,
        "format": "json"
    }

    all_trials = []
    page_num = 1
    max_trials = 100000  # Fetch up to 100k trials

    while len(all_trials) < max_trials:
        print(f"  Fetching page {page_num} (have {len(all_trials)} trials so far)...")

        current_params = params.copy()
        if page_num > 1:
            current_params["pageToken"] = page_token

        try:
            response = requests.get(base_url, params=current_params)
            response.raise_for_status()
            data = response.json()

            studies = data.get("studies", [])
            if not studies:
                break

            for study in studies:
                protocol = study.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                eligibility = protocol.get("eligibilityModule", {})
                conditions_module = protocol.get("conditionsModule", {})
                design_module = protocol.get("designModule", {})

                # Extract basic fields matching our exact schema
                trial = {
                    "nct_id": id_module.get("nctId", ""),
                    "brief_title": id_module.get("briefTitle", ""),
                    "official_title": id_module.get("officialTitle", ""),
                    "overall_status": status_module.get("overallStatus", ""),
                    "conditions": ", ".join(conditions_module.get("conditions", [])),
                    "eligibility_criteria_full": eligibility.get("eligibilityCriteria", ""),
                    "min_age_years": parse_age(eligibility.get("minimumAge", "")),
                    "max_age_years": parse_age(eligibility.get("maximumAge", "")),
                    "gender": eligibility.get("sex", "ALL"),
                    "phase": ", ".join(design_module.get("phases", [])),
                    "enrollment_count": status_module.get("enrollmentInfo", {}).get("count", 0),
                    "hemoglobin_min": None,
                    "creatinine_max": None,
                    "platelets_min": None,
                    "is_oncology_trial": is_oncology_trial(conditions_module.get("conditions", [])),
                    "is_diabetes_trial": is_diabetes_trial(conditions_module.get("conditions", [])),
                    "is_cardiac_trial": is_cardiac_trial(conditions_module.get("conditions", [])),
                    "imported_at": None
                }
                all_trials.append(trial)

                if len(all_trials) >= max_trials:
                    break

            # Check for next page
            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

            page_token = next_page_token
            page_num += 1
            time.sleep(0.1)  # Rate limiting

        except Exception as e:
            print(f"    ⚠️ Error fetching page {page_num}: {e}")
            break

    # Remove duplicates
    unique_trials = {trial['nct_id']: trial for trial in all_trials}
    trials_to_insert = list(unique_trials.values())

    print(f"  Found {len(trials_to_insert)} unique recruiting trials")

    # Insert into BigQuery
    if trials_to_insert:
        client = bigquery.Client(project=PROJECT_ID)
        table_id = f"{PROJECT_ID}.{DATASET_ID}.trials_comprehensive"

        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            schema=[
                bigquery.SchemaField("nct_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("brief_title", "STRING"),
                bigquery.SchemaField("official_title", "STRING"),
                bigquery.SchemaField("overall_status", "STRING"),
                bigquery.SchemaField("conditions", "STRING"),
                bigquery.SchemaField("eligibility_criteria_full", "STRING"),
                bigquery.SchemaField("min_age_years", "INTEGER"),
                bigquery.SchemaField("max_age_years", "INTEGER"),
                bigquery.SchemaField("gender", "STRING"),
                bigquery.SchemaField("phase", "STRING"),
                bigquery.SchemaField("enrollment_count", "INTEGER"),
                bigquery.SchemaField("hemoglobin_min", "FLOAT"),
                bigquery.SchemaField("creatinine_max", "FLOAT"),
                bigquery.SchemaField("platelets_min", "FLOAT"),
                bigquery.SchemaField("is_oncology_trial", "BOOLEAN"),
                bigquery.SchemaField("is_diabetes_trial", "BOOLEAN"),
                bigquery.SchemaField("is_cardiac_trial", "BOOLEAN"),
                bigquery.SchemaField("imported_at", "TIMESTAMP"),
            ],
        )

        job = client.load_table_from_json(trials_to_insert, table_id, job_config=job_config)
        job.result()

        print(f"✅ Successfully imported {len(trials_to_insert)} clinical trials")
        return True
    else:
        print("❌ No trials to import")
        return False

def generate_embeddings():
    """Generate embeddings for patients and trials"""
    # Check if already generated
    if check_table_exists("patient_embeddings") and check_table_exists("trial_embeddings"):
        print("✅ Embeddings already generated")
        return True

    script_path = Path("python_files/generate_embeddings.py")
    if not script_path.exists():
        script_path = Path("python_files/generate_embeddings_simple.py")

    if script_path.exists():
        return run_command(f"python {script_path}", "Generating embeddings")
    else:
        print("⚠️ Embeddings script not found, skipping")
        return True

def create_matches():
    """Create patient-trial matches"""
    # Check if already created
    if check_table_exists("match_scores_real"):
        client = bigquery.Client(project=PROJECT_ID)
        query = f"SELECT COUNT(*) as count FROM `{PROJECT_ID}.{DATASET_ID}.match_scores_real`"
        result = client.query(query).result()
        count = list(result)[0].count
        if count > 1000000:  # Expect millions of matches
            print(f"✅ Matches already created: {count:,} matches")
            return True

    script_path = Path("python_files/generate_complete_matches.py")
    if not script_path.exists():
        script_path = Path("python_files/simple_matching_pipeline.py")

    if script_path.exists():
        return run_command(f"python {script_path}", "Creating patient-trial matches")
    else:
        print("⚠️ Matching script not found, skipping")
        return True

def verify_pipeline():
    """Verify the complete pipeline"""
    print("\n" + "="*70)
    print("  PIPELINE VERIFICATION")
    print("="*70)

    client = bigquery.Client(project=PROJECT_ID)

    critical_tables = [
        "discharge_summaries_2025",
        "lab_events_2025",
        "medications_2025",
        "patient_current_status_2025",
        "trials_comprehensive"
    ]

    optional_tables = [
        "patient_embeddings",
        "trial_embeddings",
        "match_scores_real"
    ]

    results = {}
    all_critical_present = True

    print("\n📊 Critical Tables:")
    for table in critical_tables:
        exists = check_table_exists(table)
        results[table] = exists
        if not exists:
            all_critical_present = False

    print("\n📊 Optional Tables:")
    for table in optional_tables:
        exists = check_table_exists(table)
        results[table] = exists

    # Get patient stats
    if results.get("patient_current_status_2025"):
        query = f"""
        SELECT
            patient_status,
            COUNT(*) as count
        FROM `{PROJECT_ID}.{DATASET_ID}.patient_current_status_2025`
        GROUP BY patient_status
        ORDER BY count DESC
        """
        result = client.query(query).result()
        print("\n📊 Patient Status Distribution:")
        for row in result:
            print(f"  {row.patient_status}: {row.count:,}")

    return all_critical_present

def main():
    """Main pipeline execution"""
    print("\n" + "="*70)
    print("  BIGQUERY 2025 COMPETITION - COMPLETE PIPELINE")
    print("="*70)
    print(f"\nProject: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Location: {LOCATION}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check command line arguments
    skip_embeddings = "--skip-embeddings" in sys.argv
    skip_matches = "--skip-matches" in sys.argv

    if skip_embeddings:
        print("\n⚠️ Skipping embeddings generation")
    if skip_matches:
        print("⚠️ Skipping match creation")

    # Load status
    status = load_status()

    try:
        # Step 1: Create dataset
        if not status["dataset_created"]:
            if create_dataset():
                status["dataset_created"] = True
                save_status(status)
        else:
            print("\n✅ Dataset already created")

        # Step 2: Import MIMIC data
        if not status["mimic_imported"]:
            print("\n" + "-"*70)
            print("  Step 2/5: Importing MIMIC-IV data")
            print("-"*70)
            if import_mimic_data():
                status["mimic_imported"] = True
                save_status(status)
            else:
                print("\n❌ Failed to import MIMIC data")
                print("Please ensure you have PhysioNet access configured")
                return 1
        else:
            print("\n✅ MIMIC data already imported")

        # Step 3: Apply temporal transformation
        if not status["temporal_transformed"]:
            print("\n" + "-"*70)
            print("  Step 3/5: Applying temporal transformation")
            print("-"*70)
            if apply_temporal_transformation():
                status["temporal_transformed"] = True
                save_status(status)
            else:
                print("\n❌ Failed to apply temporal transformation")
                return 1
        else:
            print("\n✅ Temporal transformation already applied")

        # Step 4: Import clinical trials
        if not status["trials_imported"]:
            print("\n" + "-"*70)
            print("  Step 4/5: Importing clinical trials")
            print("-"*70)
            if import_clinical_trials():
                status["trials_imported"] = True
                save_status(status)
            else:
                print("\n⚠️ Clinical trials import failed (optional)")
        else:
            print("\n✅ Clinical trials already imported")

        # Step 5: Generate embeddings (optional)
        if not skip_embeddings and not status["embeddings_generated"]:
            print("\n" + "-"*70)
            print("  Step 5/6: Generating embeddings (optional)")
            print("-"*70)
            if generate_embeddings():
                status["embeddings_generated"] = True
                save_status(status)

        # Step 6: Create matches (optional)
        if not skip_matches and not status["matches_created"]:
            print("\n" + "-"*70)
            print("  Step 6/6: Creating matches (optional)")
            print("-"*70)
            if create_matches():
                status["matches_created"] = True
                save_status(status)

        # Verify results
        if verify_pipeline():
            print("\n" + "="*70)
            print("  ✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print("\nNext steps:")
            print("  1. Run API server: python python_files/api_main.py")
            print("  2. Access UI: http://localhost:8001/ui/")
            print("  3. Test queries: python python_files/comprehensive_data_test.py")
            return 0
        else:
            print("\n⚠️ Pipeline completed with some missing tables")
            print("Check the errors above and re-run if needed")
            return 1

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())