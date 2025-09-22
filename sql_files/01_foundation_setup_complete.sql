-- ============================================================================
-- 01_foundation_setup_complete.sql
-- COMPLETE FOUNDATION SETUP - BigQuery 2025 Competition
-- ============================================================================
-- IMPORTANT: Run 00_create_patient_demographics.sql FIRST to create patient
-- demographics table with proper age calculation (or include inline)
-- ============================================================================
-- This is the ONLY 01 file you need. It consolidates all functionality:
-- 
-- OPTION A: Use existing snapshot data (if available)
-- OPTION B: Import directly from PhysioNet MIMIC-IV
-- 
-- Includes:
-- 1. Dataset creation
-- 2. Data import (from snapshot OR PhysioNet)
-- 3. Clinical trials import (from API)
-- 4. Temporal transformation to 2025
-- 5. Patient status assessment
-- 6. Vertex AI model setup
-- 7. Validation
-- ============================================================================

-- ============================================================================
-- CONFIGURATION - JUDGES: REPLACE WITH YOUR VALUES
-- ============================================================================
-- IMPORTANT FOR JUDGES:
-- 1. Replace 'YOUR-PROJECT-ID' with your actual Google Cloud project ID
-- 2. Keep DATASET_ID as 'clinical_trial_matching' or update all references
-- 3. USE_SNAPSHOT must be FALSE (judges don't have access to our snapshots)
-- ============================================================================
DECLARE PROJECT_ID STRING DEFAULT 'YOUR-PROJECT-ID';  -- JUDGES: Replace this!
DECLARE DATASET_ID STRING DEFAULT 'clinical_trial_matching';
DECLARE USE_SNAPSHOT BOOL DEFAULT FALSE;  -- MUST be FALSE for judges

-- ============================================================================
-- SECTION 1: DATASET CREATION
-- ============================================================================
-- Create dataset if it doesn't exist
CREATE SCHEMA IF NOT EXISTS `${PROJECT_ID}.${DATASET_ID}`
OPTIONS(
  description="Clinical trial matching dataset for BigQuery 2025 competition",
  location="US"
);

-- ============================================================================
-- SECTION 2: DATA IMPORT - CHOOSE YOUR PATH
-- ============================================================================

-- PATH A: IMPORT FROM SNAPSHOT (Faster - 5 minutes)
-- COMMENTED OUT since USE_SNAPSHOT = FALSE (judges don't have snapshot)

/*
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.discharge_summaries` AS
SELECT * FROM `${PROJECT_ID}.clinical_trial_matching_snapshot_20250920.discharge_summaries`;

CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.lab_events` AS
SELECT * FROM `${PROJECT_ID}.clinical_trial_matching_snapshot_20250920.lab_events`;

CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.medications` AS
SELECT * FROM `${PROJECT_ID}.clinical_trial_matching_snapshot_20250920.medications`;

CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.radiology_reports` AS
SELECT * FROM `${PROJECT_ID}.clinical_trial_matching_snapshot_20250920.radiology_reports`;

CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.trials_comprehensive` AS
SELECT * FROM `${PROJECT_ID}.clinical_trial_matching_snapshot_20250920.trials_comprehensive`;
*/

-- PATH B: IMPORT FROM PHYSIONET (Required for judges - 2-3 hours)
-- ACTIVE PATH - Importing directly from PhysioNet MIMIC-IV
-- IMPORTANT: Judges must have PhysioNet credentials with MIMIC-IV access
-- Sign up at: https://physionet.org/

-- Import discharge summaries with diagnoses
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.discharge_summaries` AS
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
WHERE n.discharge_text IS NOT NULL;

-- Import lab events (key tests only to manage size)
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.lab_events` AS
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
  );

-- Import medications
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.medications` AS
SELECT
  CAST(p.subject_id AS STRING) AS patient_id,
  CAST(p.hadm_id AS STRING) AS hadm_id,
  p.starttime,
  p.stoptime,
  p.drug AS medication,
  p.dose_val_rx AS dose_value,
  p.dose_unit_rx AS dose_unit,
  p.route,
  p.frequency,
  CASE
    WHEN LOWER(p.drug) LIKE '%insulin%' OR LOWER(p.drug) LIKE '%metformin%' THEN 'diabetes'
    WHEN LOWER(p.drug) LIKE '%statin%' OR LOWER(p.drug) LIKE '%aspirin%' THEN 'cardiovascular'
    WHEN LOWER(p.drug) LIKE '%chemotherapy%' THEN 'oncology'
    ELSE 'other'
  END AS medication_category,
  CURRENT_TIMESTAMP() AS imported_at
FROM `physionet-data.mimiciv_3_1_hosp.prescriptions` p
WHERE p.hadm_id IS NOT NULL;

-- Import radiology reports
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.radiology_reports` AS
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
WHERE r.text IS NOT NULL AND r.hadm_id IS NOT NULL;

-- ============================================================================
-- SECTION 3: CLINICAL TRIALS IMPORT
-- ============================================================================
-- Run Python script: python python_files/import_clinical_trials_comprehensive.py
-- OR create empty table for manual import:

CREATE TABLE IF NOT EXISTS `${PROJECT_ID}.${DATASET_ID}.trials_comprehensive` (
  nct_id STRING NOT NULL,
  brief_title STRING,
  official_title STRING,
  overall_status STRING,
  conditions STRING,
  eligibility_criteria_full STRING,
  min_age_years INT64,
  max_age_years INT64,
  gender STRING,
  phase STRING,
  enrollment_count INT64,
  -- Lab requirements
  hemoglobin_min FLOAT64,
  creatinine_max FLOAT64,
  platelets_min FLOAT64,
  -- Trial classification
  is_oncology_trial BOOL,
  is_diabetes_trial BOOL,
  is_cardiac_trial BOOL,
  imported_at TIMESTAMP
);

-- ============================================================================
-- SECTION 4: TEMPORAL TRANSFORMATION TO 2025
-- ============================================================================
-- Transform all dates from 2100+ to 2025 timeframe

CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.discharge_summaries_2025` AS
SELECT
  patient_id,
  hadm_id,
  -- Simple 186-year shift: transforms 2100s dates to 1919-2026 range
  -- This gives us realistic distribution of recent vs historical data
  DATETIME_SUB(note_datetime, INTERVAL 186 YEAR) AS admission_date_2025,
  DATETIME_ADD(
    DATETIME_SUB(note_datetime, INTERVAL 186 YEAR),
    INTERVAL 7 DAY
  ) AS discharge_date_2025,
  note_datetime AS original_admission_time,
  discharge_diagnosis,
  discharge_text AS discharge_summary,
  mentions_cancer,
  mentions_diabetes,
  mentions_heart_failure,
  CURRENT_TIMESTAMP() AS transformed_at
FROM `${PROJECT_ID}.${DATASET_ID}.discharge_summaries`
WHERE patient_id IS NOT NULL;

CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.lab_events_2025` AS
SELECT
  patient_id,
  hadm_id,
  itemid,
  lab_test_name,
  lab_test_category,
  -- Simple 186-year shift for consistent temporal transformation
  DATETIME_SUB(charttime, INTERVAL 186 YEAR) AS charttime_2025,
  -- Days since test from current date (September 2025)
  DATE_DIFF(
    CURRENT_DATE(),
    DATE(DATETIME_SUB(charttime, INTERVAL 186 YEAR)),
    DAY
  ) AS days_since_test,
  lab_value_numeric,
  lab_unit,
  abnormal_flag,
  CURRENT_TIMESTAMP() AS transformed_at
FROM `${PROJECT_ID}.${DATASET_ID}.lab_events`
WHERE charttime IS NOT NULL;

CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.medications_2025` AS
SELECT
  patient_id,
  hadm_id,
  medication,
  medication_category,
  -- Simple 186-year shift for consistent temporal transformation
  DATETIME_SUB(starttime, INTERVAL 186 YEAR) AS starttime_2025,
  CASE
    WHEN stoptime IS NOT NULL THEN
      DATETIME_SUB(stoptime, INTERVAL 186 YEAR)
    ELSE NULL
  END AS stoptime_2025,
  -- Mark as active if started within 30 days and no stop time, or stop time is in future
  CASE
    WHEN stoptime IS NULL
     AND DATE_DIFF(CURRENT_DATE(), DATE(DATETIME_SUB(starttime, INTERVAL 186 YEAR)), DAY) <= 30
    THEN TRUE
    WHEN stoptime IS NOT NULL
     AND DATE(DATETIME_SUB(stoptime, INTERVAL 186 YEAR)) >= CURRENT_DATE()
    THEN TRUE
    ELSE FALSE
  END AS is_currently_active,
  dose_value,
  dose_unit,
  route,
  frequency,
  CURRENT_TIMESTAMP() AS transformed_at
FROM `${PROJECT_ID}.${DATASET_ID}.medications`
WHERE starttime IS NOT NULL;

CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.radiology_reports_2025` AS
SELECT
  patient_id,
  hadm_id,
  report_text,
  imaging_type,
  has_findings,
  mentions_mass,
  -- Simple 186-year shift for consistent temporal transformation
  DATETIME_SUB(study_datetime, INTERVAL 186 YEAR) AS study_datetime_2025,
  DATE_DIFF(
    CURRENT_DATE(),
    DATE(DATETIME_SUB(study_datetime, INTERVAL 186 YEAR)),
    DAY
  ) AS days_since_report,
  CURRENT_TIMESTAMP() AS transformed_at
FROM `${PROJECT_ID}.${DATASET_ID}.radiology_reports`
WHERE study_datetime IS NOT NULL;

-- ============================================================================
-- SECTION 5: PATIENT CURRENT STATUS ASSESSMENT
-- ============================================================================

CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.patient_current_status_2025` AS
WITH patient_activity AS (
  SELECT
    patient_id,
    MAX(DATE(admission_date_2025)) AS last_admission_date,
    COUNT(DISTINCT hadm_id) AS total_admissions,
    DATE_DIFF(CURRENT_DATE(), MAX(DATE(admission_date_2025)), DAY) AS days_since_last_admission
  FROM `${PROJECT_ID}.${DATASET_ID}.discharge_summaries_2025`
  GROUP BY patient_id
),
lab_activity AS (
  SELECT
    patient_id,
    MIN(days_since_test) AS days_since_latest_lab,
    COUNT(*) AS total_lab_tests
  FROM `${PROJECT_ID}.${DATASET_ID}.lab_events_2025`
  WHERE lab_test_category != 'other'
  GROUP BY patient_id
),
medication_activity AS (
  SELECT
    patient_id,
    COUNT(DISTINCT medication) AS total_medications,
    COUNT(DISTINCT CASE WHEN is_currently_active THEN medication END) AS active_medications
  FROM `${PROJECT_ID}.${DATASET_ID}.medications_2025`
  GROUP BY patient_id
)
SELECT
  pa.patient_id,
  pa.last_admission_date,
  pa.total_admissions,
  pa.days_since_last_admission,
  COALESCE(la.days_since_latest_lab, 9999) AS days_since_latest_lab,
  COALESCE(ma.total_medications, 0) AS total_medications,
  COALESCE(ma.active_medications, 0) AS active_medications,
  -- Determine trial readiness
  CASE
    WHEN COALESCE(la.days_since_latest_lab, 9999) <= 30
     AND pa.days_since_last_admission <= 30
    THEN 'Active_Ready'
    WHEN COALESCE(la.days_since_latest_lab, 9999) <= 90
     OR pa.days_since_last_admission <= 90
    THEN 'Recent_Screening_Needed'
    WHEN COALESCE(la.days_since_latest_lab, 9999) <= 365
     OR pa.days_since_last_admission <= 365
    THEN 'Inactive_Full_Screening'
    ELSE 'Historical_Not_Eligible'
  END AS patient_status,
  CURRENT_DATE() AS assessment_date,
  CURRENT_TIMESTAMP() AS created_at
FROM patient_activity pa
LEFT JOIN lab_activity la ON pa.patient_id = la.patient_id
LEFT JOIN medication_activity ma ON pa.patient_id = ma.patient_id;

-- ============================================================================
-- SECTION 6: VERTEX AI MODEL SETUP (OPTIONAL)
-- ============================================================================

-- Create connection (judges must run this once to enable Vertex AI)
-- JUDGES: Uncomment and run this command ONCE to create the connection:
/*
CREATE CONNECTION `${PROJECT_ID}.US.vertex_ai_connection`
LOCATION = 'US'
OPTIONS (type = 'CLOUD_RESOURCE');
*/

-- Create Gemini 2.5 Flash Lite model for AI functions
CREATE OR REPLACE MODEL `${PROJECT_ID}.${DATASET_ID}.gemini_model`
REMOTE WITH CONNECTION `${PROJECT_ID}.US.vertex_ai_connection`
OPTIONS (
  endpoint = 'gemini-2.5-flash-lite'  -- Cost-optimized: 50% cheaper, low latency
);

-- Create embedding model for ML.GENERATE_EMBEDDING
CREATE OR REPLACE MODEL `${PROJECT_ID}.${DATASET_ID}.embedding_model`
REMOTE WITH CONNECTION `${PROJECT_ID}.US.vertex_ai_connection`
OPTIONS (
  endpoint = 'gemini-embedding-001'  -- Latest: 768-3072 dimensions, MRL support
);

-- Create text_embedding_model alias for compatibility
CREATE OR REPLACE MODEL `${PROJECT_ID}.${DATASET_ID}.text_embedding_model`
REMOTE WITH CONNECTION `${PROJECT_ID}.US.vertex_ai_connection`
OPTIONS (
  endpoint = 'gemini-embedding-001'
);

-- ============================================================================
-- SECTION 7: VALIDATION
-- ============================================================================

SELECT
  '✅ FOUNDATION SETUP COMPLETE' AS status,
  STRUCT(
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.patient_current_status_2025`) AS total_patients,
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.patient_current_status_2025` WHERE patient_status = 'Active_Ready') AS active_ready_patients,
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.discharge_summaries_2025`) AS discharge_summaries,
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.lab_events_2025` WHERE days_since_test <= 30) AS recent_lab_results,
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.medications_2025` WHERE is_currently_active) AS active_medications,
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.trials_comprehensive`) AS clinical_trials
  ) AS counts,
  CURRENT_TIMESTAMP() AS completed_at;

-- ============================================================================
-- END OF FOUNDATION SETUP
-- ============================================================================
/*
Next steps:
1. Run 02_patient_profiling.sql to create patient profiles
2. Run 03_vector_embeddings.sql to generate embeddings
3. Run 04_vector_search_indexes.sql to create indexes
4. Run 05_ai_functions.sql for AI capabilities
*/