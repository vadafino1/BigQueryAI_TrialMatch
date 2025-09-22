-- ============================================================================
-- 02_patient_profiling.sql
-- COMPREHENSIVE PATIENT PROFILING - BigQuery 2025 Competition
-- ============================================================================
-- CONSOLIDATES: 03_build_patient_profiles + patient profiling from embeddings
--
-- COMPLETE PATIENT DATA PREPARATION INCLUDING:
-- ✅ Patient profile table with clinical features
-- ✅ Lab values extraction and pivoting
-- ✅ Medication tracking with washout periods
-- ✅ Diagnosis categorization
-- ✅ Trial readiness assessment
-- ✅ Temporal eligibility windows
-- ✅ Clinical risk scoring
--
-- Competition: BigQuery 2025
-- Last Updated: September 2025 - Consolidated from 2 files
-- Prerequisites: Run 01_foundation_complete.sql first

-- ============================================================================
-- CONFIGURATION VARIABLES
-- ============================================================================
-- IMPORTANT: Replace with your actual project ID or set as environment variable
DECLARE PROJECT_ID STRING DEFAULT 'YOUR_PROJECT_ID';
DECLARE DATASET_ID STRING DEFAULT 'clinical_trial_matching';

-- ============================================================================

-- ============================================================================
-- SECTION 1: PATIENT PROFILE TABLE WITH CLINICAL INTELLIGENCE
-- ============================================================================

-- Drop existing table for clean rebuild
DROP TABLE IF EXISTS `${PROJECT_ID}.${DATASET_ID}.patient_profile`;

-- Create comprehensive patient profile table
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.patient_profile`
PARTITION BY DATE(profile_created_at)
CLUSTER BY patient_id, trial_readiness, readiness_priority
OPTIONS (
  description = "Enhanced patient profiles for clinical trial matching - BigQuery 2025 Competition",
  labels = [("environment", "production"), ("competition", "bigquery_2025"), ("updated", "2025_09")]
) AS
WITH latest_labs AS (
  -- Extract most recent lab values with clinical validation
  SELECT
    patient_id,
    itemid,
    lab_value,
    lab_unit,
    days_since_test,
    -- Identify abnormal values based on standard ranges
    CASE
      WHEN itemid = 50912 AND SAFE_CAST(lab_value AS FLOAT64) > 1.5 THEN TRUE  -- Creatinine
      WHEN itemid = 50861 AND SAFE_CAST(lab_value AS FLOAT64) > 40 THEN TRUE   -- ALT
      WHEN itemid = 50878 AND SAFE_CAST(lab_value AS FLOAT64) > 40 THEN TRUE   -- AST
      WHEN itemid = 51222 AND SAFE_CAST(lab_value AS FLOAT64) < 10 THEN TRUE   -- Hemoglobin
      WHEN itemid = 51265 AND SAFE_CAST(lab_value AS FLOAT64) < 100 THEN TRUE  -- Platelets
      WHEN itemid = 51301 AND (SAFE_CAST(lab_value AS FLOAT64) < 3.5 OR
                               SAFE_CAST(lab_value AS FLOAT64) > 10.5) THEN TRUE -- WBC
      ELSE FALSE
    END AS is_abnormal,
    ROW_NUMBER() OVER (
      PARTITION BY patient_id, itemid
      ORDER BY days_since_test ASC
    ) AS rn
  FROM `${PROJECT_ID}.${DATASET_ID}.lab_events_2025`
  WHERE lab_value IS NOT NULL
    AND days_since_test <= 365  -- Only consider recent labs
),

pivoted_labs AS (
  -- Pivot key lab values into columns for easier access
  SELECT
    patient_id,
    -- Creatinine (itemid: 50912)
    MAX(CASE WHEN itemid = 50912 THEN SAFE_CAST(lab_value AS FLOAT64) END) AS creatinine,
    MAX(CASE WHEN itemid = 50912 THEN days_since_test END) AS creatinine_days_ago,
    -- ALT (itemid: 50861)
    MAX(CASE WHEN itemid = 50861 THEN SAFE_CAST(lab_value AS FLOAT64) END) AS alt,
    MAX(CASE WHEN itemid = 50861 THEN days_since_test END) AS alt_days_ago,
    -- AST (itemid: 50878)
    MAX(CASE WHEN itemid = 50878 THEN SAFE_CAST(lab_value AS FLOAT64) END) AS ast,
    MAX(CASE WHEN itemid = 50878 THEN days_since_test END) AS ast_days_ago,
    -- Hemoglobin (itemid: 51222)
    MAX(CASE WHEN itemid = 51222 THEN SAFE_CAST(lab_value AS FLOAT64) END) AS hemoglobin,
    MAX(CASE WHEN itemid = 51222 THEN days_since_test END) AS hemoglobin_days_ago,
    -- Platelets (itemid: 51265)
    MAX(CASE WHEN itemid = 51265 THEN SAFE_CAST(lab_value AS FLOAT64) END) AS platelets,
    MAX(CASE WHEN itemid = 51265 THEN days_since_test END) AS platelets_days_ago,
    -- WBC (itemid: 51301)
    MAX(CASE WHEN itemid = 51301 THEN SAFE_CAST(lab_value AS FLOAT64) END) AS wbc,
    MAX(CASE WHEN itemid = 51301 THEN days_since_test END) AS wbc_days_ago,
    -- Count of abnormal labs
    COUNTIF(is_abnormal) AS num_abnormal_labs
  FROM latest_labs
  WHERE rn = 1  -- Only most recent value per test
  GROUP BY patient_id
),

active_medications AS (
  -- Track current medications and washout periods
  SELECT
    patient_id,
    -- Current medications array
    ARRAY_AGG(DISTINCT
      IF(is_currently_active, medication, NULL)
      IGNORE NULLS
    ) AS current_medications,

    -- Medication categories for eligibility matching
    ARRAY_AGG(DISTINCT
      IF(is_currently_active,
         CASE
           WHEN UPPER(medication) LIKE '%INSULIN%' THEN 'DIABETES_MED'
           WHEN UPPER(medication) LIKE '%METFORMIN%' THEN 'DIABETES_MED'
           WHEN UPPER(medication) LIKE '%STATIN%' THEN 'LIPID_MED'
           WHEN UPPER(medication) LIKE '%LISINOPRIL%' THEN 'BP_MED'
           WHEN UPPER(medication) LIKE '%WARFARIN%' THEN 'ANTICOAGULANT'
           WHEN UPPER(medication) LIKE '%ASPIRIN%' THEN 'ANTIPLATELET'
           ELSE 'OTHER_MED'
         END,
         NULL)
      IGNORE NULLS
    ) AS medication_categories,

    -- Track recent medication stops for washout periods
    ARRAY_AGG(DISTINCT
      IF(NOT is_currently_active AND
         DATE_DIFF(CURRENT_DATE(), DATE(stoptime_2025), DAY) <= 180,
         STRUCT(
           medication AS drug_name,
           DATE_DIFF(CURRENT_DATE(), DATE(stoptime_2025), DAY) AS days_off
         ),
         NULL)
      IGNORE NULLS
    ) AS recent_washouts,

    -- Medication counts
    COUNTIF(is_currently_active) AS num_current_medications,

    -- Key medication flags for eligibility
    LOGICAL_OR(is_currently_active AND UPPER(medication) LIKE '%WARFARIN%') AS has_anticoagulant,
    LOGICAL_OR(is_currently_active AND UPPER(medication) LIKE '%INSULIN%') AS has_diabetes_med,
    LOGICAL_OR(is_currently_active AND UPPER(medication) LIKE '%CHEMOTHERAPY%') AS has_chemo

  FROM `${PROJECT_ID}.${DATASET_ID}.medications_2025`
  GROUP BY patient_id
),

diagnoses AS (
  -- Extract and categorize patient diagnoses
  SELECT
    patient_id,
    -- All unique diagnoses
    ARRAY_AGG(DISTINCT discharge_diagnosis IGNORE NULLS) AS conditions,

    -- Categorized conditions for matching
    ARRAY_AGG(DISTINCT
      CASE
        WHEN UPPER(discharge_diagnosis) LIKE '%DIABETES%' THEN 'DIABETES'
        WHEN UPPER(discharge_diagnosis) LIKE '%HYPERTENSION%' THEN 'HYPERTENSION'
        WHEN UPPER(discharge_diagnosis) LIKE '%HEART%' OR
             UPPER(discharge_diagnosis) LIKE '%CARDIAC%' THEN 'CARDIAC'
        WHEN UPPER(discharge_diagnosis) LIKE '%CANCER%' OR
             UPPER(discharge_diagnosis) LIKE '%MALIGNANT%' THEN 'ONCOLOGY'
        WHEN UPPER(discharge_diagnosis) LIKE '%KIDNEY%' OR
             UPPER(discharge_diagnosis) LIKE '%RENAL%' THEN 'RENAL'
        WHEN UPPER(discharge_diagnosis) LIKE '%LIVER%' THEN 'HEPATIC'
        WHEN UPPER(discharge_diagnosis) LIKE '%STROKE%' THEN 'NEUROLOGICAL'
        WHEN UPPER(discharge_diagnosis) LIKE '%COPD%' OR
             UPPER(discharge_diagnosis) LIKE '%ASTHMA%' THEN 'RESPIRATORY'
        ELSE 'OTHER'
      END
      IGNORE NULLS
    ) AS condition_categories,

    COUNT(DISTINCT discharge_diagnosis) AS num_conditions,
    MIN(DATE_DIFF(CURRENT_DATE(), DATE(admission_date_2025), DAY)) AS days_since_last_admission

  FROM `${PROJECT_ID}.${DATASET_ID}.discharge_summaries_2025`
  WHERE discharge_diagnosis IS NOT NULL
  GROUP BY patient_id
),

imaging_summary AS (
  -- Summarize recent imaging studies
  SELECT
    patient_id,
    COUNT(*) AS num_recent_imaging,
    MIN(days_since_report) AS days_since_latest_imaging,
    ARRAY_AGG(DISTINCT
      CASE
        WHEN UPPER(report_text) LIKE '%CT%' THEN 'CT'
        WHEN UPPER(report_text) LIKE '%MRI%' THEN 'MRI'
        WHEN UPPER(report_text) LIKE '%X-RAY%' OR
             UPPER(report_text) LIKE '%XRAY%' THEN 'XRAY'
        WHEN UPPER(report_text) LIKE '%PET%' THEN 'PET'
        WHEN UPPER(report_text) LIKE '%ULTRASOUND%' THEN 'ULTRASOUND'
        ELSE 'OTHER_IMAGING'
      END
      IGNORE NULLS
    ) AS imaging_types
  FROM `${PROJECT_ID}.${DATASET_ID}.radiology_reports_2025`
  WHERE days_since_report <= 180  -- Only recent imaging
  GROUP BY patient_id
)

-- Main patient profile assembly
SELECT
  pd.patient_id,

  -- Demographics
  pd.current_age_2025 AS age,
  pd.gender,

  -- Primary diagnosis and conditions
  pd.primary_diagnosis,
  COALESCE(d.conditions, []) AS conditions,
  COALESCE(d.condition_categories, []) AS condition_categories,
  COALESCE(d.num_conditions, 0) AS num_conditions,

  -- Medications
  COALESCE(am.current_medications, []) AS current_medications,
  COALESCE(am.medication_categories, []) AS medication_categories,
  COALESCE(am.num_current_medications, 0) AS num_current_medications,
  COALESCE(am.recent_washouts, []) AS recent_washouts,
  COALESCE(am.has_anticoagulant, FALSE) AS has_anticoagulant,
  COALESCE(am.has_diabetes_med, FALSE) AS has_diabetes_med,
  COALESCE(am.has_chemo, FALSE) AS has_chemo,

  -- Lab values with recency
  pl.creatinine,
  pl.creatinine_days_ago,
  pl.alt,
  pl.alt_days_ago,
  pl.ast,
  pl.ast_days_ago,
  pl.hemoglobin,
  pl.hemoglobin_days_ago,
  pl.platelets,
  pl.platelets_days_ago,
  pl.wbc,
  pl.wbc_days_ago,
  COALESCE(pl.num_abnormal_labs, 0) AS num_abnormal_labs,

  -- Imaging summary
  COALESCE(img.num_recent_imaging, 0) AS num_recent_imaging,
  img.days_since_latest_imaging,
  COALESCE(img.imaging_types, []) AS imaging_types,

  -- Trial readiness assessment based on recent activity
  pcs.patient_status AS trial_readiness,

  -- Readiness priority scoring (higher = more ready)
  CASE
    WHEN pcs.patient_status = 'Active_Ready' THEN
      100 - LEAST(pcs.days_since_latest_lab, 100) -- Active patients prioritized by recency
    WHEN pcs.patient_status = 'Recent_Screening_Needed' THEN
      50 - (pcs.days_since_latest_lab / 2) -- Recent patients with lower priority
    ELSE 0
  END AS readiness_priority,

  -- Temporal eligibility windows
  pcs.days_since_last_admission,
  pcs.days_since_latest_lab,
  COALESCE(d.days_since_last_admission, 999) AS days_since_discharge,

  -- Clinical complexity scoring
  CASE
    WHEN COALESCE(d.num_conditions, 0) >= 5 AND
         COALESCE(am.num_current_medications, 0) >= 5 THEN 'HIGH_COMPLEXITY'
    WHEN COALESCE(d.num_conditions, 0) >= 3 OR
         COALESCE(am.num_current_medications, 0) >= 3 THEN 'MODERATE_COMPLEXITY'
    ELSE 'LOW_COMPLEXITY'
  END AS clinical_complexity,

  -- Risk stratification
  CASE
    WHEN COALESCE(pl.num_abnormal_labs, 0) >= 3 OR
         am.has_anticoagulant OR am.has_chemo THEN 'HIGH_RISK'
    WHEN COALESCE(pl.num_abnormal_labs, 0) >= 1 OR
         COALESCE(am.num_current_medications, 0) >= 5 THEN 'MODERATE_RISK'
    ELSE 'LOW_RISK'
  END AS risk_category,

  -- Create comprehensive text description for embedding generation
  CONCAT(
    'Patient: ', pd.current_age_2025, ' year old ', pd.gender,
    ' with ', pd.primary_diagnosis,
    '. Conditions: ', ARRAY_TO_STRING(d.condition_categories, ', '),
    '. Medications: ', ARRAY_TO_STRING(am.medication_categories, ', '),
    '. Clinical complexity: ',
    CASE
      WHEN COALESCE(d.num_conditions, 0) >= 5 THEN 'high'
      WHEN COALESCE(d.num_conditions, 0) >= 3 THEN 'moderate'
      ELSE 'low'
    END,
    '. Recent labs: ',
    CASE
      WHEN pl.creatinine IS NOT NULL THEN CONCAT('Creatinine=', ROUND(pl.creatinine, 2))
      ELSE ''
    END,
    CASE
      WHEN pl.hemoglobin IS NOT NULL THEN CONCAT(' Hemoglobin=', ROUND(pl.hemoglobin, 1))
      ELSE ''
    END
  ) AS clinical_text_description,

  -- Metadata
  CURRENT_TIMESTAMP() AS profile_created_at,
  '2025-09' AS profile_version

FROM `${PROJECT_ID}.${DATASET_ID}.patient_demographics` pd
INNER JOIN `${PROJECT_ID}.${DATASET_ID}.patient_current_status_2025` pcs
  ON pd.patient_id = pcs.patient_id
LEFT JOIN pivoted_labs pl ON pd.patient_id = pl.patient_id
LEFT JOIN active_medications am ON pd.patient_id = am.patient_id
LEFT JOIN diagnoses d ON pd.patient_id = d.patient_id
LEFT JOIN imaging_summary img ON pd.patient_id = img.patient_id
-- WHERE pcs.patient_status IN ('Active_Ready', 'Recent_Screening_Needed');  -- REMOVED: Include all patients
;

-- ============================================================================
-- SECTION 2: PROFILE QUALITY METRICS AND VALIDATION
-- ============================================================================

-- Create profile quality assessment view
CREATE OR REPLACE VIEW `${PROJECT_ID}.${DATASET_ID}.v_patient_profile_quality` AS
SELECT
  COUNT(*) AS total_profiles,

  -- Demographic completeness
  COUNTIF(age IS NOT NULL) AS has_age,
  COUNTIF(gender IS NOT NULL) AS has_gender,

  -- Clinical data completeness
  COUNTIF(ARRAY_LENGTH(conditions) > 0) AS has_conditions,
  COUNTIF(ARRAY_LENGTH(current_medications) > 0) AS has_medications,
  COUNTIF(creatinine IS NOT NULL) AS has_creatinine,
  COUNTIF(hemoglobin IS NOT NULL) AS has_hemoglobin,

  -- Trial readiness distribution
  COUNTIF(trial_readiness = 'Active_Ready') AS active_ready_count,
  COUNTIF(trial_readiness = 'Recent_Screening_Needed') AS recent_screening_count,

  -- Risk distribution
  COUNTIF(risk_category = 'HIGH_RISK') AS high_risk_count,
  COUNTIF(risk_category = 'MODERATE_RISK') AS moderate_risk_count,
  COUNTIF(risk_category = 'LOW_RISK') AS low_risk_count,

  -- Complexity distribution
  COUNTIF(clinical_complexity = 'HIGH_COMPLEXITY') AS high_complexity_count,
  COUNTIF(clinical_complexity = 'MODERATE_COMPLEXITY') AS moderate_complexity_count,
  COUNTIF(clinical_complexity = 'LOW_COMPLEXITY') AS low_complexity_count,

  -- Data freshness
  AVG(days_since_discharge) AS avg_days_since_discharge,
  AVG(days_since_latest_lab) AS avg_days_since_lab,

  CURRENT_TIMESTAMP() AS quality_check_timestamp
FROM `${PROJECT_ID}.${DATASET_ID}.patient_profile`;

-- ============================================================================
-- SECTION 3: CLINICAL FEATURE AGGREGATION FOR ML
-- ============================================================================

-- Create aggregated features table for machine learning
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.patient_features_ml` AS
SELECT
  patient_id,

  -- Demographic features
  age,
  CASE WHEN gender = 'Male' THEN 1 ELSE 0 END AS gender_male,

  -- Condition features (binary flags)
  CASE WHEN 'DIABETES' IN UNNEST(condition_categories) THEN 1 ELSE 0 END AS has_diabetes,
  CASE WHEN 'HYPERTENSION' IN UNNEST(condition_categories) THEN 1 ELSE 0 END AS has_hypertension,
  CASE WHEN 'CARDIAC' IN UNNEST(condition_categories) THEN 1 ELSE 0 END AS has_cardiac,
  CASE WHEN 'ONCOLOGY' IN UNNEST(condition_categories) THEN 1 ELSE 0 END AS has_cancer,
  CASE WHEN 'RENAL' IN UNNEST(condition_categories) THEN 1 ELSE 0 END AS has_renal,

  -- Medication features
  num_current_medications,
  CAST(has_anticoagulant AS INT64) AS on_anticoagulant,
  CAST(has_diabetes_med AS INT64) AS on_diabetes_med,

  -- Lab features (normalized)
  COALESCE(creatinine, 1.0) AS creatinine_value,
  COALESCE(hemoglobin, 13.0) AS hemoglobin_value,
  COALESCE(platelets, 250.0) AS platelets_value,
  COALESCE(wbc, 7.0) AS wbc_value,

  -- Temporal features
  LEAST(days_since_discharge, 365) AS days_since_discharge_capped,
  LEAST(days_since_latest_lab, 365) AS days_since_lab_capped,

  -- Risk scores
  CASE risk_category
    WHEN 'HIGH_RISK' THEN 3
    WHEN 'MODERATE_RISK' THEN 2
    ELSE 1
  END AS risk_score,

  CASE clinical_complexity
    WHEN 'HIGH_COMPLEXITY' THEN 3
    WHEN 'MODERATE_COMPLEXITY' THEN 2
    ELSE 1
  END AS complexity_score,

  -- Readiness score
  readiness_priority,

  CURRENT_TIMESTAMP() AS features_generated_at
FROM `${PROJECT_ID}.${DATASET_ID}.patient_profile`
-- WHERE trial_readiness IN ('Active_Ready', 'Recent_Screening_Needed');  -- REMOVED: Include all patients
;

-- ============================================================================
-- SECTION 4: VALIDATION AND SUMMARY
-- ============================================================================

-- Final validation query
WITH profile_summary AS (
  SELECT
    'Patient Profiling Complete' AS status,
    COUNT(*) AS total_profiles,
    COUNTIF(trial_readiness = 'Active_Ready') AS active_patients,
    ROUND(AVG(age), 1) AS avg_age,
    ROUND(AVG(num_conditions), 1) AS avg_conditions,
    ROUND(AVG(num_current_medications), 1) AS avg_medications,
    MIN(profile_created_at) AS earliest_profile,
    MAX(profile_created_at) AS latest_profile
  FROM `${PROJECT_ID}.${DATASET_ID}.patient_profile`
)
SELECT
  status,
  total_profiles,
  active_patients,
  CONCAT(avg_age, ' years') AS average_age,
  avg_conditions AS avg_conditions_per_patient,
  avg_medications AS avg_medications_per_patient,
  CASE
    WHEN total_profiles > 100000 THEN '✅ PRODUCTION SCALE'
    WHEN total_profiles > 10000 THEN '⚠️ PILOT SCALE'
    ELSE '❌ INSUFFICIENT DATA'
  END AS scale_assessment,
  CURRENT_TIMESTAMP() AS validation_timestamp
FROM profile_summary;

-- ============================================================================
-- PATIENT PROFILING COMPLETE
-- ============================================================================

SELECT
  '🎯 PATIENT PROFILING COMPLETE' AS status,
  'Ready for embedding generation and trial matching' AS message,
  STRUCT(
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.patient_profile`) AS total_profiles,
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.patient_features_ml`) AS ml_features_ready,
    (SELECT COUNT(DISTINCT patient_id) FROM `${PROJECT_ID}.${DATASET_ID}.patient_profile`
     -- WHERE trial_readiness = 'Active_Ready'  -- REMOVED: Include all
     ) AS trial_ready_patients
  ) AS summary,
  CURRENT_TIMESTAMP() AS completed_at;

-- ============================================================================
-- NEXT STEPS
-- ============================================================================
/*
Patient profiling complete! Next file to run:

03_vector_embeddings.sql - Generate embeddings for semantic matching

This profiling provides:
✅ Comprehensive patient profiles with clinical features
✅ Lab values, medications, and diagnoses
✅ Trial readiness assessment
✅ Risk stratification and complexity scoring
✅ ML-ready feature table
✅ Text descriptions for embedding generation
*/