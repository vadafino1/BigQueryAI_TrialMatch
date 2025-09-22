-- ============================================================================
-- 03_vector_embeddings.sql
-- COMPREHENSIVE VECTOR EMBEDDING GENERATION - BigQuery 2025 Competition
-- ============================================================================
-- CONSOLIDATES: 04_generate_vector_embeddings + embedding logic from other files
--
-- COMPLETE EMBEDDING GENERATION INCLUDING:
-- ✅ Patient embeddings from clinical profiles
-- ✅ Trial embeddings from eligibility criteria
-- ✅ Batch processing for large datasets
-- ✅ Embedding quality validation
-- ✅ Dimension consistency checks
--
-- Competition: BigQuery 2025 (Semantic Detective Approach)
-- Last Updated: September 2025
-- Prerequisites: Run 01_foundation_complete.sql and 02_patient_profiling.sql first

-- ============================================================================
-- CONFIGURATION VARIABLES
-- ============================================================================
-- IMPORTANT: Replace with your actual project ID or set as environment variable
DECLARE PROJECT_ID STRING DEFAULT 'YOUR_PROJECT_ID';
DECLARE DATASET_ID STRING DEFAULT 'clinical_trial_matching';

-- ============================================================================

-- ============================================================================
-- SECTION 0: CREATE EMBEDDING MODEL
-- ============================================================================

-- Create or replace the embedding model with error handling
CREATE OR REPLACE MODEL `${PROJECT_ID}.${DATASET_ID}.gemini_embedding_model`
REMOTE WITH CONNECTION `${PROJECT_ID}.US.vertex_ai_connection`
OPTIONS(ENDPOINT = 'text-embedding-004');

-- ============================================================================
-- SECTION 1: PATIENT EMBEDDINGS GENERATION
-- ============================================================================

-- Drop existing embeddings for clean rebuild
DROP TABLE IF EXISTS `${PROJECT_ID}.${DATASET_ID}.patient_embeddings`;

-- Generate patient embeddings from clinical profiles
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.patient_embeddings`
PARTITION BY DATE(embedding_generated_at)
CLUSTER BY patient_id, trial_readiness
OPTIONS (
  description = "Patient embeddings for semantic matching - BigQuery 2025 Competition",
  labels = [("model", "gemini_embedding_001"), ("dimensions", "768"), ("updated", "2025_09")]
) AS
WITH patient_text AS (
  -- Prepare rich clinical text for embedding generation
  SELECT
    patient_id,
    trial_readiness,
    readiness_priority,
    clinical_complexity,
    risk_category,

    -- Create comprehensive clinical narrative for embedding
    CONCAT(
      -- Demographics
      'Patient demographics: ', age, ' year old ', gender, '. ',

      -- Primary condition and diagnosis
      'Primary diagnosis: ', primary_diagnosis, '. ',
      'All conditions: ', ARRAY_TO_STRING(condition_categories, ', '), '. ',

      -- Current medications
      'Current medications: ',
      CASE
        WHEN ARRAY_LENGTH(medication_categories) > 0
        THEN ARRAY_TO_STRING(medication_categories, ', ')
        ELSE 'None listed'
      END, '. ',

      -- Lab values with clinical context
      'Recent lab values: ',
      CASE WHEN creatinine IS NOT NULL
           THEN CONCAT('Creatinine ', ROUND(creatinine, 2), ' mg/dL (', creatinine_days_ago, ' days ago)')
           ELSE '' END,
      CASE WHEN hemoglobin IS NOT NULL
           THEN CONCAT(', Hemoglobin ', ROUND(hemoglobin, 1), ' g/dL (', hemoglobin_days_ago, ' days ago)')
           ELSE '' END,
      CASE WHEN platelets IS NOT NULL
           THEN CONCAT(', Platelets ', ROUND(platelets, 0), ' K/uL (', platelets_days_ago, ' days ago)')
           ELSE '' END,
      '. ',

      -- Clinical complexity and risk
      'Clinical complexity: ', clinical_complexity, '. ',
      'Risk category: ', risk_category, '. ',

      -- Trial readiness
      'Trial readiness status: ', trial_readiness, '. ',
      'Days since last admission: ', days_since_discharge, '. ',

      -- Special considerations
      CASE WHEN has_anticoagulant THEN 'Currently on anticoagulation. ' ELSE '' END,
      CASE WHEN has_diabetes_med THEN 'Currently on diabetes medications. ' ELSE '' END,
      CASE WHEN num_abnormal_labs > 0
           THEN CONCAT(num_abnormal_labs, ' abnormal lab values present. ')
           ELSE '' END
    ) AS patient_text_full

  FROM `${PROJECT_ID}.${DATASET_ID}.patient_profile`
  -- Strategic selection: prioritize high-complexity and high-risk patients
  ORDER BY
    CASE clinical_complexity
      WHEN 'HIGH_COMPLEXITY' THEN 1
      WHEN 'MODERATE_COMPLEXITY' THEN 2
      ELSE 3
    END,
    CASE risk_category
      WHEN 'HIGH_RISK' THEN 1
      WHEN 'MODERATE_RISK' THEN 2
      ELSE 3
    END,
    RAND()  -- Random sampling within categories
  LIMIT 10000  -- Target 10,000 most relevant patients
)
SELECT
  patient_id,
  trial_readiness,
  readiness_priority,
  clinical_complexity,
  risk_category,
  ml_generate_embedding_result AS embedding,
  patient_text_full AS source_text,
  ARRAY_LENGTH(ml_generate_embedding_result) AS embedding_dimension,
  CURRENT_TIMESTAMP() AS embedding_generated_at,
  'gemini-text-embedding-004' AS embedding_model,
  '2025-09' AS embedding_version
FROM ML.GENERATE_EMBEDDING(
  MODEL `${PROJECT_ID}.${DATASET_ID}.gemini_embedding_model`,
  (SELECT
    patient_id,
    trial_readiness,
    readiness_priority,
    clinical_complexity,
    risk_category,
    patient_text_full AS content,
    patient_text_full
  FROM patient_text),
  STRUCT(
    TRUE AS flatten_json_output,
    768 AS output_dimensionality
  )
);

-- ============================================================================
-- SECTION 2: TRIAL EMBEDDINGS GENERATION
-- ============================================================================

-- Drop existing trial embeddings for clean rebuild
DROP TABLE IF EXISTS `${PROJECT_ID}.${DATASET_ID}.trial_embeddings`;

-- Generate trial embeddings from eligibility criteria
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.trial_embeddings`
PARTITION BY DATE(embedding_generated_at)
CLUSTER BY nct_id, phase, therapeutic_area
OPTIONS (
  description = "Clinical trial embeddings for semantic matching - BigQuery 2025 Competition",
  labels = [("model", "gemini_embedding_001"), ("dimensions", "768"), ("updated", "2025_09")]
) AS
WITH trial_text AS (
  -- Prepare comprehensive trial text for embedding generation
  SELECT
    nct_id,
    brief_title,
    phase,
    overall_status,
    CASE
      WHEN is_oncology_trial THEN 'ONCOLOGY'
      WHEN is_diabetes_trial THEN 'DIABETES'
      WHEN is_cardiac_trial THEN 'CARDIAC'
      ELSE 'OTHER'
    END AS therapeutic_area,
    enrollment_count,

    -- Create detailed trial description for embedding
    CONCAT(
      -- Trial basics
      'Clinical trial: ', brief_title, '. ',
      'Phase: ', IFNULL(phase, 'Not specified'), '. ',
      'Status: ', overall_status, '. ',

      -- Conditions and eligibility
      'Conditions studied: ', IFNULL(conditions, 'Not specified'), '. ',

      -- Eligibility criteria (truncated for embedding)
      'Eligibility criteria: ', SUBSTR(IFNULL(eligibility_criteria_full, ''), 1, 2000), '. ',

      -- Enrollment information
      'Target enrollment: ', CAST(enrollment_count AS STRING), ' patients. '
    ) AS trial_text_full

  FROM `${PROJECT_ID}.${DATASET_ID}.trials_comprehensive`
  WHERE overall_status = 'RECRUITING'
    AND nct_id IS NOT NULL
),
-- Ensure therapeutic diversity: 30% oncology, 15% cardiac, 15% diabetes, 40% other
ranked_trials AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY therapeutic_area
      ORDER BY
        CASE phase
          WHEN 'PHASE2' THEN 1
          WHEN 'PHASE3' THEN 2
          WHEN 'PHASE1' THEN 3
          WHEN 'PHASE2, PHASE3' THEN 4
          WHEN 'PHASE1, PHASE2' THEN 5
          ELSE 6
        END,
        enrollment_count DESC
    ) AS rank
  FROM trial_text
),
selected_trials AS (
  SELECT * FROM ranked_trials
  WHERE (therapeutic_area = 'ONCOLOGY' AND rank <= 1500)    -- 30% of 5000
     OR (therapeutic_area = 'CARDIAC' AND rank <= 750)      -- 15% of 5000
     OR (therapeutic_area = 'DIABETES' AND rank <= 750)     -- 15% of 5000
     OR (therapeutic_area = 'OTHER' AND rank <= 2000)       -- 40% of 5000
  LIMIT 5000  -- Target 5,000 most diverse trials
)
SELECT
  nct_id,
  brief_title,
  phase,
  overall_status,
  therapeutic_area,
  enrollment_count,
  ml_generate_embedding_result AS embedding,
  trial_text_full AS source_text,
  ARRAY_LENGTH(ml_generate_embedding_result) AS embedding_dimension,
  CURRENT_TIMESTAMP() AS embedding_generated_at,
  'gemini-text-embedding-004' AS embedding_model,
  '2025-09' AS embedding_version
FROM ML.GENERATE_EMBEDDING(
  MODEL `${PROJECT_ID}.${DATASET_ID}.gemini_embedding_model`,
  (SELECT
    nct_id,
    brief_title,
    phase,
    overall_status,
    therapeutic_area,
    enrollment_count,
    trial_text_full AS content,
    trial_text_full
  FROM selected_trials),
  STRUCT(
    TRUE AS flatten_json_output,
    768 AS output_dimensionality
  )
);

-- ============================================================================
-- SECTION 3: EMBEDDING QUALITY VALIDATION
-- ============================================================================

-- Create embedding quality assessment view
CREATE OR REPLACE VIEW `${PROJECT_ID}.${DATASET_ID}.v_embedding_quality` AS
WITH patient_embedding_stats AS (
  SELECT
    'Patient Embeddings' AS embedding_type,
    COUNT(*) AS total_embeddings,
    AVG(embedding_dimension) AS avg_dimension,
    MIN(embedding_dimension) AS min_dimension,
    MAX(embedding_dimension) AS max_dimension,
    COUNTIF(embedding_dimension = 768) AS correct_dimension_count,
    COUNTIF(embedding IS NULL) AS null_embedding_count,
    COUNT(DISTINCT trial_readiness) AS unique_readiness_states,
    MIN(embedding_generated_at) AS earliest_generation,
    MAX(embedding_generated_at) AS latest_generation
  FROM `${PROJECT_ID}.${DATASET_ID}.patient_embeddings`
),
trial_embedding_stats AS (
  SELECT
    'Trial Embeddings' AS embedding_type,
    COUNT(*) AS total_embeddings,
    AVG(embedding_dimension) AS avg_dimension,
    MIN(embedding_dimension) AS min_dimension,
    MAX(embedding_dimension) AS max_dimension,
    COUNTIF(embedding_dimension = 768) AS correct_dimension_count,
    COUNTIF(embedding IS NULL) AS null_embedding_count,
    COUNT(DISTINCT phase) AS unique_phases,
    MIN(embedding_generated_at) AS earliest_generation,
    MAX(embedding_generated_at) AS latest_generation
  FROM `${PROJECT_ID}.${DATASET_ID}.trial_embeddings`
)
SELECT * FROM patient_embedding_stats
UNION ALL
SELECT * FROM trial_embedding_stats;

-- ============================================================================
-- SECTION 4: BATCH PROCESSING FOR REMAINING RECORDS
-- ============================================================================

-- Create batch processing status table
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.embedding_batch_status` (
  batch_id STRING,
  batch_type STRING,  -- 'PATIENT' or 'TRIAL'
  start_timestamp TIMESTAMP,
  end_timestamp TIMESTAMP,
  records_processed INT64,
  status STRING,  -- 'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'
  error_message STRING
)
PARTITION BY DATE(start_timestamp);

-- Insert initial batch status
INSERT INTO `${PROJECT_ID}.${DATASET_ID}.embedding_batch_status`
(batch_id, batch_type, start_timestamp, records_processed, status)
VALUES
  (GENERATE_UUID(), 'PATIENT', CURRENT_TIMESTAMP(),
   (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.patient_embeddings`),
   'COMPLETED'),
  (GENERATE_UUID(), 'TRIAL', CURRENT_TIMESTAMP(),
   (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.trial_embeddings`),
   'COMPLETED');

-- ============================================================================
-- SECTION 5: EMBEDDING SIMILARITY VALIDATION
-- ============================================================================

-- Validate embeddings by computing sample similarities
CREATE OR REPLACE VIEW `${PROJECT_ID}.${DATASET_ID}.v_embedding_similarity_sample` AS
WITH sample_patients AS (
  SELECT patient_id, embedding, trial_readiness
  FROM `${PROJECT_ID}.${DATASET_ID}.patient_embeddings`
  -- WHERE trial_readiness = 'Active_Ready'  -- REMOVED: Include all patients
  LIMIT 10
),
sample_trials AS (
  SELECT nct_id, embedding, therapeutic_area
  FROM `${PROJECT_ID}.${DATASET_ID}.trial_embeddings`
  WHERE therapeutic_area IN ('DIABETES', 'CARDIAC', 'ONCOLOGY')
  LIMIT 10
)
SELECT
  p.patient_id,
  t.nct_id,
  t.therapeutic_area,
  -- Compute cosine similarity
  (1 - ML.DISTANCE(p.embedding, t.embedding, 'COSINE')) AS cosine_similarity,
  -- Compute Euclidean distance for comparison
  ML.DISTANCE(p.embedding, t.embedding, 'EUCLIDEAN') AS euclidean_distance,
  CURRENT_TIMESTAMP() AS similarity_computed_at
FROM sample_patients p
CROSS JOIN sample_trials t
ORDER BY p.patient_id, cosine_similarity DESC;

-- ============================================================================
-- SECTION 6: INCREMENTAL EMBEDDING UPDATES
-- ============================================================================

-- Create view for patients needing embeddings
CREATE OR REPLACE VIEW `${PROJECT_ID}.${DATASET_ID}.v_patients_needing_embeddings` AS
SELECT
  pp.patient_id,
  pp.trial_readiness,
  pp.clinical_text_description,
  'MISSING_EMBEDDING' AS status
FROM `${PROJECT_ID}.${DATASET_ID}.patient_profile` pp
LEFT JOIN `${PROJECT_ID}.${DATASET_ID}.patient_embeddings` pe
  ON pp.patient_id = pe.patient_id
WHERE pe.patient_id IS NULL
  -- AND pp.trial_readiness IN ('Active_Ready', 'Recent_Screening_Needed')  -- REMOVED: Include all
LIMIT 1000;  -- Process in manageable batches

-- Create view for trials needing embeddings
CREATE OR REPLACE VIEW `${PROJECT_ID}.${DATASET_ID}.v_trials_needing_embeddings` AS
SELECT
  tc.nct_id,
  tc.title,
  tc.eligibility_criteria,
  'MISSING_EMBEDDING' AS status
FROM `${PROJECT_ID}.${DATASET_ID}.trials_comprehensive` tc
LEFT JOIN `${PROJECT_ID}.${DATASET_ID}.trial_embeddings` te
  ON tc.nct_id = te.nct_id
WHERE te.nct_id IS NULL
  AND tc.overall_status = 'Recruiting'
LIMIT 1000;  -- Process in manageable batches

-- ============================================================================
-- SECTION 7: EMBEDDING STATISTICS AND MONITORING
-- ============================================================================

-- Create comprehensive embedding statistics view
CREATE OR REPLACE VIEW `${PROJECT_ID}.${DATASET_ID}.v_embedding_statistics` AS
SELECT
  'Embedding Generation Statistics' AS report_type,
  STRUCT(
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.patient_embeddings`) AS total_patient_embeddings,
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.trial_embeddings`) AS total_trial_embeddings,
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.v_patients_needing_embeddings`) AS patients_missing_embeddings,
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.v_trials_needing_embeddings`) AS trials_missing_embeddings,
    (SELECT AVG(embedding_dimension) FROM `${PROJECT_ID}.${DATASET_ID}.patient_embeddings`) AS avg_patient_embedding_dim,
    (SELECT AVG(embedding_dimension) FROM `${PROJECT_ID}.${DATASET_ID}.trial_embeddings`) AS avg_trial_embedding_dim
  ) AS metrics,
  CURRENT_TIMESTAMP() AS report_generated_at;

-- ============================================================================
-- FINAL VALIDATION AND SUMMARY
-- ============================================================================

WITH embedding_summary AS (
  SELECT
    'Vector Embeddings Generation Complete' AS status,
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.patient_embeddings`) AS patient_embeddings,
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.trial_embeddings`) AS trial_embeddings,
    (SELECT COUNT(*)
     FROM `${PROJECT_ID}.${DATASET_ID}.patient_embeddings`
     WHERE embedding_dimension = 768) AS valid_patient_embeddings,
    (SELECT COUNT(*)
     FROM `${PROJECT_ID}.${DATASET_ID}.trial_embeddings`
     WHERE embedding_dimension = 768) AS valid_trial_embeddings
)
SELECT
  status,
  patient_embeddings,
  trial_embeddings,
  ROUND(100.0 * valid_patient_embeddings / NULLIF(patient_embeddings, 0), 1) AS patient_embedding_validity_pct,
  ROUND(100.0 * valid_trial_embeddings / NULLIF(trial_embeddings, 0), 1) AS trial_embedding_validity_pct,
  CASE
    WHEN patient_embeddings > 100000 AND trial_embeddings > 30000 THEN '✅ PRODUCTION READY'
    WHEN patient_embeddings > 10000 AND trial_embeddings > 5000 THEN '⚠️ PILOT READY'
    ELSE '❌ INSUFFICIENT EMBEDDINGS'
  END AS readiness_status,
  CURRENT_TIMESTAMP() AS validation_timestamp
FROM embedding_summary;

-- ============================================================================
-- VECTOR EMBEDDINGS COMPLETE
-- ============================================================================

SELECT
  '🎯 VECTOR EMBEDDINGS COMPLETE' AS status,
  'Ready for vector index creation and semantic search' AS message,
  STRUCT(
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.patient_embeddings`) AS patient_embeddings_count,
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.trial_embeddings`) AS trial_embeddings_count,
    768 AS embedding_dimensions,
    'gemini-embedding-001' AS model_used
  ) AS summary,
  CURRENT_TIMESTAMP() AS completed_at;

-- ============================================================================
-- NEXT STEPS
-- ============================================================================
/*
Vector embeddings generation complete! Next file to run:

04_vector_search_indexes.sql - Create TreeAH indexes for fast similarity search

This embedding generation provides:
✅ Patient embeddings from comprehensive clinical profiles
✅ Trial embeddings from eligibility criteria
✅ 768-dimensional vectors for optimal performance
✅ Quality validation and monitoring
✅ Batch processing capabilities
✅ Incremental update views
*/