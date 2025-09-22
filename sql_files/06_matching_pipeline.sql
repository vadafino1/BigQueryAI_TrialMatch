-- ============================================================================
-- 06_matching_pipeline.sql
-- COMPREHENSIVE PATIENT-TRIAL MATCHING PIPELINE - BigQuery 2025 Competition
-- ============================================================================
-- CONSOLIDATES: 07_vector_search_similarity_matching + matching logic from other files
--
-- COMPLETE MATCHING PIPELINE INCLUDING:
-- ✅ Vector similarity search using TreeAH indexes
-- ✅ Hybrid scoring (semantic + clinical)
-- ✅ Multi-stage matching pipeline
-- ✅ Ranking and prioritization
-- ✅ Match quality assessment
--
-- Competition: BigQuery 2025 (Semantic Detective + AI Architect)
-- Last Updated: September 2025
-- Prerequisites: Run files 01-05 first

-- ============================================================================
-- CONFIGURATION VARIABLES
-- ============================================================================
-- IMPORTANT: Replace with your actual project ID or set as environment variable
DECLARE PROJECT_ID STRING DEFAULT 'YOUR_PROJECT_ID';
DECLARE DATASET_ID STRING DEFAULT 'clinical_trial_matching';

-- ============================================================================

-- ============================================================================
-- SECTION 1: SEMANTIC SIMILARITY MATCHING WITH VECTOR SEARCH
-- ============================================================================

-- Drop existing match tables for clean rebuild
DROP TABLE IF EXISTS `${PROJECT_ID}.${DATASET_ID}.semantic_matches`;

-- Create semantic matches using VECTOR_SEARCH with TreeAH indexes
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.semantic_matches`
PARTITION BY DATE(match_generated_at)
CLUSTER BY patient_id, rank
AS
WITH active_patients AS (
  -- Select ALL patients for trial matching (removed Active_Ready filter per remediation)
  SELECT
    patient_id,
    embedding,
    trial_readiness,
    clinical_complexity,
    risk_category
  FROM `${PROJECT_ID}.${DATASET_ID}.patient_embeddings`
  -- WHERE trial_readiness = 'Active_Ready'  -- REMOVED: Only 3/10000 patients qualify
  LIMIT 10000  -- Process in batches for performance
),
-- Note: Using table-to-table VECTOR_SEARCH for batch processing
-- This approach processes all patients at once without LATERAL joins
vector_matches AS (
  SELECT
    vs.query.patient_id,
    vs.query.trial_readiness,
    vs.query.clinical_complexity,
    vs.query.risk_category,
    vs.base.nct_id AS trial_id,
    vs.base.brief_title AS trial_title,
    vs.base.phase AS trial_phase,
    vs.base.therapeutic_area,
    vs.distance,
    (1 - vs.distance) AS cosine_similarity,
    ROW_NUMBER() OVER (PARTITION BY vs.query.patient_id ORDER BY vs.distance ASC) AS rank
  FROM VECTOR_SEARCH(
    TABLE `${PROJECT_ID}.${DATASET_ID}.trial_embeddings`,
    'embedding',
    TABLE active_patients,
    'embedding',
    top_k => 50,
    options => '{"fraction_lists_to_search": 0.05, "use_brute_force": false}'
  ) AS vs
  -- WHERE vs.query.trial_readiness = 'Active_Ready'  -- REMOVED: Too restrictive
)
SELECT
  patient_id,
  trial_readiness,
  clinical_complexity,
  risk_category,
  trial_id,
  trial_title,
  trial_phase,
  therapeutic_area,
  distance,
  cosine_similarity,
  rank,

  -- Categorize match quality based on similarity
  CASE
    WHEN cosine_similarity >= 0.85 THEN 'EXCELLENT_MATCH'
    WHEN cosine_similarity >= 0.75 THEN 'GOOD_MATCH'
    WHEN cosine_similarity >= 0.65 THEN 'FAIR_MATCH'
    ELSE 'WEAK_MATCH'
  END AS match_quality,

  CURRENT_TIMESTAMP() AS match_generated_at

FROM vector_matches
WHERE rank <= 50

-- ============================================================================
-- SECTION 2: CLINICAL CRITERIA MATCHING
-- ============================================================================

-- Create clinical eligibility matches
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.clinical_matches` AS
WITH patient_clinical AS (
  SELECT
    p.patient_id,
    p.age,
    p.gender,
    p.condition_categories,
    p.current_medications,
    p.medication_categories,
    p.creatinine,
    p.hemoglobin,
    p.platelets,
    p.has_anticoagulant,
    p.clinical_complexity
  FROM `${PROJECT_ID}.${DATASET_ID}.patient_profile` p
  -- WHERE p.trial_readiness = 'Active_Ready'  -- REMOVED: Include all patients
),
trial_requirements AS (
  SELECT
    t.nct_id,
    t.condition,
    t.phase,
    t.therapeutic_area,
    -- Extract age requirements (simplified)
    CASE
      WHEN CONTAINS_SUBSTR(LOWER(t.eligibility_criteria), '18 years')
      THEN 18
      ELSE 0
    END AS min_age,
    CASE
      WHEN CONTAINS_SUBSTR(LOWER(t.eligibility_criteria), '65 years')
      THEN 65
      WHEN CONTAINS_SUBSTR(LOWER(t.eligibility_criteria), '75 years')
      THEN 75
      ELSE 120
    END AS max_age,
    -- Extract lab requirements (simplified)
    CONTAINS_SUBSTR(LOWER(t.eligibility_criteria), 'creatinine') AS requires_creatinine_check,
    CONTAINS_SUBSTR(LOWER(t.eligibility_criteria), 'hemoglobin') AS requires_hemoglobin_check,
    CONTAINS_SUBSTR(LOWER(t.eligibility_criteria), 'platelet') AS requires_platelet_check
  FROM `${PROJECT_ID}.${DATASET_ID}.trials_comprehensive` t
  WHERE t.overall_status = 'Recruiting'
)
SELECT
  pc.patient_id,
  tr.nct_id AS trial_id,

  -- Age eligibility
  CASE
    WHEN pc.age >= tr.min_age AND pc.age <= tr.max_age THEN 1.0
    ELSE 0.0
  END AS age_match_score,

  -- Condition match
  CASE
    WHEN tr.therapeutic_area IN UNNEST(pc.condition_categories) THEN 1.0
    WHEN tr.therapeutic_area = 'General' THEN 0.5
    ELSE 0.0
  END AS condition_match_score,

  -- Lab eligibility (simplified)
  CASE
    WHEN tr.requires_creatinine_check AND pc.creatinine IS NOT NULL
         AND pc.creatinine <= 1.5 THEN 1.0
    WHEN tr.requires_creatinine_check AND pc.creatinine > 1.5 THEN 0.0
    WHEN NOT tr.requires_creatinine_check THEN 1.0
    ELSE 0.5
  END AS creatinine_eligible_score,

  CASE
    WHEN tr.requires_hemoglobin_check AND pc.hemoglobin IS NOT NULL
         AND pc.hemoglobin >= 10.0 THEN 1.0
    WHEN tr.requires_hemoglobin_check AND pc.hemoglobin < 10.0 THEN 0.0
    WHEN NOT tr.requires_hemoglobin_check THEN 1.0
    ELSE 0.5
  END AS hemoglobin_eligible_score,

  CASE
    WHEN tr.requires_platelet_check AND pc.platelets IS NOT NULL
         AND pc.platelets >= 100 THEN 1.0
    WHEN tr.requires_platelet_check AND pc.platelets < 100 THEN 0.0
    WHEN NOT tr.requires_platelet_check THEN 1.0
    ELSE 0.5
  END AS platelet_eligible_score,

  -- Medication considerations
  CASE
    WHEN pc.has_anticoagulant AND
         CONTAINS_SUBSTR(LOWER(tr.eligibility_criteria), 'anticoagulant')
    THEN 0.0  -- Excluded if on anticoagulants
    ELSE 1.0
  END AS medication_eligible_score,

  -- Gender eligibility (FIXED: Map F→FEMALE, M→MALE per remediation)
  CASE
    -- Check if trial accepts all genders
    WHEN CONTAINS_SUBSTR(UPPER(tr.eligibility_criteria), 'ALL')
         OR CONTAINS_SUBSTR(UPPER(tr.eligibility_criteria), 'BOTH') THEN 1.0
    -- Map patient gender F to FEMALE
    WHEN pc.gender = 'F' AND CONTAINS_SUBSTR(UPPER(tr.eligibility_criteria), 'FEMALE') THEN 1.0
    -- Map patient gender M to MALE
    WHEN pc.gender = 'M' AND CONTAINS_SUBSTR(UPPER(tr.eligibility_criteria), 'MALE')
         AND NOT CONTAINS_SUBSTR(UPPER(tr.eligibility_criteria), 'FEMALE') THEN 1.0
    -- Default to eligible if no specific gender requirement found
    WHEN NOT CONTAINS_SUBSTR(UPPER(tr.eligibility_criteria), 'MALE')
         AND NOT CONTAINS_SUBSTR(UPPER(tr.eligibility_criteria), 'FEMALE') THEN 1.0
    ELSE 0.0
  END AS gender_eligible_score,

  CURRENT_TIMESTAMP() AS clinical_match_timestamp

FROM patient_clinical pc
CROSS JOIN trial_requirements tr
WHERE
  -- Basic filtering to reduce cross-join size
  pc.age >= tr.min_age - 5 AND pc.age <= tr.max_age + 5;

-- ============================================================================
-- SECTION 3: HYBRID SCORING AND RANKING
-- ============================================================================

-- Create hybrid match scores combining semantic and clinical
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.hybrid_match_scores`
PARTITION BY DATE(score_generated_at)
CLUSTER BY patient_id, final_rank
AS
WITH combined_scores AS (
  SELECT
    sm.patient_id,
    sm.trial_id,
    sm.trial_title,
    sm.trial_phase,
    sm.therapeutic_area,
    sm.cosine_similarity,
    sm.match_quality,

    -- Clinical scores
    COALESCE(cm.age_match_score, 0.5) AS age_score,
    COALESCE(cm.condition_match_score, 0.5) AS condition_score,
    COALESCE(cm.creatinine_eligible_score, 0.5) AS creatinine_score,
    COALESCE(cm.hemoglobin_eligible_score, 0.5) AS hemoglobin_score,
    COALESCE(cm.platelet_eligible_score, 0.5) AS platelet_score,
    COALESCE(cm.medication_eligible_score, 1.0) AS medication_score,
    COALESCE(cm.gender_eligible_score, 1.0) AS gender_score,  -- Added per remediation

    -- AI eligibility scores (if available)
    COALESCE(aes.eligibility_score / 100.0, 0.5) AS ai_eligibility_score

  FROM `${PROJECT_ID}.${DATASET_ID}.semantic_matches` sm
  LEFT JOIN `${PROJECT_ID}.${DATASET_ID}.clinical_matches` cm
    ON sm.patient_id = cm.patient_id AND sm.trial_id = cm.trial_id
  LEFT JOIN `${PROJECT_ID}.${DATASET_ID}.v_ai_eligibility_scores` aes
    ON sm.patient_id = aes.patient_id AND sm.trial_id = aes.nct_id
)
SELECT
  patient_id,
  trial_id,
  trial_title,
  trial_phase,
  therapeutic_area,

  -- Individual scores
  cosine_similarity,
  age_score,
  condition_score,
  creatinine_score,
  hemoglobin_score,
  platelet_score,
  medication_score,
  ai_eligibility_score,

  -- Calculate hybrid score with weights
  (
    0.30 * cosine_similarity +           -- 30% semantic similarity
    0.20 * condition_score +              -- 20% condition match
    0.15 * ai_eligibility_score +         -- 15% AI assessment
    0.10 * age_score +                    -- 10% age eligibility
    0.10 * medication_score +             -- 10% medication compatibility
    0.05 * creatinine_score +             -- 5% creatinine eligibility
    0.05 * hemoglobin_score +             -- 5% hemoglobin eligibility
    0.05 * platelet_score                 -- 5% platelet eligibility
  ) AS hybrid_score,

  -- Overall eligibility (all must pass)
  CASE
    WHEN age_score > 0 AND
         medication_score > 0 AND
         creatinine_score > 0 AND
         hemoglobin_score > 0 AND
         platelet_score > 0
    THEN TRUE
    ELSE FALSE
  END AS is_eligible,

  -- Match confidence
  CASE
    WHEN cosine_similarity >= 0.80 AND ai_eligibility_score >= 0.70
    THEN 'HIGH_CONFIDENCE'
    WHEN cosine_similarity >= 0.70 OR ai_eligibility_score >= 0.60
    THEN 'MEDIUM_CONFIDENCE'
    ELSE 'LOW_CONFIDENCE'
  END AS match_confidence,

  -- Ranking within patient
  ROW_NUMBER() OVER (
    PARTITION BY patient_id
    ORDER BY
      (0.30 * cosine_similarity + 0.20 * condition_score +
       0.15 * ai_eligibility_score + 0.10 * age_score +
       0.10 * medication_score + 0.05 * creatinine_score +
       0.05 * hemoglobin_score + 0.05 * platelet_score) DESC
  ) AS final_rank,

  CURRENT_TIMESTAMP() AS score_generated_at

FROM combined_scores;

-- ============================================================================
-- SECTION 4: BEST MATCHES SELECTION
-- ============================================================================

-- Create view for best patient-trial matches
CREATE OR REPLACE VIEW `${PROJECT_ID}.${DATASET_ID}.v_best_matches` AS
WITH ranked_matches AS (
  SELECT
    patient_id,
    trial_id,
    trial_title,
    trial_phase,
    therapeutic_area,
    hybrid_score,
    cosine_similarity,
    is_eligible,
    match_confidence,
    final_rank
  FROM `${PROJECT_ID}.${DATASET_ID}.hybrid_match_scores`
  WHERE is_eligible = TRUE
    AND final_rank <= 10  -- Top 10 per patient
)
SELECT
  rm.*,

  -- Add patient details
  pp.age AS patient_age,
  pp.gender AS patient_gender,
  pp.primary_diagnosis AS patient_diagnosis,
  pp.clinical_complexity AS patient_complexity,
  pp.trial_readiness AS patient_readiness,

  -- Match categorization
  CASE
    WHEN hybrid_score >= 0.80 THEN 'PRIORITY_1_EXCELLENT'
    WHEN hybrid_score >= 0.70 THEN 'PRIORITY_2_GOOD'
    WHEN hybrid_score >= 0.60 THEN 'PRIORITY_3_FAIR'
    ELSE 'PRIORITY_4_CONSIDER'
  END AS match_priority,

  -- Recommendation
  CASE
    WHEN hybrid_score >= 0.80 AND match_confidence = 'HIGH_CONFIDENCE'
    THEN 'Strongly recommend for immediate screening'
    WHEN hybrid_score >= 0.70 AND match_confidence IN ('HIGH_CONFIDENCE', 'MEDIUM_CONFIDENCE')
    THEN 'Recommend for screening'
    WHEN hybrid_score >= 0.60
    THEN 'Consider for screening if capacity allows'
    ELSE 'Keep as backup option'
  END AS recommendation

FROM ranked_matches rm
LEFT JOIN `${PROJECT_ID}.${DATASET_ID}.patient_profile` pp
  ON rm.patient_id = pp.patient_id
ORDER BY rm.patient_id, rm.final_rank;

-- ============================================================================
-- SECTION 5: MATCH DISTRIBUTION ANALYSIS
-- ============================================================================

-- Create match distribution statistics
CREATE OR REPLACE VIEW `${PROJECT_ID}.${DATASET_ID}.v_match_distribution` AS
WITH match_stats AS (
  SELECT
    patient_id,
    COUNT(*) AS total_matches,
    COUNTIF(is_eligible) AS eligible_matches,
    COUNTIF(hybrid_score >= 0.70) AS high_quality_matches,
    COUNTIF(match_confidence = 'HIGH_CONFIDENCE') AS high_confidence_matches,
    AVG(hybrid_score) AS avg_hybrid_score,
    MAX(hybrid_score) AS best_match_score,
    MIN(final_rank) AS best_rank
  FROM `${PROJECT_ID}.${DATASET_ID}.hybrid_match_scores`
  GROUP BY patient_id
)
SELECT
  COUNT(DISTINCT patient_id) AS total_patients,
  AVG(total_matches) AS avg_matches_per_patient,
  AVG(eligible_matches) AS avg_eligible_per_patient,
  AVG(high_quality_matches) AS avg_high_quality_per_patient,

  -- Distribution buckets
  COUNTIF(eligible_matches = 0) AS patients_with_no_matches,
  COUNTIF(eligible_matches BETWEEN 1 AND 5) AS patients_with_1_to_5_matches,
  COUNTIF(eligible_matches BETWEEN 6 AND 10) AS patients_with_6_to_10_matches,
  COUNTIF(eligible_matches > 10) AS patients_with_over_10_matches,

  -- Quality metrics
  AVG(avg_hybrid_score) AS overall_avg_score,
  AVG(best_match_score) AS avg_best_match_score,

  -- Success rate
  ROUND(100.0 * COUNTIF(eligible_matches > 0) / COUNT(*), 2) AS match_success_rate_pct,

  CURRENT_TIMESTAMP() AS analysis_timestamp
FROM match_stats;

-- ============================================================================
-- SECTION 6: TRIAL RECRUITMENT POTENTIAL
-- ============================================================================

-- Create trial recruitment potential view
CREATE OR REPLACE VIEW `${PROJECT_ID}.${DATASET_ID}.v_trial_recruitment` AS
WITH trial_matches AS (
  SELECT
    trial_id,
    trial_title,
    trial_phase,
    therapeutic_area,
    COUNT(DISTINCT patient_id) AS matched_patients,
    COUNTIF(is_eligible) AS eligible_patients,
    COUNTIF(hybrid_score >= 0.70) AS high_quality_candidates,
    AVG(hybrid_score) AS avg_match_score,
    MAX(hybrid_score) AS best_match_score
  FROM `${PROJECT_ID}.${DATASET_ID}.hybrid_match_scores`
  GROUP BY trial_id, trial_title, trial_phase, therapeutic_area
)
SELECT
  tm.*,
  tc.enrollment_count AS target_enrollment,

  -- Recruitment potential
  ROUND(100.0 * eligible_patients / NULLIF(tc.enrollment_count, 0), 2) AS recruitment_fill_pct,

  -- Priority for trials
  CASE
    WHEN eligible_patients >= tc.enrollment_count * 0.5 THEN 'HIGH_RECRUITMENT_POTENTIAL'
    WHEN eligible_patients >= tc.enrollment_count * 0.2 THEN 'MODERATE_RECRUITMENT_POTENTIAL'
    ELSE 'LOW_RECRUITMENT_POTENTIAL'
  END AS recruitment_potential,

  CURRENT_TIMESTAMP() AS assessment_timestamp

FROM trial_matches tm
LEFT JOIN `${PROJECT_ID}.${DATASET_ID}.trials_comprehensive` tc
  ON tm.trial_id = tc.nct_id
ORDER BY eligible_patients DESC;

-- ============================================================================
-- SECTION 7: MATCHING PIPELINE VALIDATION
-- ============================================================================

-- Create comprehensive validation summary
WITH pipeline_metrics AS (
  SELECT
    'Matching Pipeline Metrics' AS report_type,
    (SELECT COUNT(DISTINCT patient_id)
     FROM `${PROJECT_ID}.${DATASET_ID}.semantic_matches`) AS patients_with_semantic_matches,
    (SELECT COUNT(DISTINCT patient_id)
     FROM `${PROJECT_ID}.${DATASET_ID}.clinical_matches`) AS patients_with_clinical_matches,
    (SELECT COUNT(DISTINCT patient_id)
     FROM `${PROJECT_ID}.${DATASET_ID}.hybrid_match_scores`) AS patients_with_hybrid_scores,
    (SELECT COUNT(*)
     FROM `${PROJECT_ID}.${DATASET_ID}.hybrid_match_scores`
     WHERE is_eligible = TRUE) AS total_eligible_matches,
    (SELECT AVG(hybrid_score)
     FROM `${PROJECT_ID}.${DATASET_ID}.hybrid_match_scores`) AS avg_hybrid_score,
    (SELECT COUNT(DISTINCT trial_id)
     FROM `${PROJECT_ID}.${DATASET_ID}.hybrid_match_scores`
     WHERE is_eligible = TRUE) AS trials_with_matches
)
SELECT
  report_type,
  patients_with_semantic_matches,
  patients_with_clinical_matches,
  patients_with_hybrid_scores,
  total_eligible_matches,
  ROUND(avg_hybrid_score, 3) AS avg_hybrid_score,
  trials_with_matches,
  CASE
    WHEN total_eligible_matches > 10000 THEN '✅ PRODUCTION SCALE MATCHING'
    WHEN total_eligible_matches > 1000 THEN '⚠️ PILOT SCALE MATCHING'
    ELSE '❌ INSUFFICIENT MATCHES'
  END AS pipeline_status,
  CURRENT_TIMESTAMP() AS validation_timestamp
FROM pipeline_metrics;

-- ============================================================================
-- MATCHING PIPELINE COMPLETE
-- ============================================================================

SELECT
  '🎯 MATCHING PIPELINE COMPLETE' AS status,
  'Comprehensive patient-trial matching implemented' AS message,
  STRUCT(
    (SELECT COUNT(DISTINCT patient_id) FROM `${PROJECT_ID}.${DATASET_ID}.hybrid_match_scores`) AS patients_matched,
    (SELECT COUNT(DISTINCT trial_id) FROM `${PROJECT_ID}.${DATASET_ID}.hybrid_match_scores`) AS trials_matched,
    (SELECT COUNT(*) FROM `${PROJECT_ID}.${DATASET_ID}.hybrid_match_scores` WHERE is_eligible = TRUE) AS eligible_matches,
    (SELECT ROUND(AVG(hybrid_score), 3) FROM `${PROJECT_ID}.${DATASET_ID}.hybrid_match_scores`) AS avg_match_score
  ) AS summary,
  CURRENT_TIMESTAMP() AS completed_at;

-- ============================================================================
-- NEXT STEPS
-- ============================================================================
/*
Matching pipeline complete! Next file to run:

07_ai_forecast.sql - Implement AI.FORECAST for enrollment predictions

This matching pipeline provides:
✅ Semantic similarity matching with VECTOR_SEARCH
✅ Clinical criteria evaluation
✅ Hybrid scoring combining multiple factors
✅ Patient-trial ranking and prioritization
✅ Match quality assessment
✅ Trial recruitment potential analysis
✅ Comprehensive validation metrics
*/