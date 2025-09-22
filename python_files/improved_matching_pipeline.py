#!/usr/bin/env python3
"""
Improved Clinical Trial Matching Pipeline
==========================================
Three-stage retrieval→ranking→eligibility flow optimized for precision and recall
Leverages BigQuery 2025 features: VECTOR_SEARCH, ML.GENERATE_EMBEDDING, Gemini 2.5 Flash
"""

import logging
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from google.cloud import bigquery
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MatchingConfig:
    """Configuration for the matching pipeline"""
    project_id: str = "YOUR_PROJECT_ID"
    dataset_id: str = "clinical_trial_matching"

    # Stage 1: Semantic Retrieval
    max_initial_candidates: int = 100
    vector_search_fraction: float = 0.1
    min_extraction_confidence: float = 0.8

    # Stage 2: Clinical Re-ranking
    max_ranked_candidates: int = 50
    clinical_weight: float = 0.35
    semantic_weight: float = 0.40
    basic_compatibility_weight: float = 0.25

    # Stage 3: Eligibility Evaluation
    eligibility_weight: float = 0.30
    composite_weight: float = 0.70
    min_eligible_score: float = 0.7

    # Performance
    batch_size: int = 10
    parallel_workers: int = 5
    timeout_seconds: int = 300

@dataclass
class PatientProfile:
    """Structured patient profile for matching"""
    patient_id: int
    age: int
    gender: str
    ckd_stage: Optional[int]
    anemia_severity: str
    hemoglobin: Optional[float]
    platelets: Optional[float]
    creatinine: Optional[float]
    glucose: Optional[float]

    # Comorbidities
    has_diabetes: bool
    has_hypertension: bool
    has_kidney_dysfunction: bool
    has_anemia: bool

    # Risk scores
    clinical_complexity_score: int
    trial_eligibility_risk: str
    lab_completeness_score: float

@dataclass
class TrialMatch:
    """Trial match result with detailed scoring"""
    patient_id: int
    trial_id: str
    title: str

    # Scoring components
    semantic_similarity: float
    clinical_score: float
    eligibility_score: float
    final_match_score: float

    # Eligibility details
    is_eligible: bool
    match_category: str
    inclusion_met: List[str]
    inclusion_failed: List[str]
    exclusions_triggered: List[str]

    # Explanations
    physician_explanation: str
    physician_summary: str
    confidence_level: float

    # Metadata
    processing_time_ms: float
    timestamp: datetime

class ImprovedMatchingPipeline:
    """Improved three-stage clinical trial matching pipeline"""

    def __init__(self, config: Optional[MatchingConfig] = None):
        self.config = config or MatchingConfig()
        self.client = bigquery.Client(project=self.config.project_id)

        # Performance tracking
        self.stage_times = {}
        self.total_queries = 0

    def get_patient_profile(self, patient_id: int) -> Optional[PatientProfile]:
        """Retrieve structured patient profile from enhanced table"""
        query = f"""
        SELECT
            subject_id,
            anchor_age,
            gender,
            ckd_stage,
            anemia_severity,
            hemoglobin,
            platelets,
            creatinine,
            glucose,
            has_diabetes,
            has_hypertension,
            has_kidney_dysfunction,
            has_anemia,
            clinical_complexity_score,
            trial_eligibility_risk,
            lab_completeness_score
        FROM `{self.config.project_id}.{self.config.dataset_id}.patients_enhanced`
        WHERE subject_id = {patient_id}
        """

        try:
            result = self.client.query(query).to_dataframe()
            if result.empty:
                logger.warning(f"Patient {patient_id} not found")
                return None

            row = result.iloc[0]
            return PatientProfile(
                patient_id=row['subject_id'],
                age=row['anchor_age'],
                gender=row['gender'],
                ckd_stage=row['ckd_stage'] if pd.notna(row['ckd_stage']) else None,
                anemia_severity=row['anemia_severity'],
                hemoglobin=row['hemoglobin'] if pd.notna(row['hemoglobin']) else None,
                platelets=row['platelets'] if pd.notna(row['platelets']) else None,
                creatinine=row['creatinine'] if pd.notna(row['creatinine']) else None,
                glucose=row['glucose'] if pd.notna(row['glucose']) else None,
                has_diabetes=row['has_diabetes'],
                has_hypertension=row['has_hypertension'],
                has_kidney_dysfunction=row['has_kidney_dysfunction'],
                has_anemia=row['has_anemia'],
                clinical_complexity_score=row['clinical_complexity_score'],
                trial_eligibility_risk=row['trial_eligibility_risk'],
                lab_completeness_score=row['lab_completeness_score']
            )

        except Exception as e:
            logger.error(f"Error retrieving patient {patient_id}: {e}")
            return None

    def stage1_semantic_retrieval(self, patient_id: int) -> Dict[str, Any]:
        """Stage 1: Semantic retrieval using VECTOR_SEARCH"""
        start_time = time.time()
        logger.info(f"Stage 1: Semantic retrieval for patient {patient_id}")

        try:
            # Call the SQL procedure
            query = f"""
            CALL `{self.config.project_id}.{self.config.dataset_id}.stage1_semantic_retrieval`(
                {patient_id}, {self.config.max_initial_candidates}
            )
            """

            job = self.client.query(query)
            results = list(job.result())

            if results:
                result = results[0]
                stage_result = {
                    'candidates_found': result.candidates_found,
                    'avg_similarity': result.avg_similarity,
                    'highly_compatible': result.highly_compatible,
                    'best_similarity': result.best_similarity,
                    'processing_time_ms': (time.time() - start_time) * 1000
                }

                logger.info(f"Stage 1 completed: {result.candidates_found} candidates found")
                return stage_result
            else:
                raise Exception("No results returned from stage 1")

        except Exception as e:
            logger.error(f"Stage 1 failed for patient {patient_id}: {e}")
            return {'error': str(e), 'processing_time_ms': (time.time() - start_time) * 1000}

    def stage2_clinical_ranking(self, patient_id: int) -> Dict[str, Any]:
        """Stage 2: Clinical constraint re-ranking"""
        start_time = time.time()
        logger.info(f"Stage 2: Clinical ranking for patient {patient_id}")

        try:
            query = f"""
            CALL `{self.config.project_id}.{self.config.dataset_id}.stage2_clinical_ranking`(
                {patient_id}, {self.config.max_ranked_candidates}
            )
            """

            job = self.client.query(query)
            results = list(job.result())

            if results:
                result = results[0]
                stage_result = {
                    'final_candidates': result.final_candidates,
                    'avg_composite_score': result.avg_composite_score,
                    'avg_clinical_score': result.avg_clinical_score,
                    'avg_semantic_score': result.avg_semantic_score,
                    'high_quality_matches': result.high_quality_matches,
                    'no_conflicts': result.no_conflicts,
                    'best_score': result.best_score,
                    'processing_time_ms': (time.time() - start_time) * 1000
                }

                logger.info(f"Stage 2 completed: {result.final_candidates} candidates ranked")
                return stage_result
            else:
                raise Exception("No results returned from stage 2")

        except Exception as e:
            logger.error(f"Stage 2 failed for patient {patient_id}: {e}")
            return {'error': str(e), 'processing_time_ms': (time.time() - start_time) * 1000}

    def stage3_eligibility_evaluation(self, patient_id: int) -> Dict[str, Any]:
        """Stage 3: Detailed eligibility evaluation with Gemini 2.5 Flash"""
        start_time = time.time()
        logger.info(f"Stage 3: Eligibility evaluation for patient {patient_id}")

        try:
            query = f"""
            CALL `{self.config.project_id}.{self.config.dataset_id}.stage3_eligibility_evaluation`(
                {patient_id}
            )
            """

            job = self.client.query(query)
            results = list(job.result())

            if results:
                result = results[0]
                stage_result = {
                    'total_evaluated': result.total_evaluated,
                    'eligible_matches': result.eligible_matches,
                    'excellent_matches': result.excellent_matches,
                    'good_matches': result.good_matches,
                    'avg_final_score': result.avg_final_score,
                    'avg_eligible_score': result.avg_eligible_score,
                    'best_match_score': result.best_match_score,
                    'avg_confidence': result.avg_confidence,
                    'processing_time_ms': (time.time() - start_time) * 1000
                }

                logger.info(f"Stage 3 completed: {result.eligible_matches}/{result.total_evaluated} eligible matches")
                return stage_result
            else:
                raise Exception("No results returned from stage 3")

        except Exception as e:
            logger.error(f"Stage 3 failed for patient {patient_id}: {e}")
            return {'error': str(e), 'processing_time_ms': (time.time() - start_time) * 1000}

    def get_final_matches(self, patient_id: int, limit: int = 10) -> List[TrialMatch]:
        """Retrieve final ranked matches for a patient"""
        query = f"""
        SELECT
            patient_id,
            trial_id,
            title,
            semantic_similarity_score,
            clinical_score,
            eligibility_evaluation.eligibility_score,
            final_match_score,
            eligibility_evaluation.is_eligible,
            match_category,
            eligibility_evaluation.inclusion_met,
            eligibility_evaluation.inclusion_failed,
            eligibility_evaluation.exclusions_triggered,
            eligibility_evaluation.physician_explanation,
            physician_summary,
            eligibility_evaluation.confidence_level,
            TIMESTAMP_DIFF(final_evaluation_timestamp, TIMESTAMP_SUB(final_evaluation_timestamp, INTERVAL 1 SECOND), MILLISECOND) as processing_time_ms,
            final_evaluation_timestamp
        FROM `{self.config.project_id}.{self.config.dataset_id}.final_matches`
        WHERE patient_id = {patient_id}
        ORDER BY final_match_score DESC
        LIMIT {limit}
        """

        try:
            results = self.client.query(query).to_dataframe()
            matches = []

            for _, row in results.iterrows():
                match = TrialMatch(
                    patient_id=row['patient_id'],
                    trial_id=row['trial_id'],
                    title=row['title'],
                    semantic_similarity=row['semantic_similarity_score'],
                    clinical_score=row['clinical_score'],
                    eligibility_score=row['eligibility_score'],
                    final_match_score=row['final_match_score'],
                    is_eligible=row['is_eligible'],
                    match_category=row['match_category'],
                    inclusion_met=row['inclusion_met'] if row['inclusion_met'] else [],
                    inclusion_failed=row['inclusion_failed'] if row['inclusion_failed'] else [],
                    exclusions_triggered=row['exclusions_triggered'] if row['exclusions_triggered'] else [],
                    physician_explanation=row['physician_explanation'],
                    physician_summary=row['physician_summary'],
                    confidence_level=row['confidence_level'],
                    processing_time_ms=row['processing_time_ms'] if pd.notna(row['processing_time_ms']) else 0,
                    timestamp=row['final_evaluation_timestamp']
                )
                matches.append(match)

            return matches

        except Exception as e:
            logger.error(f"Error retrieving final matches for patient {patient_id}: {e}")
            return []

    def run_complete_pipeline(self, patient_id: int) -> Dict[str, Any]:
        """Run the complete three-stage matching pipeline for a patient"""
        pipeline_start = time.time()
        logger.info(f"Starting complete matching pipeline for patient {patient_id}")

        # Validate patient exists
        patient_profile = self.get_patient_profile(patient_id)
        if not patient_profile:
            return {'error': f'Patient {patient_id} not found'}

        try:
            # Run all three stages using the orchestration procedure
            query = f"""
            CALL `{self.config.project_id}.{self.config.dataset_id}.run_complete_matching_pipeline`(
                {patient_id},
                {self.config.max_initial_candidates},
                {self.config.max_ranked_candidates}
            )
            """

            job = self.client.query(query)
            results = list(job.result())

            if results:
                result = results[0]

                # Get final matches
                final_matches = self.get_final_matches(patient_id)

                pipeline_result = {
                    'patient_id': patient_id,
                    'patient_profile': patient_profile.__dict__,
                    'pipeline_performance': {
                        'total_processing_time_ms': result.total_processing_time_ms,
                        'stage1_candidates': result.stage1_candidates,
                        'stage2_candidates': result.stage2_candidates,
                        'final_matches': result.final_matches,
                        'eligible_matches': result.eligible_matches,
                        'status': result.status
                    },
                    'final_matches': [match.__dict__ for match in final_matches],
                    'summary': {
                        'total_matches_found': len(final_matches),
                        'eligible_matches': len([m for m in final_matches if m.is_eligible]),
                        'excellent_matches': len([m for m in final_matches if m.match_category == 'Excellent Match']),
                        'good_matches': len([m for m in final_matches if m.match_category == 'Good Match']),
                        'avg_match_score': np.mean([m.final_match_score for m in final_matches]) if final_matches else 0,
                        'best_match_score': max([m.final_match_score for m in final_matches]) if final_matches else 0,
                        'avg_confidence': np.mean([m.confidence_level for m in final_matches]) if final_matches else 0
                    }
                }

                total_time = time.time() - pipeline_start
                pipeline_result['total_pipeline_time_seconds'] = total_time

                logger.info(f"Pipeline completed for patient {patient_id} in {total_time:.2f}s")
                logger.info(f"Found {len(final_matches)} matches, {pipeline_result['summary']['eligible_matches']} eligible")

                return pipeline_result
            else:
                raise Exception("No results returned from pipeline orchestration")

        except Exception as e:
            logger.error(f"Pipeline failed for patient {patient_id}: {e}")
            return {
                'patient_id': patient_id,
                'error': str(e),
                'total_pipeline_time_seconds': time.time() - pipeline_start
            }

    def batch_match_patients(self, patient_ids: List[int]) -> List[Dict[str, Any]]:
        """Run matching pipeline for multiple patients in parallel"""
        logger.info(f"Starting batch matching for {len(patient_ids)} patients")
        start_time = time.time()

        results = []

        # Process in batches with parallel workers
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = []

            for patient_id in patient_ids:
                future = executor.submit(self.run_complete_pipeline, patient_id)
                futures.append((patient_id, future))

                # Rate limiting to avoid overwhelming BigQuery
                time.sleep(0.1)

            # Collect results
            for patient_id, future in futures:
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch processing failed for patient {patient_id}: {e}")
                    results.append({
                        'patient_id': patient_id,
                        'error': str(e)
                    })

        total_time = time.time() - start_time
        successful = len([r for r in results if 'error' not in r])

        logger.info(f"Batch processing completed: {successful}/{len(patient_ids)} successful in {total_time:.2f}s")

        return results

    def generate_performance_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        successful_results = [r for r in results if 'error' not in r]

        if not successful_results:
            return {'error': 'No successful results to analyze'}

        # Extract performance metrics
        processing_times = [r['pipeline_performance']['total_processing_time_ms'] for r in successful_results]
        stage1_counts = [r['pipeline_performance']['stage1_candidates'] for r in successful_results]
        stage2_counts = [r['pipeline_performance']['stage2_candidates'] for r in successful_results]
        final_counts = [r['pipeline_performance']['final_matches'] for r in successful_results]
        eligible_counts = [r['pipeline_performance']['eligible_matches'] for r in successful_results]

        match_scores = []
        confidence_scores = []

        for result in successful_results:
            for match in result['final_matches']:
                match_scores.append(match['final_match_score'])
                confidence_scores.append(match['confidence_level'])

        report = {
            'summary': {
                'total_patients_processed': len(results),
                'successful_matches': len(successful_results),
                'failed_matches': len(results) - len(successful_results),
                'success_rate': len(successful_results) / len(results) * 100
            },
            'performance_metrics': {
                'avg_processing_time_ms': np.mean(processing_times),
                'median_processing_time_ms': np.median(processing_times),
                'p95_processing_time_ms': np.percentile(processing_times, 95),
                'min_processing_time_ms': np.min(processing_times),
                'max_processing_time_ms': np.max(processing_times)
            },
            'pipeline_efficiency': {
                'avg_stage1_candidates': np.mean(stage1_counts),
                'avg_stage2_candidates': np.mean(stage2_counts),
                'avg_final_matches': np.mean(final_counts),
                'avg_eligible_matches': np.mean(eligible_counts),
                'stage1_to_stage2_ratio': np.mean([s2/s1 for s1, s2 in zip(stage1_counts, stage2_counts) if s1 > 0]),
                'stage2_to_final_ratio': np.mean([f/s2 for s2, f in zip(stage2_counts, final_counts) if s2 > 0]),
                'eligibility_rate': np.mean([e/f for e, f in zip(eligible_counts, final_counts) if f > 0]) * 100
            },
            'match_quality': {
                'avg_match_score': np.mean(match_scores) if match_scores else 0,
                'median_match_score': np.median(match_scores) if match_scores else 0,
                'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'high_quality_matches_pct': len([s for s in match_scores if s >= 0.8]) / len(match_scores) * 100 if match_scores else 0
            },
            'recommendations': []
        }

        # Add performance recommendations
        if report['performance_metrics']['avg_processing_time_ms'] > 30000:  # > 30 seconds
            report['recommendations'].append("Consider increasing parallel workers or optimizing vector search parameters")

        if report['pipeline_efficiency']['eligibility_rate'] < 50:
            report['recommendations'].append("Low eligibility rate - consider adjusting stage 1 retrieval criteria")

        if report['match_quality']['avg_match_score'] < 0.6:
            report['recommendations'].append("Low match scores - review clinical compatibility scoring weights")

        return report

def main():
    """Example usage of the improved matching pipeline"""
    logger.info("Testing Improved Clinical Trial Matching Pipeline")

    # Initialize pipeline
    config = MatchingConfig()
    pipeline = ImprovedMatchingPipeline(config)

    # Test with a few patients
    test_patient_ids = [18320612, 19489569, 14125553]  # From the sample data we saw

    # Run single patient test
    logger.info("Testing single patient matching...")
    single_result = pipeline.run_complete_pipeline(test_patient_ids[0])

    if 'error' not in single_result:
        logger.info(f"Single patient test successful:")
        logger.info(f"  Found {single_result['summary']['total_matches_found']} matches")
        logger.info(f"  {single_result['summary']['eligible_matches']} eligible")
        logger.info(f"  Best score: {single_result['summary']['best_match_score']:.3f}")
        logger.info(f"  Processing time: {single_result['total_pipeline_time_seconds']:.2f}s")
    else:
        logger.error(f"Single patient test failed: {single_result['error']}")

    # Run batch test
    logger.info("Testing batch patient matching...")
    batch_results = pipeline.batch_match_patients(test_patient_ids)

    # Generate performance report
    performance_report = pipeline.generate_performance_report(batch_results)

    logger.info("Performance Report:")
    logger.info(f"  Success Rate: {performance_report['summary']['success_rate']:.1f}%")
    logger.info(f"  Avg Processing Time: {performance_report['performance_metrics']['avg_processing_time_ms']:.1f}ms")
    logger.info(f"  Eligibility Rate: {performance_report['pipeline_efficiency']['eligibility_rate']:.1f}%")
    logger.info(f"  Avg Match Score: {performance_report['match_quality']['avg_match_score']:.3f}")

    # Save detailed results
    with open(f'improved_matching_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump({
            'single_result': single_result,
            'batch_results': batch_results,
            'performance_report': performance_report
        }, f, indent=2, default=str)

    logger.info("Results saved to improved_matching_results_*.json")

if __name__ == "__main__":
    main()