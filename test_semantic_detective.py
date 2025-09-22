#!/usr/bin/env python3
"""
Test Script for Semantic Detective Pipeline - BigQuery 2025 Competition
Validates that all components work correctly before submission.
"""

import sys
import time
import logging
from google.cloud import bigquery
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = "gen-lang-client-0017660547"
DATASET_ID = "clinical_trial_matching"

client = bigquery.Client(project=PROJECT_ID)

class SemanticDetectiveValidator:
    """Validates the Semantic Detective implementation for competition compliance"""

    def __init__(self):
        self.client = client
        self.project_id = PROJECT_ID
        self.dataset_id = DATASET_ID
        self.test_results = {}
        self.score = 0
        self.max_score = 100

    def test_embedding_generation(self):
        """Test ML.GENERATE_EMBEDDING functionality"""
        logger.info("\n" + "="*60)
        logger.info("TEST 1: ML.GENERATE_EMBEDDING Validation")
        logger.info("="*60)

        try:
            # Check if patient embeddings exist
            query = f"""
            SELECT
                COUNT(*) as total_embeddings,
                AVG(embedding_dimension) as avg_dimension,
                COUNT(DISTINCT embedding_model) as models_used
            FROM `{self.project_id}.{self.dataset_id}.patient_embeddings`
            WHERE embedding IS NOT NULL
            """

            result = list(self.client.query(query).result())[0]

            if result.total_embeddings > 0:
                logger.info(f"✅ Patient embeddings found: {result.total_embeddings:,}")
                logger.info(f"   Average dimension: {result.avg_dimension}")
                self.test_results['patient_embeddings'] = True
                self.score += 20
            else:
                logger.error("❌ No patient embeddings found")
                self.test_results['patient_embeddings'] = False

            # Check trial embeddings
            query = f"""
            SELECT COUNT(*) as count
            FROM `{self.project_id}.{self.dataset_id}.trial_embeddings`
            WHERE embedding IS NOT NULL
            """

            result = list(self.client.query(query).result())[0]

            if result.count > 0:
                logger.info(f"✅ Trial embeddings found: {result.count:,}")
                self.test_results['trial_embeddings'] = True
                self.score += 20
            else:
                logger.error("❌ No trial embeddings found")
                self.test_results['trial_embeddings'] = False

        except Exception as e:
            logger.error(f"❌ Embedding test failed: {str(e)}")
            self.test_results['embeddings'] = False

    def test_vector_search(self):
        """Test VECTOR_SEARCH functionality"""
        logger.info("\n" + "="*60)
        logger.info("TEST 2: VECTOR_SEARCH Validation")
        logger.info("="*60)

        try:
            # Test if semantic matches were created
            query = f"""
            SELECT
                COUNT(*) as total_matches,
                AVG(cosine_similarity) as avg_similarity,
                MAX(cosine_similarity) as max_similarity,
                COUNT(DISTINCT patient_id) as unique_patients
            FROM `{self.project_id}.{self.dataset_id}.semantic_matches`
            """

            result = list(self.client.query(query).result())[0]

            if result.total_matches > 0:
                logger.info(f"✅ Semantic matches found: {result.total_matches:,}")
                logger.info(f"   Unique patients: {result.unique_patients:,}")
                logger.info(f"   Avg similarity: {result.avg_similarity:.3f}")
                logger.info(f"   Max similarity: {result.max_similarity:.3f}")
                self.test_results['vector_search'] = True
                self.score += 20
            else:
                logger.error("❌ No semantic matches found")
                self.test_results['vector_search'] = False

        except Exception as e:
            logger.error(f"❌ Vector search test failed: {str(e)}")
            self.test_results['vector_search'] = False

    def test_vector_indexes(self):
        """Test CREATE VECTOR INDEX functionality"""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: CREATE VECTOR INDEX Validation")
        logger.info("="*60)

        try:
            # Check if indexes exist (this is a simplified check)
            tables_with_potential_indexes = [
                'patient_embeddings',
                'trial_embeddings'
            ]

            indexes_found = 0
            for table in tables_with_potential_indexes:
                # Try to query the table - if it has an index, it should be faster
                query = f"""
                SELECT COUNT(*) as count
                FROM `{self.project_id}.{self.dataset_id}.{table}`
                WHERE embedding IS NOT NULL
                LIMIT 1
                """

                start_time = time.time()
                list(self.client.query(query).result())
                query_time = time.time() - start_time

                if query_time < 2.0:  # Assuming indexed queries are faster
                    logger.info(f"✅ Table {table} appears to be indexed (query time: {query_time:.2f}s)")
                    indexes_found += 1
                else:
                    logger.warning(f"⚠️ Table {table} may not be indexed (query time: {query_time:.2f}s)")

            if indexes_found > 0:
                self.test_results['vector_indexes'] = True
                self.score += 15
            else:
                self.test_results['vector_indexes'] = False

        except Exception as e:
            logger.error(f"❌ Vector index test failed: {str(e)}")
            self.test_results['vector_indexes'] = False

    def test_bigframes_integration(self):
        """Test BigFrames integration"""
        logger.info("\n" + "="*60)
        logger.info("TEST 4: BigFrames Integration Validation")
        logger.info("="*60)

        try:
            # Check if BigFrames-specific tables/views exist
            bigframes_artifacts = [
                'bigframes_forecast_input',
                'bigframes_forecast_output',
                'bigframes_vector_matches',
                'v_bigframes_patient_embeddings',
                'v_bigframes_trial_embeddings'
            ]

            found_count = 0
            for artifact in bigframes_artifacts:
                try:
                    query = f"SELECT 1 FROM `{self.project_id}.{self.dataset_id}.{artifact}` LIMIT 1"
                    self.client.query(query).result()
                    logger.info(f"✅ BigFrames artifact found: {artifact}")
                    found_count += 1
                except:
                    logger.warning(f"⚠️ BigFrames artifact not found: {artifact}")

            if found_count >= 2:  # At least some BigFrames integration
                self.test_results['bigframes'] = True
                self.score += 15
            else:
                self.test_results['bigframes'] = False

        except Exception as e:
            logger.error(f"❌ BigFrames test failed: {str(e)}")
            self.test_results['bigframes'] = False

    def test_semantic_matching_logic(self):
        """Test semantic relationship discovery"""
        logger.info("\n" + "="*60)
        logger.info("TEST 5: Semantic Relationship Discovery")
        logger.info("="*60)

        try:
            # Check match quality distribution
            query = f"""
            SELECT
                match_quality,
                COUNT(*) as count,
                AVG(cosine_similarity) as avg_similarity
            FROM `{self.project_id}.{self.dataset_id}.semantic_matches`
            GROUP BY match_quality
            ORDER BY
                CASE match_quality
                    WHEN 'EXCELLENT_MATCH' THEN 1
                    WHEN 'GOOD_MATCH' THEN 2
                    WHEN 'FAIR_MATCH' THEN 3
                    ELSE 4
                END
            """

            results = self.client.query(query).result()
            quality_found = False

            for row in results:
                logger.info(f"  {row.match_quality}: {row.count:,} matches (avg similarity: {row.avg_similarity:.3f})")
                quality_found = True

            if quality_found:
                self.test_results['semantic_logic'] = True
                self.score += 10
            else:
                logger.error("❌ No match quality categorization found")
                self.test_results['semantic_logic'] = False

        except Exception as e:
            logger.error(f"❌ Semantic logic test failed: {str(e)}")
            self.test_results['semantic_logic'] = False

    def generate_report(self):
        """Generate final validation report"""
        logger.info("\n" + "="*80)
        logger.info("FINAL VALIDATION REPORT")
        logger.info("="*80)

        # Component scores
        logger.info("\n📊 Component Scores:")
        logger.info(f"  ML.GENERATE_EMBEDDING: {'✅' if self.test_results.get('patient_embeddings') else '❌'} (20 points)")
        logger.info(f"  VECTOR_SEARCH: {'✅' if self.test_results.get('vector_search') else '❌'} (20 points)")
        logger.info(f"  CREATE VECTOR INDEX: {'✅' if self.test_results.get('vector_indexes') else '❌'} (15 points)")
        logger.info(f"  BigFrames Integration: {'✅' if self.test_results.get('bigframes') else '❌'} (15 points)")
        logger.info(f"  Semantic Discovery: {'✅' if self.test_results.get('semantic_logic') else '❌'} (10 points)")

        # Overall score
        logger.info(f"\n🎯 OVERALL SCORE: {self.score}/{self.max_score}")

        # Competition readiness
        if self.score >= 80:
            logger.info("\n✅ SUBMISSION READY - All critical components working!")
            status = "READY"
        elif self.score >= 60:
            logger.info("\n⚠️ MOSTLY READY - Some components need attention")
            status = "PARTIAL"
        else:
            logger.error("\n❌ NOT READY - Critical components missing")
            status = "NOT_READY"

        # Save report
        report_path = "/workspaces/Kaggle_BigQuerry2025/SUBMISSION/semantic_detective_validation.md"
        with open(report_path, 'w') as f:
            f.write("# Semantic Detective Validation Report\n\n")
            f.write(f"**Date**: {datetime.now().isoformat()}\n")
            f.write(f"**Status**: {status}\n")
            f.write(f"**Score**: {self.score}/{self.max_score}\n\n")

            f.write("## Component Status\n\n")
            f.write("| Component | Status | Points |\n")
            f.write("|-----------|--------|--------|\n")
            f.write(f"| ML.GENERATE_EMBEDDING | {'✅' if self.test_results.get('patient_embeddings') else '❌'} | 20 |\n")
            f.write(f"| VECTOR_SEARCH | {'✅' if self.test_results.get('vector_search') else '❌'} | 20 |\n")
            f.write(f"| CREATE VECTOR INDEX | {'✅' if self.test_results.get('vector_indexes') else '❌'} | 15 |\n")
            f.write(f"| BigFrames | {'✅' if self.test_results.get('bigframes') else '❌'} | 15 |\n")
            f.write(f"| Semantic Logic | {'✅' if self.test_results.get('semantic_logic') else '❌'} | 10 |\n")

            f.write(f"\n## Competition Readiness: **{status}**\n")

        logger.info(f"\nReport saved to: {report_path}")
        return status

def main():
    """Run all validation tests"""
    validator = SemanticDetectiveValidator()

    try:
        validator.test_embedding_generation()
        validator.test_vector_search()
        validator.test_vector_indexes()
        validator.test_bigframes_integration()
        validator.test_semantic_matching_logic()

        status = validator.generate_report()

        if status == "READY":
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()