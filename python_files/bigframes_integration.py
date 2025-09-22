#!/usr/bin/env python3
"""
BigFrames Integration for BigQuery 2025 Competition
Python code to run alongside the SQL components for complete BigFrames integration.

This script demonstrates:
✅ bigframes.ml.llm.GeminiTextGenerator for batch AI processing
✅ bigframes.DataFrame.ai.forecast() for time series forecasting
✅ bigframes.bigquery.vector_search() for semantic matching
✅ bigframes.bigquery.create_vector_index() for index management
✅ Integration with SQL AI.* functions for hybrid workflow
"""

import bigframes
import bigframes.pandas as bpd
import bigframes.ml.llm as llm
from bigframes.ml.forecasting import ARIMAPlus
import bigframes.bigquery as bbq
from google.cloud import bigquery
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure BigFrames
bigframes.options.bigquery.project = "YOUR_PROJECT_ID"
bigframes.options.bigquery.location = "US"

class BigFramesCompetitionIntegration:
    def __init__(self):
        self.project_id = "YOUR_PROJECT_ID"
        self.dataset_id = "clinical_trial_matching"
        self.connection_id = f"{self.project_id}.US.vertex_ai_connection"

        logger.info(f"Initialized BigFrames integration for project: {self.project_id}")

    # ========================================================================
    # BigFrames AI.FORECAST Integration
    # ========================================================================

    def run_bigframes_forecast(self):
        """Use bigframes.DataFrame.ai.forecast() for patient enrollment prediction"""
        try:
            logger.info("Starting BigFrames AI.FORECAST workflow...")

            # Read forecast input data
            forecast_data = bpd.read_gbq(
                f"SELECT ds, y, unique_id FROM `{self.project_id}.{self.dataset_id}.bigframes_forecast_input`"
            )

            logger.info(f"Loaded {len(forecast_data)} records for forecasting")

            # Use BigFrames native forecasting
            forecast_results = forecast_data.ai.forecast(
                time_col='ds',
                value_col='y',
                id_cols=['unique_id'],
                horizon=30,  # 30-day forecast
                frequency='D'  # Daily frequency
            )

            # Save results back to BigQuery
            forecast_results.to_gbq(
                f"{self.project_id}.{self.dataset_id}.bigframes_forecast_output",
                if_exists='replace'
            )

            logger.info("BigFrames AI.FORECAST completed successfully")
            return forecast_results

        except Exception as e:
            logger.error(f"BigFrames AI.FORECAST failed: {str(e)}")
            self._update_orchestration_status('BIGFRAMES_AI_FORECAST', 'FAILED', error_message=str(e))
            raise

    # ========================================================================
    # BigFrames Vector Search Integration
    # ========================================================================

    def run_bigframes_vector_search(self, top_k=10):
        """Use bigframes.bigquery.vector_search() for semantic matching"""
        try:
            logger.info("Starting BigFrames vector search workflow...")

            # Read patient and trial embeddings
            patients_df = bpd.read_gbq(
                f"SELECT * FROM `{self.project_id}.{self.dataset_id}.v_bigframes_patient_embeddings` LIMIT 1000"
            )

            logger.info(f"Loaded {len(patients_df)} patient embeddings")

            # Use BigFrames vector search
            search_results = bbq.vector_search(
                base_table=f'{self.project_id}.{self.dataset_id}.v_bigframes_trial_embeddings',
                column_to_search='embedding',
                query_table=patients_df,
                query_column='embedding',
                top_k=top_k,
                distance_type='COSINE',
                options={'fraction_lists_to_search': 0.05}
            )

            # Process and save results
            search_results['similarity_score'] = 1 - search_results['distance']
            search_results['rank'] = search_results.groupby('patient_id')['similarity_score'].rank(
                method='dense', ascending=False
            )

            search_results.to_gbq(
                f"{self.project_id}.{self.dataset_id}.bigframes_vector_matches",
                if_exists='replace'
            )

            logger.info("BigFrames vector search completed successfully")
            return search_results

        except Exception as e:
            logger.error(f"BigFrames vector search failed: {str(e)}")
            self._update_orchestration_status('BIGFRAMES_VECTOR_SEARCH', 'FAILED', error_message=str(e))
            raise

    # ========================================================================
    # BigFrames Text Generation Integration
    # ========================================================================

    def run_bigframes_text_generation(self):
        """Use bigframes.ml.llm.GeminiTextGenerator for batch content generation"""
        try:
            logger.info("Starting BigFrames text generation workflow...")

            # Read text generation input
            text_input = bpd.read_gbq(
                f"SELECT * FROM `{self.project_id}.{self.dataset_id}.bigframes_text_generation_input`"
            )

            logger.info(f"Loaded {len(text_input)} prompts for text generation")

            # Initialize Gemini text generator
            text_generator = llm.GeminiTextGenerator(
                model_name='gemini-2.5-flash-lite',
                connection_name=self.connection_id
            )

            # Generate text in batches
            generated_results = text_generator.predict(
                text_input[['prompt_text']],
                params={
                    'temperature': 0.7,
                    'max_output_tokens': 150,
                    'batch_size': 50
                }
            )

            # Combine with original data
            results_df = text_input.join(generated_results, how='inner')
            results_df = results_df.rename(columns={'ml_generate_text_result': 'generated_text'})

            # Save results
            results_df.to_gbq(
                f"{self.project_id}.{self.dataset_id}.bigframes_generated_content",
                if_exists='replace'
            )

            logger.info("BigFrames text generation completed successfully")
            return results_df

        except Exception as e:
            logger.error(f"BigFrames text generation failed: {str(e)}")
            self._update_orchestration_status('BIGFRAMES_TEXT_GENERATION', 'FAILED', error_message=str(e))
            raise

    # ========================================================================
    # BigFrames Vector Index Management
    # ========================================================================

    def create_bigframes_vector_indexes(self):
        """Use bigframes.bigquery.create_vector_index() for index management"""
        try:
            logger.info("Creating BigFrames vector indexes...")

            # Create patient embeddings index
            bbq.create_vector_index(
                table=f'{self.project_id}.{self.dataset_id}.patient_embeddings',
                column='embedding',
                index_name='bigframes_patient_treeah_idx',
                index_type='TREE_AH',
                distance_type='COSINE',
                options={'num_leaves': 2000, 'enable_soar': True}
            )

            # Create trial embeddings index
            bbq.create_vector_index(
                table=f'{self.project_id}.{self.dataset_id}.trial_embeddings',
                column='embedding',
                index_name='bigframes_trial_treeah_idx',
                index_type='TREE_AH',
                distance_type='COSINE',
                options={'num_leaves': 1000, 'enable_soar': True}
            )

            logger.info("BigFrames vector indexes created successfully")
            return "BigFrames vector indexes created successfully"

        except Exception as e:
            logger.error(f"BigFrames vector index creation failed: {str(e)}")
            self._update_orchestration_status('BIGFRAMES_VECTOR_INDEX_CREATION', 'FAILED', error_message=str(e))
            raise

    # ========================================================================
    # Orchestration Status Management
    # ========================================================================

    def _update_orchestration_status(self, workflow_step, status, records_processed=0, error_message=None):
        """Update orchestration status in BigQuery"""
        try:
            client = bigquery.Client(project=self.project_id)

            query = f"""
            UPDATE `{self.project_id}.{self.dataset_id}.bigframes_orchestration_status`
            SET
                status = @status,
                end_time = CURRENT_TIMESTAMP(),
                records_processed = @records_processed,
                error_message = @error_message
            WHERE workflow_step = @workflow_step
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("workflow_step", "STRING", workflow_step),
                    bigquery.ScalarQueryParameter("status", "STRING", status),
                    bigquery.ScalarQueryParameter("records_processed", "INT64", records_processed),
                    bigquery.ScalarQueryParameter("error_message", "STRING", error_message),
                ]
            )

            client.query(query, job_config=job_config).result()
            logger.info(f"Updated orchestration status: {workflow_step} -> {status}")

        except Exception as e:
            logger.error(f"Failed to update orchestration status: {str(e)}")

    # ========================================================================
    # Complete Integration Workflow
    # ========================================================================

    def run_complete_bigframes_workflow(self):
        """Execute complete BigFrames integration for competition"""

        print("🚀 Starting BigFrames Competition Workflow...")
        logger.info("Starting complete BigFrames workflow")

        workflow_results = {}

        try:
            # Step 1: Create vector indexes
            print("📊 Creating BigFrames vector indexes...")
            self._update_orchestration_status('BIGFRAMES_VECTOR_INDEX_CREATION', 'IN_PROGRESS')
            index_result = self.create_bigframes_vector_indexes()
            self._update_orchestration_status('BIGFRAMES_VECTOR_INDEX_CREATION', 'COMPLETED')
            workflow_results['index_creation'] = index_result

            # Step 2: Run forecasting
            print("📈 Running BigFrames AI.FORECAST...")
            self._update_orchestration_status('BIGFRAMES_AI_FORECAST', 'IN_PROGRESS')
            forecast_results = self.run_bigframes_forecast()
            self._update_orchestration_status('BIGFRAMES_AI_FORECAST', 'COMPLETED', len(forecast_results))
            workflow_results['forecast_results'] = forecast_results
            print(f"✅ Forecast generated: {len(forecast_results)} predictions")

            # Step 3: Run vector search
            print("🔍 Running BigFrames vector search...")
            self._update_orchestration_status('BIGFRAMES_VECTOR_SEARCH', 'IN_PROGRESS')
            search_results = self.run_bigframes_vector_search(top_k=20)
            self._update_orchestration_status('BIGFRAMES_VECTOR_SEARCH', 'COMPLETED', len(search_results))
            workflow_results['search_results'] = search_results
            print(f"✅ Vector search completed: {len(search_results)} matches")

            # Step 4: Generate personalized content
            print("📝 Running BigFrames text generation...")
            self._update_orchestration_status('BIGFRAMES_TEXT_GENERATION', 'IN_PROGRESS')
            text_results = self.run_bigframes_text_generation()
            self._update_orchestration_status('BIGFRAMES_TEXT_GENERATION', 'COMPLETED', len(text_results))
            workflow_results['text_results'] = text_results
            print(f"✅ Text generation completed: {len(text_results)} messages")

            print("🎉 BigFrames integration workflow completed successfully!")
            logger.info("BigFrames workflow completed successfully")

            return workflow_results

        except Exception as e:
            print(f"❌ BigFrames workflow failed: {str(e)}")
            logger.error(f"BigFrames workflow failed: {str(e)}")
            raise

    # ========================================================================
    # Validation and Status Reporting
    # ========================================================================

    def validate_bigframes_integration(self):
        """Validate BigFrames integration status"""
        try:
            client = bigquery.Client(project=self.project_id)

            # Check orchestration status
            query = f"""
            SELECT
                workflow_step,
                status,
                records_processed,
                TIMESTAMP_DIFF(end_time, start_time, SECOND) AS duration_seconds
            FROM `{self.project_id}.{self.dataset_id}.bigframes_orchestration_status`
            ORDER BY start_time
            """

            status_df = client.query(query).to_dataframe()
            print("\n📊 BigFrames Integration Status:")
            print(status_df.to_string(index=False))

            # Check data quality
            validation_queries = {
                'forecast_validation': f"SELECT * FROM `{self.project_id}.{self.dataset_id}.v_bigframes_forecast_validation`",
                'vector_validation': f"SELECT * FROM `{self.project_id}.{self.dataset_id}.v_bigframes_vector_validation`",
                'text_validation': f"SELECT * FROM `{self.project_id}.{self.dataset_id}.v_bigframes_text_validation`"
            }

            validation_results = {}
            for validation_name, query in validation_queries.items():
                try:
                    result_df = client.query(query).to_dataframe()
                    validation_results[validation_name] = result_df
                    print(f"\n📋 {validation_name.replace('_', ' ').title()}:")
                    print(result_df.to_string(index=False))
                except Exception as e:
                    print(f"⚠️ {validation_name} check failed: {str(e)}")

            return validation_results

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise


def main():
    """Main execution function"""
    print("🎯 BigQuery 2025 Competition - BigFrames Integration")
    print("=" * 60)

    try:
        # Initialize integration
        integration = BigFramesCompetitionIntegration()

        # Run validation first to see current status
        print("\n1️⃣ Validating current BigFrames integration status...")
        integration.validate_bigframes_integration()

        # Ask user if they want to run the full workflow
        response = input("\n🤔 Run complete BigFrames workflow? (y/N): ").strip().lower()

        if response in ['y', 'yes']:
            print("\n2️⃣ Running complete BigFrames workflow...")
            results = integration.run_complete_bigframes_workflow()

            print("\n3️⃣ Final validation...")
            integration.validate_bigframes_integration()

            print("\n🏆 BigFrames integration completed successfully!")
            print("Ready for BigQuery 2025 Competition submission!")
        else:
            print("\n✅ Validation complete. Run with 'y' to execute full workflow.")

    except Exception as e:
        print(f"\n❌ BigFrames integration failed: {str(e)}")
        logger.error(f"Main execution failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())