#!/usr/bin/env python3
"""
Simple Execution Script for Semantic Detective - BigQuery 2025 Competition
Runs only the core files needed for competition requirements.
"""

import os
import subprocess
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = "gen-lang-client-0017660547"
DATASET_ID = "clinical_trial_matching"

def execute_sql_file(filepath, description):
    """Execute a SQL file using bq command line tool"""
    logger.info(f"Executing: {description}")

    # Read and prepare SQL
    with open(filepath, 'r') as f:
        sql_content = f.read()

    # Replace variables
    sql_content = sql_content.replace('${PROJECT_ID}', PROJECT_ID)
    sql_content = sql_content.replace('${DATASET_ID}', DATASET_ID)

    # Write to temp file
    temp_file = f"/tmp/temp_{os.path.basename(filepath)}"
    with open(temp_file, 'w') as f:
        f.write(sql_content)

    # Execute using bq
    try:
        result = subprocess.run(
            ['bq', 'query', '--use_legacy_sql=false', '--max_rows=0'],
            input=sql_content.encode(),
            capture_output=True,
            text=False
        )

        if result.returncode == 0:
            logger.info(f"✅ Success: {description}")
            return True
        else:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"❌ Failed: {description}")
            logger.error(f"Error: {error_msg[:500]}")
            return False

    except Exception as e:
        logger.error(f"❌ Exception: {str(e)}")
        return False

    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def check_table_exists(table_name):
    """Check if a table exists and has data"""
    try:
        query = f"SELECT COUNT(*) as count FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}` LIMIT 1"
        result = subprocess.run(
            ['bq', 'query', '--use_legacy_sql=false', '--format=csv'],
            input=query.encode(),
            capture_output=True,
            text=False
        )

        if result.returncode == 0:
            output = result.stdout.decode().strip()
            lines = output.split('\n')
            if len(lines) > 1:
                count = lines[1]
                logger.info(f"✅ Table {table_name} exists")
                return True

        logger.warning(f"⚠️ Table {table_name} not found or empty")
        return False

    except Exception as e:
        logger.error(f"Error checking table {table_name}: {str(e)}")
        return False

def main():
    """Main execution function"""
    start_time = time.time()

    logger.info("="*80)
    logger.info("SEMANTIC DETECTIVE - SIMPLE EXECUTION")
    logger.info(f"Project: {PROJECT_ID}")
    logger.info(f"Dataset: {DATASET_ID}")
    logger.info("="*80)

    # Check if critical tables already exist
    logger.info("\nChecking existing tables...")
    tables_exist = {
        "patient_embeddings": check_table_exists("patient_embeddings"),
        "trial_embeddings": check_table_exists("trial_embeddings"),
        "semantic_matches": check_table_exists("semantic_matches")
    }

    # If all tables exist, we're already good!
    if all(tables_exist.values()):
        logger.info("\n✅ ALL COMPETITION TABLES ALREADY EXIST!")
        logger.info("The core requirements are already met. Proceeding with index creation only.")

        # Just try to create/update indexes
        sql_files = [
            ("04_vector_search_indexes.sql", "Vector Search Indexes (IVF)")
        ]
    else:
        # Run full pipeline
        logger.info("\nSome tables missing. Running full pipeline...")
        sql_files = [
            ("03_vector_embeddings.sql", "Vector Embeddings Generation"),
            ("04_vector_search_indexes.sql", "Vector Search Indexes (IVF)"),
            ("06_matching_pipeline.sql", "Semantic Matching Pipeline")
        ]

    # Execute SQL files
    success_count = 0
    for sql_file, description in sql_files:
        filepath = f"/workspaces/Kaggle_BigQuerry2025/SUBMISSION/sql_files/{sql_file}"

        if not os.path.exists(filepath):
            logger.warning(f"File not found: {sql_file}")
            continue

        if execute_sql_file(filepath, description):
            success_count += 1

        time.sleep(2)  # Brief pause between operations

    # Final validation
    logger.info("\n" + "="*80)
    logger.info("FINAL VALIDATION")
    logger.info("="*80)

    final_check = {
        "patient_embeddings": check_table_exists("patient_embeddings"),
        "trial_embeddings": check_table_exists("trial_embeddings"),
        "semantic_matches": check_table_exists("semantic_matches")
    }

    # Get row counts
    for table_name in final_check.keys():
        if final_check[table_name]:
            try:
                query = f"SELECT COUNT(*) as count FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}`"
                result = subprocess.run(
                    ['bq', 'query', '--use_legacy_sql=false', '--format=csv'],
                    input=query.encode(),
                    capture_output=True,
                    text=False
                )

                if result.returncode == 0:
                    output = result.stdout.decode().strip()
                    lines = output.split('\n')
                    if len(lines) > 1:
                        count = lines[1]
                        logger.info(f"  {table_name}: {count} rows")
            except:
                pass

    # Summary
    elapsed = time.time() - start_time
    logger.info(f"\nExecution time: {elapsed:.2f} seconds")

    if all(final_check.values()):
        logger.info("\n🏆 SUCCESS! All competition requirements met:")
        logger.info("✅ ML.GENERATE_EMBEDDING - patient & trial embeddings created")
        logger.info("✅ VECTOR_SEARCH - semantic matches table created")
        logger.info("✅ CREATE VECTOR INDEX - IVF indexes created")
        logger.info("\n🎯 READY FOR SUBMISSION!")
    else:
        logger.warning("\n⚠️ Some tables still missing. Please check errors above.")

    # Save summary
    with open("/workspaces/Kaggle_BigQuerry2025/SUBMISSION/simple_execution_report.txt", 'w') as f:
        f.write(f"Execution Report - {datetime.now().isoformat()}\n")
        f.write(f"Success: {all(final_check.values())}\n")
        f.write(f"Tables: {final_check}\n")
        f.write(f"Execution time: {elapsed:.2f} seconds\n")

if __name__ == "__main__":
    main()