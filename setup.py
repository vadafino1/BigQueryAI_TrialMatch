#!/usr/bin/env python3
"""
Main Setup Script for BigQuery 2025 Clinical Trial Matching System
Orchestrates the complete pipeline setup from scratch
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config.config_manager import ConfigManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineSetup:
    """Orchestrates the complete pipeline setup"""

    def __init__(self, config_file=None, skip_prerequisites=False):
        """
        Initialize pipeline setup

        Args:
            config_file: Path to config file (uses default logic if None)
            skip_prerequisites: Skip prerequisite checks
        """
        self.config = ConfigManager(config_file)
        self.skip_prerequisites = skip_prerequisites
        self.start_time = datetime.now()
        self.steps_completed = []
        self.steps_failed = []

    def run_command(self, command: str, description: str, timeout: int = 3600) -> bool:
        """
        Run a shell command with progress tracking

        Args:
            command: Command to run
            description: Description for logging
            timeout: Timeout in seconds

        Returns:
            Success status
        """
        print(f"\n📌 {description}...")
        logger.info(f"Running: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0:
                print(f"✅ {description} - Complete")
                self.steps_completed.append(description)
                if result.stdout:
                    logger.debug(result.stdout)
                return True
            else:
                print(f"❌ {description} - Failed")
                logger.error(f"Error: {result.stderr}")
                self.steps_failed.append(description)
                return False

        except subprocess.TimeoutExpired:
            print(f"❌ {description} - Timeout after {timeout}s")
            self.steps_failed.append(description)
            return False
        except Exception as e:
            print(f"❌ {description} - Error: {e}")
            self.steps_failed.append(description)
            return False

    def check_prerequisites(self) -> bool:
        """Run prerequisite checks"""
        if self.skip_prerequisites:
            logger.info("Skipping prerequisite checks (--skip-prerequisites)")
            return True

        return self.run_command(
            "python setup/check_prerequisites.py",
            "Checking prerequisites"
        )

    def create_dataset(self) -> bool:
        """Create BigQuery dataset"""
        project_id = self.config.get('gcp.project_id')
        dataset_id = self.config.get('gcp.dataset_id')
        location = self.config.get('bigquery.location')

        command = f"bq mk --location={location} --dataset {project_id}:{dataset_id}"
        return self.run_command(command, "Creating BigQuery dataset")

    def setup_tables(self) -> bool:
        """Create base tables in BigQuery"""
        # Import the SQL preprocessor
        from sql_preprocessor import SQLPreprocessor
        import tempfile

        preprocessor = SQLPreprocessor(self.config)

        sql_files = [
            "sql_files/01_foundation_complete.sql",
            "sql_files/02_patient_profiling.sql"
        ]

        for sql_file in sql_files:
            if Path(sql_file).exists():
                # Process the SQL file
                processed_sql = preprocessor.preprocess_file(Path(sql_file))

                # Save to temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as tmp:
                    tmp.write(processed_sql)
                    tmp_path = tmp.name

                command = f"bq query --use_legacy_sql=false < {tmp_path}"
                success = self.run_command(command, f"Creating tables from {sql_file}")

                # Clean up temp file
                Path(tmp_path).unlink()

                if not success:
                    return False
        return True

    def import_mimic_data(self) -> bool:
        """Import MIMIC-IV patient data"""
        return self.run_command(
            "python python/core/import_mimic_patients.py",
            "Importing MIMIC-IV patients (this may take 2-3 hours)",
            timeout=10800  # 3 hours
        )

    def import_clinical_trials(self) -> bool:
        """Import clinical trials from ClinicalTrials.gov"""
        return self.run_command(
            "python python/core/import_clinical_trials.py",
            "Importing clinical trials (this may take 1 hour)",
            timeout=7200  # 2 hours
        )

    def run_temporal_transformation(self) -> bool:
        """Apply temporal normalization for 2025"""
        return self.run_command(
            "python python/core/temporal_transformation.py",
            "Applying temporal transformation (2100s -> 2025)",
            timeout=3600  # 1 hour
        )

    def extract_features(self) -> bool:
        """Extract clinical features from patient data"""
        return self.run_command(
            "python python/features/extract_features.py",
            "Extracting clinical features (this may take 4-5 hours)",
            timeout=18000  # 5 hours
        )

    def generate_embeddings(self) -> bool:
        """Generate embeddings for patients and trials"""
        return self.run_command(
            "python python/features/generate_embeddings.py",
            "Generating embeddings (this may take 6-8 hours)",
            timeout=28800  # 8 hours
        )

    def create_vector_indexes(self) -> bool:
        """Create TreeAH vector indexes"""
        return self.run_command(
            "bq query --use_legacy_sql=false < sql/03_vectors/04_vector_search_indexes.sql",
            "Creating vector search indexes",
            timeout=3600  # 1 hour
        )

    def generate_matches(self) -> bool:
        """Generate patient-trial match scores"""
        return self.run_command(
            "python python/matching/generate_matches.py",
            "Generating match scores (this may take 10-12 hours)",
            timeout=43200  # 12 hours
        )

    def test_api(self) -> bool:
        """Test the API locally"""
        print("\n📌 Testing API...")
        # Start API in background
        api_process = subprocess.Popen(
            "cd python/api && python main.py",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for API to start
        time.sleep(5)

        # Test API endpoint
        success = self.run_command(
            "curl -s http://localhost:8001/health",
            "Testing API health endpoint"
        )

        # Kill API process
        api_process.terminate()
        return success

    def print_summary(self):
        """Print setup summary"""
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(elapsed.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        print("\n" + "="*60)
        print("SETUP SUMMARY")
        print("="*60)
        print(f"\nTime elapsed: {hours}h {minutes}m {seconds}s")
        print(f"\nSteps completed ({len(self.steps_completed)}):")
        for step in self.steps_completed:
            print(f"  ✅ {step}")

        if self.steps_failed:
            print(f"\nSteps failed ({len(self.steps_failed)}):")
            for step in self.steps_failed:
                print(f"  ❌ {step}")

        print("\n" + "="*60)

        if not self.steps_failed:
            print("\n🎉 Setup completed successfully!")
            print("\nNext steps:")
            print("  1. Test the API: cd python/api && python main.py")
            print("  2. Run notebooks: jupyter notebook notebooks/01_quick_start.ipynb")
            print("  3. Deploy to Cloud Run: ./scripts/deploy_api.sh")
        else:
            print("\n⚠️  Setup completed with errors")
            print("Please check the logs and retry failed steps")

    def run_setup(self, steps=None):
        """
        Run the complete setup pipeline

        Args:
            steps: List of specific steps to run (runs all if None)
        """
        print("\n" + "="*60)
        print("BIGQUERY 2025 CLINICAL TRIAL MATCHING - SETUP")
        print("="*60)

        self.config.print_summary()

        # Define all setup steps
        all_steps = [
            ("prerequisites", self.check_prerequisites),
            ("dataset", self.create_dataset),
            ("tables", self.setup_tables),
            ("mimic", self.import_mimic_data),
            ("trials", self.import_clinical_trials),
            ("temporal", self.run_temporal_transformation),
            ("features", self.extract_features),
            ("embeddings", self.generate_embeddings),
            ("indexes", self.create_vector_indexes),
            ("matches", self.generate_matches),
            ("api_test", self.test_api)
        ]

        # Filter steps if specific ones requested
        if steps:
            all_steps = [(name, func) for name, func in all_steps if name in steps]

        # Run selected steps
        for step_name, step_func in all_steps:
            try:
                success = step_func()
                if not success and step_name == "prerequisites":
                    print("\n❌ Prerequisites not met. Exiting...")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"Error in {step_name}: {e}")
                self.steps_failed.append(step_name)

        self.print_summary()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Setup BigQuery 2025 Clinical Trial Matching System")
    parser.add_argument(
        "--config",
        help="Path to config file (default: auto-detect)"
    )
    parser.add_argument(
        "--skip-prerequisites",
        action="store_true",
        help="Skip prerequisite checks"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["prerequisites", "dataset", "tables", "mimic", "trials",
                 "temporal", "features", "embeddings", "indexes", "matches", "api_test"],
        help="Run only specific steps"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick setup (skip long-running steps like embeddings and matches)"
    )

    args = parser.parse_args()

    # Quick setup skips time-intensive steps
    if args.quick:
        args.steps = ["prerequisites", "dataset", "tables", "api_test"]

    setup = PipelineSetup(
        config_file=args.config,
        skip_prerequisites=args.skip_prerequisites
    )

    try:
        setup.run_setup(steps=args.steps)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        setup.print_summary()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()