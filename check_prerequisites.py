#!/usr/bin/env python3
"""
Prerequisites Checker for BigQuery 2025 Clinical Trial Matching System
Verifies all requirements are met before running the pipeline
"""

import subprocess
import sys
import os
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config.config_manager import ConfigManager


class PrerequisiteChecker:
    """Check all prerequisites for running the pipeline"""

    def __init__(self):
        self.checks_passed = True
        self.config = ConfigManager()

    def run_command(self, command: str) -> tuple[bool, str]:
        """Run a shell command and return success status and output"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def check_python_version(self):
        """Check Python version >= 3.9"""
        print("Checking Python version... ", end="")
        version = sys.version_info
        if version.major >= 3 and version.minor >= 9:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            print(f"❌ Python {version.major}.{version.minor} (need >= 3.9)")
            return False

    def check_gcloud(self):
        """Check if gcloud CLI is installed and authenticated"""
        print("Checking Google Cloud CLI... ", end="")

        # Check if gcloud is installed
        success, output = self.run_command("gcloud --version")
        if not success:
            print("❌ gcloud CLI not installed")
            print("  Install from: https://cloud.google.com/sdk/docs/install")
            return False

        # Check if authenticated
        success, output = self.run_command("gcloud auth list --filter=status:ACTIVE --format='value(account)'")
        if not success or not output:
            print("❌ Not authenticated")
            print("  Run: gcloud auth login")
            return False

        print(f"✅ Authenticated as: {output}")
        return True

    def check_gcp_project(self):
        """Check if GCP project is set correctly"""
        print("Checking GCP project... ", end="")

        project_id = self.config.get('gcp.project_id')
        if "YOUR_PROJECT_ID" in project_id:
            print(f"❌ Project not configured in config")
            return False

        # Verify project exists and is accessible
        success, output = self.run_command(f"gcloud projects describe {project_id} --format='value(projectId)'")
        if not success:
            print(f"❌ Cannot access project: {project_id}")
            print(f"  Set project: gcloud config set project {project_id}")
            return False

        print(f"✅ {project_id}")
        return True

    def check_bigquery_access(self):
        """Check BigQuery access"""
        print("Checking BigQuery access... ", end="")

        # Test BigQuery access
        success, output = self.run_command("bq ls -n 1")
        if not success:
            print("❌ Cannot access BigQuery")
            print("  Enable API: gcloud services enable bigquery.googleapis.com")
            return False

        print("✅ BigQuery accessible")
        return True

    def check_physionet_access(self):
        """Check access to MIMIC-IV data"""
        print("Checking MIMIC-IV access... ", end="")

        # Check if physionet-data project is accessible
        query = "SELECT COUNT(*) as count FROM `physionet-data.mimiciv_hosp.patients` LIMIT 1"
        success, output = self.run_command(f'bq query --use_legacy_sql=false "{query}"')

        if not success:
            print("❌ Cannot access MIMIC-IV data")
            print("  1. Get PhysioNet access: https://physionet.org/")
            print("  2. Complete CITI training")
            print("  3. Request MIMIC-IV access")
            print("  4. Link to BigQuery: https://mimic.mit.edu/docs/gettingstarted/cloud/bigquery/")
            return False

        print("✅ MIMIC-IV accessible")
        return True

    def check_vertex_ai(self):
        """Check Vertex AI access for embeddings"""
        print("Checking Vertex AI... ", end="")

        # Check if Vertex AI API is enabled
        project_id = self.config.get('gcp.project_id')
        success, output = self.run_command(
            f"gcloud services list --enabled --filter='name:aiplatform.googleapis.com' --project={project_id} --format='value(name)'"
        )

        if not success or not output:
            print("⚠️  Vertex AI not enabled")
            print(f"  Enable with: gcloud services enable aiplatform.googleapis.com --project={project_id}")
            print("  (Optional - only needed for embedding generation)")
            return True  # Not critical for basic setup

        print("✅ Vertex AI enabled")
        return True

    def check_python_packages(self):
        """Check required Python packages"""
        print("Checking Python packages... ", end="")

        required_packages = [
            'google-cloud-bigquery',
            'pandas',
            'numpy',
            'fastapi',
            'uvicorn',
            'pydantic'
        ]

        missing = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing.append(package)

        if missing:
            print(f"❌ Missing packages: {', '.join(missing)}")
            print(f"  Install with: pip install {' '.join(missing)}")
            return False

        print("✅ All required packages installed")
        return True

    def check_api_credentials(self):
        """Check Application Default Credentials"""
        print("Checking Application Default Credentials... ", end="")

        adc_path = Path.home() / '.config' / 'gcloud' / 'application_default_credentials.json'
        if not adc_path.exists():
            print("❌ ADC not configured")
            print("  Run: gcloud auth application-default login")
            return False

        print("✅ ADC configured")
        return True

    def run_all_checks(self):
        """Run all prerequisite checks"""
        print("\n" + "="*60)
        print("PREREQUISITE CHECKS")
        print("="*60 + "\n")

        checks = [
            ("Python Version", self.check_python_version),
            ("Google Cloud CLI", self.check_gcloud),
            ("GCP Project", self.check_gcp_project),
            ("BigQuery Access", self.check_bigquery_access),
            ("MIMIC-IV Access", self.check_physionet_access),
            ("Vertex AI", self.check_vertex_ai),
            ("Python Packages", self.check_python_packages),
            ("API Credentials", self.check_api_credentials),
        ]

        results = []
        for name, check_func in checks:
            try:
                result = check_func()
                results.append((name, result))
                if not result:
                    self.checks_passed = False
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append((name, False))
                self.checks_passed = False

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        passed = sum(1 for _, result in results if result)
        total = len(results)

        print(f"\nChecks passed: {passed}/{total}")

        if self.checks_passed:
            print("\n✅ All prerequisites met! Ready to run the pipeline.")
            print("\nNext step: python setup/setup.py")
        else:
            print("\n⚠️  Some prerequisites are missing. Please address the issues above.")
            print("\nFor detailed setup instructions, see docs/SETUP_GUIDE.md")

        return self.checks_passed


if __name__ == "__main__":
    checker = PrerequisiteChecker()
    success = checker.run_all_checks()
    sys.exit(0 if success else 1)