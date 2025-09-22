#!/usr/bin/env python3
"""
SQL Preprocessor for BigQuery 2025 Clinical Trial Matching
Handles replacement of hardcoded values with configuration values
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLPreprocessor:
    """Preprocesses SQL files to replace hardcoded values with configuration"""

    # Default replacements for backward compatibility
    DEFAULT_REPLACEMENTS = {
        'gen-lang-client-0017660547': '{{PROJECT_ID}}',
        'clinical_trial_matching': '{{DATASET_ID}}',
        'us-central1': '{{LOCATION}}',
        'vertex-ai-connection': '{{CONNECTION_NAME}}'
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor with configuration

        Args:
            config: Configuration dictionary or ConfigManager instance
        """
        self.config = config or {}

    def preprocess_sql(self, sql_content: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Process SQL content and replace hardcoded values

        Args:
            sql_content: Raw SQL content
            config: Configuration to use (overrides instance config)

        Returns:
            Processed SQL with replaced values
        """
        config = config or self.config

        # Get configuration values
        if hasattr(config, 'get'):  # ConfigManager instance
            project_id = config.get('gcp.project_id', 'gen-lang-client-0017660547')
            dataset_id = config.get('gcp.dataset_id', 'clinical_trial_matching')
            location = config.get('bigquery.location', 'us-central1')
            connection = config.get('gcp.connection_name', 'vertex-ai-connection')
        else:  # Dictionary
            project_id = config.get('project_id', 'gen-lang-client-0017660547')
            dataset_id = config.get('dataset_id', 'clinical_trial_matching')
            location = config.get('location', 'us-central1')
            connection = config.get('connection_name', 'vertex-ai-connection')

        # Replace hardcoded project references
        processed = sql_content

        # Pattern 1: Full table references with backticks
        # `gen-lang-client-0017660547.clinical_trial_matching.table_name`
        pattern1 = r'`gen-lang-client-0017660547\.clinical_trial_matching\.([^`]+)`'
        replacement1 = f'`{project_id}.{dataset_id}.\\1`'
        processed = re.sub(pattern1, replacement1, processed)

        # Pattern 2: CREATE statements
        # CREATE TABLE/MODEL/VIEW `project.dataset.name`
        pattern2 = r'(CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|MODEL|VIEW|FUNCTION|PROCEDURE)(?:\s+IF\s+NOT\s+EXISTS)?\s+)`gen-lang-client-0017660547\.clinical_trial_matching\.([^`]+)`'
        replacement2 = f'\\1`{project_id}.{dataset_id}.\\2`'
        processed = re.sub(pattern2, replacement2, processed, flags=re.IGNORECASE)

        # Pattern 3: FROM/JOIN statements
        pattern3 = r'(FROM|JOIN)\s+`gen-lang-client-0017660547\.clinical_trial_matching\.([^`]+)`'
        replacement3 = f'\\1 `{project_id}.{dataset_id}.\\2`'
        processed = re.sub(pattern3, replacement3, processed, flags=re.IGNORECASE)

        # Pattern 4: INSERT/UPDATE/DELETE statements
        pattern4 = r'(INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+`gen-lang-client-0017660547\.clinical_trial_matching\.([^`]+)`'
        replacement4 = f'\\1 `{project_id}.{dataset_id}.\\2`'
        processed = re.sub(pattern4, replacement4, processed, flags=re.IGNORECASE)

        # Pattern 5: Dataset-only references (without project)
        # Careful not to replace if already part of a full path
        pattern5 = r'(?<!\.)`clinical_trial_matching\.([^`]+)`'
        replacement5 = f'`{dataset_id}.\\1`'
        processed = re.sub(pattern5, replacement5, processed)

        # Pattern 6: Python code in BigFrames sections
        pattern6 = r'["\']\s*gen-lang-client-0017660547\s*["\']'
        replacement6 = f'"{project_id}"'
        processed = re.sub(pattern6, replacement6, processed)

        # Pattern 7: Connection references
        pattern7 = r'`projects/gen-lang-client-0017660547/locations/([^/]+)/connections/([^`]+)`'
        replacement7 = f'`projects/{project_id}/locations/{location}/connections/\\2`'
        processed = re.sub(pattern7, replacement7, processed)

        # Pattern 8: Model endpoints
        pattern8 = r'endpoint\s*=\s*["\']projects/gen-lang-client-0017660547/'
        replacement8 = f'endpoint = "projects/{project_id}/'
        processed = re.sub(pattern8, replacement8, processed)

        # Count replacements made
        replacements = len(re.findall('gen-lang-client-0017660547', sql_content))
        if replacements > 0:
            remaining = len(re.findall('gen-lang-client-0017660547', processed))
            logger.info(f"Replaced {replacements - remaining} occurrences of hardcoded project ID")
            if remaining > 0:
                logger.warning(f"Warning: {remaining} occurrences could not be replaced")

        return processed

    def preprocess_file(self, input_file: Path, output_file: Optional[Path] = None,
                       config: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a SQL file and optionally save to output

        Args:
            input_file: Path to input SQL file
            output_file: Optional path to save processed SQL
            config: Configuration to use

        Returns:
            Processed SQL content
        """
        logger.info(f"Processing file: {input_file}")

        with open(input_file, 'r') as f:
            sql_content = f.read()

        processed = self.preprocess_sql(sql_content, config)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(processed)
            logger.info(f"Saved processed SQL to: {output_file}")

        return processed

    def validate_preprocessing(self, sql_content: str) -> bool:
        """
        Validate that no hardcoded values remain

        Args:
            sql_content: Processed SQL content

        Returns:
            True if validation passes
        """
        hardcoded_patterns = [
            'gen-lang-client-0017660547',
            # Don't check for clinical_trial_matching as it might be the actual dataset name
        ]

        issues = []
        for pattern in hardcoded_patterns:
            matches = re.findall(pattern, sql_content)
            if matches:
                issues.append(f"Found {len(matches)} occurrences of '{pattern}'")

        if issues:
            logger.warning("Validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            return False

        return True

    def create_template(self, sql_content: str) -> str:
        """
        Create a template version of SQL with placeholders

        Args:
            sql_content: Original SQL content

        Returns:
            Template SQL with {{PLACEHOLDER}} style variables
        """
        template = sql_content

        # Replace all occurrences with template variables
        replacements = [
            (r'gen-lang-client-0017660547', '{{PROJECT_ID}}'),
            (r'clinical_trial_matching', '{{DATASET_ID}}'),
            (r'us-central1', '{{LOCATION}}'),
            (r'vertex-ai-connection', '{{CONNECTION_NAME}}')
        ]

        for pattern, replacement in replacements:
            template = re.sub(pattern, replacement, template)

        # Add configuration header
        header = """-- ============================================================================
-- CONFIGURATION TEMPLATE
-- Replace these values with your project configuration:
-- {{PROJECT_ID}} = Your GCP Project ID
-- {{DATASET_ID}} = Your BigQuery Dataset (default: clinical_trial_matching)
-- {{LOCATION}} = Your GCP Region (default: us-central1)
-- {{CONNECTION_NAME}} = Your Vertex AI Connection (default: vertex-ai-connection)
-- ============================================================================

"""
        return header + template


def main():
    """Test preprocessor with sample SQL"""
    sample_sql = """
    CREATE OR REPLACE TABLE `gen-lang-client-0017660547.clinical_trial_matching.patients`
    AS SELECT * FROM `gen-lang-client-0017660547.clinical_trial_matching.source_patients`;
    """

    # Test with custom config
    config = {
        'project_id': 'my-project',
        'dataset_id': 'my_dataset'
    }

    preprocessor = SQLPreprocessor(config)
    processed = preprocessor.preprocess_sql(sample_sql)

    print("Original SQL:")
    print(sample_sql)
    print("\nProcessed SQL:")
    print(processed)

    # Validate
    if preprocessor.validate_preprocessing(processed):
        print("\n✅ Validation passed - no hardcoded values found")
    else:
        print("\n⚠️ Validation failed - hardcoded values remain")


if __name__ == "__main__":
    main()