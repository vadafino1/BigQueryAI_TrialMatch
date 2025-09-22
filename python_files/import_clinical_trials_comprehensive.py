#!/usr/bin/env python3
"""
Import comprehensive clinical trials from ClinicalTrials.gov API
Focuses on recruiting trials with full eligibility criteria
"""

import json
import re
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import requests
from google.cloud import bigquery
from google.auth import default

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalTrialsImporter:
    def __init__(self):
        credentials, _ = default()
        self.client = bigquery.Client(
            project="YOUR_PROJECT_ID",
            credentials=credentials,
            location="US"
        )
        self.api_base = "https://clinicaltrials.gov/api/v2"

    def fetch_recruiting_trials(self, conditions: List[str], max_trials: int = 5000) -> List[Dict]:
        """Fetch recruiting trials from ClinicalTrials.gov API"""
        all_trials = []

        for condition in conditions:
            logger.info(f"Fetching {condition} trials...")
            page_token = None
            condition_trials = 0

            while condition_trials < max_trials // len(conditions):
                params = {
                    "query.cond": condition,
                    "filter.overallStatus": "RECRUITING",
                    "fields": "NCTId,BriefTitle,OfficialTitle,OverallStatus,Condition,EligibilityCriteria,MinimumAge,MaximumAge,Gender,StartDate,PrimaryCompletionDate,StudyType,Phase,EnrollmentCount,LocationCountry",
                    "pageSize": 100,
                    "format": "json"
                }

                if page_token:
                    params["pageToken"] = page_token

                try:
                    response = requests.get(f"{self.api_base}/studies", params=params)
                    response.raise_for_status()
                    data = response.json()

                    studies = data.get("studies", [])
                    if not studies:
                        break

                    all_trials.extend(studies)
                    condition_trials += len(studies)

                    # Check for next page
                    page_token = data.get("nextPageToken")
                    if not page_token:
                        break

                    # Rate limiting
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error fetching trials: {e}")
                    break

            logger.info(f"Fetched {condition_trials} {condition} trials")

        return all_trials

    def extract_lab_requirements(self, eligibility_text: str) -> Dict[str, Optional[float]]:
        """Extract laboratory value requirements from eligibility criteria"""
        lab_values = {
            "hemoglobin_min": None,
            "hemoglobin_max": None,
            "platelets_min": None,
            "platelets_max": None,
            "creatinine_min": None,
            "creatinine_max": None,
            "wbc_min": None,
            "wbc_max": None,
            "neutrophils_min": None,
            "neutrophils_max": None,
            "bilirubin_max": None,
            "alt_max": None,
            "ast_max": None,
            "glucose_min": None,
            "glucose_max": None,
            "hba1c_min": None,
            "hba1c_max": None
        }

        if not eligibility_text:
            return lab_values

        # Patterns for common lab values
        patterns = {
            "hemoglobin": [
                r"hemoglobin\s*[≥>=]\s*([\d.]+)\s*g/dl",
                r"hgb\s*[≥>=]\s*([\d.]+)",
                r"hemoglobin.*at least\s*([\d.]+)"
            ],
            "platelets": [
                r"platelets?\s*[≥>=]\s*([\d,]+)\s*(?:×|x)?\s*10[³3⁹9]?/[µu]l",
                r"platelets?\s*[≥>=]\s*([\d,]+)",
                r"platelet count\s*[≥>=]\s*([\d,]+)"
            ],
            "creatinine": [
                r"creatinine\s*[≤<=]\s*([\d.]+)\s*mg/dl",
                r"creatinine\s*[≤<=]\s*([\d.]+)",
                r"serum creatinine.*[≤<=]\s*([\d.]+)"
            ],
            "wbc": [
                r"wbc\s*[≥>=]\s*([\d.]+)\s*(?:×|x)?\s*10[³3⁹9]?/[µu]l",
                r"white blood cells?\s*[≥>=]\s*([\d.]+)",
                r"leukocytes?\s*[≥>=]\s*([\d.]+)"
            ],
            "neutrophils": [
                r"anc\s*[≥>=]\s*([\d.]+)\s*(?:×|x)?\s*10[³3⁹9]?/[µu]l",
                r"neutrophils?\s*[≥>=]\s*([\d.]+)",
                r"absolute neutrophil count\s*[≥>=]\s*([\d.]+)"
            ],
            "bilirubin": [
                r"bilirubin\s*[≤<=]\s*([\d.]+)\s*(?:×|x)?\s*uln",
                r"total bilirubin\s*[≤<=]\s*([\d.]+)",
                r"bilirubin.*[≤<=]\s*([\d.]+)"
            ],
            "glucose": [
                r"glucose\s*(?:between|from)?\s*([\d.]+)\s*(?:to|-)\s*([\d.]+)",
                r"blood sugar\s*[≥>=]\s*([\d.]+)",
                r"fasting glucose\s*[≤<=]\s*([\d.]+)"
            ],
            "hba1c": [
                r"hba1c\s*(?:between|from)?\s*([\d.]+)\s*(?:to|-)\s*([\d.]+)%?",
                r"a1c\s*[≥>=]\s*([\d.]+)%?",
                r"glycated hemoglobin\s*[≤<=]\s*([\d.]+)%?"
            ]
        }

        text_lower = eligibility_text.lower()

        for lab_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text_lower)
                if matches:
                    try:
                        if lab_type == "hemoglobin":
                            lab_values["hemoglobin_min"] = float(matches[0])
                        elif lab_type == "platelets":
                            # Convert to standard units (× 10^3/µL)
                            value = float(matches[0].replace(",", ""))
                            if value > 1000:  # Likely in cells/µL
                                value = value / 1000
                            lab_values["platelets_min"] = value
                        elif lab_type == "creatinine":
                            lab_values["creatinine_max"] = float(matches[0])
                        elif lab_type == "wbc":
                            lab_values["wbc_min"] = float(matches[0])
                        elif lab_type == "neutrophils":
                            lab_values["neutrophils_min"] = float(matches[0])
                        elif lab_type == "bilirubin":
                            lab_values["bilirubin_max"] = float(matches[0])
                        elif lab_type == "glucose":
                            if len(matches[0]) == 2:  # Range
                                lab_values["glucose_min"] = float(matches[0][0])
                                lab_values["glucose_max"] = float(matches[0][1])
                            else:
                                lab_values["glucose_max"] = float(matches[0])
                        elif lab_type == "hba1c":
                            if len(matches[0]) == 2:  # Range
                                lab_values["hba1c_min"] = float(matches[0][0])
                                lab_values["hba1c_max"] = float(matches[0][1])
                            else:
                                lab_values["hba1c_max"] = float(matches[0])
                    except (ValueError, IndexError):
                        continue
                    break  # Use first matching pattern

        return lab_values

    def extract_temporal_requirements(self, eligibility_text: str) -> Dict[str, bool]:
        """Extract temporal requirements from eligibility criteria"""
        temporal = {
            "requires_washout": False,
            "has_temporal_req": False,
            "excludes_recent_chemo": False,
            "excludes_recent_surgery": False
        }

        if not eligibility_text:
            return temporal

        text_lower = eligibility_text.lower()

        # Check for washout requirements
        if any(term in text_lower for term in ["washout", "wash-out", "discontinu", "stop medication"]):
            temporal["requires_washout"] = True
            temporal["has_temporal_req"] = True

        # Check for recent chemotherapy exclusion
        if any(term in text_lower for term in ["chemotherapy within", "chemo within", "prior therapy within"]):
            temporal["excludes_recent_chemo"] = True
            temporal["has_temporal_req"] = True

        # Check for recent surgery exclusion
        if any(term in text_lower for term in ["surgery within", "surgical procedure within", "operation within"]):
            temporal["excludes_recent_surgery"] = True
            temporal["has_temporal_req"] = True

        return temporal

    def parse_age(self, age_str: str) -> Optional[int]:
        """Parse age string to years"""
        if not age_str:
            return None

        # Extract number
        match = re.search(r"(\d+)", age_str)
        if not match:
            return None

        value = int(match.group(1))

        # Convert to years if needed
        if "month" in age_str.lower():
            value = value // 12
        elif "week" in age_str.lower():
            value = 0
        elif "day" in age_str.lower():
            value = 0

        return value

    def classify_trial(self, conditions: List[str], title: str) -> Dict[str, bool]:
        """Classify trial by therapeutic area"""
        classification = {
            "is_oncology_trial": False,
            "is_diabetes_trial": False,
            "is_cardiac_trial": False,
            "is_respiratory_trial": False,
            "is_neurological_trial": False
        }

        # Combine conditions and title for classification
        text = " ".join(conditions + [title]).lower() if conditions else title.lower()

        # Oncology keywords
        if any(term in text for term in ["cancer", "carcinoma", "tumor", "oncol", "leukemia",
                                         "lymphoma", "melanoma", "sarcoma", "metasta", "chemo"]):
            classification["is_oncology_trial"] = True

        # Diabetes keywords
        if any(term in text for term in ["diabet", "glucose", "insulin", "glycem", "hba1c",
                                         "hyperglycemia", "hypoglycemia"]):
            classification["is_diabetes_trial"] = True

        # Cardiac keywords
        if any(term in text for term in ["cardiac", "heart", "coronary", "myocardial", "atrial",
                                         "ventricular", "angina", "arrhythm", "cardiomyopathy"]):
            classification["is_cardiac_trial"] = True

        # Respiratory keywords
        if any(term in text for term in ["asthma", "copd", "respiratory", "lung", "pulmonary",
                                         "bronch", "pneumonia", "covid"]):
            classification["is_respiratory_trial"] = True

        # Neurological keywords
        if any(term in text for term in ["alzheimer", "parkinson", "epilep", "seizure", "stroke",
                                         "neurolog", "brain", "cognitive", "dementia"]):
            classification["is_neurological_trial"] = True

        return classification

    def create_comprehensive_trials_table(self, trials_data: List[Dict]):
        """Create comprehensive trials table in BigQuery"""
        logger.info("Creating comprehensive trials table...")

        # Prepare data for BigQuery
        rows_to_insert = []

        for trial in trials_data:
            try:
                # Extract study info
                study_info = trial.get("protocolSection", {})
                id_module = study_info.get("identificationModule", {})
                status_module = study_info.get("statusModule", {})
                eligibility_module = study_info.get("eligibilityModule", {})
                conditions_module = study_info.get("conditionsModule", {})
                design_module = study_info.get("designModule", {})

                nct_id = id_module.get("nctId", "")
                if not nct_id:
                    continue

                # Extract eligibility criteria
                eligibility_text = eligibility_module.get("eligibilityCriteria", "")

                # Extract lab requirements
                lab_reqs = self.extract_lab_requirements(eligibility_text)

                # Extract temporal requirements
                temporal_reqs = self.extract_temporal_requirements(eligibility_text)

                # Parse ages
                min_age = self.parse_age(eligibility_module.get("minimumAge", ""))
                max_age = self.parse_age(eligibility_module.get("maximumAge", ""))

                # Classify trial
                conditions = conditions_module.get("conditions", [])
                title = id_module.get("briefTitle", "")
                classification = self.classify_trial(conditions, title)

                # Build row
                row = {
                    "nct_id": nct_id,
                    "brief_title": title[:500],  # Truncate to 500 chars
                    "official_title": id_module.get("officialTitle", "")[:1000],
                    "overall_status": status_module.get("overallStatus", ""),
                    "conditions": ", ".join(conditions)[:2000] if conditions else None,
                    "eligibility_criteria_full": eligibility_text[:30000],  # Truncate to 30K chars
                    "min_age_years": min_age,
                    "max_age_years": max_age,
                    "gender": eligibility_module.get("sex", "ALL"),
                    "study_type": design_module.get("studyType", ""),
                    "phase": ", ".join(design_module.get("phases", [])) if design_module.get("phases") else None,
                    "enrollment_count": design_module.get("enrollmentInfo", {}).get("count") if design_module.get("enrollmentInfo") else None,
                    "start_date": status_module.get("startDateStruct", {}).get("date"),
                    "completion_date": status_module.get("completionDateStruct", {}).get("date"),

                    # Lab requirements
                    "hemoglobin_min": lab_reqs["hemoglobin_min"],
                    "hemoglobin_max": lab_reqs["hemoglobin_max"],
                    "platelets_min": lab_reqs["platelets_min"],
                    "platelets_max": lab_reqs["platelets_max"],
                    "creatinine_min": lab_reqs["creatinine_min"],
                    "creatinine_max": lab_reqs["creatinine_max"],
                    "wbc_min": lab_reqs["wbc_min"],
                    "wbc_max": lab_reqs["wbc_max"],
                    "neutrophils_min": lab_reqs["neutrophils_min"],
                    "neutrophils_max": lab_reqs["neutrophils_max"],
                    "bilirubin_max": lab_reqs["bilirubin_max"],
                    "alt_max": lab_reqs["alt_max"],
                    "ast_max": lab_reqs["ast_max"],
                    "glucose_min": lab_reqs["glucose_min"],
                    "glucose_max": lab_reqs["glucose_max"],
                    "hba1c_min": lab_reqs["hba1c_min"],
                    "hba1c_max": lab_reqs["hba1c_max"],

                    # Temporal requirements
                    "has_temporal_req": temporal_reqs["has_temporal_req"],
                    "requires_washout": temporal_reqs["requires_washout"],
                    "excludes_recent_chemo": temporal_reqs["excludes_recent_chemo"],
                    "excludes_recent_surgery": temporal_reqs["excludes_recent_surgery"],

                    # Classification
                    "is_oncology_trial": classification["is_oncology_trial"],
                    "is_diabetes_trial": classification["is_diabetes_trial"],
                    "is_cardiac_trial": classification["is_cardiac_trial"],
                    "is_respiratory_trial": classification["is_respiratory_trial"],
                    "is_neurological_trial": classification["is_neurological_trial"],

                    # Metadata
                    "excludes_pregnancy": "pregnan" in eligibility_text.lower(),
                    "accepts_healthy": eligibility_module.get("healthyVolunteers", False),
                    "imported_at": datetime.utcnow().isoformat()
                }

                rows_to_insert.append(row)

            except Exception as e:
                logger.error(f"Error processing trial {trial.get('nctId', 'unknown')}: {e}")
                continue

        # Create table with schema
        table_id = "YOUR_PROJECT_ID.clinical_trial_matching.trials_comprehensive"

        schema = [
            bigquery.SchemaField("nct_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("brief_title", "STRING"),
            bigquery.SchemaField("official_title", "STRING"),
            bigquery.SchemaField("overall_status", "STRING"),
            bigquery.SchemaField("conditions", "STRING"),
            bigquery.SchemaField("eligibility_criteria_full", "STRING"),
            bigquery.SchemaField("min_age_years", "INTEGER"),
            bigquery.SchemaField("max_age_years", "INTEGER"),
            bigquery.SchemaField("gender", "STRING"),
            bigquery.SchemaField("study_type", "STRING"),
            bigquery.SchemaField("phase", "STRING"),
            bigquery.SchemaField("enrollment_count", "INTEGER"),
            bigquery.SchemaField("start_date", "STRING"),
            bigquery.SchemaField("completion_date", "STRING"),

            # Lab requirements
            bigquery.SchemaField("hemoglobin_min", "FLOAT"),
            bigquery.SchemaField("hemoglobin_max", "FLOAT"),
            bigquery.SchemaField("platelets_min", "FLOAT"),
            bigquery.SchemaField("platelets_max", "FLOAT"),
            bigquery.SchemaField("creatinine_min", "FLOAT"),
            bigquery.SchemaField("creatinine_max", "FLOAT"),
            bigquery.SchemaField("wbc_min", "FLOAT"),
            bigquery.SchemaField("wbc_max", "FLOAT"),
            bigquery.SchemaField("neutrophils_min", "FLOAT"),
            bigquery.SchemaField("neutrophils_max", "FLOAT"),
            bigquery.SchemaField("bilirubin_max", "FLOAT"),
            bigquery.SchemaField("alt_max", "FLOAT"),
            bigquery.SchemaField("ast_max", "FLOAT"),
            bigquery.SchemaField("glucose_min", "FLOAT"),
            bigquery.SchemaField("glucose_max", "FLOAT"),
            bigquery.SchemaField("hba1c_min", "FLOAT"),
            bigquery.SchemaField("hba1c_max", "FLOAT"),

            # Temporal requirements
            bigquery.SchemaField("has_temporal_req", "BOOLEAN"),
            bigquery.SchemaField("requires_washout", "BOOLEAN"),
            bigquery.SchemaField("excludes_recent_chemo", "BOOLEAN"),
            bigquery.SchemaField("excludes_recent_surgery", "BOOLEAN"),

            # Classification
            bigquery.SchemaField("is_oncology_trial", "BOOLEAN"),
            bigquery.SchemaField("is_diabetes_trial", "BOOLEAN"),
            bigquery.SchemaField("is_cardiac_trial", "BOOLEAN"),
            bigquery.SchemaField("is_respiratory_trial", "BOOLEAN"),
            bigquery.SchemaField("is_neurological_trial", "BOOLEAN"),

            # Other
            bigquery.SchemaField("excludes_pregnancy", "BOOLEAN"),
            bigquery.SchemaField("accepts_healthy", "BOOLEAN"),
            bigquery.SchemaField("imported_at", "TIMESTAMP"),
        ]

        # Create table
        table = bigquery.Table(table_id, schema=schema)
        table = self.client.create_table(table, exists_ok=True)

        # Insert data
        if rows_to_insert:
            errors = self.client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                logger.error(f"Failed to insert rows: {errors}")
            else:
                logger.info(f"✅ Inserted {len(rows_to_insert)} trials into comprehensive table")

        return len(rows_to_insert)

    def run_import(self):
        """Run the complete import process"""
        print("""
        ╔═══════════════════════════════════════════════════╗
        ║  COMPREHENSIVE CLINICAL TRIALS IMPORT            ║
        ║  From ClinicalTrials.gov API                     ║
        ╚═══════════════════════════════════════════════════╝
        """)

        # Define conditions to search
        conditions = [
            "cancer",
            "diabetes",
            "heart disease",
            "lung disease",
            "alzheimer",
            "covid-19"
        ]

        # Fetch trials
        logger.info("Fetching recruiting trials from ClinicalTrials.gov...")
        trials = self.fetch_recruiting_trials(conditions, max_trials=3000)
        logger.info(f"Fetched {len(trials)} total trials")

        # Create comprehensive table
        count = self.create_comprehensive_trials_table(trials)

        # Create summary view
        logger.info("Creating summary view...")
        query = """
        CREATE OR REPLACE VIEW `YOUR_PROJECT_ID.clinical_trial_matching.trials_summary` AS
        SELECT
            COUNT(*) as total_trials,
            SUM(CAST(is_oncology_trial AS INT64)) as oncology_trials,
            SUM(CAST(is_diabetes_trial AS INT64)) as diabetes_trials,
            SUM(CAST(is_cardiac_trial AS INT64)) as cardiac_trials,
            SUM(CAST(has_temporal_req AS INT64)) as trials_with_temporal_req,
            SUM(CAST(requires_washout AS INT64)) as trials_requiring_washout,
            SUM(CASE WHEN hemoglobin_min IS NOT NULL THEN 1 ELSE 0 END) as trials_with_hemoglobin_req,
            SUM(CASE WHEN platelets_min IS NOT NULL THEN 1 ELSE 0 END) as trials_with_platelet_req,
            SUM(CASE WHEN creatinine_max IS NOT NULL THEN 1 ELSE 0 END) as trials_with_creatinine_req,
            AVG(CASE WHEN min_age_years IS NOT NULL THEN min_age_years END) as avg_min_age,
            AVG(CASE WHEN max_age_years IS NOT NULL THEN max_age_years END) as avg_max_age
        FROM `YOUR_PROJECT_ID.clinical_trial_matching.trials_comprehensive`
        """

        job = self.client.query(query)
        job.result()

        # Get summary stats
        stats_query = "SELECT * FROM `YOUR_PROJECT_ID.clinical_trial_matching.trials_summary`"
        stats = list(self.client.query(stats_query).result())[0]

        print(f"""
        ✅ Import Complete!

        Total Trials Imported: {count}
        - Oncology: {stats.oncology_trials}
        - Diabetes: {stats.diabetes_trials}
        - Cardiac: {stats.cardiac_trials}

        Lab Requirements:
        - With hemoglobin criteria: {stats.trials_with_hemoglobin_req}
        - With platelet criteria: {stats.trials_with_platelet_req}
        - With creatinine criteria: {stats.trials_with_creatinine_req}

        Temporal Requirements:
        - With temporal requirements: {stats.trials_with_temporal_req}
        - Requiring washout: {stats.trials_requiring_washout}

        Age Range:
        - Average min age: {stats.avg_min_age:.1f} years
        - Average max age: {stats.avg_max_age:.1f} years
        """)

        return count

if __name__ == "__main__":
    importer = ClinicalTrialsImporter()
    importer.run_import()