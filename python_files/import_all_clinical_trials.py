#!/usr/bin/env python3
"""
Complete Clinical Trials Bulk Import Script
===========================================
Imports ALL recruiting trials from ClinicalTrials.gov (~70,000+)
Plus historical trials for comprehensive matching
"""

import logging
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from google.cloud import bigquery
from google.auth import default
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteClinicalTrialsImporter:
    """Import all clinical trials from ClinicalTrials.gov"""

    def __init__(self, project_id: str = "gen-lang-client-0017660547"):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        self.dataset_id = "clinical_trial_matching_complete"
        self.api_base = "https://clinicaltrials.gov/api/v2"
        self.batch_size = 1000  # API max page size
        self.max_workers = 10  # Parallel API calls

        # Request session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BigQuery2025-ClinicalTrialMatcher/1.0'
        })

    def get_total_trial_counts(self) -> Dict[str, int]:
        """Get counts of trials by status"""
        logger.info("Getting trial counts from ClinicalTrials.gov...")

        # Query for recruiting trials
        recruiting_params = {
            'query.term': 'AREA[OverallStatus]Recruiting',
            'pageSize': 1,
            'format': 'json'
        }

        response = self.session.get(
            f"{self.api_base}/studies",
            params=recruiting_params
        )
        recruiting_count = response.json()['totalCount']

        # Query for all trials
        all_params = {
            'pageSize': 1,
            'format': 'json'
        }

        response = self.session.get(
            f"{self.api_base}/studies",
            params=all_params
        )
        total_count = response.json()['totalCount']

        logger.info(f"""
        ClinicalTrials.gov Statistics:
        - Total Trials: {total_count:,}
        - Recruiting: {recruiting_count:,}
        - Other Status: {total_count - recruiting_count:,}
        """)

        return {
            'total': total_count,
            'recruiting': recruiting_count,
            'other': total_count - recruiting_count
        }

    def fetch_trials_batch(self, page_token: Optional[str] = None,
                          status_filter: str = "Recruiting") -> Tuple[List[Dict], Optional[str]]:
        """Fetch a batch of trials"""
        params = {
            'query.term': f'AREA[OverallStatus]{status_filter}' if status_filter else '',
            'pageSize': self.batch_size,
            'fields': 'NCTId,BriefTitle,OfficialTitle,OverallStatus,StartDate,CompletionDate,'
                     'Condition,Phase,StudyType,EnrollmentCount,EligibilityCriteria,'
                     'PrimaryOutcomeMeasure,SecondaryOutcomeMeasure,InterventionName,'
                     'InterventionType,LocationCountry,LocationCity,LocationFacility,'
                     'ResponsiblePartyName,LeadSponsorName',
            'format': 'json'
        }

        if page_token:
            params['pageToken'] = page_token

        try:
            response = self.session.get(
                f"{self.api_base}/studies",
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            trials = data.get('studies', [])
            next_token = data.get('nextPageToken')

            return trials, next_token

        except Exception as e:
            logger.error(f"Error fetching trials batch: {e}")
            return [], None

    def import_all_recruiting_trials(self):
        """Import all recruiting trials (~70,000)"""
        logger.info("Starting import of ALL RECRUITING trials...")

        # Create trials table
        self.create_trials_table()

        # Get total count
        counts = self.get_total_trial_counts()
        total_recruiting = counts['recruiting']

        logger.info(f"Importing {total_recruiting:,} recruiting trials...")

        all_trials = []
        page_token = None
        batch_num = 0

        while True:
            batch_num += 1
            trials, next_token = self.fetch_trials_batch(page_token, "Recruiting")

            if not trials:
                break

            all_trials.extend(trials)
            logger.info(f"Batch {batch_num}: Fetched {len(trials)} trials (Total: {len(all_trials):,}/{total_recruiting:,})")

            # Insert batch into BigQuery
            if len(all_trials) >= 10000:  # Insert every 10K trials
                self.insert_trials_to_bigquery(all_trials)
                all_trials = []

            page_token = next_token
            if not page_token:
                break

            # Rate limiting
            time.sleep(0.1)

        # Insert remaining trials
        if all_trials:
            self.insert_trials_to_bigquery(all_trials)

        logger.info(f"✅ Imported {len(all_trials):,} recruiting trials")
        return len(all_trials)

    def import_recent_completed_trials(self, days_back: int = 365):
        """Import recently completed trials for historical analysis"""
        logger.info(f"Importing trials completed in last {days_back} days...")

        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        params = {
            'query.term': f'AREA[CompletionDate]RANGE[{cutoff_date},MAX]',
            'pageSize': self.batch_size,
            'format': 'json'
        }

        response = self.session.get(f"{self.api_base}/studies", params=params)
        total_completed = response.json()['totalCount']

        logger.info(f"Found {total_completed:,} recently completed trials")

        # Fetch and import completed trials
        all_trials = []
        page_token = None

        while len(all_trials) < min(total_completed, 50000):  # Limit to 50K completed
            trials, next_token = self.fetch_trials_batch(page_token, None)
            if not trials:
                break

            all_trials.extend(trials)
            page_token = next_token

            if not page_token:
                break

        self.insert_trials_to_bigquery(all_trials, status="Completed")
        logger.info(f"✅ Imported {len(all_trials):,} completed trials")

        return len(all_trials)

    def create_trials_table(self):
        """Create comprehensive trials table"""
        create_query = f"""
        CREATE TABLE IF NOT EXISTS `{self.project_id}.{self.dataset_id}.clinical_trials_complete` (
            trial_id STRING NOT NULL,
            trial_hash STRING,
            brief_title STRING,
            official_title STRING,
            overall_status STRING,
            start_date DATE,
            completion_date DATE,
            primary_completion_date DATE,

            -- Study details
            phase STRING,
            study_type STRING,
            study_design STRING,
            enrollment_count INT64,
            enrollment_type STRING,

            -- Conditions and interventions
            conditions ARRAY<STRING>,
            condition_categories ARRAY<STRING>,
            interventions ARRAY<STRUCT<
                name STRING,
                type STRING,
                description STRING
            >>,

            -- Eligibility
            eligibility_criteria STRING,
            eligibility_minimum_age STRING,
            eligibility_maximum_age STRING,
            eligibility_gender STRING,
            eligibility_healthy_volunteers BOOL,
            eligibility_structured STRUCT<
                inclusion_criteria ARRAY<STRING>,
                exclusion_criteria ARRAY<STRING>
            >,

            -- Outcomes
            primary_outcomes ARRAY<STRUCT<
                measure STRING,
                description STRING,
                timeframe STRING
            >>,
            secondary_outcomes ARRAY<STRUCT<
                measure STRING,
                description STRING,
                timeframe STRING
            >>,

            -- Locations
            locations ARRAY<STRUCT<
                facility STRING,
                city STRING,
                state STRING,
                country STRING,
                status STRING
            >>,
            location_countries ARRAY<STRING>,

            -- Sponsors
            lead_sponsor STRING,
            lead_sponsor_class STRING,
            collaborators ARRAY<STRING>,
            responsible_party_name STRING,
            responsible_party_type STRING,

            -- Metadata
            first_posted_date DATE,
            last_update_posted_date DATE,
            results_first_posted_date DATE,
            study_first_submitted_date DATE,

            -- Import tracking
            import_timestamp TIMESTAMP,
            import_batch STRING,
            data_version STRING
        )
        PARTITION BY DATE(import_timestamp)
        CLUSTER BY overall_status, phase, trial_id
        """

        self.client.query(create_query).result()
        logger.info("✅ Trials table created/verified")

    def parse_trial_data(self, trial: Dict) -> Dict:
        """Parse raw trial data into structured format"""
        protocol = trial.get('protocolSection', {})
        identification = protocol.get('identificationModule', {})
        status = protocol.get('statusModule', {})
        description = protocol.get('descriptionModule', {})
        conditions = protocol.get('conditionsModule', {})
        design = protocol.get('designModule', {})
        eligibility = protocol.get('eligibilityModule', {})
        outcomes = protocol.get('outcomesModule', {})
        contacts = protocol.get('contactsLocationsModule', {})
        sponsors = protocol.get('sponsorCollaboratorsModule', {})

        # Parse structured eligibility
        eligibility_text = eligibility.get('eligibilityCriteria', '')
        inclusion, exclusion = self.parse_eligibility_criteria(eligibility_text)

        # Generate unique hash for deduplication
        trial_hash = hashlib.md5(
            f"{identification.get('nctId')}_{status.get('lastUpdatePostDateStruct', {}).get('date', '')}".encode()
        ).hexdigest()

        return {
            'trial_id': identification.get('nctId'),
            'trial_hash': trial_hash,
            'brief_title': identification.get('briefTitle'),
            'official_title': identification.get('officialTitle'),
            'overall_status': status.get('overallStatus'),
            'start_date': self.parse_date(status.get('startDateStruct')),
            'completion_date': self.parse_date(status.get('completionDateStruct')),
            'primary_completion_date': self.parse_date(status.get('primaryCompletionDateStruct')),

            # Study details
            'phase': ', '.join(design.get('phases', []) or []),
            'study_type': design.get('studyType'),
            'enrollment_count': design.get('enrollmentInfo', {}).get('count'),
            'enrollment_type': design.get('enrollmentInfo', {}).get('type'),

            # Conditions
            'conditions': conditions.get('conditions', []),
            'condition_categories': self.categorize_conditions(conditions.get('conditions', [])),

            # Interventions
            'interventions': [
                {
                    'name': i.get('name'),
                    'type': i.get('type'),
                    'description': i.get('description')
                }
                for i in (design.get('interventions') or [])
            ],

            # Eligibility
            'eligibility_criteria': eligibility_text,
            'eligibility_minimum_age': eligibility.get('minimumAge'),
            'eligibility_maximum_age': eligibility.get('maximumAge'),
            'eligibility_gender': eligibility.get('sex'),
            'eligibility_healthy_volunteers': eligibility.get('healthyVolunteers'),
            'eligibility_structured': {
                'inclusion_criteria': inclusion,
                'exclusion_criteria': exclusion
            },

            # Outcomes
            'primary_outcomes': [
                {
                    'measure': o.get('measure'),
                    'description': o.get('description'),
                    'timeframe': o.get('timeFrame')
                }
                for o in (outcomes.get('primaryOutcomes') or [])
            ],
            'secondary_outcomes': [
                {
                    'measure': o.get('measure'),
                    'description': o.get('description'),
                    'timeframe': o.get('timeFrame')
                }
                for o in (outcomes.get('secondaryOutcomes') or [])
            ],

            # Locations
            'locations': [
                {
                    'facility': loc.get('facility'),
                    'city': loc.get('city'),
                    'state': loc.get('state'),
                    'country': loc.get('country'),
                    'status': loc.get('status')
                }
                for loc in (contacts.get('locations') or [])
            ],
            'location_countries': list(set([
                loc.get('country') for loc in (contacts.get('locations') or [])
                if loc.get('country')
            ])),

            # Sponsors
            'lead_sponsor': sponsors.get('leadSponsor', {}).get('name'),
            'lead_sponsor_class': sponsors.get('leadSponsor', {}).get('class'),
            'collaborators': [c.get('name') for c in (sponsors.get('collaborators') or [])],
            'responsible_party_name': sponsors.get('responsibleParty', {}).get('name'),
            'responsible_party_type': sponsors.get('responsibleParty', {}).get('type'),

            # Metadata
            'import_timestamp': datetime.now().isoformat(),
            'import_batch': f"bulk_{datetime.now().strftime('%Y%m%d')}",
            'data_version': '2.0'
        }

    def parse_eligibility_criteria(self, criteria_text: str) -> Tuple[List[str], List[str]]:
        """Parse eligibility criteria into inclusion/exclusion lists"""
        if not criteria_text:
            return [], []

        inclusion = []
        exclusion = []
        current_section = None

        lines = criteria_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if 'inclusion criteria' in line.lower():
                current_section = 'inclusion'
            elif 'exclusion criteria' in line.lower():
                current_section = 'exclusion'
            elif line and current_section:
                if current_section == 'inclusion':
                    inclusion.append(line)
                else:
                    exclusion.append(line)

        return inclusion, exclusion

    def categorize_conditions(self, conditions: List[str]) -> List[str]:
        """Categorize conditions into major disease areas"""
        categories = set()

        condition_text = ' '.join(conditions).lower()

        category_keywords = {
            'oncology': ['cancer', 'tumor', 'carcinoma', 'leukemia', 'lymphoma', 'melanoma'],
            'cardiovascular': ['heart', 'cardiac', 'coronary', 'hypertension', 'stroke', 'vascular'],
            'respiratory': ['lung', 'asthma', 'copd', 'pneumonia', 'respiratory', 'pulmonary'],
            'neurological': ['alzheimer', 'parkinson', 'epilepsy', 'multiple sclerosis', 'neurological'],
            'infectious': ['covid', 'hiv', 'hepatitis', 'tuberculosis', 'infection', 'virus'],
            'metabolic': ['diabetes', 'obesity', 'metabolic', 'thyroid'],
            'psychiatric': ['depression', 'anxiety', 'schizophrenia', 'bipolar', 'mental'],
            'renal': ['kidney', 'renal', 'dialysis', 'nephropathy'],
            'gastrointestinal': ['crohn', 'colitis', 'ibd', 'liver', 'hepatic', 'gastro'],
            'rheumatologic': ['arthritis', 'lupus', 'rheumatoid', 'autoimmune']
        }

        for category, keywords in category_keywords.items():
            if any(keyword in condition_text for keyword in keywords):
                categories.add(category)

        return list(categories)

    def parse_date(self, date_struct: Optional[Dict]) -> Optional[str]:
        """Parse date structure from API"""
        if not date_struct:
            return None

        date_str = date_struct.get('date')
        if not date_str:
            return None

        try:
            # Parse various date formats
            for fmt in ['%Y-%m-%d', '%B %Y', '%Y-%m', '%Y']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%Y-%m-%d')
                except:
                    continue
        except:
            pass

        return None

    def insert_trials_to_bigquery(self, trials: List[Dict], status: str = "Recruiting"):
        """Insert trials batch to BigQuery"""
        if not trials:
            return

        logger.info(f"Inserting {len(trials)} trials to BigQuery...")

        # Parse all trials
        parsed_trials = []
        for trial in trials:
            try:
                parsed = self.parse_trial_data(trial)
                parsed_trials.append(parsed)
            except Exception as e:
                logger.warning(f"Error parsing trial: {e}")
                continue

        if not parsed_trials:
            return

        # Convert to DataFrame
        df = pd.DataFrame(parsed_trials)

        # Insert to BigQuery
        table_id = f"{self.project_id}.{self.dataset_id}.clinical_trials_complete"

        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
        )

        job = self.client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        job.result()

        logger.info(f"✅ Inserted {len(parsed_trials)} trials")

    def generate_summary_statistics(self):
        """Generate comprehensive statistics for imported trials"""
        logger.info("Generating trial statistics...")

        stats_query = f"""
        WITH trial_stats AS (
            SELECT
                COUNT(DISTINCT trial_id) as total_trials,
                COUNTIF(overall_status = 'Recruiting') as recruiting_trials,
                COUNTIF(overall_status = 'Completed') as completed_trials,
                COUNTIF(overall_status = 'Active, not recruiting') as active_not_recruiting,
                COUNTIF(overall_status = 'Terminated') as terminated_trials,

                -- Phase distribution
                COUNTIF(phase LIKE '%Phase 1%') as phase_1_trials,
                COUNTIF(phase LIKE '%Phase 2%') as phase_2_trials,
                COUNTIF(phase LIKE '%Phase 3%') as phase_3_trials,
                COUNTIF(phase LIKE '%Phase 4%') as phase_4_trials,

                -- Study types
                COUNTIF(study_type = 'Interventional') as interventional_trials,
                COUNTIF(study_type = 'Observational') as observational_trials,

                -- Geographic distribution
                COUNT(DISTINCT UNNEST(location_countries)) as unique_countries,

                -- Enrollment
                SUM(enrollment_count) as total_enrollment,
                AVG(enrollment_count) as avg_enrollment,

                -- Eligibility
                COUNTIF(eligibility_healthy_volunteers = TRUE) as accepts_healthy_volunteers,
                COUNTIF(ARRAY_LENGTH(eligibility_structured.inclusion_criteria) > 0) as with_structured_criteria

            FROM `{self.project_id}.{self.dataset_id}.clinical_trials_complete`
        ),

        condition_stats AS (
            SELECT
                COUNT(DISTINCT condition) as unique_conditions,
                COUNT(DISTINCT category) as unique_categories
            FROM `{self.project_id}.{self.dataset_id}.clinical_trials_complete`,
                UNNEST(conditions) as condition,
                UNNEST(condition_categories) as category
        )

        SELECT * FROM trial_stats CROSS JOIN condition_stats
        """

        stats = self.client.query(stats_query).to_dataframe().iloc[0]

        logger.info(f"""
        ==========================================
        CLINICAL TRIALS IMPORT SUMMARY
        ==========================================

        TRIAL COUNTS:
        -------------
        Total Trials: {stats['total_trials']:,}
        Recruiting: {stats['recruiting_trials']:,}
        Completed: {stats['completed_trials']:,}
        Active (not recruiting): {stats['active_not_recruiting']:,}
        Terminated: {stats['terminated_trials']:,}

        PHASE DISTRIBUTION:
        -------------------
        Phase 1: {stats['phase_1_trials']:,}
        Phase 2: {stats['phase_2_trials']:,}
        Phase 3: {stats['phase_3_trials']:,}
        Phase 4: {stats['phase_4_trials']:,}

        STUDY TYPES:
        ------------
        Interventional: {stats['interventional_trials']:,}
        Observational: {stats['observational_trials']:,}

        ENROLLMENT:
        -----------
        Total Enrollment: {stats['total_enrollment']:,.0f}
        Average per Trial: {stats['avg_enrollment']:.0f}

        GEOGRAPHIC REACH:
        -----------------
        Countries: {stats['unique_countries']:,}

        CONDITIONS:
        -----------
        Unique Conditions: {stats['unique_conditions']:,}
        Disease Categories: {stats['unique_categories']:,}

        ELIGIBILITY:
        ------------
        Accept Healthy Volunteers: {stats['accepts_healthy_volunteers']:,}
        With Structured Criteria: {stats['with_structured_criteria']:,}
        ==========================================
        """)

        return stats.to_dict()

def main():
    """Execute complete trial import"""
    start_time = time.time()

    logger.info("""
    ╔══════════════════════════════════════════════╗
    ║  COMPLETE CLINICAL TRIALS IMPORT SCRIPT     ║
    ║      Importing 70,000+ Active Trials        ║
    ╚══════════════════════════════════════════════╝
    """)

    importer = CompleteClinicalTrialsImporter()

    # Import all recruiting trials
    recruiting_count = importer.import_all_recruiting_trials()

    # Import recent completed trials
    completed_count = importer.import_recent_completed_trials()

    # Generate statistics
    stats = importer.generate_summary_statistics()

    elapsed_time = time.time() - start_time

    logger.info(f"""
    ╔══════════════════════════════════════════════╗
    ║           IMPORT COMPLETED!                  ║
    ║   Total Time: {elapsed_time/60:.1f} minutes              ║
    ║   Trials Imported: {recruiting_count + completed_count:,}           ║
    ╚══════════════════════════════════════════════╝

    Next Steps:
    1. Process clinical notes: python process_all_clinical_notes.py
    2. Generate patient-trial matches: python generate_complete_matches.py
    """)

    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'recruiting_imported': recruiting_count,
        'completed_imported': completed_count,
        'total_imported': recruiting_count + completed_count,
        'statistics': {k: float(v) if pd.api.types.is_number(v) else v
                      for k, v in stats.items()},
        'elapsed_time_minutes': elapsed_time / 60
    }

    with open('complete_trials_import_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    logger.info("Report saved to: complete_trials_import_report.json")

if __name__ == "__main__":
    main()