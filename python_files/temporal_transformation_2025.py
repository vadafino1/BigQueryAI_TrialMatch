#!/usr/bin/env python3
"""
MIMIC-IV Temporal Transformation for 2025 Clinical Trial Matching
==================================================================
Transforms MIMIC-IV future timestamps (2100-2200) to current 2025 timeframe
Preserves relative timing while making data appear current for trial matching

Author: BigQuery 2025 Competition Team
Date: September 20, 2025
"""

from google.cloud import bigquery
from datetime import datetime, date
import logging
import json
from typing import Dict, Any

# Constants
CURRENT_DATE = date(2025, 9, 20)
PROJECT_ID = "YOUR_PROJECT_ID"
DATASET_ID = "clinical_trial_matching"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TemporalTransformer2025:
    """Transform MIMIC-IV timestamps to 2025 for accurate trial matching"""

    def __init__(self):
        self.client = bigquery.Client(project=PROJECT_ID)
        self.logger = logging.getLogger(__name__)
        self.transformation_stats = {}

    def create_normalization_mapping(self) -> None:
        """Create mapping table for temporal shifts - SIMPLIFIED VERSION"""

        self.logger.info("Creating temporal normalization mapping...")

        query = f"""
        CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.temporal_normalization_2025`
        PARTITION BY RANGE_BUCKET(CAST(patient_id AS INT64), GENERATE_ARRAY(0, 1000000, 10000))
        AS
        WITH patient_anchors AS (
            -- Get patient anchor information from MIMIC-IV
            SELECT DISTINCT
                CAST(subject_id AS STRING) as patient_id,
                anchor_year,
                anchor_year_group,
                anchor_age
            FROM `physionet-data.mimiciv_3_1_hosp.patients`
        )
        SELECT
            patient_id,
            anchor_year,
            anchor_year_group,
            anchor_age,

            -- SIMPLIFIED: Use consistent 186-year shift for all patients
            -- This brings timestamps from 2100s to 1919-2026 range
            186 as years_to_subtract,

            -- No additional months needed with simple shift
            0 as months_to_add,

            -- Calculate current age in 2025 based on anchor_age
            CASE
                WHEN anchor_year_group = '2008 - 2010' THEN anchor_age + (2025 - 2009)
                WHEN anchor_year_group = '2011 - 2013' THEN anchor_age + (2025 - 2012)
                WHEN anchor_year_group = '2014 - 2016' THEN anchor_age + (2025 - 2015)
                WHEN anchor_year_group = '2017 - 2019' THEN anchor_age + (2025 - 2018)
                ELSE anchor_age + 15  -- Default
            END as age_in_2025,

            -- Metadata
            CURRENT_TIMESTAMP() as transformation_timestamp,
            'MIMIC-IV to 2025 simplified normalization - 186 year shift' as transformation_type
        FROM patient_anchors
        """

        job = self.client.query(query)
        job.result()

        # Get statistics
        stats_query = f"""
        SELECT
            COUNT(*) as total_patients,
            AVG(years_to_subtract) as avg_years_shifted,
            MIN(age_in_2025) as min_age_2025,
            MAX(age_in_2025) as max_age_2025,
            AVG(age_in_2025) as avg_age_2025
        FROM `{PROJECT_ID}.{DATASET_ID}.temporal_normalization_2025`
        """

        stats = self.client.query(stats_query).to_dataframe().iloc[0]
        self.transformation_stats['mapping'] = stats.to_dict()

        self.logger.info(f"✅ Created temporal normalization mapping for {stats['total_patients']:,.0f} patients")

    def transform_discharge_summaries(self) -> None:
        """Transform discharge summaries to 2025 timeframe"""

        self.logger.info("Transforming discharge summaries...")

        query = f"""
        CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.discharge_summaries_2025`
        PARTITION BY DATE(note_datetime_normalized)
        CLUSTER BY patient_id
        AS
        SELECT
            d.*,
            -- Keep original timestamp for reference
            d.note_datetime as note_datetime_original,

            -- Normalized to 2025 timeframe
            TIMESTAMP_ADD(
                TIMESTAMP_SUB(d.note_datetime, INTERVAL t.years_to_subtract YEAR),
                INTERVAL t.months_to_add MONTH
            ) as note_datetime_normalized,

            -- Days from current date (September 20, 2025)
            DATE_DIFF(
                DATE('2025-09-20'),
                DATE(TIMESTAMP_ADD(
                    TIMESTAMP_SUB(d.note_datetime, INTERVAL t.years_to_subtract YEAR),
                    INTERVAL t.months_to_add MONTH
                )),
                DAY
            ) as days_ago,

            -- Recency category for eligibility
            CASE
                WHEN DATE_DIFF(DATE('2025-09-20'),
                     DATE(TIMESTAMP_ADD(
                       TIMESTAMP_SUB(d.note_datetime, INTERVAL t.years_to_subtract YEAR),
                       INTERVAL t.months_to_add MONTH
                     )), DAY) <= 30 THEN 'Current'
                WHEN DATE_DIFF(DATE('2025-09-20'),
                     DATE(TIMESTAMP_ADD(
                       TIMESTAMP_SUB(d.note_datetime, INTERVAL t.years_to_subtract YEAR),
                       INTERVAL t.months_to_add MONTH
                     )), DAY) <= 90 THEN 'Recent'
                WHEN DATE_DIFF(DATE('2025-09-20'),
                     DATE(TIMESTAMP_ADD(
                       TIMESTAMP_SUB(d.note_datetime, INTERVAL t.years_to_subtract YEAR),
                       INTERVAL t.months_to_add MONTH
                     )), DAY) <= 365 THEN 'Within_Year'
                ELSE 'Historical'
            END as recency_category,

            -- Add transformation metadata
            t.age_in_2025 as patient_age_2025

        FROM `{PROJECT_ID}.{DATASET_ID}.discharge_summaries` d
        JOIN `{PROJECT_ID}.{DATASET_ID}.temporal_normalization_2025` t
            ON d.patient_id = t.patient_id
        """

        job = self.client.query(query)
        job.result()

        # Get statistics
        stats_query = f"""
        SELECT
            COUNT(*) as total_records,
            COUNT(DISTINCT patient_id) as unique_patients,
            COUNTIF(recency_category = 'Current') as current_records,
            COUNTIF(recency_category = 'Recent') as recent_records,
            MIN(days_ago) as most_recent_days,
            MAX(days_ago) as oldest_days
        FROM `{PROJECT_ID}.{DATASET_ID}.discharge_summaries_2025`
        """

        stats = self.client.query(stats_query).to_dataframe().iloc[0]
        self.transformation_stats['discharge_summaries'] = stats.to_dict()

        self.logger.info(f"✅ Transformed {stats['total_records']:,.0f} discharge summaries")

    def transform_lab_events(self) -> None:
        """Transform lab events to 2025 timeframe"""

        self.logger.info("Transforming lab events...")

        query = f"""
        CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.lab_events_2025`
        PARTITION BY DATE(lab_datetime_normalized)
        CLUSTER BY patient_id, lab_name
        AS
        SELECT
            l.*,
            l.lab_datetime as lab_datetime_original,

            -- Normalized timestamp
            TIMESTAMP_ADD(
                TIMESTAMP_SUB(l.lab_datetime, INTERVAL t.years_to_subtract YEAR),
                INTERVAL t.months_to_add MONTH
            ) as lab_datetime_normalized,

            -- Days since test (from September 20, 2025)
            DATE_DIFF(DATE('2025-09-20'),
                DATE(TIMESTAMP_ADD(
                    TIMESTAMP_SUB(l.lab_datetime, INTERVAL t.years_to_subtract YEAR),
                    INTERVAL t.months_to_add MONTH
                )), DAY) as days_since_test,

            -- Eligibility windows for clinical trials
            CASE
                WHEN DATE_DIFF(DATE('2025-09-20'),
                    DATE(TIMESTAMP_ADD(
                        TIMESTAMP_SUB(l.lab_datetime, INTERVAL t.years_to_subtract YEAR),
                        INTERVAL t.months_to_add MONTH
                    )), DAY) <= 7 THEN 'Within_Week'
                WHEN DATE_DIFF(DATE('2025-09-20'),
                    DATE(TIMESTAMP_ADD(
                        TIMESTAMP_SUB(l.lab_datetime, INTERVAL t.years_to_subtract YEAR),
                        INTERVAL t.months_to_add MONTH
                    )), DAY) <= 30 THEN 'Within_Month'
                WHEN DATE_DIFF(DATE('2025-09-20'),
                    DATE(TIMESTAMP_ADD(
                        TIMESTAMP_SUB(l.lab_datetime, INTERVAL t.years_to_subtract YEAR),
                        INTERVAL t.months_to_add MONTH
                    )), DAY) <= 90 THEN 'Within_3_Months'
                WHEN DATE_DIFF(DATE('2025-09-20'),
                    DATE(TIMESTAMP_ADD(
                        TIMESTAMP_SUB(l.lab_datetime, INTERVAL t.years_to_subtract YEAR),
                        INTERVAL t.months_to_add MONTH
                    )), DAY) <= 180 THEN 'Within_6_Months'
                ELSE 'Older'
            END as eligibility_window,

            t.age_in_2025 as patient_age_2025

        FROM `{PROJECT_ID}.{DATASET_ID}.lab_events` l
        JOIN `{PROJECT_ID}.{DATASET_ID}.temporal_normalization_2025` t
            ON l.patient_id = t.patient_id
        """

        job = self.client.query(query)
        job.result()

        # Get statistics
        stats_query = f"""
        SELECT
            COUNT(*) as total_records,
            COUNT(DISTINCT patient_id) as unique_patients,
            COUNTIF(eligibility_window = 'Within_Week') as within_week,
            COUNTIF(eligibility_window = 'Within_Month') as within_month,
            COUNTIF(eligibility_window IN ('Within_Week', 'Within_Month')) as current_labs,
            MIN(days_since_test) as most_recent_days
        FROM `{PROJECT_ID}.{DATASET_ID}.lab_events_2025`
        """

        stats = self.client.query(stats_query).to_dataframe().iloc[0]
        self.transformation_stats['lab_events'] = stats.to_dict()

        self.logger.info(f"✅ Transformed {stats['total_records']:,.0f} lab events")

    def transform_medications(self) -> None:
        """Transform medications to 2025 timeframe"""

        self.logger.info("Transforming medications...")

        query = f"""
        CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.medications_2025`
        CLUSTER BY patient_id, drug
        AS
        SELECT
            m.*,
            m.medication_start as medication_start_original,
            m.medication_stop as medication_stop_original,

            -- Normalized start time
            TIMESTAMP_ADD(
                TIMESTAMP_SUB(m.medication_start, INTERVAL t.years_to_subtract YEAR),
                INTERVAL t.months_to_add MONTH
            ) as medication_start_normalized,

            -- Normalized stop time (if exists)
            CASE
                WHEN m.medication_stop IS NOT NULL THEN
                    TIMESTAMP_ADD(
                        TIMESTAMP_SUB(m.medication_stop, INTERVAL t.years_to_subtract YEAR),
                        INTERVAL t.months_to_add MONTH
                    )
                ELSE NULL
            END as medication_stop_normalized,

            -- Is medication currently active (as of September 2025)?
            CASE
                WHEN m.medication_stop IS NULL THEN FALSE  -- Old data, assume discontinued
                WHEN DATE(TIMESTAMP_ADD(
                       TIMESTAMP_SUB(m.medication_stop, INTERVAL t.years_to_subtract YEAR),
                       INTERVAL t.months_to_add MONTH
                     )) >= DATE('2025-09-20') THEN TRUE
                ELSE FALSE
            END as is_currently_active,

            -- Days since stopped (for washout period calculations)
            CASE
                WHEN m.medication_stop IS NOT NULL THEN
                    DATE_DIFF(DATE('2025-09-20'),
                        DATE(TIMESTAMP_ADD(
                            TIMESTAMP_SUB(m.medication_stop, INTERVAL t.years_to_subtract YEAR),
                            INTERVAL t.months_to_add MONTH
                        )), DAY)
                ELSE 9999  -- Not stopped or very old
            END as days_since_stopped,

            t.age_in_2025 as patient_age_2025

        FROM `{PROJECT_ID}.{DATASET_ID}.medications` m
        JOIN `{PROJECT_ID}.{DATASET_ID}.temporal_normalization_2025` t
            ON m.patient_id = t.patient_id
        """

        job = self.client.query(query)
        job.result()

        # Get statistics
        stats_query = f"""
        SELECT
            COUNT(*) as total_records,
            COUNT(DISTINCT patient_id) as unique_patients,
            COUNTIF(is_currently_active) as active_medications,
            COUNT(DISTINCT CASE WHEN is_currently_active THEN drug END) as unique_active_drugs,
            MIN(CASE WHEN days_since_stopped < 9999 THEN days_since_stopped END) as min_washout_days
        FROM `{PROJECT_ID}.{DATASET_ID}.medications_2025`
        """

        stats = self.client.query(stats_query).to_dataframe().iloc[0]
        self.transformation_stats['medications'] = stats.to_dict()

        self.logger.info(f"✅ Transformed {stats['total_records']:,.0f} medication records")

    def transform_radiology_reports(self) -> None:
        """Transform radiology reports to 2025 timeframe"""

        self.logger.info("Transforming radiology reports...")

        query = f"""
        CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.radiology_reports_2025`
        PARTITION BY DATE(report_datetime_normalized)
        CLUSTER BY patient_id
        AS
        SELECT
            r.*,
            r.report_datetime as report_datetime_original,

            -- Normalized timestamp
            TIMESTAMP_ADD(
                TIMESTAMP_SUB(r.report_datetime, INTERVAL t.years_to_subtract YEAR),
                INTERVAL t.months_to_add MONTH
            ) as report_datetime_normalized,

            -- Days since report
            DATE_DIFF(DATE('2025-09-20'),
                DATE(TIMESTAMP_ADD(
                    TIMESTAMP_SUB(r.report_datetime, INTERVAL t.years_to_subtract YEAR),
                    INTERVAL t.months_to_add MONTH
                )), DAY) as days_since_report,

            -- Recency for eligibility
            CASE
                WHEN DATE_DIFF(DATE('2025-09-20'),
                    DATE(TIMESTAMP_ADD(
                        TIMESTAMP_SUB(r.report_datetime, INTERVAL t.years_to_subtract YEAR),
                        INTERVAL t.months_to_add MONTH
                    )), DAY) <= 90 THEN 'Recent_Imaging'
                WHEN DATE_DIFF(DATE('2025-09-20'),
                    DATE(TIMESTAMP_ADD(
                        TIMESTAMP_SUB(r.report_datetime, INTERVAL t.years_to_subtract YEAR),
                        INTERVAL t.months_to_add MONTH
                    )), DAY) <= 365 THEN 'Within_Year'
                ELSE 'Historical'
            END as imaging_recency,

            t.age_in_2025 as patient_age_2025

        FROM `{PROJECT_ID}.{DATASET_ID}.radiology_reports` r
        JOIN `{PROJECT_ID}.{DATASET_ID}.temporal_normalization_2025` t
            ON r.patient_id = t.patient_id
        """

        job = self.client.query(query)
        job.result()

        self.logger.info("✅ Transformed radiology reports")

    def create_patient_current_status(self) -> None:
        """Create comprehensive patient status as of September 2025"""

        self.logger.info("Creating patient current status table...")

        query = f"""
        CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.patient_current_status_2025`
        CLUSTER BY patient_id, trial_readiness
        AS
        WITH latest_labs AS (
            SELECT
                patient_id,
                MIN(days_since_test) as most_recent_lab_days,
                MAX(CASE WHEN days_since_test <= 30 THEN 1 ELSE 0 END) as has_current_labs,
                MAX(CASE WHEN days_since_test <= 90 THEN 1 ELSE 0 END) as has_recent_labs
            FROM `{PROJECT_ID}.{DATASET_ID}.lab_events_2025`
            GROUP BY patient_id
        ),
        latest_notes AS (
            SELECT
                patient_id,
                MIN(days_ago) as most_recent_note_days,
                MAX(CASE WHEN days_ago <= 30 THEN 1 ELSE 0 END) as has_current_notes,
                MAX(CASE WHEN days_ago <= 90 THEN 1 ELSE 0 END) as has_recent_notes
            FROM `{PROJECT_ID}.{DATASET_ID}.discharge_summaries_2025`
            GROUP BY patient_id
        ),
        current_medications AS (
            SELECT
                patient_id,
                COUNT(DISTINCT CASE WHEN is_currently_active THEN drug END) as active_medication_count,
                STRING_AGG(DISTINCT CASE WHEN is_currently_active THEN drug END, ', ' LIMIT 10) as active_medications,
                MIN(CASE WHEN NOT is_currently_active AND days_since_stopped < 9999
                    THEN days_since_stopped END) as shortest_washout_days
            FROM `{PROJECT_ID}.{DATASET_ID}.medications_2025`
            GROUP BY patient_id
        ),
        recent_imaging AS (
            SELECT
                patient_id,
                MIN(days_since_report) as most_recent_imaging_days,
                MAX(CASE WHEN imaging_recency = 'Recent_Imaging' THEN 1 ELSE 0 END) as has_recent_imaging
            FROM `{PROJECT_ID}.{DATASET_ID}.radiology_reports_2025`
            GROUP BY patient_id
        )
        SELECT
            t.patient_id,
            t.age_in_2025 as current_age,
            t.anchor_year_group as original_timeframe,

            -- Activity status
            COALESCE(l.most_recent_lab_days, 9999) as days_since_last_lab,
            COALESCE(n.most_recent_note_days, 9999) as days_since_last_note,
            COALESCE(i.most_recent_imaging_days, 9999) as days_since_last_imaging,

            -- Current status flags
            COALESCE(l.has_current_labs, 0) as has_current_labs,
            COALESCE(n.has_current_notes, 0) as has_current_notes,
            COALESCE(l.has_recent_labs, 0) as has_recent_labs,
            COALESCE(n.has_recent_notes, 0) as has_recent_notes,
            COALESCE(i.has_recent_imaging, 0) as has_recent_imaging,

            -- Medications
            COALESCE(m.active_medication_count, 0) as active_medication_count,
            m.active_medications,
            COALESCE(m.shortest_washout_days, 9999) as shortest_washout_days,

            -- Trial readiness assessment
            CASE
                WHEN l.has_current_labs = 1 AND n.has_current_notes = 1
                    THEN 'Active_Ready'
                WHEN l.has_recent_labs = 1 OR n.has_recent_notes = 1
                    THEN 'Recent_Screening_Needed'
                WHEN l.most_recent_lab_days <= 365 OR n.most_recent_note_days <= 365
                    THEN 'Inactive_Full_Screening'
                ELSE 'Historical_Not_Eligible'
            END as trial_readiness,

            -- Eligibility score (0-100)
            CAST(
                (CASE WHEN l.has_current_labs = 1 THEN 40 ELSE 0 END) +
                (CASE WHEN n.has_current_notes = 1 THEN 30 ELSE 0 END) +
                (CASE WHEN i.has_recent_imaging = 1 THEN 20 ELSE 0 END) +
                (CASE WHEN m.active_medication_count > 0 THEN 10 ELSE 0 END)
            AS INT64) as eligibility_score,

            DATE('2025-09-20') as status_date,
            CURRENT_TIMESTAMP() as created_at

        FROM `{PROJECT_ID}.{DATASET_ID}.temporal_normalization_2025` t
        LEFT JOIN latest_labs l ON t.patient_id = l.patient_id
        LEFT JOIN latest_notes n ON t.patient_id = n.patient_id
        LEFT JOIN current_medications m ON t.patient_id = m.patient_id
        LEFT JOIN recent_imaging i ON t.patient_id = i.patient_id
        """

        job = self.client.query(query)
        job.result()

        # Get final statistics
        stats_query = f"""
        SELECT
            COUNT(DISTINCT patient_id) as total_patients,
            COUNTIF(trial_readiness = 'Active_Ready') as active_patients,
            COUNTIF(trial_readiness = 'Recent_Screening_Needed') as recent_patients,
            COUNTIF(has_current_labs = 1) as with_current_labs,
            COUNTIF(has_current_notes = 1) as with_current_notes,
            AVG(current_age) as avg_age_2025,
            MIN(days_since_last_lab) as most_recent_lab_days,
            AVG(eligibility_score) as avg_eligibility_score,
            ROUND(COUNTIF(trial_readiness = 'Active_Ready') / COUNT(*) * 100, 2) as pct_trial_ready
        FROM `{PROJECT_ID}.{DATASET_ID}.patient_current_status_2025`
        """

        stats = self.client.query(stats_query).to_dataframe().iloc[0]
        self.transformation_stats['patient_status'] = stats.to_dict()

        self.logger.info(f"✅ Created patient status for {stats['total_patients']:,.0f} patients")

    def validate_transformation(self) -> Dict[str, Any]:
        """Validate the transformation results"""

        self.logger.info("Validating transformation...")

        validation_results = {}

        # Check date ranges
        date_check_query = f"""
        WITH date_ranges AS (
            SELECT
                'discharge_summaries' as table_name,
                MIN(EXTRACT(YEAR FROM note_datetime_original)) as min_original_year,
                MAX(EXTRACT(YEAR FROM note_datetime_original)) as max_original_year,
                MIN(EXTRACT(YEAR FROM note_datetime_normalized)) as min_normalized_year,
                MAX(EXTRACT(YEAR FROM note_datetime_normalized)) as max_normalized_year
            FROM `{PROJECT_ID}.{DATASET_ID}.discharge_summaries_2025`

            UNION ALL

            SELECT
                'lab_events' as table_name,
                MIN(EXTRACT(YEAR FROM lab_datetime_original)) as min_original_year,
                MAX(EXTRACT(YEAR FROM lab_datetime_original)) as max_original_year,
                MIN(EXTRACT(YEAR FROM lab_datetime_normalized)) as min_normalized_year,
                MAX(EXTRACT(YEAR FROM lab_datetime_normalized)) as max_normalized_year
            FROM `{PROJECT_ID}.{DATASET_ID}.lab_events_2025`
        )
        SELECT * FROM date_ranges
        """

        date_ranges = self.client.query(date_check_query).to_dataframe()
        validation_results['date_ranges'] = date_ranges.to_dict('records')

        # Check data distribution
        distribution_query = f"""
        SELECT
            trial_readiness,
            COUNT(*) as patient_count,
            ROUND(COUNT(*) / SUM(COUNT(*)) OVER() * 100, 2) as percentage
        FROM `{PROJECT_ID}.{DATASET_ID}.patient_current_status_2025`
        GROUP BY trial_readiness
        ORDER BY patient_count DESC
        """

        distribution = self.client.query(distribution_query).to_dataframe()
        validation_results['readiness_distribution'] = distribution.to_dict('records')

        return validation_results

    def save_transformation_report(self) -> None:
        """Save transformation report to JSON file"""

        report = {
            'transformation_date': str(CURRENT_DATE),
            'statistics': self.transformation_stats,
            'validation': self.validate_transformation(),
            'tables_created': [
                'temporal_normalization_2025',
                'discharge_summaries_2025',
                'lab_events_2025',
                'medications_2025',
                'radiology_reports_2025',
                'patient_current_status_2025'
            ]
        }

        with open('temporal_transformation_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info("📊 Saved transformation report to temporal_transformation_report.json")

    def run_full_transformation(self) -> Dict[str, Any]:
        """Execute complete temporal transformation pipeline"""

        self.logger.info("=" * 60)
        self.logger.info("Starting temporal transformation to 2025...")
        self.logger.info("=" * 60)

        try:
            # Step 1: Create mapping
            self.create_normalization_mapping()

            # Step 2: Transform clinical data tables
            self.transform_discharge_summaries()
            self.transform_lab_events()
            self.transform_medications()
            self.transform_radiology_reports()

            # Step 3: Create patient status table
            self.create_patient_current_status()

            # Step 4: Save report
            self.save_transformation_report()

            # Final statistics
            final_stats = self.transformation_stats.get('patient_status', {})

            self.logger.info("=" * 60)
            self.logger.info("✅ TRANSFORMATION COMPLETE")
            self.logger.info("=" * 60)
            self.logger.info(f"Total Patients: {final_stats.get('total_patients', 0):,.0f}")
            self.logger.info(f"Trial Ready: {final_stats.get('active_patients', 0):,.0f} ({final_stats.get('pct_trial_ready', 0)}%)")
            self.logger.info(f"With Current Labs: {final_stats.get('with_current_labs', 0):,.0f}")
            self.logger.info(f"Average Age in 2025: {final_stats.get('avg_age_2025', 0):.1f}")
            self.logger.info(f"Most Recent Lab: {final_stats.get('most_recent_lab_days', 0):.0f} days ago")
            self.logger.info("=" * 60)

            return self.transformation_stats

        except Exception as e:
            self.logger.error(f"❌ Transformation failed: {str(e)}")
            raise


def main():
    """Main execution function"""

    print("""
    ╔════════════════════════════════════════════════════════╗
    ║  MIMIC-IV TEMPORAL TRANSFORMATION FOR 2025            ║
    ║  Converting 2100s timestamps to current 2025 timeframe ║
    ║                                                        ║
    ║  Purpose: Enable accurate clinical trial eligibility   ║
    ║  matching with proper temporal criteria evaluation     ║
    ╚════════════════════════════════════════════════════════╝
    """)

    try:
        transformer = TemporalTransformer2025()
        stats = transformer.run_full_transformation()

        print("\n✅ Transformation successful! Data now appears current for September 2025.")
        print("\n📊 Tables created:")
        print("  - temporal_normalization_2025 (mapping table)")
        print("  - discharge_summaries_2025 (normalized clinical notes)")
        print("  - lab_events_2025 (normalized lab results)")
        print("  - medications_2025 (normalized medications)")
        print("  - radiology_reports_2025 (normalized imaging)")
        print("  - patient_current_status_2025 (trial readiness assessment)")
        print("\n📝 Report saved to: temporal_transformation_report.json")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())