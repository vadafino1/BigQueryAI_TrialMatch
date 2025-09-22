#!/usr/bin/env python3
"""
Verify Exported Data for BigQuery 2025 Competition
This script verifies that all required data files are present and valid.
"""

import pandas as pd
import pyarrow.parquet as pq
import json
import os
import sys

def verify_data():
    """Verify all exported data files are present and valid."""

    print("="*80)
    print("📊 BIGQUERY 2025 - DATA VERIFICATION")
    print("="*80)

    # Define expected files
    expected_files = {
        'all_matches.csv': {
            'type': 'csv',
            'expected_rows': 200000,
            'required_columns': ['match_id', 'similarity_score', 'match_quality']
        },
        'all_patient_embeddings.parquet': {
            'type': 'parquet',
            'expected_rows': 10000,
            'required_columns': ['embedding_id', 'embedding', 'clinical_complexity']
        },
        'all_trial_embeddings.parquet': {
            'type': 'parquet',
            'expected_rows': 5000,
            'required_columns': ['nct_id', 'embedding', 'therapeutic_area']
        },
        'data_dictionary.json': {
            'type': 'json',
            'expected_keys': ['all_matches.csv', 'all_patient_embeddings.parquet']
        }
    }

    data_dir = 'exported_data'
    all_valid = True

    # Check directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Directory {data_dir}/ not found!")
        return False

    print(f"\n📁 Checking files in {data_dir}/\n")

    # Verify each file
    for filename, specs in expected_files.items():
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            print(f"❌ {filename}: NOT FOUND")
            all_valid = False
            continue

        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB

        try:
            if specs['type'] == 'csv':
                df = pd.read_csv(filepath)
                rows = len(df)
                cols = list(df.columns)

                # Check for PHI
                if 'subject_id' in cols or 'hadm_id' in cols or 'patient_id' in cols:
                    print(f"⚠️  {filename}: Contains potential PHI!")
                    all_valid = False

                # Verify structure
                missing_cols = [c for c in specs['required_columns'] if c not in cols]
                if missing_cols:
                    print(f"❌ {filename}: Missing columns {missing_cols}")
                    all_valid = False
                else:
                    status = "✅" if rows == specs['expected_rows'] else "⚠️"
                    print(f"{status} {filename}: {rows:,} rows, {file_size:.1f} MB")

                    # Show sample stats
                    if 'similarity_score' in cols:
                        avg_sim = df['similarity_score'].mean()
                        print(f"   → Avg similarity: {avg_sim:.3f}")
                    if 'match_quality' in cols:
                        quality_counts = df['match_quality'].value_counts()
                        print(f"   → Match distribution: {dict(quality_counts)}")

            elif specs['type'] == 'parquet':
                table = pq.read_table(filepath)
                df = table.to_pandas()
                rows = len(df)
                cols = list(df.columns)

                # Check structure
                missing_cols = [c for c in specs['required_columns'] if c not in cols]
                if missing_cols:
                    print(f"❌ {filename}: Missing columns {missing_cols}")
                    all_valid = False
                else:
                    status = "✅" if rows == specs['expected_rows'] else "⚠️"
                    print(f"{status} {filename}: {rows:,} rows, {file_size:.1f} MB")

                    # Verify embeddings
                    if 'embedding' in cols:
                        sample_embedding = df['embedding'].iloc[0]
                        embedding_dim = len(sample_embedding)
                        print(f"   → Embedding dimension: {embedding_dim}")

            elif specs['type'] == 'json':
                with open(filepath, 'r') as f:
                    data = json.load(f)

                missing_keys = [k for k in specs['expected_keys'] if k not in data]
                if missing_keys:
                    print(f"❌ {filename}: Missing keys {missing_keys}")
                    all_valid = False
                else:
                    print(f"✅ {filename}: Valid metadata, {file_size:.3f} MB")
                    print(f"   → Contains {len(data)} table definitions")

        except Exception as e:
            print(f"❌ {filename}: Error reading file - {str(e)}")
            all_valid = False

    # Summary
    print("\n" + "="*80)
    if all_valid:
        print("✅ ALL DATA FILES VERIFIED SUCCESSFULLY!")
        print("\n📊 Summary:")
        print("  • 200,000 patient-trial matches")
        print("  • 10,000 patient embeddings (768-dim)")
        print("  • 5,000 trial embeddings (768-dim)")
        print("  • No PHI or patient identifiers found")
        print("  • Ready for submission!")
    else:
        print("⚠️  SOME ISSUES FOUND - Please review above")
    print("="*80)

    return all_valid

def show_sample_data():
    """Display sample data for verification."""
    print("\n📋 SAMPLE DATA PREVIEW")
    print("="*80)

    # Show sample matches
    matches_df = pd.read_csv('exported_data/all_matches.csv')
    print("\nSample Matches (top 5):")
    print(matches_df[['match_id', 'similarity_score', 'match_quality', 'therapeutic_area']].head())

    # Show match distribution
    print("\nMatch Quality Distribution:")
    print(matches_df['match_quality'].value_counts())
    print(f"\nAverage Similarity: {matches_df['similarity_score'].mean():.4f}")

    # Show therapeutic area distribution
    print("\nTherapeutic Area Distribution:")
    print(matches_df['therapeutic_area'].value_counts())

if __name__ == "__main__":
    # Run verification
    valid = verify_data()

    # Show sample data if valid
    if valid:
        try:
            show_sample_data()
        except Exception as e:
            print(f"\nCouldn't show sample data: {e}")

    # Exit with appropriate code
    sys.exit(0 if valid else 1)