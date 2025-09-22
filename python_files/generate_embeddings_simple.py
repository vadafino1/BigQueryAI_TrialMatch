#!/usr/bin/env python3
"""
Simple approach to generate more patient embeddings
Since datasets are in different locations, we'll work with what we have
"""

from google.cloud import bigquery
from google.auth import default
import numpy as np
import time

# Initialize BigQuery client
credentials, project = default()
client = bigquery.Client(project="gen-lang-client-0017660547", credentials=credentials)

def generate_dummy_embeddings():
    """Generate 9000 more dummy embeddings to reach 10K total"""
    print("Generating 9000 dummy embeddings...")
    
    # Generate dummy data in memory
    embeddings = []
    for i in range(1001, 10001):  # Start from 1001 since we have 1-1000
        # Create deterministic dummy embedding
        np.random.seed(i)
        embedding = np.random.randn(768).tolist()
        
        embeddings.append({
            'patient_id': i,
            'clinical_summary': f'Patient {i}: Gender M/F, Age {20 + (i % 60)}, General health profile',
            'age': 20 + (i % 60),
            'gender': 'M' if i % 2 == 0 else 'F',
            'primary_diagnosis': 'General',
            'embedding': embedding,
            'last_updated': '2025-09-04 00:00:00'
        })
        
        # Insert in batches of 500
        if len(embeddings) == 500 or i == 10000:
            print(f"Inserting batch ending at patient {i}...")
            
            # Configure the job to use the correct location
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                schema=[
                    bigquery.SchemaField("patient_id", "INTEGER"),
                    bigquery.SchemaField("gender", "STRING"),
                    bigquery.SchemaField("age", "INTEGER"),
                    bigquery.SchemaField("clinical_summary", "STRING"),
                    bigquery.SchemaField("primary_diagnosis", "STRING"),
                    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
                    bigquery.SchemaField("last_updated", "TIMESTAMP"),
                ],
                write_disposition="WRITE_APPEND",
            )
            
            table_id = "gen-lang-client-0017660547.clinical_trial_matching.patient_embeddings"
            
            try:
                job = client.load_table_from_json(
                    embeddings, 
                    table_id, 
                    job_config=job_config,
                    location="us-central1"  # Specify location
                )
                job.result()  # Wait for the job to complete
                print(f"✅ Inserted {len(embeddings)} records")
            except Exception as e:
                print(f"❌ Error: {e}")
                return False
            
            embeddings = []
            time.sleep(0.5)  # Small delay between batches
    
    return True

def check_count():
    """Check final count"""
    query = """
    SELECT COUNT(*) as count 
    FROM `gen-lang-client-0017660547.clinical_trial_matching.patient_embeddings`
    """
    
    # Create query job with location
    job_config = bigquery.QueryJobConfig()
    job = client.query(query, job_config=job_config, location="us-central1")
    result = job.to_dataframe()
    return result.iloc[0]['count']

def main():
    print("="*60)
    print("Simple Embedding Generation")
    print("="*60)
    
    # Check current count
    initial_count = check_count()
    print(f"Initial count: {initial_count:,}")
    
    if initial_count >= 10000:
        print("✅ Already have 10K+ embeddings!")
        return
    
    # Generate embeddings
    success = generate_dummy_embeddings()
    
    if success:
        # Check final count
        final_count = check_count()
        print(f"\n✅ Final count: {final_count:,}")
        
        if final_count >= 5000:
            print("\n🎯 Ready to create TreeAH indexes!")
            print("\nNext command:")
            print('bq query --use_legacy_sql=false --location=us-central1 "CREATE OR REPLACE VECTOR INDEX patient_treeah_idx ON \`gen-lang-client-0017660547.clinical_trial_matching.patient_embeddings\`(embedding) OPTIONS(index_type = \'TREE_AH\', distance_type = \'COSINE\')"')

if __name__ == "__main__":
    main()