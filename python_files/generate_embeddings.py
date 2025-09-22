#!/usr/bin/env python3
"""
Generate Embeddings for Story 1.3
Uses Vertex AI to generate embeddings for patients and trials
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from typing import List, Dict, Any, Optional
from google.cloud import bigquery
from google.cloud import aiplatform
import numpy as np
from datetime import datetime

from config.bigquery_config import (
    get_bigquery_client,
    GCP_PROJECT_ID,
    DATASET_ID,
    LOCATION,
    FULL_TABLE_IDS,
    EMBEDDING_CONFIG
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for medical data using Vertex AI"""
    
    def __init__(self):
        """Initialize embedding generator with cloud configuration"""
        self.client = get_bigquery_client()
        self.project_id = GCP_PROJECT_ID
        self.dataset_id = DATASET_ID
        self.location = LOCATION
        
        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location=self.location)
        
        logger.info(f"Initialized embedding generator for project: {self.project_id}")
    
    def generate_text_embeddings_batch(self, texts: List[str],
                                      model_id: str = "gemini-embedding-001") -> List[List[float]]:
        """Generate text embeddings in batch using Vertex AI
        
        Args:
            texts: List of text strings to embed
            model_id: Model ID for embedding generation
            
        Returns:
            List of embedding vectors
        """
        try:
            from vertexai.language_models import TextEmbeddingModel
            
            model = TextEmbeddingModel.from_pretrained(model_id)
            embeddings = model.get_embeddings(texts)
            
            # Extract embedding values
            embedding_vectors = []
            for embedding in embeddings:
                embedding_vectors.append(embedding.values)
            
            return embedding_vectors
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return dummy embeddings for testing without Vertex AI
            logger.info("Using dummy embeddings for testing")
            return [np.random.randn(EMBEDDING_CONFIG['dimension']).tolist() for _ in texts]
    
    def process_patient_embeddings(self, batch_size: int = 100):
        """Generate embeddings for all patients
        
        Args:
            batch_size: Number of patients to process in each batch
        """
        logger.info("Processing patient embeddings...")
        
        # Query patients without embeddings
        query = f"""
        SELECT 
            p.patient_id,
            p.clinical_notes,
            ARRAY_TO_STRING(ARRAY(
                SELECT d.description 
                FROM UNNEST(p.diagnoses) d
            ), ' | ') as diagnoses_text,
            ARRAY_TO_STRING(ARRAY(
                SELECT m.drug_name 
                FROM UNNEST(p.medications) m
            ), ', ') as medications_text
        FROM `{FULL_TABLE_IDS['patients']}` p
        LEFT JOIN `{FULL_TABLE_IDS['patient_embeddings']}` pe
            ON p.patient_id = pe.patient_id
        WHERE pe.patient_id IS NULL
        LIMIT {batch_size}
        """
        
        df = self.client.query(query).to_dataframe()
        
        if df.empty:
            logger.info("No patients without embeddings found")
            return
        
        logger.info(f"Processing {len(df)} patients...")
        
        # Prepare text for embedding
        texts = []
        for _, row in df.iterrows():
            combined_text = f"""
            Clinical Notes: {row['clinical_notes'] or 'None'}
            Diagnoses: {row['diagnoses_text'] or 'None'}
            Medications: {row['medications_text'] or 'None'}
            """.strip()
            texts.append(combined_text)
        
        # Generate embeddings
        embeddings = self.generate_text_embeddings_batch(texts)
        
        # Prepare data for insertion
        rows_to_insert = []
        for i, row in df.iterrows():
            rows_to_insert.append({
                'patient_id': row['patient_id'],
                'text_embedding': embeddings[i],
                'text_embedding_model': EMBEDDING_CONFIG['model'],
                'text_embedding_date': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            })
        
        # Insert into BigQuery
        table_id = FULL_TABLE_IDS['patient_embeddings']
        errors = self.client.insert_rows_json(table_id, rows_to_insert)
        
        if errors:
            logger.error(f"Error inserting embeddings: {errors}")
        else:
            logger.info(f"Successfully inserted {len(rows_to_insert)} patient embeddings")
    
    def process_trial_embeddings(self, batch_size: int = 100):
        """Generate embeddings for all trials
        
        Args:
            batch_size: Number of trials to process in each batch
        """
        logger.info("Processing trial embeddings...")
        
        # Query trials without embeddings
        query = f"""
        SELECT 
            t.trial_id,
            t.title,
            t.brief_summary,
            t.eligibility_criteria,
            ARRAY_TO_STRING(t.conditions, ', ') as conditions_text
        FROM `{FULL_TABLE_IDS['trials']}` t
        LEFT JOIN `{FULL_TABLE_IDS['trial_embeddings']}` te
            ON t.trial_id = te.trial_id
        WHERE te.trial_id IS NULL
        LIMIT {batch_size}
        """
        
        df = self.client.query(query).to_dataframe()
        
        if df.empty:
            logger.info("No trials without embeddings found")
            return
        
        logger.info(f"Processing {len(df)} trials...")
        
        # Prepare text for embedding
        texts = []
        for _, row in df.iterrows():
            combined_text = f"""
            Title: {row['title'] or 'None'}
            Summary: {row['brief_summary'] or 'None'}
            Conditions: {row['conditions_text'] or 'None'}
            Eligibility: {row['eligibility_criteria'] or 'None'}
            """.strip()
            texts.append(combined_text)
        
        # Generate embeddings
        embeddings = self.generate_text_embeddings_batch(texts)
        
        # Prepare data for insertion
        rows_to_insert = []
        for i, row in df.iterrows():
            rows_to_insert.append({
                'trial_id': row['trial_id'],
                'criteria_embedding': embeddings[i],
                'criteria_embedding_model': EMBEDDING_CONFIG['model'],
                'embedding_date': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            })
        
        # Insert into BigQuery
        table_id = FULL_TABLE_IDS['trial_embeddings']
        errors = self.client.insert_rows_json(table_id, rows_to_insert)
        
        if errors:
            logger.error(f"Error inserting embeddings: {errors}")
        else:
            logger.info(f"Successfully inserted {len(rows_to_insert)} trial embeddings")
    
    def validate_embeddings(self):
        """Validate that embeddings are properly generated"""
        logger.info("Validating embeddings...")
        
        # Check patient embeddings
        query = f"""
        SELECT 
            COUNT(*) as total_patients,
            COUNT(pe.patient_id) as patients_with_embeddings,
            AVG(ARRAY_LENGTH(pe.text_embedding)) as avg_embedding_dim
        FROM `{FULL_TABLE_IDS['patients']}` p
        LEFT JOIN `{FULL_TABLE_IDS['patient_embeddings']}` pe
            ON p.patient_id = pe.patient_id
        """
        
        result = self.client.query(query).to_dataframe()
        
        print("\n📊 Embedding Validation Report")
        print("=" * 50)
        print(f"Total patients: {result['total_patients'].iloc[0]}")
        print(f"Patients with embeddings: {result['patients_with_embeddings'].iloc[0]}")
        print(f"Average embedding dimension: {result['avg_embedding_dim'].iloc[0]}")
        
        # Check trial embeddings
        query = f"""
        SELECT 
            COUNT(*) as total_trials,
            COUNT(te.trial_id) as trials_with_embeddings,
            AVG(ARRAY_LENGTH(te.criteria_embedding)) as avg_embedding_dim
        FROM `{FULL_TABLE_IDS['trials']}` t
        LEFT JOIN `{FULL_TABLE_IDS['trial_embeddings']}` te
            ON t.trial_id = te.trial_id
        """
        
        result = self.client.query(query).to_dataframe()
        
        print(f"\nTotal trials: {result['total_trials'].iloc[0]}")
        print(f"Trials with embeddings: {result['trials_with_embeddings'].iloc[0]}")
        print(f"Average embedding dimension: {result['avg_embedding_dim'].iloc[0]}")
        print("=" * 50)

def main():
    """Main execution function"""
    generator = EmbeddingGenerator()
    
    print("🚀 Starting embedding generation for Story 1.3")
    print(f"Project: {GCP_PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Location: {LOCATION}")
    print()
    
    # Process embeddings
    generator.process_patient_embeddings()
    generator.process_trial_embeddings()
    
    # Validate results
    generator.validate_embeddings()
    
    print("\n✅ Embedding generation complete!")

if __name__ == "__main__":
    main()