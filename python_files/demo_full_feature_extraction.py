#!/usr/bin/env python3
"""
Comprehensive Eligibility Feature Extraction Pipeline
BigQuery 2025 Competition - Extract structured features from 238+ clinical trials
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from google.cloud import bigquery
from google.auth import default
import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EligibilityFeatureExtractor:
    """Extract structured features from clinical trial eligibility criteria"""
    
    def __init__(self):
        """Initialize BigQuery client and extraction patterns"""
        credentials, _ = default()
        self.client = bigquery.Client(
            project='gen-lang-client-0017660547',
            credentials=credentials
        )
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, Any]:
        """Initialize regex patterns for feature extraction"""
        return {
            'numeric': {
                'age': r'age[d\s]*(?:of\s+)?(\d+)\s*(?:to|-)\s*(\d+)',
                'hemoglobin': r'hemoglobin[^0-9]*([>≥<≤]=?)\s*(\d+\.?\d*)\s*(g/dL|g/dl|gm/dl)?',
                'platelets': r'platelet[^0-9]*([>≥<≤]=?)\s*(\d+,?\d*)\s*(x10|/μL|/mm3)?',
                'creatinine': r'creatinine[^0-9]*([>≥<≤]=?)\s*(\d+\.?\d*)\s*(mg/dL|mg/dl|μmol/L)?',
                'bilirubin': r'bilirubin[^0-9]*([>≥<≤]=?)\s*(\d+\.?\d*)\s*(mg/dL|mg/dl|μmol/L)?',
                'neutrophils': r'neutrophil[^0-9]*([>≥<≤]=?)\s*(\d+,?\d*)\s*(x10|/μL|/mm3)?',
                'wbc': r'white blood cell[^0-9]*([>≥<≤]=?)\s*(\d+,?\d*)\s*(x10|/μL|/mm3)?',
                'ecog': r'ECOG[^0-9]*([0-4])',
                'karnofsky': r'Karnofsky[^0-9]*([>≥]=?)\s*(\d+)',
                'bmi': r'BMI[^0-9]*(\d+\.?\d*)\s*(?:to|-)\s*(\d+\.?\d*)',
                'life_expectancy': r'life expectancy[^0-9]*([>≥]=?)\s*(\d+)\s*(weeks?|months?|years?)',
                'prior_therapies': r'(\d+)\s*(?:or fewer|or less)?\s*prior.*?(?:therap|treatment|regimen)',
            },
            'boolean': {
                'pregnancy_excluded': r'(?i)(pregnant|pregnancy|nursing|lactating)',
                'treatment_naive': r'(?i)(treatment.naive|no prior|untreated)',
                'prior_therapy_required': r'(?i)(prior therapy|previous treatment|failed)',
                'measurable_disease': r'(?i)measurable disease',
                'metastatic': r'(?i)(metastatic|advanced|stage IV)',
                'brain_metastases': r'(?i)brain metastas',
                'adequate_organ_function': r'(?i)adequate (organ|hepatic|renal|bone marrow)',
                'informed_consent': r'(?i)informed consent',
                'contraception': r'(?i)contraception',
                'hiv_hepatitis': r'(?i)(HIV|hepatitis|HBV|HCV)',
                'biopsy_required': r'(?i)(biopsy|tissue|specimen) required',
                'surgery_eligible': r'(?i)(resectable|surgical candidate|operable)',
                'unresectable': r'(?i)unresectable',
            },
            'categorical': {
                'stage': {
                    'Stage I': r'(?i)stage (I|1)([^IVX]|$)',
                    'Stage II': r'(?i)stage (II|2)([^IVX]|$)',
                    'Stage III': r'(?i)stage (III|3)([^VX]|$)',
                    'Stage IV': r'(?i)stage (IV|4)',
                    'Advanced': r'(?i)(advanced|metastatic)',
                    'Early': r'(?i)early.stage',
                },
                'treatment_line': {
                    'First Line': r'(?i)(first.line|1st.line|treatment.naive)',
                    'Second Line': r'(?i)(second.line|2nd.line)',
                    'Third Line': r'(?i)(third.line|3rd.line)',
                    'Refractory': r'(?i)(refractory|resistant|failed)',
                },
                'biomarker': {
                    'HER2+': r'(?i)(HER2.positive|HER2\+)',
                    'ER+': r'(?i)(ER.positive|ER\+)',
                    'PR+': r'(?i)(PR.positive|PR\+)',
                    'PD-L1+': r'(?i)(PD.L1.positive|PD-L1\+)',
                    'EGFR': r'(?i)EGFR.mutation',
                    'ALK': r'(?i)ALK.(positive|rearrangement)',
                    'BRAF': r'(?i)BRAF.(mutation|V600)',
                    'MSI-H': r'(?i)(MSI.high|microsatellite.instability)',
                },
            }
        }
    
    def extract_numeric_features(self, text: str) -> Dict[str, Any]:
        """Extract numeric values and ranges from eligibility text"""
        features = {}
        
        for feature_name, pattern in self.patterns['numeric'].items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if feature_name == 'age':
                    if len(matches[0]) >= 2:
                        features[f'{feature_name}_min'] = float(matches[0][0])
                        features[f'{feature_name}_max'] = float(matches[0][1])
                elif feature_name in ['hemoglobin', 'platelets', 'creatinine', 'bilirubin', 'neutrophils', 'wbc']:
                    # Extract operator and value
                    if len(matches[0]) >= 2:
                        operator = matches[0][0]
                        value = float(matches[0][1].replace(',', ''))
                        if '>' in operator or '≥' in operator:
                            features[f'{feature_name}_min'] = value
                        else:
                            features[f'{feature_name}_max'] = value
                elif feature_name == 'ecog':
                    features[f'{feature_name}_max'] = int(matches[0])
                elif feature_name == 'karnofsky':
                    if len(matches[0]) >= 2:
                        features[f'{feature_name}_min'] = int(matches[0][1])
                elif feature_name == 'life_expectancy':
                    if len(matches[0]) >= 2:
                        features[f'{feature_name}_months'] = self._convert_to_months(
                            float(matches[0][1]), matches[0][2]
                        )
                elif feature_name == 'prior_therapies':
                    features[f'{feature_name}_max'] = int(matches[0])
                else:
                    # Generic numeric extraction
                    try:
                        features[feature_name] = float(matches[0]) if isinstance(matches[0], str) else float(matches[0][0])
                    except:
                        pass
        
        return features
    
    def extract_boolean_features(self, text: str) -> Dict[str, bool]:
        """Extract boolean features from eligibility text"""
        features = {}
        
        for feature_name, pattern in self.patterns['boolean'].items():
            features[feature_name] = bool(re.search(pattern, text))
        
        return features
    
    def extract_categorical_features(self, text: str) -> Dict[str, str]:
        """Extract categorical features from eligibility text"""
        features = {}
        
        for category, patterns in self.patterns['categorical'].items():
            for value, pattern in patterns.items():
                if re.search(pattern, text):
                    features[category] = value
                    break
            if category not in features:
                features[category] = 'Not Specified'
        
        return features
    
    def _convert_to_months(self, value: float, unit: str) -> float:
        """Convert time duration to months"""
        unit = unit.lower()
        if 'week' in unit:
            return value / 4.33
        elif 'year' in unit:
            return value * 12
        else:
            return value
    
    def extract_all_features(self, trial_data: Dict) -> Dict[str, Any]:
        """Extract all features from a clinical trial"""
        
        # Combine all text fields
        text_fields = [
            trial_data.get('eligibility_criteria_full_text', ''),
            trial_data.get('inclusion_criteria_raw', ''),
            trial_data.get('exclusion_criteria_raw', ''),
            trial_data.get('inclusion_performance_status', ''),
        ]
        combined_text = ' '.join([str(t) for t in text_fields if t])
        
        # Extract features
        numeric_features = self.extract_numeric_features(combined_text)
        boolean_features = self.extract_boolean_features(combined_text)
        categorical_features = self.extract_categorical_features(combined_text)
        
        # Parse existing structured data
        structured_features = self._parse_structured_data(trial_data)
        
        # Combine all features
        all_features = {
            'trial_id': trial_data.get('trial_id'),
            'extraction_timestamp': datetime.now().isoformat(),
            **numeric_features,
            **boolean_features,
            **categorical_features,
            **structured_features,
            'total_features_extracted': len(numeric_features) + len(boolean_features) + len(categorical_features)
        }
        
        return all_features
    
    def _parse_structured_data(self, trial_data: Dict) -> Dict:
        """Parse already structured data from existing extraction"""
        features = {}
        
        # Parse biomarkers
        if trial_data.get('inclusion_biomarkers'):
            try:
                biomarkers = json.loads(trial_data['inclusion_biomarkers'])
                features['biomarker_count'] = len(biomarkers)
                features['biomarker_names'] = [b.get('name', '') for b in biomarkers]
            except:
                pass
        
        # Parse lab values
        if trial_data.get('inclusion_lab_values'):
            try:
                labs = json.loads(trial_data['inclusion_lab_values'])
                features['lab_criteria_count'] = len(labs)
            except:
                pass
        
        # Use existing age data
        if trial_data.get('inclusion_age_min'):
            features['age_min_structured'] = trial_data['inclusion_age_min']
        if trial_data.get('inclusion_age_max'):
            features['age_max_structured'] = trial_data['inclusion_age_max']
        
        return features
    
    def process_trials_batch(self, limit: int = 100) -> pd.DataFrame:
        """Process a batch of trials and extract features"""
        
        logger.info(f"Loading {limit} trials for feature extraction...")
        
        # Query trials
        query = f"""
        SELECT *
        FROM `gen-lang-client-0017660547.clinical_trial_matching.clinical_trials_eligibility`
        WHERE extraction_confidence >= 0.8
        ORDER BY extraction_confidence DESC
        LIMIT {limit}
        """
        
        trials_df = self.client.query(query).to_dataframe()
        logger.info(f"Loaded {len(trials_df)} trials")
        
        # Extract features for each trial
        all_features = []
        for idx, trial in trials_df.iterrows():
            features = self.extract_all_features(trial.to_dict())
            all_features.append(features)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(trials_df)} trials")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Calculate statistics
        self._print_extraction_statistics(features_df)
        
        return features_df
    
    def _print_extraction_statistics(self, df: pd.DataFrame):
        """Print statistics about extracted features"""
        
        print("\n" + "=" * 70)
        print("📊 FEATURE EXTRACTION STATISTICS")
        print("=" * 70)
        
        # Numeric features
        numeric_cols = [col for col in df.columns if any(x in col for x in ['_min', '_max', '_count', 'ecog', 'karnofsky'])]
        print(f"\n📈 Numeric Features: {len(numeric_cols)}")
        for col in numeric_cols[:5]:
            non_null = df[col].notna().sum()
            if non_null > 0:
                print(f"  • {col}: {non_null}/{len(df)} trials ({non_null/len(df)*100:.1f}%)")
        
        # Boolean features
        boolean_cols = [col for col in df.columns if df[col].dtype == bool]
        print(f"\n✅ Boolean Features: {len(boolean_cols)}")
        for col in boolean_cols[:5]:
            true_count = df[col].sum()
            print(f"  • {col}: {true_count}/{len(df)} trials ({true_count/len(df)*100:.1f}%)")
        
        # Categorical features
        categorical_cols = ['stage', 'treatment_line', 'biomarker']
        print(f"\n📋 Categorical Features:")
        for col in categorical_cols:
            if col in df.columns:
                value_counts = df[col].value_counts()
                print(f"  • {col}:")
                for val, count in value_counts.head(3).items():
                    print(f"    - {val}: {count} ({count/len(df)*100:.1f}%)")
        
        # Overall statistics
        print(f"\n📊 Overall Statistics:")
        print(f"  • Total Trials Processed: {len(df)}")
        print(f"  • Average Features per Trial: {df['total_features_extracted'].mean():.1f}")
        print(f"  • Max Features in a Trial: {df['total_features_extracted'].max()}")
        print(f"  • Trials with Age Criteria: {df['age_min_structured'].notna().sum()}")
        print(f"  • Trials with Biomarkers: {df['biomarker_count'].notna().sum()}")
    
    def save_to_bigquery(self, features_df: pd.DataFrame, table_name: str):
        """Save extracted features to BigQuery"""
        
        table_id = f"gen-lang-client-0017660547.clinical_trial_matching.{table_name}"
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
        )
        
        try:
            job = self.client.load_table_from_dataframe(
                features_df, table_id, job_config=job_config
            )
            job.result()
            logger.info(f"✅ Saved {len(features_df)} rows to {table_id}")
        except Exception as e:
            logger.error(f"❌ Error saving to BigQuery: {e}")

def main():
    """Main execution function"""
    
    print("=" * 70)
    print("🚀 ELIGIBILITY FEATURE EXTRACTION PIPELINE")
    print("BigQuery 2025 Competition")
    print("=" * 70)
    
    extractor = EligibilityFeatureExtractor()
    
    # Process trials
    print("\n📥 Extracting features from clinical trials...")
    features_df = extractor.process_trials_batch(limit=50)
    
    # Show sample results
    print("\n📋 Sample Extracted Features:")
    print("-" * 70)
    
    sample_trial = features_df.iloc[0]
    print(f"Trial: {sample_trial['trial_id']}")
    
    # Show numeric features
    print("\nNumeric Features:")
    for col in features_df.columns:
        if any(x in col for x in ['_min', '_max', 'ecog', 'karnofsky']) and pd.notna(sample_trial[col]):
            print(f"  • {col}: {sample_trial[col]}")
    
    # Show boolean features
    print("\nBoolean Features:")
    for col in features_df.columns:
        if features_df[col].dtype == bool and sample_trial[col]:
            print(f"  • {col}: ✅")
    
    # Show categorical features
    print("\nCategorical Features:")
    for col in ['stage', 'treatment_line', 'biomarker']:
        if col in features_df.columns:
            print(f"  • {col}: {sample_trial[col]}")
    
    print("\n" + "=" * 70)
    print("✅ Feature extraction complete!")
    print(f"📊 Extracted {features_df['total_features_extracted'].sum()} total features")
    print(f"📈 Average {features_df['total_features_extracted'].mean():.1f} features per trial")

if __name__ == "__main__":
    main()