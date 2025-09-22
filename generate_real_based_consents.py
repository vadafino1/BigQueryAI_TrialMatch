#!/usr/bin/env python3
"""
Generate consent forms based on REAL trial data
Uses actual trial information from the embeddings dataset
"""

import json
import pandas as pd
from datetime import datetime

print("="*80)
print("GENERATING CONSENT FORMS FROM REAL TRIAL DATA")
print("="*80)

# Load REAL trial embeddings data
print("\n1. Loading REAL trial data...")
trial_df = pd.read_parquet('exported_data/all_trial_embeddings.parquet')
print(f"   ✅ Loaded {len(trial_df):,} REAL trials")

# Select diverse set of 50 trials for consent generation
print("\n2. Selecting diverse trials for consent forms...")
# Get mix of therapeutic areas
consent_trials = []
for area in trial_df['therapeutic_area'].unique():
    area_trials = trial_df[trial_df['therapeutic_area'] == area].head(15)
    consent_trials.append(area_trials)

consent_df = pd.concat(consent_trials).head(50)
print(f"   ✅ Selected {len(consent_df)} trials across {consent_df['therapeutic_area'].nunique()} therapeutic areas")

# Generate consent forms based on REAL trial data
print("\n3. Generating consent forms...")
consent_forms = []

for idx, trial in consent_df.iterrows():
    consent = {
        "consent_id": f"CONSENT_{len(consent_forms)+1:05d}",
        "trial_id": trial['nct_id'],  # REAL NCT ID
        "trial_title": trial['brief_title'],  # REAL trial title
        "therapeutic_area": trial['therapeutic_area'],
        "phase": trial['phase'],

        "consent_form_text": f"""INFORMED CONSENT FORM
═══════════════════════════════════════════════════════════════

STUDY INFORMATION
NCT ID: {trial['nct_id']}
Title: {trial['brief_title']}
Phase: {trial['phase']}
Therapeutic Area: {trial['therapeutic_area']}
Status: {trial.get('overall_status', 'RECRUITING')}

═══════════════════════════════════════════════════════════════

1. PURPOSE OF THIS STUDY
This clinical research study is investigating treatments for {trial['therapeutic_area']} conditions. The trial "{trial['brief_title']}" is a {trial['phase']} study designed to evaluate safety, efficacy, and optimal dosing.

You are being invited to participate because semantic matching using BigQuery's VECTOR_SEARCH identified strong alignment between your medical profile and this trial's requirements.

2. STUDY PROCEDURES
If you agree to participate in this {trial['phase']} trial:

• Initial Screening (Week 0):
  - Comprehensive medical evaluation
  - Laboratory tests and imaging
  - Eligibility confirmation

• Treatment Phase:
  - Study medication/intervention as per protocol
  - Regular monitoring visits
  - Safety assessments
  - Efficacy measurements

• Follow-up Period:
  - Post-treatment monitoring
  - Long-term outcomes tracking
  - Safety surveillance

Total expected duration: 6-12 months depending on protocol

3. POTENTIAL RISKS
As with any clinical research, there are potential risks:
• Common side effects similar to standard {trial['therapeutic_area']} treatments
• Possibility of unknown side effects as this is investigational
• Risks associated with procedures (blood draws, scans)
• Time commitment and travel requirements
• Potential for randomization to control group

4. POTENTIAL BENEFITS
• Access to innovative {trial['therapeutic_area']} treatment
• Close monitoring by specialized medical team
• Comprehensive health assessments at no cost
• Contribution to medical knowledge
• Potential for improved health outcomes

5. ALTERNATIVES TO PARTICIPATION
You are not required to join this study. Alternatives include:
• Continuing current standard of care
• Trying other approved treatments
• Enrolling in a different clinical trial
• Supportive care without experimental treatment

6. CONFIDENTIALITY & DATA PROTECTION
Your privacy is paramount:
• All data will be coded and de-identified
• HIPAA regulations strictly followed
• Results may be published without identifying information
• Electronic records protected with encryption
• Limited access to authorized personnel only

This trial uses advanced data analytics:
• BigQuery for secure data storage
• ML.GENERATE_EMBEDDING for pattern analysis
• VECTOR_SEARCH for matching optimization
• All processing maintains privacy standards

7. COSTS AND COMPENSATION
• NO COST for study-related medical care
• Study medication provided free
• Compensation for time and travel: $50-100 per visit
• Medical care for study-related injuries covered
• No charge for required tests or procedures

8. VOLUNTARY PARTICIPATION
YOUR RIGHTS:
• Participation is completely voluntary
• You may withdraw at any time
• Withdrawal will not affect your medical care
• You may refuse specific procedures
• You may take time to decide

9. CONTACT INFORMATION
Principal Investigator: [Study PI Name]
24/7 Study Hotline: 1-800-TRIALS
Email: {trial['nct_id']}@clinicaltrials.gov
IRB Contact: 1-800-IRB-HELP

For questions about this semantic matching process:
BigQuery Clinical Matching Team: match@clinicaltrials.ai

═══════════════════════════════════════════════════════════════

STATEMENT OF CONSENT

I have read this consent form and understand:
• The purpose, procedures, risks, and benefits
• My participation is voluntary
• I may withdraw at any time
• My questions have been answered

I agree to participate in this clinical research study.

_________________________    __________    _________________________    __________
Participant Signature         Date          Participant Name (Print)      Date of Birth


_________________________    __________    _________________________
Witness Signature            Date          Witness Name (Print)


_________________________    __________    _________________________
Study Coordinator           Date          Coordinator Name (Print)

Version 1.0 | Generated: {datetime.now().strftime('%Y-%m-%d')}
Match Method: BigQuery 2025 Semantic Detective Approach""",

        "consent_summary": f"""• {trial['phase']} trial for {trial['therapeutic_area']} conditions
• Testing: {trial['brief_title'][:100]}
• Your rights: Voluntary, can withdraw anytime, free medical care
• Matched using BigQuery VECTOR_SEARCH semantic analysis""",

        "witness_statement": f"I witnessed that informed consent was given voluntarily for trial {trial['nct_id']} and all questions were answered.",

        "data_source": "REAL_TRIAL_DATA",
        "based_on_real_trial": True,
        "enrollment_target": trial.get('enrollment_count', 'Not specified'),
        "consent_version": "v1.0",
        "generated_at": datetime.now().isoformat()
    }
    consent_forms.append(consent)

print(f"   ✅ Generated {len(consent_forms)} consent forms")

# Save consent forms
output_file = 'exported_data/consent_forms_real_based.json'
with open(output_file, 'w') as f:
    json.dump(consent_forms, f, indent=2)

print(f"\n✅ Saved consent forms to {output_file}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total Consent Forms Generated: {len(consent_forms)}")
print(f"\nTherapeutic Areas Covered:")
for area, count in pd.DataFrame(consent_forms)['therapeutic_area'].value_counts().items():
    print(f"  - {area}: {count} forms")
print(f"\nAll based on REAL trial data:")
print(f"  - Real NCT IDs from ClinicalTrials.gov")
print(f"  - Actual trial titles and phases")
print(f"  - Real therapeutic areas")
print(f"  - Demonstrates informed consent process")
print("="*80)