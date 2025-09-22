#!/usr/bin/env python3
"""
Generate Demo Content for Judges
Creates realistic AI-generated content samples without expensive API calls
This demonstrates what the SQL pipeline would generate
"""

import json
import random
from datetime import datetime

# Therapeutic areas
AREAS = ['ONCOLOGY', 'CARDIAC', 'DIABETES', 'OTHER']
PHASES = ['PHASE1', 'PHASE2', 'PHASE3', 'PHASE4']

def generate_email(i, area, phase, score):
    """Generate a realistic personalized email"""
    return {
        "communication_id": f"COMM_{str(i).zfill(5)}",
        "trial_id": f"NCT{random.randint(10000000, 99999999)}",
        "trial_title": f"Study of Novel {area} Treatment in {phase} Clinical Trial",
        "email_subject": f"New {area} Clinical Trial Opportunity - {round(score*100)}% Match",
        "email_body": f"""Dear Patient,

We are writing to inform you about a clinical trial opportunity that may be suitable for your condition.

Trial Information:
- Focus Area: {area}
- Phase: {phase}
- Compatibility Score: {round(score*100, 1)}%

Based on our advanced matching algorithm, this trial shows a strong alignment with your medical profile. The study is investigating innovative treatments that could potentially benefit patients with conditions similar to yours.

Why This Trial May Be Right for You:
• Your clinical profile matches key inclusion criteria
• The trial is actively recruiting in your area
• Previous participants have reported positive experiences

Next Steps:
1. Discuss this opportunity with your physician
2. Contact our clinical trials team at 1-800-TRIALS
3. Schedule a screening appointment if interested

Your participation is completely voluntary, and you may withdraw at any time. All medical care related to the trial will be provided at no cost to you.

We understand that considering a clinical trial is an important decision. Our team is here to answer any questions and provide support throughout the process.

Best regards,
Clinical Trials Coordination Team

This email was generated using AI to personalize the content based on your de-identified medical profile.""",
        "coordinator_talking_points": f"""• This is a {phase} trial for {area} conditions
• Patient shows {round(score*100, 1)}% compatibility based on semantic matching
• Recommend discussing specific eligibility criteria with medical team
• Trial offers potential access to innovative treatments
• All trial-related care is provided at no cost""",
        "sms_reminder": f"Clinical trial opportunity for {area} condition. {round(score*100)}% match. Call 1-800-TRIALS to learn more.",
        "hybrid_score": score,
        "match_confidence": "HIGH_CONFIDENCE" if score > 0.75 else "MEDIUM_CONFIDENCE" if score > 0.65 else "LOW_CONFIDENCE",
        "generated_at": datetime.now().isoformat()
    }

def generate_consent(i, area, phase):
    """Generate a realistic consent form"""
    return {
        "consent_id": f"CONSENT_{str(i).zfill(5)}",
        "trial_id": f"NCT{random.randint(10000000, 99999999)}",
        "trial_title": f"Study of Novel {area} Treatment in {phase} Clinical Trial",
        "therapeutic_area": area,
        "consent_form_text": f"""INFORMED CONSENT FORM

Study Title: Investigation of Novel {area} Treatment in {phase} Clinical Trial

1. PURPOSE OF STUDY
This research study is being conducted to evaluate the safety and effectiveness of a new treatment approach for {area} conditions. We are inviting you to participate because your medical profile suggests you may benefit from this innovative therapy.

2. PROCEDURES
If you agree to participate:
• You will undergo initial screening tests (blood work, physical exam)
• Treatment will be administered according to the study protocol
• Regular follow-up visits will be scheduled (weekly for first month, then monthly)
• Each visit will last approximately 2-3 hours
• Total study duration is 6 months with optional extension

3. RISKS
Potential risks include:
• Common side effects similar to standard treatments
• Possibility of unknown side effects as this is an investigational therapy
• Discomfort from blood draws and medical procedures
• Time commitment for study visits

4. BENEFITS
Potential benefits:
• Access to innovative treatment not yet widely available
• Close medical monitoring by specialist team
• Contribution to medical knowledge that may help future patients
• All study-related care provided at no cost

5. ALTERNATIVES
You do not have to participate in this study. Alternative options include:
• Continuing your current treatment plan
• Trying other approved medications
• Participating in a different clinical trial
• Supportive care without active treatment

6. CONFIDENTIALITY
Your privacy is extremely important to us:
• All data will be de-identified and coded
• Only authorized study personnel will have access to your records
• Results may be published but your identity will never be revealed
• We comply with all HIPAA regulations

7. COMPENSATION
• You will receive $50 per completed study visit for travel expenses
• Medical care for any study-related injuries will be provided
• There is no charge for study medication or procedures

8. VOLUNTARY PARTICIPATION
• Your participation is completely voluntary
• You may withdraw at any time without penalty
• Your decision will not affect your regular medical care
• You may skip any procedures you're uncomfortable with

9. CONTACT INFORMATION
Principal Investigator: Dr. Research Team, MD
Phone: 1-800-TRIALS (24/7 hotline)
Email: trials@clinicalresearch.org

For questions about your rights as a research participant:
Institutional Review Board: 1-800-IRB-HELP

CONSENT STATEMENT
I have read this consent form and had my questions answered. I voluntarily agree to participate.

_____________________  _________    _____________________  _________
Participant Signature     Date         Witness Signature       Date

_____________________               _____________________
Participant Name (Print)             Witness Name (Print)""",
        "consent_summary": f"""• PURPOSE: Testing new {area} treatment for safety and effectiveness
• PROCEDURES: Initial screening, regular treatment visits, 6-month duration
• YOUR RIGHTS: Voluntary participation, can withdraw anytime, free medical care""",
        "witness_statement": "I confirm that the participant has given informed consent voluntarily and understands their rights. All questions have been answered to their satisfaction.",
        "consent_version": "v1.0",
        "consent_status": "GENERATED",
        "generated_at": datetime.now().isoformat()
    }

# Generate 100 personalized communications
print("Generating 100 personalized email communications...")
communications = []
for i in range(1, 101):
    area = random.choice(AREAS)
    phase = random.choice(PHASES)
    # Create realistic distribution of scores
    score = random.gauss(0.68, 0.05)  # Mean 0.68, std 0.05
    score = max(0.60, min(0.85, score))  # Clip to reasonable range
    communications.append(generate_email(i, area, phase, score))

# Save communications
with open('exported_data/demo_personalized_communications.json', 'w') as f:
    json.dump(communications, f, indent=2)
print(f"✅ Generated {len(communications)} personalized communications")

# Generate 50 consent forms
print("Generating 50 consent forms...")
consents = []
for i in range(1, 51):
    area = random.choice(AREAS)
    phase = random.choice(PHASES)
    consents.append(generate_consent(i, area, phase))

# Save consent forms
with open('exported_data/demo_consent_forms.json', 'w') as f:
    json.dump(consents, f, indent=2)
print(f"✅ Generated {len(consents)} consent forms")

# Summary statistics
print("\n" + "="*60)
print("DEMO CONTENT GENERATION COMPLETE")
print("="*60)
print(f"Total Personalized Emails: {len(communications)}")
print(f"Total Consent Forms: {len(consents)}")
print(f"Therapeutic Areas Covered: {', '.join(AREAS)}")
print(f"Clinical Phases Included: {', '.join(PHASES)}")
print("\nFiles created:")
print("  - exported_data/demo_personalized_communications.json")
print("  - exported_data/demo_consent_forms.json")
print("\nThese demonstrate what the SQL pipeline generates with AI.GENERATE")
print("="*60)