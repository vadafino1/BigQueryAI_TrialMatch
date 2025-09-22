# ⚠️ IMPORTANT: Data Files Not in GitHub

## The `exported_data/` folder is NOT in this GitHub repository

### Why?
- Total size: 116MB (exceeds GitHub's 100MB file limit)
- Individual files up to 48MB each

### Where to Get the Data?

#### Option 1: Auto-Download (Recommended)
Run the judge notebook which auto-downloads:
```bash
jupyter notebook demo_judge_complete.ipynb
```

#### Option 2: Manual Download
Download from Google Drive:
https://drive.google.com/drive/folders/1YCSzH2GA-GTf_x6JNOI4K4isayfZhUYK

### Files Needed:
- `all_matches.csv` (46MB) - 200,000 matches
- `all_patient_embeddings.parquet` (46MB) - 10,000 embeddings
- `all_trial_embeddings.parquet` (24MB) - 5,000 embeddings
- `performance_metrics.json` - System metrics
- `ai_eligibility_assessments.json` - AI assessments
- Additional JSON files for AI-generated content

### After Download:
1. Create `exported_data/` folder in project root
2. Place all downloaded files there
3. Run any notebook

## ✅ The notebooks will work perfectly once data is downloaded!