"""Consolidated Clinical Trial Matching API

Unified API combining all functionality from:
- main_enhanced.py
- main_hospital.py
- main_production.py
- hospital_production_api.py
- clinical_trial_matching_api.py
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import logging
import os

from api.routers import health, patients, trials, analytics
from api.adapters.real_data_only_adapter import RealDataOnlyAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="🏥 Clinical Trial Matching API",
    description="""
    **Enterprise Clinical Trial Patient Matching Platform**
    
    Production-ready API showcasing advanced multimodal AI capabilities for matching patients 
    to clinical trials using real MIMIC-IV data and Google Cloud AI.
    
    🚀 **Production System** - Enterprise-grade clinical trial matching
    
    **Key Features:**
    * 🔍 **Patient-Trial Matching** - AI-powered similarity matching with confidence scores
    * 🏥 **Real Medical Data** - Access to 364K+ patients from MIMIC-IV database
    * 🧠 **Multimodal Embeddings** - Text, structured, and unified vector representations  
    * 📊 **Analytics Dashboard** - Real-time performance and cost monitoring
    * 🔬 **Clinical Trial Search** - Search and analyze 1,250+ active trials
    * ⚡ **Fast Performance** - Sub-second response times with BigQuery optimization
    
    **Production Workflow:**
    1. 💊 Get patient details: `GET /api/v1/patients/{patient_id}`
    2. 🎯 Find trial matches: `POST /api/v1/patients/match` 
    3. 🔍 Search trials: `GET /api/v1/trials/search`
    4. 📈 View analytics: `GET /api/v1/analytics/dashboard`
    
    **Sample Patient IDs to try:** `10000032`, `10000045`, `10000123`
    """,
    version="1.0.0",
    contact={
        "name": "Clinical Trial Matching Team",
        "url": "https://github.com/anthropics/claude-code",
        "email": "api@clinicaltrialmatch.ai"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Health", 
            "description": "System health and status monitoring"
        },
        {
            "name": "Patients", 
            "description": "Patient data retrieval, matching, and embeddings generation"
        },
        {
            "name": "Clinical Trials", 
            "description": "Trial search, details, and patient eligibility analysis"
        },
        {
            "name": "Analytics", 
            "description": "Performance metrics, costs, and quality monitoring"
        }
    ]
)

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions"""
    logger.error(f"Global exception on {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": request.url.path,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(patients.router, prefix="/api/v1", tags=["Patients"])
app.include_router(trials.router, prefix="/api/v1", tags=["Clinical Trials"])
app.include_router(analytics.router, prefix="/api/v1", tags=["Analytics"])

# Mount static files for the consolidated web UI
if os.path.exists("web"):
    app.mount("/", StaticFiles(directory="web", html=True), name="static")
    logger.info("Mounted web directory for static files")
else:
    # Fallback to old locations if web directory doesn't exist yet
    if os.path.exists("frontend/physician-trial-dashboard"):
        app.mount("/ui", StaticFiles(directory="frontend/physician-trial-dashboard", html=True), name="static")
    if os.path.exists("frontend/static"):
        app.mount("/studio", StaticFiles(directory="frontend/static", html=True), name="studio")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Clinical Trial Matching API",
        "version": "0.1.0",
        "status": "running",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "user_interfaces": {
            "web_app": "/ui/",
            "model_studio": "/studio/model_studio.html",
            "api_docs": "/docs",
            "api_redoc": "/redoc"
        },
        "endpoints": {
            "health": "/health",
            "patient_matching": "/api/v1/patients/match",
            "patient_details": "/api/v1/patients/{patient_id}",
            "patient_embeddings": "/api/v1/patients/embeddings",
            "trial_details": "/api/v1/trials/{trial_id}",
            "trial_search": "/api/v1/trials/search",
            "eligible_patients": "/api/v1/trials/eligible-patients",
            "dashboard": "/api/v1/analytics/dashboard",
            "performance": "/api/v1/analytics/performance"
        },
        "timestamp": datetime.utcnow()
    }