#!/usr/bin/env python3
"""
Production REST API for Contract Correspondence Multi-Category Classification System
Built with FastAPI for high performance and automatic documentation
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import asyncio
import uuid
import time
import logging
from datetime import datetime
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.config_manager import ConfigManager
from classifier.issue_mapper import IssueCategoryMapper
from classifier.validation import ValidationEngine
from classifier.data_sufficiency import DataSufficiencyAnalyzer
from classifier.pure_llm import PureLLMClassifier
from classifier.hybrid_rag import HybridRAGClassifier

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
Path('logs').mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Contract Correspondence Classification API",
    description="Multi-category classification system for contract correspondence using LLM and RAG approaches",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for classifiers
config_manager = None
classifiers = {}
components = {}

# Request/Response Models
class ClassificationRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000, description="Text to classify")
    approach: str = Field("hybrid_rag", regex="^(pure_llm|hybrid_rag|ensemble)$", description="Classification approach")
    confidence_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_results: Optional[int] = Field(5, ge=1, le=20, description="Maximum number of results to return")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

class BatchClassificationRequest(BaseModel):
    texts: List[str] = Field(..., max_items=100, description="List of texts to classify")
    approach: str = Field("hybrid_rag", regex="^(pure_llm|hybrid_rag|ensemble)$", description="Classification approach")
    confidence_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_results: Optional[int] = Field(5, ge=1, le=20, description="Maximum number of results to return")

class IssueResult(BaseModel):
    issue_type: str
    confidence: float
    source: Optional[str] = None

class CategoryResult(BaseModel):
    category: str
    confidence: float
    source_issues: Optional[List[str]] = None

class ClassificationResponse(BaseModel):
    id: str
    status: str
    processing_time: float
    approach_used: str
    identified_issues: List[IssueResult]
    categories: List[CategoryResult]
    confidence_score: float
    llm_provider_used: Optional[str] = None
    data_sufficiency_warnings: List[Dict[str, Any]]
    validation_report: Dict[str, Any]
    timestamp: str

class BatchClassificationResponse(BaseModel):
    batch_id: str
    status: str
    total_items: int
    completed_items: int
    failed_items: int
    results: List[ClassificationResponse]
    total_processing_time: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    system_info: Dict[str, Any]
    classifiers_loaded: Dict[str, bool]

# Background task storage
background_tasks = {}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize classifiers and components on startup."""
    global config_manager, classifiers, components
    
    logger.info("🚀 Starting Contract Correspondence Classification API...")
    
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        
        if not config_manager.validate_config():
            logger.error("❌ Configuration validation failed")
            raise RuntimeError("Configuration validation failed")
        
        # Load training data
        training_path = './data/synthetic/combined_training_data.xlsx'
        if not Path(training_path).exists():
            training_path = './data/raw/Consolidated_labeled_data.xlsx'
            
        if not Path(training_path).exists():
            logger.error(f"❌ Training data not found at {training_path}")
            raise FileNotFoundError("Training data not found")
        
        logger.info(f"📊 Loading training data from: {training_path}")
        
        # Initialize core components
        components = {
            'issue_mapper': IssueCategoryMapper(training_path),
            'validator': ValidationEngine(training_path),
            'data_analyzer': DataSufficiencyAnalyzer(training_path)
        }
        
        logger.info(f"✅ Loaded {len(components['issue_mapper'].get_all_issue_types())} issue types")
        logger.info(f"✅ Loaded {len(components['issue_mapper'].get_all_categories())} categories")
        
        # Initialize classifiers based on enabled approaches
        enabled_approaches = config_manager.get_enabled_approaches()
        logger.info(f"🔧 Initializing classifiers: {enabled_approaches}")
        
        if 'pure_llm' in enabled_approaches:
            config = config_manager.get_approach_config('pure_llm')
            if config.get('api_key'):
                classifiers['pure_llm'] = PureLLMClassifier(
                    config=config,
                    issue_mapper=components['issue_mapper'],
                    validator=components['validator'],
                    data_analyzer=components['data_analyzer']
                )
                logger.info("✅ Pure LLM Classifier initialized")
            else:
                logger.warning("⚠️  Pure LLM classifier skipped - no API key")
        
        if 'hybrid_rag' in enabled_approaches:
            config = config_manager.get_approach_config('hybrid_rag')
            classifier = HybridRAGClassifier(
                config=config,
                issue_mapper=components['issue_mapper'],
                validator=components['validator'],
                data_analyzer=components['data_analyzer']
            )
            
            # Build or load index
            index_path = Path('./data/embeddings/rag_index')
            if not index_path.with_suffix('.faiss').exists():
                logger.info("🔨 Building vector index for RAG approach...")
                classifier.build_index(training_path, save_path=str(index_path))
                logger.info("✅ Vector index built and saved")
            else:
                logger.info("✅ Using existing vector index")
            
            classifiers['hybrid_rag'] = classifier
            logger.info("✅ Hybrid RAG Classifier initialized")
        
        if not classifiers:
            logger.error("❌ No classifiers initialized")
            raise RuntimeError("No classifiers available")
        
        logger.info("🎉 API startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Get API health status."""
    uptime = time.time() - _start_time
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=uptime,
        system_info={
            "classifiers_available": list(classifiers.keys()),
            "training_samples": len(components['data_analyzer'].df) if components else 0,
            "issue_types": len(components['issue_mapper'].get_all_issue_types()) if components else 0,
            "categories": len(components['issue_mapper'].get_all_categories()) if components else 0
        },
        classifiers_loaded={approach: approach in classifiers for approach in ['pure_llm', 'hybrid_rag']}
    )

# Classification endpoint
@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """Classify a single text document."""
    start_time = time.time()
    classification_id = str(uuid.uuid4())
    
    logger.info(f"📝 Classification request {classification_id}: {request.approach}")
    
    try:
        # Validate approach is available
        if request.approach not in classifiers:
            available = list(classifiers.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Approach '{request.approach}' not available. Available: {available}"
            )
        
        # Get classifier
        classifier = classifiers[request.approach]
        
        # Perform classification
        result = classifier.classify(request.text)
        
        # Filter results by confidence threshold
        filtered_issues = [
            IssueResult(
                issue_type=issue['issue_type'],
                confidence=issue['confidence'],
                source=issue.get('source', None)
            )
            for issue in result.get('identified_issues', [])
            if issue['confidence'] >= request.confidence_threshold
        ][:request.max_results]
        
        filtered_categories = [
            CategoryResult(
                category=cat['category'],
                confidence=cat['confidence'],
                source_issues=[
                    issue['issue_type'] if isinstance(issue, dict) else str(issue)
                    for issue in cat.get('source_issues', [])
                ]
            )
            for cat in result.get('categories', [])
            if cat['confidence'] >= request.confidence_threshold
        ][:request.max_results]
        
        # Calculate overall confidence
        overall_confidence = max(
            [issue.confidence for issue in filtered_issues] + 
            [cat.confidence for cat in filtered_categories] + 
            [0.0]
        )
        
        processing_time = time.time() - start_time
        
        response = ClassificationResponse(
            id=classification_id,
            status="success",
            processing_time=processing_time,
            approach_used=request.approach,
            identified_issues=filtered_issues,
            categories=filtered_categories,
            confidence_score=overall_confidence,
            llm_provider_used=result.get('llm_provider_used'),
            data_sufficiency_warnings=result.get('data_sufficiency_warnings', []),
            validation_report=result.get('validation_report', {}),
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Classification {classification_id} completed in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Classification {classification_id} failed: {e}")
        
        return ClassificationResponse(
            id=classification_id,
            status="error",
            processing_time=processing_time,
            approach_used=request.approach,
            identified_issues=[],
            categories=[],
            confidence_score=0.0,
            data_sufficiency_warnings=[{"message": f"Classification failed: {str(e)}"}],
            validation_report={"error": str(e)},
            timestamp=datetime.now().isoformat()
        )

# Batch classification endpoint
@app.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch(request: BatchClassificationRequest, background_tasks: BackgroundTasks):
    """Classify multiple texts in batch."""
    batch_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"📦 Batch classification {batch_id}: {len(request.texts)} items")
    
    try:
        # Validate approach is available
        if request.approach not in classifiers:
            available = list(classifiers.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Approach '{request.approach}' not available. Available: {available}"
            )
        
        results = []
        failed_items = 0
        
        # Process each text
        for i, text in enumerate(request.texts):
            try:
                # Create individual classification request
                individual_request = ClassificationRequest(
                    text=text,
                    approach=request.approach,
                    confidence_threshold=request.confidence_threshold,
                    max_results=request.max_results
                )
                
                # Classify
                result = await classify_text(individual_request)
                results.append(result)
                
            except Exception as e:
                logger.error(f"❌ Batch item {i} failed: {e}")
                failed_items += 1
                
                # Add error result
                results.append(ClassificationResponse(
                    id=str(uuid.uuid4()),
                    status="error",
                    processing_time=0.0,
                    approach_used=request.approach,
                    identified_issues=[],
                    categories=[],
                    confidence_score=0.0,
                    data_sufficiency_warnings=[{"message": f"Item {i} failed: {str(e)}"}],
                    validation_report={"error": str(e)},
                    timestamp=datetime.now().isoformat()
                ))
        
        total_processing_time = time.time() - start_time
        
        response = BatchClassificationResponse(
            batch_id=batch_id,
            status="completed",
            total_items=len(request.texts),
            completed_items=len(results) - failed_items,
            failed_items=failed_items,
            results=results,
            total_processing_time=total_processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Batch {batch_id} completed: {response.completed_items}/{response.total_items} successful")
        return response
        
    except Exception as e:
        logger.error(f"❌ Batch {batch_id} failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# Get available categories
@app.get("/categories")
async def get_categories():
    """Get list of available categories."""
    if not components:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    categories = components['issue_mapper'].get_all_categories()
    return {
        "categories": sorted(categories),
        "total_count": len(categories),
        "standard_categories": [
            "EoT", "Dispute Resolution", "Contractor's Obligations",
            "Payments", "Authority's Obligations", "Change of Scope",
            "Others", "Appointed Date"
        ]
    }

# Get available issue types
@app.get("/issue-types")
async def get_issue_types():
    """Get list of available issue types."""
    if not components:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    issue_types = components['issue_mapper'].get_all_issue_types()
    return {
        "issue_types": sorted(issue_types),
        "total_count": len(issue_types)
    }

# Get system statistics
@app.get("/stats")
async def get_system_stats():
    """Get system statistics."""
    if not components:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    analyzer = components['data_analyzer']
    report = analyzer.generate_sufficiency_report()
    
    return {
        "training_data": {
            "total_samples": report['summary']['total_samples'],
            "unique_issue_types": report['summary']['unique_issue_types'],
            "unique_categories": report['summary']['unique_categories']
        },
        "data_sufficiency": {
            "critical_issues": len(report['critical_issues']),
            "warning_issues": len(report['warning_issues']),
            "good_issues": report['summary']['total_samples'] - len(report['critical_issues']) - len(report['warning_issues'])
        },
        "classifiers": {
            "available": list(classifiers.keys()),
            "total": len(classifiers)
        }
    }

# Initialize start time for uptime tracking at module level
import time
_start_time = time.time()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "production_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )