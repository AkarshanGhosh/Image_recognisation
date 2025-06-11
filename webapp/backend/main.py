# main.py - Enhanced Multi-Model AI Backend
import os
import sys
import logging
import asyncio
import uuid
import zipfile
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional
import base64
import io
from concurrent.futures import ThreadPoolExecutor

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
import motor.motor_asyncio
import pymongo

# Import your existing schemas
from schemas import (
    PredictionRequest, 
    PredictionResponse, 
    ImageUpload, 
    ModelInfo,
    TrainingRequest,
    TrainingStatus
)

# Enhanced schemas for the full application
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class MultiModelPrediction(BaseModel):
    """Response from multiple models running in parallel"""
    image_id: str
    predictions: List[Dict[str, Any]]
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

class CustomClass(BaseModel):
    """Custom class for training"""
    class_name: str
    metadata: Dict[str, Any]  # student_name, dob, etc.
    images: List[Dict[str, str]]  # [{"url": "...", "type": "front"}, ...]

class CustomTrainingRequest(BaseModel):
    """Request to train custom model"""
    model_name: str
    classes: List[CustomClass]
    training_config: Optional[Dict[str, Any]] = {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001
    }

class UserModel(BaseModel):
    """User's custom trained model"""
    model_id: str
    model_name: str
    classes: List[str]
    accuracy: Optional[float]
    created_at: datetime
    download_count: int = 0

class SingleModelPrediction(BaseModel):
    """Request for single model prediction"""
    image: str  # base64 encoded image
    model_name: str  # specific model to use

class ModelStatus(BaseModel):
    """Model status information"""
    name: str
    loaded: bool
    classes: List[str]
    device: str
    file_size_mb: float

# Import your model manager with better error handling
try:
    from model_manager import model_manager
    MODEL_MANAGER_AVAILABLE = True
    print("✅ Model manager loaded successfully")
    available_models = model_manager.get_available_models()
    print(f"📋 Available models: {available_models}")
    
    # Print detailed model information
    for model_name in available_models:
        info = model_manager.get_model_info(model_name)
        print(f"  - {model_name}: {len(info['classes'])} classes on {info['device']}")
        
except ImportError as e:
    print(f"❌ Failed to import model manager: {e}")
    MODEL_MANAGER_AVAILABLE = False
    model_manager = None
except Exception as e:
    print(f"⚠️  Model manager initialization error: {e}")
    MODEL_MANAGER_AVAILABLE = False
    model_manager = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
db = client.ai_app

# Collections
users_collection = db.users
training_jobs_collection = db.training_jobs
models_collection = db.models
predictions_collection = db.predictions

# Create FastAPI app
app = FastAPI(
    title="AI Multi-Model Web Application",
    description="Full-stack AI app with multi-model prediction and custom training",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("🚀 Starting AI Multi-Model Web Application...")

    # Test MongoDB connection
    try:
        await db.command("ping")
        logger.info("✅ MongoDB connection successful")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")

    if MODEL_MANAGER_AVAILABLE and model_manager:
        logger.info("✅ Model manager initialized")
        available_models = model_manager.get_available_models()
        logger.info(f"📋 Available built-in models: {available_models}")
        
        # Print detailed model info
        for model_name in available_models:
            info = model_manager.get_model_info(model_name)
            logger.info(f"  🤖 {model_name}: {len(info['classes'])} classes on {info['device']}")
    else:
        logger.warning("⚠️  Model manager not available - running in limited mode")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down AI Multi-Model Web Application...")
    if client:
        client.close()

@app.get("/")
async def root():
    """Root endpoint"""
    status = {
        "message": "AI Multi-Model Web Application",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Multi-model parallel prediction",
            "Single model prediction",
            "Custom model training",
            "Model download with web app",
            "MongoDB integration"
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    if MODEL_MANAGER_AVAILABLE and model_manager:
        status["available_models"] = model_manager.get_available_models()
        status["model_manager"] = "available"
    else:
        status["model_manager"] = "unavailable"
    
    return status

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_status = "healthy"
    try:
        await db.command("ping")
    except:
        db_status = "unhealthy"

    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_manager": MODEL_MANAGER_AVAILABLE,
        "database": db_status,
        "components": {
            "models": "available" if MODEL_MANAGER_AVAILABLE else "unavailable",
            "database": db_status,
            "api": "healthy"
        }
    }

    if MODEL_MANAGER_AVAILABLE and model_manager:
        status["available_models"] = model_manager.get_available_models()

    return status

# ============ MODEL INFORMATION ENDPOINTS ============

@app.get("/models/info")
async def get_models_info():
    """Get information about available models"""
    if not MODEL_MANAGER_AVAILABLE or not model_manager:
        return {
            "error": "Model manager not available", 
            "models": [],
            "total_models": 0
        }
    
    models_info = []
    for model_name in model_manager.get_available_models():
        info = model_manager.get_model_info(model_name)
        # Add file size information
        try:
            file_size = os.path.getsize(info['path']) / (1024 * 1024)  # MB
            info['file_size_mb'] = round(file_size, 1)
        except:
            info['file_size_mb'] = 0
        models_info.append(info)
    
    return {
        "models": models_info,
        "total_models": len(models_info),
        "status": "success"
    }

@app.get("/models/status")
async def get_models_status():
    """Get detailed status of all models"""
    if not MODEL_MANAGER_AVAILABLE or not model_manager:
        return {"error": "Model manager not available"}
    
    models_status = []
    for model_name in model_manager.get_available_models():
        info = model_manager.get_model_info(model_name)
        try:
            file_size = os.path.getsize(info['path']) / (1024 * 1024)
        except:
            file_size = 0
            
        models_status.append(ModelStatus(
            name=model_name,
            loaded=True,
            classes=info['classes'],
            device=info['device'],
            file_size_mb=round(file_size, 1)
        ))
    
    return {
        "models": models_status,
        "total_loaded": len(models_status)
    }

# ============ PREDICTION ENDPOINTS ============

@app.post("/predict/multi", response_model=MultiModelPrediction)
async def predict_multi_models(request: PredictionRequest):
    """Run prediction on all available models simultaneously"""
    if not MODEL_MANAGER_AVAILABLE or not model_manager:
        raise HTTPException(
            status_code=503, 
            detail="Model manager not available. Please check server logs."
        )

    try:
        start_time = datetime.now()
        image_id = str(uuid.uuid4())
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
        
        image_stream = io.BytesIO(image_data)
        
        # Validate image
        try:
            test_image = Image.open(io.BytesIO(image_data))
            test_image.verify()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        # Reset stream position
        image_stream.seek(0)
        
        # Run predictions on all models
        results = model_manager.predict_all(image_stream)
        
        # Check for errors in results
        if isinstance(results, dict) and "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        processing_time = (datetime.now() - start_time).total_seconds()

        # Save to database
        try:
            await predictions_collection.insert_one({
                "image_id": image_id,
                "predictions": results,
                "processing_time": processing_time,
                "prediction_type": "multi_model",
                "created_at": datetime.now()
            })
        except Exception as e:
            logger.warning(f"Failed to save prediction to database: {e}")

        # Format response
        formatted_predictions = []
        for model_name, result in results.items():
            if isinstance(result, dict) and "error" not in result:
                formatted_predictions.append({
                    "model_type": model_name,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "all_probabilities": result.get("all_probabilities", {})
                })
            else:
                error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
                formatted_predictions.append({
                    "model_type": model_name,
                    "error": error_msg
                })

        return MultiModelPrediction(
            image_id=image_id,
            predictions=formatted_predictions,
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-model prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/single")
async def predict_single_model(request: SingleModelPrediction):
    """Run prediction on a specific model"""
    if not MODEL_MANAGER_AVAILABLE or not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not available")
    
    # Check if model exists
    available_models = model_manager.get_available_models()
    if request.model_name not in available_models:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{request.model_name}' not found. Available models: {available_models}"
        )
    
    try:
        start_time = datetime.now()
        image_id = str(uuid.uuid4())
        
        # Decode and validate image
        image_data = base64.b64decode(request.image)
        image_stream = io.BytesIO(image_data)
        
        # Run prediction on all models (we'll filter for the specific one)
        all_results = model_manager.predict_all(image_stream)
        
        if request.model_name not in all_results:
            raise HTTPException(status_code=500, detail="Model prediction failed")
        
        result = all_results[request.model_name]
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Save to database
        try:
            await predictions_collection.insert_one({
                "image_id": image_id,
                "model_name": request.model_name,
                "prediction": result,
                "processing_time": processing_time,
                "prediction_type": "single_model",
                "created_at": datetime.now()
            })
        except Exception as e:
            logger.warning(f"Failed to save prediction to database: {e}")
        
        return {
            "image_id": image_id,
            "model_name": request.model_name,
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "all_probabilities": result.get("all_probabilities", {}),
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single model prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/upload")
async def predict_from_upload(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None)
):
    """Upload an image file and run prediction"""
    if not MODEL_MANAGER_AVAILABLE or not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not available")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Convert to base64 for processing
        image_b64 = base64.b64encode(file_content).decode('utf-8')
        
        if model_name:
            # Single model prediction
            request = SingleModelPrediction(image=image_b64, model_name=model_name)
            return await predict_single_model(request)
        else:
            # Multi-model prediction
            request = PredictionRequest(image=image_b64)
            return await predict_multi_models(request)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload prediction failed: {str(e)}")

# ============ STATISTICS AND ADMIN ENDPOINTS ============

@app.get("/admin/stats")
async def get_admin_stats():
    """Get admin statistics"""
    try:
        total_users = await users_collection.count_documents({})
        total_models = await models_collection.count_documents({})
        total_predictions = await predictions_collection.count_documents({})
        
        # Get predictions by type
        multi_model_predictions = await predictions_collection.count_documents({
            "prediction_type": "multi_model"
        })
        single_model_predictions = await predictions_collection.count_documents({
            "prediction_type": "single_model"
        })
        
        # Get recent predictions
        recent_predictions = await predictions_collection.find(
            {}, 
            {"_id": 0, "created_at": 1, "processing_time": 1, "prediction_type": 1}
        ).sort("created_at", -1).limit(10).to_list(length=10)

        return {
            "total_users": total_users,
            "total_models": total_models,
            "total_predictions": total_predictions,
            "prediction_breakdown": {
                "multi_model": multi_model_predictions,
                "single_model": single_model_predictions
            },
            "recent_predictions": recent_predictions,
            "available_models": model_manager.get_available_models() if MODEL_MANAGER_AVAILABLE else [],
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Admin stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get("/stats/predictions")
async def get_prediction_stats():
    """Get prediction statistics"""
    try:
        # Total predictions
        total = await predictions_collection.count_documents({})
        
        # Predictions by date (last 7 days)
        from datetime import timedelta
        week_ago = datetime.now() - timedelta(days=7)
        recent = await predictions_collection.count_documents({
            "created_at": {"$gte": week_ago}
        })
        
        # Average processing time
        pipeline = [
            {"$group": {
                "_id": None,
                "avg_processing_time": {"$avg": "$processing_time"},
                "min_processing_time": {"$min": "$processing_time"},
                "max_processing_time": {"$max": "$processing_time"}
            }}
        ]
        
        processing_stats = await predictions_collection.aggregate(pipeline).to_list(length=1)
        
        stats = {
            "total_predictions": total,
            "recent_predictions": recent,
            "processing_stats": processing_stats[0] if processing_stats else None,
            "timestamp": datetime.now().isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Prediction stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ UTILITY ENDPOINTS ============

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {
        "message": "API is working!",
        "timestamp": datetime.now().isoformat(),
        "model_manager_available": MODEL_MANAGER_AVAILABLE,
        "available_models": model_manager.get_available_models() if MODEL_MANAGER_AVAILABLE else []
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting AI Multi-Model Web Application")
    print("=" * 50)
    print(f"📊 Model Manager Available: {MODEL_MANAGER_AVAILABLE}")
    if MODEL_MANAGER_AVAILABLE and model_manager:
        models = model_manager.get_available_models()
        print(f"🤖 Available Models: {models}")
    print("=" * 50)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )