# schemas.py
from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict, Union
from datetime import datetime
import base64

class ImageUpload(BaseModel):
    """Schema for image upload"""
    image: str = Field(..., description="Base64 encoded image")
    filename: Optional[str] = Field(None, description="Original filename")

class PredictionRequest(BaseModel):
    """Schema for prediction request"""
    image: str = Field(..., description="Base64 encoded image")
    model_type: str = Field(..., description="Type of model to use (animals, gender, emotion)")
    confidence_threshold: Optional[float] = Field(0.5, description="Minimum confidence threshold")

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    prediction: Any  # Can be float, int, string, etc.
    confidence: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    model_used: Optional[str] = None
    processing_time: Optional[float] = None

class PredictionResult(BaseModel):
    """Result from model prediction - alias for PredictionResponse"""
    prediction: Any
    confidence: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    model_used: Optional[str] = None
    processing_time: Optional[float] = None
    raw_output: Optional[Dict[str, Any]] = None

class ModelInfo(BaseModel):
    """Information about available models"""
    name: str
    type: str
    path: str
    loaded: bool = False
    last_used: Optional[datetime] = None
    accuracy: Optional[float] = None
    description: Optional[str] = None

class TrainingRequest(BaseModel):
    """Schema for model training request"""
    model_type: str = Field(..., description="Type of model to train")
    dataset_path: Optional[str] = None
    epochs: int = Field(10, description="Number of training epochs")
    batch_size: int = Field(32, description="Training batch size")
    learning_rate: float = Field(0.001, description="Learning rate")

class TrainingStatus(BaseModel):
    """Schema for training status"""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class ModelMetrics(BaseModel):
    """Schema for model performance metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    loss: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None

class DatasetInfo(BaseModel):
    """Information about datasets"""
    name: str
    path: str
    total_samples: int
    classes: List[str]
    split_info: Optional[Dict[str, int]] = None  # train, val, test counts