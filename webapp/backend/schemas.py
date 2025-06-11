# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class User(BaseModel):
    username: str
    email: str
    subscription: str = "free"

class ProjectCreate(BaseModel):
    projectName: str
    description: str
    modelType: str
    classes: List[str]
    config: Dict[str, Any]

class TrainingDataCreate(BaseModel):
    className: str
    metadata: Dict[str, Any]

class PredictionRequest(BaseModel):
    projectId: Optional[str] = None
    imageData: str  # base64 encoded
    inputType: str = "upload"

class PredictionResponse(BaseModel):
    results: List[Dict[str, Any]]
    inferenceTime: float
    timestamp: datetime
