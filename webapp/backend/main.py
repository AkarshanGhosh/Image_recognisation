from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from typing import List, Optional
from database import db
from schemas import User, ProjectCreate, PredictionRequest, PredictionResponse
from model_manager import model_manager
from train_model_task import train_model_task
from app_generator import CustomAppGenerator
import cloudinary.uploader

app = FastAPI(
    title="AI Training Platform API",
    description="Multi-model AI training and prediction platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Mock user
    return {"user_id": "mock_user_id", "username": "mock_user"}

@app.get("/")
async def root():
    return {"message": "AI Training Platform API", "version": "1.0.0"}

@app.post("/auth/register")
async def register(user: User):
    existing_user = await db.users.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    user_doc = {
        "username": user.username,
        "email": user.email,
        "password": "hashed_password",  # TODO: hash properly
        "subscription": user.subscription,
        "createdAt": datetime.utcnow(),
        "usage": {"modelsCreated": 0, "trainingHours": 0, "predictions": 0}
    }

    result = await db.users.insert_one(user_doc)
    return {"message": "User created successfully", "user_id": str(result.inserted_id)}

@app.post("/predict")
async def predict(request: PredictionRequest, current_user=Depends(get_current_user)):
    start_time = datetime.utcnow()
    results = await model_manager.predict(request.imageData, request.projectId)
    inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000

    prediction_doc = {
        "userId": current_user["user_id"],
        "projectId": request.projectId,
        "inputType": request.inputType,
        "results": results,
        "inferenceTime": inference_time,
        "timestamp": start_time
    }
    await db.predictions.insert_one(prediction_doc)
    await db.users.update_one(
        {"_id": current_user["user_id"]},
        {"$inc": {"usage.predictions": 1}}
    )

    return PredictionResponse(
        results=results,
        inferenceTime=inference_time,
        timestamp=start_time
    )

@app.post("/projects")
async def create_project(project: ProjectCreate, current_user=Depends(get_current_user)):
    project_doc = {
        "userId": current_user["user_id"],
        "projectName": project.projectName,
        "description": project.description,
        "modelType": project.modelType,
        "classes": project.classes,
        "config": project.config,
        "status": "creating",
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow()
    }

    result = await db.projects.insert_one(project_doc)
    return {"message": "Project created", "project_id": str(result.inserted_id)}

@app.get("/projects")
async def get_projects(current_user=Depends(get_current_user)):
    projects = []
    async for project in db.projects.find({"userId": current_user["user_id"]}):
        project["_id"] = str(project["_id"])
        projects.append(project)
    return projects

@app.post("/projects/{project_id}/training-data")
async def upload_training_data(
    project_id: str,
    files: List[UploadFile] = File(...),
    className: str = None,
    metadata: str = None,
    current_user=Depends(get_current_user)
):
    uploaded_images = []

    for file in files:
        result = cloudinary.uploader.upload(
            file.file,
            folder=f"ai_training/{current_user['user_id']}/{project_id}",
            public_id=f"{className}_{len(uploaded_images)}"
        )
        uploaded_images.append({
            "imageId": result["public_id"],
            "originalName": file.filename,
            "imageUrl": result["secure_url"],
            "thumbnailUrl": result["secure_url"].replace("/upload/", "/upload/w_150,h_150,c_thumb/"),
            "size": len(await file.read()),
            "dimensions": {"width": result["width"], "height": result["height"]},
            "uploadedAt": datetime.utcnow(),
            "tags": [],
            "isValidated": False
        })

    training_data_doc = {
        "projectId": project_id,
        "userId": current_user["user_id"],
        "className": className,
        "metadata": eval(metadata) if metadata else {},
        "images": uploaded_images,
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow()
    }

    await db.trainingData.insert_one(training_data_doc)
    return {"message": "Training data uploaded successfully", "imageCount": len(uploaded_images)}

@app.post("/projects/{project_id}/train")
async def start_training(project_id: str, background_tasks: BackgroundTasks, current_user=Depends(get_current_user)):
    await db.projects.update_one(
        {"_id": project_id},
        {"$set": {"status": "training", "updatedAt": datetime.utcnow()}}
    )
    background_tasks.add_task(train_model_task, project_id, current_user["user_id"])
    return {"message": "Training started", "project_id": project_id}

@app.get("/projects/{project_id}/download")
async def download_model_app(project_id: str, current_user=Depends(get_current_user)):
    project = await db.projects.find_one({"_id": project_id, "userId": current_user["user_id"]})
    if not project or project['status'] != 'completed':
        raise HTTPException(status_code=404, detail="Project not found or not completed")

    app_generator = CustomAppGenerator(project)
    zip_file_path = await app_generator.generate()

    return {"download_url": zip_file_path, "expires_at": datetime.utcnow()}
