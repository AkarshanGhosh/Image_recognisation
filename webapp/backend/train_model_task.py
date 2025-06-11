from database import db
from time import sleep
from datetime import datetime  # ✅ ADD THIS

# Background training task
async def train_model_task(project_id: str, user_id: str):
    """
    Background task to simulate model training.
    This should be replaced with your actual model training logic.
    """
    try:
        print(f"[Training Task] Starting training for project {project_id}")

        # Simulate training (replace with actual training logic)
        sleep(60)  # Simulate time delay for training

        # Update project status and dummy training results
        await db.projects.update_one(
            {"_id": project_id},
            {"$set": {
                "status": "completed",
                "training": {
                    "accuracy": 0.92,
                    "loss": 0.18,
                    "trainedAt": datetime.utcnow(),
                    "trainingTime": 60,
                    "modelPath": f"models/trained_model_{project_id}.pth",
                    "metricsPath": f"metrics/metrics_{project_id}.json"
                },
                "updatedAt": datetime.utcnow()
            }}
        )

        print(f"[Training Task] Training completed for project {project_id}")

    except Exception as e:
        print(f"[Training Task] Failed to train project {project_id}: {e}")
        await db.projects.update_one(
            {"_id": project_id},
            {"$set": {"status": "failed", "updatedAt": datetime.utcnow()}}
        )
