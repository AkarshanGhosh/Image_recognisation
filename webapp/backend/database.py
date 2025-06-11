# database.py
import os
from motor.motor_asyncio import AsyncIOMotorClient
from celery import Celery
import redis
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB client setup
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
mongodb_client = AsyncIOMotorClient(MONGODB_URL)
db = mongodb_client.ai_platform

# Redis client for caching and Celery broker
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)

# Celery configuration
celery_app = Celery(
    'ai_training',
    broker=f'redis://{redis_host}:{redis_port}/0',
    backend=f'redis://{redis_host}:{redis_port}/0'
)

celery_app.conf.task_routes = {
    'train_model_task': {'queue': 'training'}
}
