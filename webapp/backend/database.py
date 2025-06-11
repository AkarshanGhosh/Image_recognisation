# database.py
import os
from motor.motor_asyncio import AsyncIOMotorClient
from celery import Celery
import redis
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB client setup using Atlas connection string
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://ronighosh494:<roniempire>@userdetail.flx0d.mongodb.net/?retryWrites=true&w=majority&appName=UserDetail")
mongodb_client = AsyncIOMotorClient(MONGODB_URL)
db = mongodb_client.UserDetail  # Use the actual database name on Atlas

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
