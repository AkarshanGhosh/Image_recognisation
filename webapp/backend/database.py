# database.py
import os
from motor.motor_asyncio import AsyncIOMotorClient
from celery import Celery
import redis
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB client setup using Atlas connection string
# Fixed: Added database name and URL-encoded password
MONGODB_URL = os.getenv("MONGODB_URL", 
    "mongodb+srv://ronighosh494:roniempire@userdetail.flx0d.mongodb.net/UserDetail?retryWrites=true&w=majority&appName=UserDetail"
)

try:
    mongodb_client = AsyncIOMotorClient(MONGODB_URL)
    db = mongodb_client.UserDetail  # Use the actual database name on Atlas
    logger.info("MongoDB Atlas connection initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MongoDB connection: {e}")
    raise

# Redis client for caching and Celery broker (optional - only if you need Redis)
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))

# Make Redis optional - won't crash if Redis is not available
try:
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, socket_connect_timeout=5)
    # Test Redis connection
    redis_client.ping()
    logger.info("Redis connection established successfully")
    
    # Celery configuration (only if Redis is available)
    celery_app = Celery(
        'ai_training',
        broker=f'redis://{redis_host}:{redis_port}/0',
        backend=f'redis://{redis_host}:{redis_port}/0'
    )
    
    celery_app.conf.task_routes = {
        'train_model_task': {'queue': 'training'}
    }
    logger.info("Celery configured successfully")
    
except redis.ConnectionError:
    logger.warning("Redis is not available. Running without Redis/Celery support.")
    redis_client = None
    celery_app = None
except Exception as e:
    logger.error(f"Redis configuration error: {e}")
    redis_client = None
    celery_app = None

# Test MongoDB connection function
async def test_database_connection():
    """Test the database connection"""
    try:
        # Test the connection
        await mongodb_client.admin.command('ping')
        logger.info("MongoDB Atlas connection test successful")
        return True
    except Exception as e:
        logger.error(f"MongoDB Atlas connection test failed: {e}")
        return False

# Graceful shutdown
async def close_database_connection():
    """Close the database connection"""
    if mongodb_client:
        mongodb_client.close()
        logger.info("MongoDB connection closed")