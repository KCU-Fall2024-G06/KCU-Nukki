import redis
from fastapi import Depends

# Connect to Redis
def get_redis():
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    try:
        yield redis_client  # Yield Redis client to be used as a dependency
    finally:
        redis_client.close()  # Make sure to close the connection after use