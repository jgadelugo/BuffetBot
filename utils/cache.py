import hashlib
import json
import os
import pickle
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Union

from utils.config import CACHE_CONFIG, CACHE_DIR
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class Cache:
    """
    Cache manager for storing and retrieving data.
    Supports both file-based and memory-based caching.
    """

    def __init__(self, cache_type: str = "file"):
        """
        Initialize cache manager.

        Args:
            cache_type: Type of cache ("file" or "memory")
        """
        self.cache_type = cache_type
        self.memory_cache: dict[str, dict[str, Any]] = {}
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)

    def _generate_key(self, data: Any) -> str:
        """
        Generate a unique cache key for the given data.

        Args:
            data: Data to generate key for

        Returns:
            str: Unique cache key
        """
        try:
            # Convert data to string and hash it
            data_str = str(data).encode("utf-8")
            return hashlib.md5(data_str).hexdigest()

        except Exception as e:
            logger.error(f"Error generating cache key: {str(e)}")
            raise

    def _get_cache_path(self, key: str) -> Path:
        """
        Get the file path for a cache key.

        Args:
            key: Cache key

        Returns:
            Path: Cache file path
        """
        return self.cache_dir / f"{key}.cache"

    def _is_expired(self, timestamp: float, expiration: int) -> bool:
        """
        Check if cached data has expired.

        Args:
            timestamp: Cache timestamp
            expiration: Expiration time in seconds

        Returns:
            bool: True if expired, False otherwise
        """
        return time.time() - timestamp > expiration

    def get(self, key: str, expiration: int | None = None) -> Any | None:
        """
        Retrieve data from cache.

        Args:
            key: Cache key
            expiration: Expiration time in seconds

        Returns:
            Optional[Any]: Cached data if found and not expired, None otherwise
        """
        try:
            if expiration is None:
                expiration = CACHE_CONFIG["expiration"]

            if self.cache_type == "memory":
                if key in self.memory_cache:
                    cache_data = self.memory_cache[key]
                    if not self._is_expired(cache_data["timestamp"], expiration):
                        logger.info(f"Cache hit for key: {key}")
                        return cache_data["data"]
                    else:
                        del self.memory_cache[key]

            else:  # file cache
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    with open(cache_path, "rb") as f:
                        cache_data = pickle.load(f)
                        if not self._is_expired(cache_data["timestamp"], expiration):
                            logger.info(f"Cache hit for key: {key}")
                            return cache_data["data"]
                        else:
                            cache_path.unlink()

            logger.info(f"Cache miss for key: {key}")
            return None

        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None

    def set(self, key: str, data: Any, expiration: int | None = None) -> bool:
        """
        Store data in cache.

        Args:
            key: Cache key
            data: Data to cache
            expiration: Expiration time in seconds

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if expiration is None:
                expiration = CACHE_CONFIG["expiration"]

            cache_data = {
                "data": data,
                "timestamp": time.time(),
                "expiration": expiration,
            }

            if self.cache_type == "memory":
                self.memory_cache[key] = cache_data

            else:  # file cache
                cache_path = self._get_cache_path(key)
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)

            logger.info(f"Data cached for key: {key}")
            return True

        except Exception as e:
            logger.error(f"Error storing in cache: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete data from cache.

        Args:
            key: Cache key

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.cache_type == "memory":
                if key in self.memory_cache:
                    del self.memory_cache[key]

            else:  # file cache
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    cache_path.unlink()

            logger.info(f"Cache deleted for key: {key}")
            return True

        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")
            return False

    def clear(self) -> bool:
        """
        Clear all cached data.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.cache_type == "memory":
                self.memory_cache.clear()

            else:  # file cache
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()

            logger.info("Cache cleared")
            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict containing cache statistics
        """
        try:
            stats = {"type": self.cache_type, "size": 0, "items": 0, "expired": 0}

            if self.cache_type == "memory":
                stats["items"] = len(self.memory_cache)
                stats["size"] = sum(len(str(v)) for v in self.memory_cache.values())

            else:  # file cache
                for cache_file in self.cache_dir.glob("*.cache"):
                    stats["items"] += 1
                    stats["size"] += cache_file.stat().st_size

                    # Check for expired items
                    with open(cache_file, "rb") as f:
                        cache_data = pickle.load(f)
                        if self._is_expired(
                            cache_data["timestamp"], cache_data["expiration"]
                        ):
                            stats["expired"] += 1

            return stats

        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}


def cache_result(expiration: int | None = None):
    """
    Decorator for caching function results.

    Args:
        expiration: Cache expiration time in seconds

    Returns:
        Decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            key = hashlib.md5("|".join(key_parts).encode()).hexdigest()

            # Try to get from cache
            cache = Cache(CACHE_CONFIG["type"])
            result = cache.get(key, expiration)

            if result is None:
                # Cache miss, execute function
                result = func(*args, **kwargs)
                cache.set(key, result, expiration)

            return result

        return wrapper

    return decorator
