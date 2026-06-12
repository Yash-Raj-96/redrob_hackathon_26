"""
Caching utilities for improved performance
"""

import json
import time
from typing import Any, Optional, Callable

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class CacheManager:
    """
    Unified cache manager supporting:
    - In-memory cache
    - Redis cache
    """

    def __init__(
        self,
        cache_type: str = "memory",
        ttl_seconds: int = 3600
    ):

        self.cache_type = cache_type.lower()
        self.ttl = ttl_seconds

        # Memory cache
        self.cache = {}
        self.timestamps = {}

        # Redis client
        self.redis_client = None

        # ==========================================
        # Redis initialization
        # ==========================================

        if self.cache_type == "redis":

            try:
                import redis

                self.redis_client = redis.Redis(
                    host="localhost",
                    port=6379,
                    db=0,
                    decode_responses=True
                )

                self.redis_client.ping()

                logger.info("Redis cache initialized")

            except Exception as e:

                logger.warning(
                    f"Redis unavailable: {str(e)}"
                )

                logger.warning(
                    "Falling back to in-memory cache"
                )

                self.cache_type = "memory"

        logger.info(
            f"CacheManager initialized "
            f"(type={self.cache_type}, ttl={self.ttl}s)"
        )

    # =====================================================
    # Internal Helpers
    # =====================================================

    def _is_expired(self, key: str) -> bool:
        """
        Check if memory cache entry expired
        """

        if key not in self.timestamps:
            return True

        return (
            time.time() - self.timestamps[key]
        ) > self.ttl

    def _serialize(self, value: Any) -> str:
        """
        Serialize object for Redis
        """

        try:
            return json.dumps(value, default=str)

        except Exception:
            return json.dumps(str(value))

    def _deserialize(self, value: str) -> Any:
        """
        Deserialize Redis object
        """

        try:
            return json.loads(value)

        except Exception:
            return value

    # =====================================================
    # Get Cache
    # =====================================================

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value
        """

        try:

            # ======================================
            # Redis
            # ======================================

            if self.cache_type == "redis":

                value = self.redis_client.get(key)

                if value is None:
                    return None

                logger.debug(f"Redis cache hit: {key}")

                return self._deserialize(value)

            # ======================================
            # Memory cache
            # ======================================

            if key not in self.cache:
                return None

            if self._is_expired(key):

                logger.debug(f"Cache expired: {key}")

                self.delete(key)

                return None

            logger.debug(f"Memory cache hit: {key}")

            return self.cache.get(key)

        except Exception as e:

            logger.error(
                f"Cache GET failed for key={key}: {str(e)}"
            )

            return None

    # =====================================================
    # Set Cache
    # =====================================================

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store value in cache
        """

        ttl = ttl or self.ttl

        try:

            # ======================================
            # Redis
            # ======================================

            if self.cache_type == "redis":

                self.redis_client.setex(
                    key,
                    ttl,
                    self._serialize(value)
                )

                logger.debug(f"Redis cache set: {key}")

                return True

            # ======================================
            # Memory
            # ======================================

            self.cache[key] = value
            self.timestamps[key] = time.time()

            logger.debug(f"Memory cache set: {key}")

            return True

        except Exception as e:

            logger.error(
                f"Cache SET failed for key={key}: {str(e)}"
            )

            return False

    # =====================================================
    # Delete Cache
    # =====================================================

    def delete(self, key: str) -> bool:
        """
        Delete cache key
        """

        try:

            if self.cache_type == "redis":

                self.redis_client.delete(key)

            else:

                self.cache.pop(key, None)
                self.timestamps.pop(key, None)

            logger.debug(f"Cache deleted: {key}")

            return True

        except Exception as e:

            logger.error(
                f"Cache DELETE failed for key={key}: {str(e)}"
            )

            return False

    # =====================================================
    # Clear Cache
    # =====================================================

    def clear(self) -> bool:
        """
        Clear all cache entries
        """

        try:

            if self.cache_type == "redis":

                self.redis_client.flushdb()

            else:

                self.cache.clear()
                self.timestamps.clear()

            logger.info("Cache cleared")

            return True

        except Exception as e:

            logger.error(
                f"Cache CLEAR failed: {str(e)}"
            )

            return False

    # =====================================================
    # Get or Set
    # =====================================================

    def get_or_set(
        self,
        key: str,
        func: Callable,
        *args,
        ttl: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Retrieve from cache or compute and store
        """

        cached = self.get(key)

        if cached is not None:
            return cached

        logger.debug(f"Cache miss: {key}")

        value = func(*args, **kwargs)

        self.set(key, value, ttl=ttl)

        return value

    # =====================================================
    # Cache Stats
    # =====================================================

    def stats(self) -> dict:
        """
        Return cache statistics
        """

        if self.cache_type == "redis":

            try:
                info = self.redis_client.info()

                return {
                    "type": "redis",
                    "connected_clients": info.get(
                        "connected_clients"
                    ),
                    "used_memory_human": info.get(
                        "used_memory_human"
                    ),
                    "keys": self.redis_client.dbsize(),
                }

            except Exception as e:

                return {
                    "type": "redis",
                    "error": str(e)
                }

        return {
            "type": "memory",
            "keys": len(self.cache),
            "ttl_seconds": self.ttl
        }