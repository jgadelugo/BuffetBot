"""
Connection Pool Management for GCS Operations

Manages connection pooling and resource optimization for GCS client connections.
"""

import asyncio
import logging
import threading
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for connection pool"""

    max_connections: int = 50
    min_connections: int = 5
    connection_timeout_seconds: int = 60
    idle_timeout_seconds: int = 300  # 5 minutes
    max_lifetime_seconds: int = 3600  # 1 hour
    health_check_interval_seconds: int = 30


@dataclass
class ConnectionStats:
    """Statistics for connection pool"""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    created_connections: int = 0
    destroyed_connections: int = 0
    total_requests: int = 0
    current_requests: int = 0


class PooledConnection:
    """Represents a pooled connection with metadata"""

    def __init__(self, connection: Any, pool: "ConnectionPool"):
        self.connection = connection
        self.pool = weakref.ref(pool)
        self.created_at = datetime.now()
        self.last_used_at = datetime.now()
        self.is_active = False
        self.use_count = 0
        self.connection_id = id(connection)

    def mark_active(self):
        """Mark connection as active"""
        self.is_active = True
        self.last_used_at = datetime.now()
        self.use_count += 1

    def mark_idle(self):
        """Mark connection as idle"""
        self.is_active = False
        self.last_used_at = datetime.now()

    def is_expired(self, config: ConnectionConfig) -> bool:
        """Check if connection has expired"""
        now = datetime.now()

        # Check maximum lifetime
        if (now - self.created_at).total_seconds() > config.max_lifetime_seconds:
            return True

        # Check idle timeout
        if (
            not self.is_active
            and (now - self.last_used_at).total_seconds() > config.idle_timeout_seconds
        ):
            return True

        return False

    def is_healthy(self) -> bool:
        """Check if connection is healthy (basic check)"""
        # This would typically ping the connection or check its state
        # For now, we'll assume the connection is healthy if it exists
        return self.connection is not None


class ConnectionPool:
    """Thread-safe connection pool for GCS operations"""

    def __init__(self, connection_factory, config: Optional[ConnectionConfig] = None):
        self.connection_factory = connection_factory
        self.config = config or ConnectionConfig()
        self.logger = logging.getLogger(__name__)

        # Connection storage
        self._pool: Queue = Queue(maxsize=self.config.max_connections)
        self._active_connections: dict[int, PooledConnection] = {}
        self._lock = threading.RLock()

        # Statistics
        self.stats = ConnectionStats()

        # Pool state
        self._shutdown = False
        self._initialized = False

        # Background maintenance
        self._maintenance_thread = None

    async def initialize(self):
        """Initialize the connection pool"""
        if self._initialized:
            return

        self.logger.info(
            f"Initializing connection pool with {self.config.min_connections} minimum connections"
        )

        # Create minimum connections
        for _ in range(self.config.min_connections):
            try:
                await self._create_connection()
            except Exception as e:
                self.logger.error(f"Failed to create initial connection: {str(e)}")

        # Start maintenance thread
        self._start_maintenance_thread()

        self._initialized = True
        self.logger.info("Connection pool initialized successfully")

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool (async context manager)"""
        connection = None
        try:
            connection = await self._acquire_connection()
            yield connection.connection
        finally:
            if connection:
                await self._release_connection(connection)

    async def _acquire_connection(self) -> PooledConnection:
        """Acquire a connection from the pool"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        timeout = self.config.connection_timeout_seconds

        while time.time() - start_time < timeout:
            # Try to get an idle connection
            try:
                pooled_conn = self._pool.get_nowait()

                # Check if connection is still healthy
                if pooled_conn.is_healthy() and not pooled_conn.is_expired(self.config):
                    pooled_conn.mark_active()

                    with self._lock:
                        self._active_connections[
                            pooled_conn.connection_id
                        ] = pooled_conn
                        self.stats.active_connections += 1
                        self.stats.idle_connections = self._pool.qsize()
                        self.stats.current_requests += 1
                        self.stats.total_requests += 1

                    self.logger.debug(
                        f"Acquired existing connection {pooled_conn.connection_id}"
                    )
                    return pooled_conn
                else:
                    # Connection is unhealthy or expired, destroy it
                    await self._destroy_connection(pooled_conn)

            except Empty:
                # No idle connections available
                pass

            # Try to create a new connection if under limit
            if self.stats.total_connections < self.config.max_connections:
                try:
                    pooled_conn = await self._create_connection()
                    pooled_conn.mark_active()

                    with self._lock:
                        self._active_connections[
                            pooled_conn.connection_id
                        ] = pooled_conn
                        self.stats.active_connections += 1
                        self.stats.current_requests += 1
                        self.stats.total_requests += 1

                    self.logger.debug(
                        f"Created new connection {pooled_conn.connection_id}"
                    )
                    return pooled_conn

                except Exception as e:
                    self.logger.error(f"Failed to create new connection: {str(e)}")
                    self.stats.failed_connections += 1

            # Wait a bit before retrying
            await asyncio.sleep(0.1)

        raise TimeoutError(f"Failed to acquire connection within {timeout} seconds")

    async def _release_connection(self, pooled_conn: PooledConnection):
        """Release a connection back to the pool"""
        try:
            pooled_conn.mark_idle()

            with self._lock:
                if pooled_conn.connection_id in self._active_connections:
                    del self._active_connections[pooled_conn.connection_id]
                self.stats.active_connections = len(self._active_connections)
                self.stats.current_requests = max(0, self.stats.current_requests - 1)

            # Check if connection should be destroyed
            if pooled_conn.is_expired(self.config) or not pooled_conn.is_healthy():
                await self._destroy_connection(pooled_conn)
                return

            # Return to pool
            try:
                self._pool.put_nowait(pooled_conn)
                self.stats.idle_connections = self._pool.qsize()
                self.logger.debug(
                    f"Released connection {pooled_conn.connection_id} to pool"
                )
            except:
                # Pool is full, destroy the connection
                await self._destroy_connection(pooled_conn)

        except Exception as e:
            self.logger.error(f"Error releasing connection: {str(e)}")
            await self._destroy_connection(pooled_conn)

    async def _create_connection(self) -> PooledConnection:
        """Create a new connection"""
        try:
            # Create the actual connection
            if asyncio.iscoroutinefunction(self.connection_factory):
                connection = await self.connection_factory()
            else:
                connection = self.connection_factory()

            pooled_conn = PooledConnection(connection, self)

            with self._lock:
                self.stats.total_connections += 1
                self.stats.created_connections += 1

            self.logger.debug(f"Created connection {pooled_conn.connection_id}")
            return pooled_conn

        except Exception as e:
            self.logger.error(f"Failed to create connection: {str(e)}")
            raise

    async def _destroy_connection(self, pooled_conn: PooledConnection):
        """Destroy a connection"""
        try:
            # Close the connection if it has a close method
            if hasattr(pooled_conn.connection, "close"):
                if asyncio.iscoroutinefunction(pooled_conn.connection.close):
                    await pooled_conn.connection.close()
                else:
                    pooled_conn.connection.close()

            with self._lock:
                self.stats.total_connections = max(0, self.stats.total_connections - 1)
                self.stats.destroyed_connections += 1

                # Remove from active connections if present
                if pooled_conn.connection_id in self._active_connections:
                    del self._active_connections[pooled_conn.connection_id]
                    self.stats.active_connections = len(self._active_connections)

            self.logger.debug(f"Destroyed connection {pooled_conn.connection_id}")

        except Exception as e:
            self.logger.error(f"Error destroying connection: {str(e)}")

    def _start_maintenance_thread(self):
        """Start background maintenance thread"""
        if self._maintenance_thread and self._maintenance_thread.is_alive():
            return

        self._maintenance_thread = threading.Thread(
            target=self._maintenance_worker,
            daemon=True,
            name="ConnectionPoolMaintenance",
        )
        self._maintenance_thread.start()
        self.logger.debug("Started connection pool maintenance thread")

    def _maintenance_worker(self):
        """Background worker for pool maintenance"""
        while not self._shutdown:
            try:
                self._cleanup_expired_connections()
                self._ensure_minimum_connections()
                time.sleep(self.config.health_check_interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in maintenance worker: {str(e)}")
                time.sleep(5)  # Wait before retrying

    def _cleanup_expired_connections(self):
        """Clean up expired connections"""
        expired_connections = []

        # Check idle connections
        temp_queue = Queue()
        while not self._pool.empty():
            try:
                pooled_conn = self._pool.get_nowait()
                if pooled_conn.is_expired(self.config) or not pooled_conn.is_healthy():
                    expired_connections.append(pooled_conn)
                else:
                    temp_queue.put(pooled_conn)
            except Empty:
                break

        # Put back non-expired connections
        while not temp_queue.empty():
            self._pool.put(temp_queue.get())

        # Destroy expired connections
        for pooled_conn in expired_connections:
            asyncio.create_task(self._destroy_connection(pooled_conn))

        if expired_connections:
            self.logger.info(
                f"Cleaned up {len(expired_connections)} expired connections"
            )

    def _ensure_minimum_connections(self):
        """Ensure minimum number of connections"""
        current_total = self.stats.total_connections
        if current_total < self.config.min_connections:
            needed = self.config.min_connections - current_total
            self.logger.debug(f"Creating {needed} connections to maintain minimum")

            for _ in range(needed):
                try:
                    asyncio.create_task(self._create_connection())
                except Exception as e:
                    self.logger.error(f"Failed to create minimum connection: {str(e)}")
                    break

    async def shutdown(self):
        """Shutdown the connection pool"""
        self.logger.info("Shutting down connection pool")
        self._shutdown = True

        # Close all active connections
        active_connections = list(self._active_connections.values())
        for pooled_conn in active_connections:
            await self._destroy_connection(pooled_conn)

        # Close all idle connections
        while not self._pool.empty():
            try:
                pooled_conn = self._pool.get_nowait()
                await self._destroy_connection(pooled_conn)
            except Empty:
                break

        self.logger.info("Connection pool shutdown complete")

    def get_stats(self) -> ConnectionStats:
        """Get current pool statistics"""
        with self._lock:
            # Update current counts
            self.stats.active_connections = len(self._active_connections)
            self.stats.idle_connections = self._pool.qsize()
            return ConnectionStats(
                total_connections=self.stats.total_connections,
                active_connections=self.stats.active_connections,
                idle_connections=self.stats.idle_connections,
                failed_connections=self.stats.failed_connections,
                created_connections=self.stats.created_connections,
                destroyed_connections=self.stats.destroyed_connections,
                total_requests=self.stats.total_requests,
                current_requests=self.stats.current_requests,
            )
