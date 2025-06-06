#!/usr/bin/env python3
"""
Pub/Sub Market Data Publisher

Publishes market events to Pub/Sub for real-time processing.
Integrates with Phase 1 schema validation for data quality assurance.

Features:
- Real-time market data publishing
- Schema validation using Phase 1 system
- Batch publishing for high throughput
- Error handling and retry logic
- Message ordering for time-series data
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from google.api_core import retry
from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.types import PubsubMessage

# Import Phase 1 components - DO NOT MODIFY
from buffetbot.storage.schemas.manager import SchemaManager


@dataclass
class MarketData:
    """Market data structure for publishing."""

    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    market_cap: Optional[float] = None
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO string
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class PublishResult:
    """Result from publishing operation."""

    message_id: str
    topic: str
    success: bool
    error: Optional[str] = None
    publish_time: Optional[datetime] = None


@dataclass
class BatchPublishResult:
    """Result from batch publishing operation."""

    total_messages: int
    successful: int
    failed: int
    message_ids: list[str]
    errors: list[str]
    publish_time: datetime


class MarketDataPublisher:
    """
    Publishes market events to Pub/Sub for real-time processing.
    Integrates with Phase 1 schema validation.
    """

    def __init__(self, project_id: str, default_topic: str = "market-data"):
        """
        Initialize Market Data Publisher.

        Args:
            project_id: GCP project ID
            default_topic: Default Pub/Sub topic for market data
        """
        self.project_id = project_id
        self.default_topic = default_topic
        self.publisher = pubsub_v1.PublisherClient()

        # Initialize Phase 1 integration
        self.schema_manager = SchemaManager()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=10)

        # Topic paths cache
        self._topic_paths = {}

        # Initialize default topic
        asyncio.create_task(self._ensure_topic_exists(default_topic))

    def _get_topic_path(self, topic: str) -> str:
        """Get full topic path."""
        if topic not in self._topic_paths:
            self._topic_paths[topic] = self.publisher.topic_path(self.project_id, topic)
        return self._topic_paths[topic]

    async def _ensure_topic_exists(self, topic: str) -> None:
        """Ensure topic exists, create if not."""
        try:
            topic_path = self._get_topic_path(topic)

            # Check if topic exists
            try:
                self.publisher.get_topic(request={"topic": topic_path})
                self.logger.info(f"Topic {topic} already exists")
            except Exception:
                # Topic doesn't exist, create it
                self.publisher.create_topic(request={"name": topic_path})
                self.logger.info(f"Created topic {topic}")

        except Exception as e:
            self.logger.error(f"Failed to ensure topic {topic} exists: {e}")

    async def _validate_market_data(self, data: MarketData) -> bool:
        """
        Validate market data using Phase 1 schema system.

        Args:
            data: Market data to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Convert to dict for validation
            data_dict = data.to_dict()

            # Use Phase 1 schema validation
            # Note: This assumes a market_data schema exists in Phase 1
            is_valid = await self.schema_manager.validate_data(
                data_dict, schema_name="market_data"
            )

            if not is_valid:
                self.logger.warning(f"Market data validation failed for {data.symbol}")

            return is_valid

        except Exception as e:
            self.logger.error(f"Schema validation error: {e}")
            return False

    async def publish_market_data(
        self,
        data: MarketData,
        topic: Optional[str] = None,
        validate_schema: bool = True,
    ) -> PublishResult:
        """
        Publish validated market data with ordering keys.

        Args:
            data: Market data to publish
            topic: Pub/Sub topic (uses default if None)
            validate_schema: Whether to validate against schema

        Returns:
            PublishResult with operation details
        """
        start_time = datetime.now()
        topic = topic or self.default_topic

        try:
            # Validate schema if requested
            if validate_schema:
                is_valid = await self._validate_market_data(data)
                if not is_valid:
                    return PublishResult(
                        message_id="",
                        topic=topic,
                        success=False,
                        error="Schema validation failed",
                    )

            # Prepare message
            message_data = json.dumps(data.to_dict()).encode("utf-8")

            # Create message with attributes
            message = PubsubMessage(
                data=message_data,
                attributes={
                    "symbol": data.symbol,
                    "source": data.source,
                    "timestamp": data.timestamp.isoformat(),
                    "message_type": "market_data",
                },
                ordering_key=data.symbol,  # Ensure ordered delivery per symbol
            )

            # Publish message
            topic_path = self._get_topic_path(topic)

            # Use retry policy for reliability
            future = self.publisher.publish(
                topic_path,
                message.data,
                ordering_key=message.ordering_key,
                **message.attributes,
            )

            # Wait for result
            message_id = future.result(timeout=30)

            return PublishResult(
                message_id=message_id,
                topic=topic,
                success=True,
                publish_time=start_time,
            )

        except Exception as e:
            self.logger.error(f"Failed to publish market data for {data.symbol}: {e}")
            return PublishResult(
                message_id="", topic=topic, success=False, error=str(e)
            )

    async def publish_batch(
        self,
        data_batch: list[MarketData],
        topic: Optional[str] = None,
        validate_schema: bool = True,
    ) -> BatchPublishResult:
        """
        Batch publishing for high-throughput scenarios.

        Args:
            data_batch: List of market data to publish
            topic: Pub/Sub topic (uses default if None)
            validate_schema: Whether to validate against schema

        Returns:
            BatchPublishResult with batch operation details
        """
        start_time = datetime.now()
        topic = topic or self.default_topic

        successful = 0
        failed = 0
        message_ids = []
        errors = []

        # Ensure topic exists
        await self._ensure_topic_exists(topic)

        # Process batch
        publish_tasks = []
        for data in data_batch:
            task = self.publish_market_data(data, topic, validate_schema)
            publish_tasks.append(task)

        # Execute all publishes concurrently
        results = await asyncio.gather(*publish_tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed += 1
                errors.append(f"Message {i}: {str(result)}")
            elif isinstance(result, PublishResult):
                if result.success:
                    successful += 1
                    message_ids.append(result.message_id)
                else:
                    failed += 1
                    errors.append(f"Message {i}: {result.error}")
            else:
                failed += 1
                errors.append(f"Message {i}: Unknown error")

        return BatchPublishResult(
            total_messages=len(data_batch),
            successful=successful,
            failed=failed,
            message_ids=message_ids,
            errors=errors,
            publish_time=start_time,
        )

    async def publish_json_data(
        self,
        json_data: dict[str, Any],
        topic: Optional[str] = None,
        ordering_key: Optional[str] = None,
    ) -> PublishResult:
        """
        Publish raw JSON data to Pub/Sub.

        Args:
            json_data: JSON data to publish
            topic: Pub/Sub topic (uses default if None)
            ordering_key: Optional ordering key for message ordering

        Returns:
            PublishResult with operation details
        """
        start_time = datetime.now()
        topic = topic or self.default_topic

        try:
            # Ensure topic exists
            await self._ensure_topic_exists(topic)

            # Prepare message
            message_data = json.dumps(json_data).encode("utf-8")

            # Create attributes from data
            attributes = {
                "message_type": "json_data",
                "timestamp": datetime.now().isoformat(),
            }

            # Add symbol if present for ordering
            if "symbol" in json_data:
                attributes["symbol"] = str(json_data["symbol"])
                ordering_key = ordering_key or str(json_data["symbol"])

            # Publish message
            topic_path = self._get_topic_path(topic)

            publish_kwargs = {"ordering_key": ordering_key} if ordering_key else {}

            future = self.publisher.publish(
                topic_path, message_data, **attributes, **publish_kwargs
            )

            # Wait for result
            message_id = future.result(timeout=30)

            return PublishResult(
                message_id=message_id,
                topic=topic,
                success=True,
                publish_time=start_time,
            )

        except Exception as e:
            self.logger.error(f"Failed to publish JSON data: {e}")
            return PublishResult(
                message_id="", topic=topic, success=False, error=str(e)
            )

    async def create_topic(self, topic: str) -> bool:
        """
        Create a new Pub/Sub topic.

        Args:
            topic: Topic name to create

        Returns:
            True if successful, False otherwise
        """
        try:
            topic_path = self._get_topic_path(topic)
            self.publisher.create_topic(request={"name": topic_path})
            self.logger.info(f"Created topic {topic}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create topic {topic}: {e}")
            return False

    async def delete_topic(self, topic: str) -> bool:
        """
        Delete a Pub/Sub topic.

        Args:
            topic: Topic name to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            topic_path = self._get_topic_path(topic)
            self.publisher.delete_topic(request={"topic": topic_path})
            self.logger.info(f"Deleted topic {topic}")

            # Remove from cache
            if topic in self._topic_paths:
                del self._topic_paths[topic]

            return True
        except Exception as e:
            self.logger.error(f"Failed to delete topic {topic}: {e}")
            return False

    async def list_topics(self) -> list[str]:
        """
        List all topics in the project.

        Returns:
            List of topic names
        """
        try:
            project_path = f"projects/{self.project_id}"
            topics = self.publisher.list_topics(request={"project": project_path})

            topic_names = []
            for topic in topics:
                # Extract topic name from full path
                topic_name = topic.name.split("/")[-1]
                topic_names.append(topic_name)

            return topic_names
        except Exception as e:
            self.logger.error(f"Failed to list topics: {e}")
            return []

    def close(self):
        """Close the publisher and cleanup resources."""
        try:
            self.publisher.transport.close()
            self.executor.shutdown(wait=True)
            self.logger.info("Publisher closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing publisher: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.close()
