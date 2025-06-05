"""
Security Management for Storage System

Handles encryption, access control, and security utilities for GCS storage.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


@dataclass
class AccessPolicy:
    """Represents an access control policy"""

    resource_pattern: str
    allowed_operations: list[str]
    principals: list[str]
    conditions: dict[str, Any] = None
    expires_at: Optional[datetime] = None


@dataclass
class SecurityContext:
    """Security context for operations"""

    user_id: str
    roles: list[str]
    permissions: list[str]
    session_id: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class SecurityManager:
    """Comprehensive security management for storage operations"""

    def __init__(self, encryption_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        # Initialize encryption
        if encryption_key:
            self.encryption_key = encryption_key.encode()
        else:
            self.encryption_key = self._generate_encryption_key()

        self.cipher_suite = self._create_cipher_suite()

        # Access policies
        self.access_policies: list[AccessPolicy] = []

        # Security audit log
        self.audit_log: list[dict[str, Any]] = []

        # Initialize default policies
        self._initialize_default_policies()

    def encrypt_data(self, data: Union[str, bytes]) -> str:
        """Encrypt sensitive data"""
        try:
            if isinstance(data, str):
                data = data.encode("utf-8")

            encrypted_data = self.cipher_suite.encrypt(data)
            return base64.b64encode(encrypted_data).decode("utf-8")

        except Exception as e:
            self.logger.error(f"Encryption failed: {str(e)}")
            raise

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode("utf-8"))
            decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted_data.decode("utf-8")

        except Exception as e:
            self.logger.error(f"Decryption failed: {str(e)}")
            raise

    def hash_data(self, data: str, salt: Optional[str] = None) -> str:
        """Create secure hash of data"""
        if salt is None:
            salt = os.urandom(32)
        elif isinstance(salt, str):
            salt = salt.encode("utf-8")

        # Use PBKDF2 for secure hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        key = kdf.derive(data.encode("utf-8"))
        return base64.b64encode(salt + key).decode("utf-8")

    def verify_hash(self, data: str, hashed_data: str) -> bool:
        """Verify data against hash"""
        try:
            decoded = base64.b64decode(hashed_data.encode("utf-8"))
            salt = decoded[:32]
            stored_key = decoded[32:]

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )

            key = kdf.derive(data.encode("utf-8"))
            return hmac.compare_digest(stored_key, key)

        except Exception as e:
            self.logger.error(f"Hash verification failed: {str(e)}")
            return False

    def check_access(
        self, context: SecurityContext, resource: str, operation: str
    ) -> bool:
        """Check if access is allowed for given context and resource"""
        try:
            # Log access attempt
            self._log_access_attempt(context, resource, operation)

            # Check each policy
            for policy in self.access_policies:
                if self._matches_policy(context, resource, operation, policy):
                    self.logger.debug(
                        f"Access granted: {context.user_id} -> {resource} ({operation})"
                    )
                    return True

            self.logger.warning(
                f"Access denied: {context.user_id} -> {resource} ({operation})"
            )
            return False

        except Exception as e:
            self.logger.error(f"Access check failed: {str(e)}")
            return False

    def add_access_policy(self, policy: AccessPolicy) -> None:
        """Add a new access policy"""
        self.access_policies.append(policy)
        self.logger.info(f"Added access policy for {policy.resource_pattern}")

    def remove_access_policy(self, resource_pattern: str) -> bool:
        """Remove access policy by resource pattern"""
        initial_count = len(self.access_policies)
        self.access_policies = [
            p for p in self.access_policies if p.resource_pattern != resource_pattern
        ]

        removed = len(self.access_policies) < initial_count
        if removed:
            self.logger.info(f"Removed access policy for {resource_pattern}")

        return removed

    def create_signed_url(
        self, resource: str, operation: str, expires_in_hours: int = 1
    ) -> str:
        """Create a signed URL for temporary access"""
        expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)

        # Create payload
        payload = {
            "resource": resource,
            "operation": operation,
            "expires_at": expires_at.isoformat(),
        }

        # Sign the payload
        payload_str = json.dumps(payload, sort_keys=True)
        signature = self._sign_data(payload_str)

        # Encode for URL
        encoded_payload = base64.urlsafe_b64encode(payload_str.encode()).decode()
        encoded_signature = base64.urlsafe_b64encode(signature.encode()).decode()

        return f"{encoded_payload}.{encoded_signature}"

    def verify_signed_url(self, signed_url: str, resource: str, operation: str) -> bool:
        """Verify a signed URL"""
        try:
            # Split URL
            parts = signed_url.split(".")
            if len(parts) != 2:
                return False

            encoded_payload, encoded_signature = parts

            # Decode
            payload_str = base64.urlsafe_b64decode(encoded_payload.encode()).decode()
            signature = base64.urlsafe_b64decode(encoded_signature.encode()).decode()

            # Verify signature
            if not self._verify_signature(payload_str, signature):
                return False

            # Parse payload
            payload = json.loads(payload_str)

            # Check resource and operation
            if payload["resource"] != resource or payload["operation"] != operation:
                return False

            # Check expiration
            expires_at = datetime.fromisoformat(payload["expires_at"])
            if datetime.utcnow() > expires_at:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Signed URL verification failed: {str(e)}")
            return False

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove dangerous characters
        dangerous_chars = ["..", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        sanitized = filename

        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "_")

        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[: 255 - len(ext)] + ext

        return sanitized

    def validate_data_classification(
        self, data: dict[str, Any], classification: str
    ) -> bool:
        """Validate data meets classification requirements"""
        classification_rules = {
            "public": {
                "allowed_fields": ["symbol", "price", "volume", "timestamp"],
                "forbidden_patterns": [],
            },
            "internal": {
                "allowed_fields": ["*"],  # All fields allowed
                "forbidden_patterns": ["ssn", "credit_card", "password"],
            },
            "confidential": {
                "allowed_fields": ["*"],
                "forbidden_patterns": ["ssn", "credit_card", "password"],
                "encryption_required": True,
            },
        }

        if classification not in classification_rules:
            return False

        rules = classification_rules[classification]

        # Check forbidden patterns
        data_str = json.dumps(data).lower()
        for pattern in rules.get("forbidden_patterns", []):
            if pattern in data_str:
                self.logger.warning(f"Data contains forbidden pattern: {pattern}")
                return False

        return True

    def get_audit_log(
        self, since: Optional[datetime] = None, user_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Get security audit log"""
        logs = self.audit_log

        if since:
            logs = [log for log in logs if log["timestamp"] >= since]

        if user_id:
            logs = [log for log in logs if log.get("user_id") == user_id]

        return logs

    def _generate_encryption_key(self) -> bytes:
        """Generate a new encryption key"""
        return Fernet.generate_key()

    def _create_cipher_suite(self) -> Fernet:
        """Create cipher suite for encryption/decryption"""
        return Fernet(self.encryption_key)

    def _initialize_default_policies(self) -> None:
        """Initialize default access policies"""
        # Public read access to market data
        self.add_access_policy(
            AccessPolicy(
                resource_pattern="market_data/*",
                allowed_operations=["read"],
                principals=["public"],
            )
        )

        # Admin full access
        self.add_access_policy(
            AccessPolicy(
                resource_pattern="*",
                allowed_operations=["read", "write", "delete"],
                principals=["admin"],
            )
        )

        # Service accounts read/write access to their data types
        self.add_access_policy(
            AccessPolicy(
                resource_pattern="forecasts/*",
                allowed_operations=["read", "write"],
                principals=["ml_service", "analytics_service"],
            )
        )

    def _matches_policy(
        self,
        context: SecurityContext,
        resource: str,
        operation: str,
        policy: AccessPolicy,
    ) -> bool:
        """Check if context matches policy"""
        # Check expiration
        if policy.expires_at and datetime.utcnow() > policy.expires_at:
            return False

        # Check resource pattern
        if not self._matches_pattern(resource, policy.resource_pattern):
            return False

        # Check operation
        if operation not in policy.allowed_operations:
            return False

        # Check principals (roles or user IDs)
        if not any(
            principal in context.roles or principal == context.user_id
            for principal in policy.principals
        ):
            return False

        # Check conditions if any
        if policy.conditions:
            if not self._check_conditions(context, policy.conditions):
                return False

        return True

    def _matches_pattern(self, resource: str, pattern: str) -> bool:
        """Check if resource matches pattern (supports wildcards)"""
        if pattern == "*":
            return True

        if pattern.endswith("*"):
            return resource.startswith(pattern[:-1])

        return resource == pattern

    def _check_conditions(
        self, context: SecurityContext, conditions: dict[str, Any]
    ) -> bool:
        """Check policy conditions"""
        # Time-based conditions
        if "time_range" in conditions:
            current_hour = datetime.utcnow().hour
            start_hour = conditions["time_range"].get("start", 0)
            end_hour = conditions["time_range"].get("end", 23)

            if not (start_hour <= current_hour <= end_hour):
                return False

        # IP-based conditions
        if "allowed_ips" in conditions and context.ip_address:
            if context.ip_address not in conditions["allowed_ips"]:
                return False

        return True

    def _sign_data(self, data: str) -> str:
        """Create HMAC signature for data"""
        signature = hmac.new(
            self.encryption_key, data.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return signature

    def _verify_signature(self, data: str, signature: str) -> bool:
        """Verify HMAC signature"""
        expected_signature = self._sign_data(data)
        return hmac.compare_digest(signature, expected_signature)

    def _log_access_attempt(
        self, context: SecurityContext, resource: str, operation: str
    ) -> None:
        """Log access attempt for audit"""
        log_entry = {
            "timestamp": datetime.utcnow(),
            "user_id": context.user_id,
            "resource": resource,
            "operation": operation,
            "ip_address": context.ip_address,
            "user_agent": context.user_agent,
            "session_id": context.session_id,
        }

        self.audit_log.append(log_entry)

        # Limit audit log size
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]  # Keep last 5000 entries
