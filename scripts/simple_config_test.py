#!/usr/bin/env python3
"""Simple configuration test to verify the system works."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.config import DatabaseConfig, DatabaseEnvironment


def main():
    """Test configuration with explicit parameters."""
    print("Testing database configuration with explicit parameters...")

    try:
        # Test with explicit parameters (not environment variables)
        config = DatabaseConfig(
            username="test_user",
            password="test_pass",
            database="test_db",
            host="localhost",
            port=5432,
        )

        print(f"✅ Configuration created successfully:")
        print(f"   Username: {config.username}")
        print(f"   Database: {config.database}")
        print(f"   Host: {config.host}:{config.port}")
        print(f"   Environment: {config.environment}")

        # Test URL generation
        async_url = config.get_database_url(async_driver=True)
        sync_url = config.get_database_url(async_driver=False)

        print(f"✅ URLs generated successfully:")
        print(f"   Async URL: {async_url}")
        print(f"   Sync URL: {sync_url}")

        # Test engine kwargs
        kwargs = config.engine_kwargs
        print(f"✅ Engine kwargs: {kwargs}")

        # Test validation
        try:
            invalid_config = DatabaseConfig(
                username="test",
                password="test",
                database="test",
                port=99999,  # Invalid port
            )
            print("❌ Should have failed validation")
            return 1
        except ValueError as e:
            print(f"✅ Validation works: {e}")

        print("\n🎉 All basic configuration tests passed!")
        return 0

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
