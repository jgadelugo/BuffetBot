"""
Compression Management

Handles data compression and decompression for storage optimization.
"""

import gzip
import io
import logging
from enum import Enum
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Supported compression types"""

    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    SNAPPY = "snappy"
    NONE = "none"


class CompressionManager:
    """Manages data compression and decompression operations"""

    def __init__(self, default_compression: CompressionType = CompressionType.GZIP):
        self.default_compression = default_compression
        self.logger = logging.getLogger(__name__)

        # Compression level settings
        self.compression_levels = {
            CompressionType.GZIP: 6,
            CompressionType.LZ4: 0,  # LZ4 doesn't use levels
            CompressionType.ZSTD: 3,
            CompressionType.SNAPPY: 0,  # Snappy doesn't use levels
            CompressionType.NONE: 0,
        }

    def compress(
        self,
        data: Union[str, bytes],
        compression_type: Optional[CompressionType] = None,
    ) -> bytes:
        """Compress data using specified compression type"""
        compression_type = compression_type or self.default_compression

        try:
            # Convert string to bytes if needed
            if isinstance(data, str):
                data = data.encode("utf-8")

            if compression_type == CompressionType.GZIP:
                return self._compress_gzip(data)
            elif compression_type == CompressionType.LZ4:
                return self._compress_lz4(data)
            elif compression_type == CompressionType.ZSTD:
                return self._compress_zstd(data)
            elif compression_type == CompressionType.SNAPPY:
                return self._compress_snappy(data)
            elif compression_type == CompressionType.NONE:
                return data
            else:
                raise ValueError(f"Unsupported compression type: {compression_type}")

        except Exception as e:
            self.logger.error(f"Compression failed with {compression_type}: {str(e)}")
            raise

    def decompress(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using specified compression type"""
        try:
            if compression_type == CompressionType.GZIP:
                return self._decompress_gzip(data)
            elif compression_type == CompressionType.LZ4:
                return self._decompress_lz4(data)
            elif compression_type == CompressionType.ZSTD:
                return self._decompress_zstd(data)
            elif compression_type == CompressionType.SNAPPY:
                return self._decompress_snappy(data)
            elif compression_type == CompressionType.NONE:
                return data
            else:
                raise ValueError(f"Unsupported compression type: {compression_type}")

        except Exception as e:
            self.logger.error(f"Decompression failed with {compression_type}: {str(e)}")
            raise

    def _compress_gzip(self, data: bytes) -> bytes:
        """Compress using GZIP"""
        level = self.compression_levels[CompressionType.GZIP]
        return gzip.compress(data, compresslevel=level)

    def _decompress_gzip(self, data: bytes) -> bytes:
        """Decompress GZIP data"""
        return gzip.decompress(data)

    def _compress_lz4(self, data: bytes) -> bytes:
        """Compress using LZ4"""
        try:
            import lz4.frame

            return lz4.frame.compress(data)
        except ImportError:
            self.logger.warning("LZ4 not available, falling back to GZIP")
            return self._compress_gzip(data)

    def _decompress_lz4(self, data: bytes) -> bytes:
        """Decompress LZ4 data"""
        try:
            import lz4.frame

            return lz4.frame.decompress(data)
        except ImportError:
            self.logger.warning("LZ4 not available, trying GZIP")
            return self._decompress_gzip(data)

    def _compress_zstd(self, data: bytes) -> bytes:
        """Compress using Zstandard"""
        try:
            import zstandard as zstd

            level = self.compression_levels[CompressionType.ZSTD]
            cctx = zstd.ZstdCompressor(level=level)
            return cctx.compress(data)
        except ImportError:
            self.logger.warning("Zstandard not available, falling back to GZIP")
            return self._compress_gzip(data)

    def _decompress_zstd(self, data: bytes) -> bytes:
        """Decompress Zstandard data"""
        try:
            import zstandard as zstd

            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        except ImportError:
            self.logger.warning("Zstandard not available, trying GZIP")
            return self._decompress_gzip(data)

    def _compress_snappy(self, data: bytes) -> bytes:
        """Compress using Snappy"""
        try:
            import snappy

            return snappy.compress(data)
        except ImportError:
            self.logger.warning("Snappy not available, falling back to GZIP")
            return self._compress_gzip(data)

    def _decompress_snappy(self, data: bytes) -> bytes:
        """Decompress Snappy data"""
        try:
            import snappy

            return snappy.decompress(data)
        except ImportError:
            self.logger.warning("Snappy not available, trying GZIP")
            return self._decompress_gzip(data)

    def get_compression_ratio(
        self, original_data: Union[str, bytes], compressed_data: bytes
    ) -> float:
        """Calculate compression ratio"""
        try:
            if isinstance(original_data, str):
                original_size = len(original_data.encode("utf-8"))
            else:
                original_size = len(original_data)

            compressed_size = len(compressed_data)

            if compressed_size == 0:
                return 0.0

            return original_size / compressed_size

        except Exception as e:
            self.logger.error(f"Failed to calculate compression ratio: {str(e)}")
            return 1.0

    def get_compression_info(
        self,
        original_data: Union[str, bytes],
        compression_type: Optional[CompressionType] = None,
    ) -> dict[str, Any]:
        """Get detailed compression information"""
        compression_type = compression_type or self.default_compression

        try:
            # Get original size
            if isinstance(original_data, str):
                original_size = len(original_data.encode("utf-8"))
            else:
                original_size = len(original_data)

            # Compress the data
            compressed_data = self.compress(original_data, compression_type)
            compressed_size = len(compressed_data)

            # Calculate metrics
            compression_ratio = self.get_compression_ratio(
                original_data, compressed_data
            )
            space_saved = original_size - compressed_size
            space_saved_percent = (
                (space_saved / original_size * 100) if original_size > 0 else 0
            )

            return {
                "compression_type": compression_type.value,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
                "space_saved_bytes": space_saved,
                "space_saved_percent": space_saved_percent,
            }

        except Exception as e:
            self.logger.error(f"Failed to get compression info: {str(e)}")
            return {"compression_type": compression_type.value, "error": str(e)}

    def benchmark_compression_types(
        self, data: Union[str, bytes]
    ) -> dict[str, dict[str, Any]]:
        """Benchmark different compression types on the given data"""
        results = {}

        for comp_type in CompressionType:
            if comp_type == CompressionType.NONE:
                continue

            try:
                info = self.get_compression_info(data, comp_type)
                results[comp_type.value] = info
            except Exception as e:
                results[comp_type.value] = {"error": str(e)}

        return results

    def recommend_compression(self, data: Union[str, bytes]) -> CompressionType:
        """Recommend the best compression type for the given data"""
        try:
            benchmark = self.benchmark_compression_types(data)

            best_compression = CompressionType.GZIP
            best_ratio = 1.0

            for comp_type_str, info in benchmark.items():
                if "error" not in info:
                    ratio = info.get("compression_ratio", 1.0)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_compression = CompressionType(comp_type_str)

            self.logger.info(
                f"Recommended compression: {best_compression.value} (ratio: {best_ratio:.2f})"
            )
            return best_compression

        except Exception as e:
            self.logger.error(f"Failed to recommend compression: {str(e)}")
            return CompressionType.GZIP  # Safe default
