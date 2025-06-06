"""
Cloud Upgrade Interfaces - Prepare for future cloud ML services
Zero cost implementations with easy upgrade paths
"""

from .bigquery_interface import BigQueryMLInterface
from .ml_interface import MLInterface
from .vertex_interface import VertexAIInterface

__all__ = ["MLInterface", "BigQueryMLInterface", "VertexAIInterface"]
