"""
Standalone Vision Recognition and Visual Character Analysis Package.
Operates independently from the core story generation pipeline.
"""

from .analyzer import VisionAnalyzer, analyze_image

__all__ = ["VisionAnalyzer", "analyze_image"]
