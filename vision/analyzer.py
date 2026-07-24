"""
Unified Vision Analyzer Orchestrator.
Dispatches to local or cloud vision providers dynamically.
"""

import logging
from typing import Dict, Any, Optional

import config
from .schema import PersonAnalysisResult
from .local_provider import LocalVisionProvider
from .cloud_provider import CloudVisionProvider

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """Orchestrator for standalone Person Recognition and Vision Analysis."""

    def __init__(self, backend_mode: str = None):
        self.backend_mode = (backend_mode or getattr(config, "VISION_BACKEND", "local")).lower()
        self.local_provider = LocalVisionProvider()
        self.cloud_provider = CloudVisionProvider()

    def get_available_backends(self) -> Dict[str, Any]:
        """Query availability of local and cloud vision providers."""
        local_status = self.local_provider.check_availability()
        cloud_status = self.cloud_provider.check_availability()

        return {
            "active_mode": self.backend_mode,
            "local": local_status,
            "cloud": cloud_status
        }

    def analyze(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        mode: str = None,
        prompt_extra: str = ""
    ) -> PersonAnalysisResult:
        """
        Analyze an image for person recognition, body structure, facial features, and style.
        Mode can be 'local', 'cloud', 'gemini', or 'auto'.
        """
        active_mode = (mode or self.backend_mode).lower()

        if active_mode in ("cloud", "gemini"):
            logger.info("Running vision analysis with Cloud Provider (Gemini)...")
            return self.cloud_provider.analyze(image_bytes, mime_type=mime_type, prompt_extra=prompt_extra)

        elif active_mode in ("local", "ollama"):
            logger.info("Running vision analysis with Local Provider (Ollama)...")
            try:
                return self.local_provider.analyze(image_bytes, mime_type=mime_type, prompt_extra=prompt_extra)
            except Exception as e:
                # If local is requested but fails, attempt graceful fallback to cloud if API key is available
                logger.warning("Local vision failed: %s. Checking if cloud fallback is available...", e)
                cloud_avail = self.cloud_provider.check_availability()
                if cloud_avail.get("available"):
                    logger.info("Falling back to Cloud Vision (Gemini)...")
                    res = self.cloud_provider.analyze(image_bytes, mime_type=mime_type, prompt_extra=prompt_extra)
                    res.backend_used += " (cloud fallback)"
                    return res
                raise e

        elif active_mode == "auto":
            # Check local first
            local_avail = self.local_provider.check_availability()
            if local_avail.get("available"):
                return self.local_provider.analyze(image_bytes, mime_type=mime_type, prompt_extra=prompt_extra)
            else:
                return self.cloud_provider.analyze(image_bytes, mime_type=mime_type, prompt_extra=prompt_extra)

        else:
            raise ValueError(f"Unknown vision backend mode '{active_mode}'. Use 'local', 'cloud', or 'auto'.")


def analyze_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    mode: str = None,
    prompt_extra: str = ""
) -> PersonAnalysisResult:
    """Convenience function for visual person analysis."""
    analyzer = VisionAnalyzer()
    return analyzer.analyze(image_bytes, mime_type=mime_type, mode=mode, prompt_extra=prompt_extra)
