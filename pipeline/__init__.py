"""
Pipeline package.

Imports are lazy so that `pipeline.errors` can be imported from `models`
without triggering a circular import through the (heavy) orchestrator.
"""
__all__ = ["PipelineOrchestrator"]


def __getattr__(name):
    if name == "PipelineOrchestrator":
        from .orchestrator import PipelineOrchestrator
        return PipelineOrchestrator
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")
