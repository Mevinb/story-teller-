from abc import ABC, abstractmethod


class AgentContract(ABC):
    """Strict multi-agent contract used by the orchestrator."""

    name: str = "agent"

    @abstractmethod
    def run(self, state: dict) -> dict:
        """
        Execute one deterministic agent step.

        Must return:
        {
            "output": ...,
            "confidence": float,
            "next_action": str
        }
        """
        ...
