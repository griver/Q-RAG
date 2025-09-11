from abc import ABC, abstractmethod


class AnswerMetric(ABC):
    """Abstract base class for answer quality metrics."""

    @abstractmethod
    def __call__(self, prediction: str, target: str) -> float:
        """Compute metric value for *prediction* against *target*."""
        raise NotImplementedError