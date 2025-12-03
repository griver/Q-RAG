"""Utilities for post-processing retriever outputs before LLM answering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


class ChunkFilter:
    """Base class for filtering retriever-selected chunks."""

    def __call__(self, retriever_results: Dict) -> Dict[str, List]:  # pragma: no cover - interface definition
        raise NotImplementedError

    @staticmethod
    def _get_predicted_chunks(retriever_results: Dict) -> Tuple[List[int], List[str]]:
        idx = retriever_results.get("pred_idx") or []
        texts = (
            retriever_results.get("pred_text")
            or retriever_results.get("pred_texts")
            or []
        )
        return list(idx), list(texts)

    @staticmethod
    def _get_support_chunks(retriever_results: Dict) -> Tuple[List[int], List[str]]:
        idx = retriever_results.get("sf_idx") or []
        texts = retriever_results.get("sf_texts") or []
        return list(idx), list(texts)

    @staticmethod
    def _format_output(idx: Sequence[int], texts: Sequence[str]) -> Dict[str, List]:
        return {
            "filtered_idx": list(idx),
            "filtered_texts": list(texts),
        }


@dataclass
class EarlyStopChunkFilter(ChunkFilter):
    """Keep chunks only up until the last support fact encountered."""

    empty_if_no_sf: bool = False

    def __call__(self, retriever_results: Dict) -> Dict[str, List]:
        pred_idx, pred_texts = self._get_predicted_chunks(retriever_results)
        support_idx, _ = self._get_support_chunks(retriever_results)
        support_set = set(int(idx) for idx in support_idx)

        last_support_pos = None
        for pos, idx in enumerate(pred_idx):
            if idx in support_set:
                last_support_pos = pos

        if last_support_pos is None:
            if self.empty_if_no_sf:
                filtered_idx: List[int] = []
                filtered_texts: List[str] = []
            else:
                filtered_idx = pred_idx
                filtered_texts = pred_texts
        else:
            cutoff = last_support_pos + 1
            filtered_idx = pred_idx[:cutoff]
            filtered_texts = pred_texts[:cutoff]

        return self._format_output(filtered_idx, filtered_texts)


@dataclass
class QValueChunkFilter(ChunkFilter):
    """
    Filter chunks by Q-function value.
    If the Q(s_t,a_t) <= stopping_threshold then all chunks starting from a_t are removed.
    """

    stopping_threshold: int = -0.5 #value to never stop retriever

    def __call__(self, retriever_results: Dict) -> Dict[str, List]:
        pred_idx, pred_texts = self._get_predicted_chunks(retriever_results)
        q_values = retriever_results['q_values']

        i = 0
        while i < len(q_values):
            if q_values[i] <= self.stopping_threshold: break
            i += 1

        return self._format_output(pred_idx[:i], pred_texts[:i])




class NoChunkFilter(ChunkFilter):
    """Return retriever-selected chunks unchanged."""

    def __call__(self, retriever_results: Dict) -> Dict[str, List]:
        pred_idx, pred_texts = self._get_predicted_chunks(retriever_results)
        return self._format_output(pred_idx, pred_texts)


class GroundTruthChunkFilter(ChunkFilter):
    """Use only ground-truth support facts for answering."""

    def __call__(self, retriever_results: Dict) -> Dict[str, List]:
        sf_idx, sf_texts = self._get_support_chunks(retriever_results)
        return self._format_output(sf_idx, sf_texts)


class NoNoiseChunkFilter(ChunkFilter):
    """Remove non-support chunks from the retriever output."""

    def __call__(self, retriever_results: Dict) -> Dict[str, List]:
        pred_idx, pred_texts = self._get_predicted_chunks(retriever_results)
        support_idx, _ = self._get_support_chunks(retriever_results)
        support_set = set(support_idx)

        filtered_idx: List[int] = []
        filtered_texts: List[str] = []
        for idx, text in zip(pred_idx, pred_texts):
            if idx in support_set:
                filtered_idx.append(idx)
                filtered_texts.append(text)

        return self._format_output(filtered_idx, filtered_texts)


class LLMChunkFilter(ChunkFilter):
    """Placeholder for LLM-based filtering (not implemented)."""

    def __call__(self, retriever_results: Dict) -> Dict[str, List]:  # pragma: no cover - simple exception
        raise NotImplementedError(
            "LLM-based chunk filtering is not implemented yet."
        )


_FILTERS = {
    "early_stop": EarlyStopChunkFilter,
    "none": NoChunkFilter,
    "gt": GroundTruthChunkFilter,
    "no_noise": NoNoiseChunkFilter,
    "llm": LLMChunkFilter,
    "qvalue": QValueChunkFilter,
}


def build_chunk_filter(name: str, **kwargs) -> ChunkFilter:
    """Factory helper to instantiate a chunk filter by name."""

    try:
        filter_cls = _FILTERS[name]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unknown chunk filter: {name}") from exc
    return filter_cls(**kwargs)


__all__ = [
    "ChunkFilter",
    "EarlyStopChunkFilter",
    "GroundTruthChunkFilter",
    "NoNoiseChunkFilter",
    "NoChunkFilter",
    "LLMChunkFilter",
    "QValueChunkFilter",
    "build_chunk_filter",
]