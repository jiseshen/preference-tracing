from dataclasses import dataclass
from typing import Optional
from model import BaseModel, GenerationConfig
from .hypothesis_set import HypothesisSet, WorkingBelief


@dataclass
class TracerConfig:
    n_hypotheses: int = 5
    importance: float = 0.5
    consolidate_alpha: float = 0.5
    similarity_threshold: float = 0.8
    perturb_alpha: float = 0.3
    max_history_turns: int = 3


@dataclass
class TracerContext:
    model: BaseModel
    hypothesis_set: HypothesisSet
    current_belief: Optional[WorkingBelief] = None
    tracer_config: TracerConfig = TracerConfig()
    generation_config: Optional[GenerationConfig] = GenerationConfig()