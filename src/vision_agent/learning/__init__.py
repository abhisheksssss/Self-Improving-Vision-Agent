"""Phase 7: Self-Improvement & Reflection Subsystem for SIVAC.

Provides post-task cognitive reflection, heuristic lesson extraction, and
adaptive strategy learning:
    - evaluator         : Trajectory efficiency scoring & friction detection
    - reflection        : LLM cognitive post-mortem analysis
    - lesson_extractor  : Reusable IF [Condition] THEN [Strategy] rule synthesis
    - strategy          : Modality prioritization & feedback reinforcement
    - self_improver     : Unified post-task self-improvement coordinator
"""

from .schema import (
    TrajectoryEvaluation,
    ExtractedLesson,
    ReflectionOutput,
)
from .evaluator import TrajectoryEvaluator
from .reflection import ReflectionEngine
from .lesson_extractor import LessonExtractor
from .strategy import StrategyManager
from .self_improver import SelfImprovementPipeline

__all__ = [
    "TrajectoryEvaluation",
    "ExtractedLesson",
    "ReflectionOutput",
    "TrajectoryEvaluator",
    "ReflectionEngine",
    "LessonExtractor",
    "StrategyManager",
    "SelfImprovementPipeline",
]
