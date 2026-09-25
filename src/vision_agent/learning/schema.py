"""Data Schemas and Transfer Models for SIVAC Phase 7 Learning Subsystem.

Defines strongly-typed Pydantic models for trajectory evaluations, cognitive
post-task reflections, and extracted operational heuristic lessons.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TrajectoryEvaluation(BaseModel):
    """Diagnostic evaluation metrics assessing trajectory efficiency and friction."""
    task_id: str
    overall_efficiency: float = 1.0  # 0.0 to 1.0
    total_steps: int = 0
    successful_steps: int = 0
    recovery_count: int = 0
    friction_points: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class ExtractedLesson(BaseModel):
    """An atomic operational heuristic rule synthesized from task experience."""
    application_name: str
    trigger_condition: str
    heuristic_rule: str
    confidence_score: float = 0.80
    rationale: str = ""


class ReflectionOutput(BaseModel):
    """Complete cognitive post-mortem analysis of a completed task."""
    task_id: str
    overall_success: bool
    root_cause_analysis: str = ""
    friction_points: List[str] = Field(default_factory=list)
    lessons: List[ExtractedLesson] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
