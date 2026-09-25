"""Data Schemas and Transfer Models for SIVAC Phase 5 Memory Subsystem.

Defines strongly-typed Pydantic models for episodic trajectories, structured
task execution steps, verifications, reflections, and learned heuristics.
"""

from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Execution status for high-level user tasks."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskRecord(BaseModel):
    """Relational model representation of a high-level task session."""
    id: str
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    total_steps: int = 0
    total_duration_seconds: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class TrajectoryRecord(BaseModel):
    """Model usage, token metrics, and cost metadata for a task trajectory."""
    id: str
    task_id: str
    model_used: str = "default"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class StepRecord(BaseModel):
    """Detailed audit record of a single step within a task trajectory."""
    id: str
    trajectory_id: str
    step_number: int
    screenshot_before: str = ""
    screenshot_after: Optional[str] = None
    observation_elements: Optional[List[Dict[str, Any]]] = None
    planned_subgoal: str = ""
    action_command: Dict[str, Any] = Field(default_factory=dict)
    execution_result: Dict[str, Any] = Field(default_factory=dict)
    executed_at: datetime = Field(default_factory=datetime.utcnow)


class VerificationRecord(BaseModel):
    """Audit record of the outcome verification for a single step."""
    id: str
    step_id: str
    passed: bool
    visual_diff_score: float = 0.0
    dom_delta: Optional[Dict[str, Any]] = None
    failure_reason: Optional[str] = None
    retry_count: int = 0


class ReflectionRecord(BaseModel):
    """Post-task cognitive reflection evaluating success and friction points."""
    id: str
    task_id: str
    overall_success: bool
    root_cause_analysis: str
    friction_points: Optional[List[str]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LearnedSkill(BaseModel):
    """An abstracted IF [Condition] THEN [Strategy] heuristic rule."""
    id: str
    reflection_id: Optional[str] = None
    application_name: str
    trigger_condition: str
    heuristic_rule: str
    success_count: int = 1
    failure_count: int = 0
    confidence_score: float = 0.8
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Retrieval Data Models
# ---------------------------------------------------------------------------

class SimilarTaskResult(BaseModel):
    """Semantic vector search match for a past task demonstration."""
    task_id: str
    goal: str
    application: str = "Desktop"
    status: str = "COMPLETED"
    total_steps: int = 0
    duration_seconds: float = 0.0
    similarity_score: float = 0.0


class HeuristicQueryResult(BaseModel):
    """Semantic vector search match for an application heuristic rule."""
    skill_id: str
    application_name: str
    trigger_condition: str
    heuristic_rule: str
    confidence_score: float = 0.8
    similarity_score: float = 0.0


class RetrievedExperience(BaseModel):
    """Dual-retrieval context container combining past tasks and heuristics."""
    query_goal: str
    target_application: str
    similar_tasks: List[SimilarTaskResult] = Field(default_factory=list)
    heuristics: List[HeuristicQueryResult] = Field(default_factory=list)
