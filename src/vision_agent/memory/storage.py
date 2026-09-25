"""Relational Storage Subsystem for SIVAC Phase 5 (SQLite + SQLAlchemy 2.0).

Provides transactional audit logging of tasks, trajectory steps, raw actions,
verifications, post-task reflections, and learned heuristic skills.
"""

import json
import uuid
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Generator

from sqlalchemy import (
    create_engine,
    event,
    Column,
    String,
    Integer,
    Float,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
    select,
    update,
    delete,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    relationship,
    sessionmaker,
    Session,
)

from ..config import settings
from .schema import (
    TaskStatus,
    TaskRecord,
    TrajectoryRecord,
    StepRecord,
    VerificationRecord,
    ReflectionRecord,
    LearnedSkill,
)

logger = logging.getLogger("sivac.memory.storage")


# ---------------------------------------------------------------------------
# SQLAlchemy 2.0 Declarative Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class TaskModel(Base):
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    goal = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default=TaskStatus.PENDING.value)
    total_steps = Column(Integer, nullable=False, default=0)
    total_duration_seconds = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    trajectories = relationship("TrajectoryModel", back_populates="task", cascade="all, delete-orphan")
    reflection = relationship("ReflectionModel", back_populates="task", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_tasks_status", "status"),
    )


class TrajectoryModel(Base):
    __tablename__ = "trajectories"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String(36), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    model_used = Column(String(100), nullable=False, default="default")
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    estimated_cost_usd = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    task = relationship("TaskModel", back_populates="trajectories")
    steps = relationship("StepModel", back_populates="trajectory", cascade="all, delete-orphan", order_by="StepModel.step_number")


class StepModel(Base):
    __tablename__ = "steps"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    trajectory_id = Column(String(36), ForeignKey("trajectories.id", ondelete="CASCADE"), nullable=False)
    step_number = Column(Integer, nullable=False)
    screenshot_before = Column(Text, nullable=False, default="")
    screenshot_after = Column(Text, nullable=True)
    observation_elements = Column(Text, nullable=True)  # JSON string
    planned_subgoal = Column(Text, nullable=False, default="")
    action_command = Column(Text, nullable=False, default="{}")  # JSON string
    execution_result = Column(Text, nullable=False, default="{}")  # JSON string
    executed_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    trajectory = relationship("TrajectoryModel", back_populates="steps")
    verification = relationship("VerificationModel", back_populates="step", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("trajectory_id", "step_number", name="uq_trajectory_step"),
        Index("idx_steps_trajectory", "trajectory_id", "step_number"),
    )


class VerificationModel(Base):
    __tablename__ = "verifications"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    step_id = Column(String(36), ForeignKey("steps.id", ondelete="CASCADE"), nullable=False, unique=True)
    passed = Column(Integer, nullable=False, default=1)  # 1 = True, 0 = False
    visual_diff_score = Column(Float, nullable=False, default=0.0)
    dom_delta = Column(Text, nullable=True)  # JSON string
    failure_reason = Column(Text, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)

    # Relationships
    step = relationship("StepModel", back_populates="verification")


class ReflectionModel(Base):
    __tablename__ = "reflections"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String(36), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, unique=True)
    overall_success = Column(Integer, nullable=False, default=1)  # 1 = True, 0 = False
    root_cause_analysis = Column(Text, nullable=False, default="")
    friction_points = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    task = relationship("TaskModel", back_populates="reflection")
    learned_skills = relationship("LearnedSkillModel", back_populates="reflection")


class LearnedSkillModel(Base):
    __tablename__ = "learned_skills"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    reflection_id = Column(String(36), ForeignKey("reflections.id", ondelete="SET NULL"), nullable=True)
    application_name = Column(String(100), nullable=False)
    trigger_condition = Column(Text, nullable=False)
    heuristic_rule = Column(Text, nullable=False)
    success_count = Column(Integer, nullable=False, default=1)
    failure_count = Column(Integer, nullable=False, default=0)
    confidence_score = Column(Float, nullable=False, default=0.8)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    reflection = relationship("ReflectionModel", back_populates="learned_skills")

    __table_args__ = (
        Index("idx_skills_app", "application_name"),
    )


# ---------------------------------------------------------------------------
# Database Manager
# ---------------------------------------------------------------------------

class DatabaseManager:
    """Encapsulates engine creation, connection pooling, and CRUD operations."""

    def __init__(self, db_url: Optional[str] = None) -> None:
        self.db_url = db_url or settings.sqlite_db_url
        self.engine = create_engine(
            self.db_url,
            connect_args={"check_same_thread": False} if "sqlite" in self.db_url else {},
            echo=False,
        )

        # Enforce foreign keys and WAL mode on SQLite
        if "sqlite" in self.db_url:
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()

        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.init_db()

    def init_db(self) -> None:
        """Create all tables and indexes if they do not exist."""
        Base.metadata.create_all(self.engine)
        logger.info(f"Database tables initialized at: {self.db_url}")

    def close(self) -> None:
        """Dispose the SQLAlchemy engine and close active connections."""
        if self.engine:
            self.engine.dispose()
            logger.debug("Database engine connection pool disposed")

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session rolled back due to error: {e}")
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Task Operations
    # ------------------------------------------------------------------

    def create_task(self, goal: str, task_id: Optional[str] = None) -> TaskRecord:
        """Create and persist a new task session."""
        tid = task_id or str(uuid.uuid4())
        with self.session_scope() as session:
            task = TaskModel(id=tid, goal=goal, status=TaskStatus.RUNNING.value)
            session.add(task)
            session.flush()
            return TaskRecord(
                id=task.id,
                goal=task.goal,
                status=TaskStatus(task.status),
                total_steps=task.total_steps,
                total_duration_seconds=task.total_duration_seconds,
                created_at=task.created_at,
            )

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        total_steps: int = 0,
        total_duration_seconds: float = 0.0,
    ) -> Optional[TaskRecord]:
        """Update task completion status, steps, and duration."""
        with self.session_scope() as session:
            task = session.get(TaskModel, task_id)
            if not task:
                return None
            task.status = status.value
            task.total_steps = total_steps
            task.total_duration_seconds = total_duration_seconds
            if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                task.completed_at = datetime.utcnow()
            session.flush()
            return TaskRecord(
                id=task.id,
                goal=task.goal,
                status=TaskStatus(task.status),
                total_steps=task.total_steps,
                total_duration_seconds=task.total_duration_seconds,
                created_at=task.created_at,
                completed_at=task.completed_at,
            )

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Retrieve task by ID."""
        with self.session_scope() as session:
            task = session.get(TaskModel, task_id)
            if not task:
                return None
            return TaskRecord(
                id=task.id,
                goal=task.goal,
                status=TaskStatus(task.status),
                total_steps=task.total_steps,
                total_duration_seconds=task.total_duration_seconds,
                created_at=task.created_at,
                completed_at=task.completed_at,
            )

    def list_tasks(self, limit: int = 20, status: Optional[TaskStatus] = None) -> List[TaskRecord]:
        """List recent tasks, optionally filtered by status."""
        with self.session_scope() as session:
            stmt = select(TaskModel).order_by(TaskModel.created_at.desc()).limit(limit)
            if status:
                stmt = stmt.where(TaskModel.status == status.value)
            tasks = session.scalars(stmt).all()
            return [
                TaskRecord(
                    id=t.id,
                    goal=t.goal,
                    status=TaskStatus(t.status),
                    total_steps=t.total_steps,
                    total_duration_seconds=t.total_duration_seconds,
                    created_at=t.created_at,
                    completed_at=t.completed_at,
                )
                for t in tasks
            ]

    def delete_task(self, task_id: str) -> bool:
        """Delete task and all associated trajectories/steps via cascade."""
        with self.session_scope() as session:
            task = session.get(TaskModel, task_id)
            if task:
                session.delete(task)
                return True
            return False

    # ------------------------------------------------------------------
    # Trajectory & Step Operations
    # ------------------------------------------------------------------

    def create_trajectory(
        self,
        task_id: str,
        model_used: str = "default",
        trajectory_id: Optional[str] = None,
    ) -> TrajectoryRecord:
        """Create a new trajectory entry for a task."""
        tr_id = trajectory_id or str(uuid.uuid4())
        with self.session_scope() as session:
            traj = TrajectoryModel(
                id=tr_id,
                task_id=task_id,
                model_used=model_used,
            )
            session.add(traj)
            session.flush()
            return TrajectoryRecord(
                id=traj.id,
                task_id=traj.task_id,
                model_used=traj.model_used,
                prompt_tokens=traj.prompt_tokens,
                completion_tokens=traj.completion_tokens,
                estimated_cost_usd=traj.estimated_cost_usd,
                created_at=traj.created_at,
            )

    def get_trajectory(self, task_id: str) -> Optional[TrajectoryRecord]:
        """Get the primary trajectory for a task."""
        with self.session_scope() as session:
            stmt = select(TrajectoryModel).where(TrajectoryModel.task_id == task_id).order_by(TrajectoryModel.created_at.desc())
            traj = session.scalars(stmt).first()
            if not traj:
                return None
            return TrajectoryRecord(
                id=traj.id,
                task_id=traj.task_id,
                model_used=traj.model_used,
                prompt_tokens=traj.prompt_tokens,
                completion_tokens=traj.completion_tokens,
                estimated_cost_usd=traj.estimated_cost_usd,
                created_at=traj.created_at,
            )

    def add_step(
        self,
        trajectory_id: str,
        step_number: int,
        screenshot_before: str = "",
        screenshot_after: Optional[str] = None,
        observation_elements: Optional[List[Dict[str, Any]]] = None,
        planned_subgoal: str = "",
        action_command: Optional[Dict[str, Any]] = None,
        execution_result: Optional[Dict[str, Any]] = None,
    ) -> StepRecord:
        """Persist a single execution step to the trajectory audit log."""
        step_id = str(uuid.uuid4())
        obs_json = json.dumps(observation_elements) if observation_elements else None
        cmd_json = json.dumps(action_command or {})
        res_json = json.dumps(execution_result or {})

        with self.session_scope() as session:
            step = StepModel(
                id=step_id,
                trajectory_id=trajectory_id,
                step_number=step_number,
                screenshot_before=screenshot_before,
                screenshot_after=screenshot_after,
                observation_elements=obs_json,
                planned_subgoal=planned_subgoal,
                action_command=cmd_json,
                execution_result=res_json,
            )
            session.add(step)
            session.flush()
            return StepRecord(
                id=step.id,
                trajectory_id=step.trajectory_id,
                step_number=step.step_number,
                screenshot_before=step.screenshot_before,
                screenshot_after=step.screenshot_after,
                observation_elements=observation_elements,
                planned_subgoal=step.planned_subgoal,
                action_command=action_command or {},
                execution_result=execution_result or {},
                executed_at=step.executed_at,
            )

    def add_verification(
        self,
        step_id: str,
        passed: bool,
        visual_diff_score: float = 0.0,
        dom_delta: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None,
        retry_count: int = 0,
    ) -> VerificationRecord:
        """Persist verification results linked to a step."""
        verif_id = str(uuid.uuid4())
        delta_json = json.dumps(dom_delta) if dom_delta else None

        with self.session_scope() as session:
            verif = VerificationModel(
                id=verif_id,
                step_id=step_id,
                passed=1 if passed else 0,
                visual_diff_score=visual_diff_score,
                dom_delta=delta_json,
                failure_reason=failure_reason,
                retry_count=retry_count,
            )
            session.add(verif)
            session.flush()
            return VerificationRecord(
                id=verif.id,
                step_id=verif.step_id,
                passed=bool(verif.passed),
                visual_diff_score=verif.visual_diff_score,
                dom_delta=dom_delta,
                failure_reason=verif.failure_reason,
                retry_count=verif.retry_count,
            )

    def get_steps_for_trajectory(self, trajectory_id: str) -> List[StepRecord]:
        """Retrieve ordered steps for a trajectory."""
        with self.session_scope() as session:
            stmt = select(StepModel).where(StepModel.trajectory_id == trajectory_id).order_by(StepModel.step_number)
            steps = session.scalars(stmt).all()
            result = []
            for s in steps:
                obs = json.loads(s.observation_elements) if s.observation_elements else None
                cmd = json.loads(s.action_command) if s.action_command else {}
                res = json.loads(s.execution_result) if s.execution_result else {}
                result.append(
                    StepRecord(
                        id=s.id,
                        trajectory_id=s.trajectory_id,
                        step_number=s.step_number,
                        screenshot_before=s.screenshot_before,
                        screenshot_after=s.screenshot_after,
                        observation_elements=obs,
                        planned_subgoal=s.planned_subgoal,
                        action_command=cmd,
                        execution_result=res,
                        executed_at=s.executed_at,
                    )
                )
            return result

    # ------------------------------------------------------------------
    # Learned Skills / Heuristics Operations
    # ------------------------------------------------------------------

    def save_learned_skill(
        self,
        application_name: str,
        trigger_condition: str,
        heuristic_rule: str,
        reflection_id: Optional[str] = None,
        confidence_score: float = 0.8,
    ) -> LearnedSkill:
        """Persist a new learned operational heuristic rule."""
        skill_id = str(uuid.uuid4())
        with self.session_scope() as session:
            skill = LearnedSkillModel(
                id=skill_id,
                reflection_id=reflection_id,
                application_name=application_name,
                trigger_condition=trigger_condition,
                heuristic_rule=heuristic_rule,
                confidence_score=confidence_score,
            )
            session.add(skill)
            session.flush()
            return LearnedSkill(
                id=skill.id,
                reflection_id=skill.reflection_id,
                application_name=skill.application_name,
                trigger_condition=skill.trigger_condition,
                heuristic_rule=skill.heuristic_rule,
                success_count=skill.success_count,
                failure_count=skill.failure_count,
                confidence_score=skill.confidence_score,
                created_at=skill.created_at,
                updated_at=skill.updated_at,
            )

    def get_learned_skills(
        self, application_name: Optional[str] = None, limit: int = 50
    ) -> List[LearnedSkill]:
        """Fetch learned skills, optionally filtered by application name."""
        with self.session_scope() as session:
            stmt = select(LearnedSkillModel).order_by(LearnedSkillModel.confidence_score.desc()).limit(limit)
            if application_name:
                stmt = stmt.where(LearnedSkillModel.application_name.ilike(f"%{application_name}%"))
            skills = session.scalars(stmt).all()
            return [
                LearnedSkill(
                    id=s.id,
                    reflection_id=s.reflection_id,
                    application_name=s.application_name,
                    trigger_condition=s.trigger_condition,
                    heuristic_rule=s.heuristic_rule,
                    success_count=s.success_count,
                    failure_count=s.failure_count,
                    confidence_score=s.confidence_score,
                    created_at=s.created_at,
                    updated_at=s.updated_at,
                )
                for s in skills
            ]

    def update_skill_stats(self, skill_id: str, success: bool) -> Optional[LearnedSkill]:
        """Reinforce or penalize a skill based on task outcome."""
        with self.session_scope() as session:
            skill = session.get(LearnedSkillModel, skill_id)
            if not skill:
                return None
            if success:
                skill.success_count += 1
                skill.confidence_score = round(min(1.0, skill.confidence_score + 0.05), 4)
            else:
                skill.failure_count += 1
                skill.confidence_score = round(max(0.1, skill.confidence_score - 0.10), 4)
            skill.updated_at = datetime.utcnow()
            session.flush()
            return LearnedSkill(
                id=skill.id,
                reflection_id=skill.reflection_id,
                application_name=skill.application_name,
                trigger_condition=skill.trigger_condition,
                heuristic_rule=skill.heuristic_rule,
                success_count=skill.success_count,
                failure_count=skill.failure_count,
                confidence_score=skill.confidence_score,
                created_at=skill.created_at,
                updated_at=skill.updated_at,
            )
