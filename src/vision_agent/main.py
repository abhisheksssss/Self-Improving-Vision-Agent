"""SIVAC Interactive CLI & Execution Entry Point.

Provides command-line interaction with the autonomous computer-use vision agent:
- Run individual task instructions via CLI.
- Launch an interactive terminal session.
- Stream step-by-step perception, planning, safety, action, and verification.
- Inspect dual-layer memory (past tasks & learned operational heuristics).
"""

import sys
import argparse
import logging
from typing import Optional

from .agent import SIVACAgent, AgentState
from .actions.schema import ActionType
from .memory import DatabaseManager, SemanticMemoryManager, ChromaVectorStore
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def print_banner():
    banner = r"""
  ____ _____     ___    ____ 
 / ___|_ _\ \   / / \  / ___|
 \___ \| | \ \ / / _ \| |    
  ___) | |  \ V / ___ \ |___ 
 |____/___|  \_/_/   \_\____|
  Autonomous Self-Improving Vision Agent
"""
    print(banner)
    print("  Controls: Press [Ctrl+Alt+Esc] anytime for Emergency Kill-Switch")
    print("  Data Dir: " + settings.DATA_DIR)
    print("=" * 64 + "\n")


def format_node_update(node_name: str, state_update: dict):
    """Format and print real-time node state transitions."""
    if node_name == "initialize":
        tid = state_update.get("task_id", "")
        print(f"\n[INIT] Task allocated session ID: {tid[:8]}...")

    elif node_name == "retrieve":
        guidance = state_update.get("retrieved_guidance", "")
        if guidance:
            print("[MEMORY] Retrieved prior experience and heuristics for this context")
        else:
            print("[MEMORY] Starting with clean slate (no previous matching heuristics)")

    elif node_name == "perceive":
        ui = state_update.get("current_ui_state")
        elem_count = len(ui.elements) if ui and ui.elements else 0
        app = ui.application if ui else "Desktop"
        shot = ui.screenshot_path if ui else None
        shot_info = f" | Screenshot: {shot.split('/')[-1] if shot else 'none'}"
        print(f"[PERCEIVE] Application: {app} | Detected {elem_count} interactive elements{shot_info}")

    elif node_name == "plan":
        subgoal = state_update.get("planned_subgoal", "")
        cmd = state_update.get("current_action")
        if cmd:
            action_desc = f"{cmd.action.value.upper()}"
            if cmd.target_id:
                action_desc += f" (target='{cmd.target_id}')"
            elif cmd.x is not None and cmd.y is not None:
                action_desc += f" at ({cmd.x}, {cmd.y})"
            if cmd.text:
                action_desc += f" text='{cmd.text}'"
            if cmd.key:
                action_desc += f" key='{cmd.key}'"
            if cmd.url:
                action_desc += f" url='{cmd.url}'"
            if cmd.app_name:
                action_desc += f" app='{cmd.app_name}'"
            reason_short = (cmd.reason or "")[:80]
            print(f"[PLAN] Subgoal: \"{subgoal}\" -> {action_desc}")
            if reason_short:
                print(f"       Reason: {reason_short}")

    elif node_name == "validate_safety":
        is_safe = state_update.get("is_safe", True)
        if is_safe:
            print("[SAFETY] Guardrails approved planned action")
        else:
            reason = state_update.get("safety_reason", "Safety violation")
            print(f"[SAFETY BLOCKED] {reason}")

    elif node_name == "act":
        res = state_update.get("last_execution_result")
        if res:
            status = "SUCCESS" if res.success else "FAILED"
            print(f"[EXECUTE] Action dispatched ({status} in {res.execution_time_ms:.1f}ms)")

    elif node_name == "verify":
        v_res = state_update.get("last_verification_result")
        if v_res:
            if v_res.passed:
                print(f"[VERIFY] Outcome verified: PASSED (delta visual={v_res.visual_diff_score:.2f})")
            else:
                print(f"[VERIFY] Outcome check FAILED: mode='{v_res.failure_mode.value}' - {v_res.details}")

    elif node_name == "recover":
        plan = state_update.get("recovery_plan")
        if plan:
            print(f"[RECOVER] Injected {len(plan.steps)} corrective steps for mode '{plan.failure_mode.value}'")

    elif node_name == "finalize":
        complete = state_update.get("is_complete", False)
        status = "COMPLETED" if complete else "TERMINATED / FAILED"
        print(f"\n[FINALIZE] Task {status}")
        print("[SELF-IMPROVE] Trajectory evaluated, reflection stored, operational rules indexed.")


def run_single_task(
    goal: str,
    application: str = "Desktop",
    dry_run: bool = False,
    max_steps: int = 25,
):
    """Execute a task with streaming feedback."""
    mode_str = "DRY-RUN (Simulated)" if dry_run else "LIVE (Direct Computer Control)"
    print(f"\nGoal       : \"{goal}\"")
    print(f"Application: {application}")
    print(f"Mode       : {mode_str}")
    print(f"Max Steps  : {max_steps}")
    print("-" * 64)

    agent = SIVACAgent(dry_run=dry_run)

    step_count = 0
    final_state = None

    try:
        for update in agent.stream(
            goal=goal,
            application=application,
            max_steps=max_steps,
        ):
            for node_name, partial_state in update.items():
                format_node_update(node_name, partial_state)
                final_state = partial_state
                if "step_count" in partial_state:
                    step_count = partial_state["step_count"]

        print("\n" + "=" * 64)
        print(f"Execution finished in {step_count} step(s).")
        print("=" * 64)

    except KeyboardInterrupt:
        print("\n[ABORTED] Task cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] Task execution failed: {e}")


def interactive_session(dry_run: bool = False):
    """Interactive continuous REPL loop."""
    print_banner()
    mode_str = "DRY-RUN (Simulated)" if dry_run else "LIVE (Direct Mouse & Keyboard Control)"
    print(f"Session started in {mode_str} mode.")
    print("Type your instruction and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            goal = input("\nsivac> ").strip()
            if not goal:
                continue
            if goal.lower() in ("exit", "quit", "q"):
                print("Exiting SIVAC. Goodbye!")
                break
            if goal.lower() == "skills":
                show_skills()
                continue
            if goal.lower() == "history":
                show_history()
                continue

            app = "Chrome" if ("http" in goal or "browser" in goal.lower()) else "Desktop"
            run_single_task(goal=goal, application=app, dry_run=dry_run)

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def show_skills():
    """List all operational heuristics stored in memory."""
    db = DatabaseManager()
    skills = db.get_learned_skills(limit=25)
    db.close()

    print("\n" + "=" * 64)
    print(f"  SIVAC LEARNED OPERATIONAL HEURISTICS ({len(skills)} stored)")
    print("=" * 64)

    if not skills:
        print("  No operational heuristics recorded yet.")
        print("  Run tasks to have SIVAC reflect and extract operational rules automatically!")
    else:
        for idx, s in enumerate(skills, 1):
            print(f"\n[{idx}] App: {s.application_name} | Confidence: {s.confidence_score:.2f} (Success: {s.success_count}, Fail: {s.failure_count})")
            print(f"    Trigger : {s.trigger_condition}")
            print(f"    Strategy: {s.heuristic_rule}")
    print("\n" + "=" * 64)


def show_history():
    """List recent completed tasks and reflections."""
    db = DatabaseManager()
    from .memory.storage import TaskModel, ReflectionModel
    with db.session_scope() as session:
        tasks = session.query(TaskModel).order_by(TaskModel.created_at.desc()).limit(10).all()
        print("\n" + "=" * 64)
        print(f"  SIVAC TASK HISTORY (Last {len(tasks)} tasks)")
        print("=" * 64)

        if not tasks:
            print("  No tasks executed yet.")
        else:
            for t in tasks:
                status_str = t.status.upper() if t.status else "UNKNOWN"
                steps = t.total_steps or 0
                print(f"- [{status_str}] \"{t.goal}\" (Steps: {steps}, Date: {t.created_at.strftime('%Y-%m-%d %H:%M')})")
                if t.reflection:
                    refl = t.reflection
                    print(f"  Reflection: {refl.root_cause_analysis[:100]}...")
    db.close()
    print("\n" + "=" * 64)


def main():
    parser = argparse.ArgumentParser(
        description="SIVAC — Self-Improving Vision Agent for Autonomous Computer Use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a single task instruction")
    run_parser.add_argument("goal", type=str, help="Natural language goal instruction")
    run_parser.add_argument("--app", type=str, default="Desktop", help="Target application name (default: Desktop)")
    run_parser.add_argument("--max-steps", type=int, default=25, help="Maximum execution steps (default: 25)")
    run_parser.add_argument("--dry-run", action="store_true", help="Simulate execution without moving cursor/typing")
    run_parser.add_argument("--live", dest="dry_run", action="store_false", help="Direct live mouse and keyboard control")
    run_parser.set_defaults(dry_run=False)

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive prompt session")
    interactive_parser.add_argument("--dry-run", action="store_true", help="Run session in simulated mode")
    interactive_parser.add_argument("--live", dest="dry_run", action="store_false", help="Run session in live control mode")
    interactive_parser.set_defaults(dry_run=False)

    # Skills command
    subparsers.add_parser("skills", help="Display learned operational heuristics")

    # History command
    subparsers.add_parser("history", help="Display recent task execution history")

    args = parser.parse_args()

    if args.command == "run":
        print_banner()
        run_single_task(
            goal=args.goal,
            application=args.app,
            dry_run=args.dry_run,
            max_steps=args.max_steps,
        )
    elif args.command == "interactive":
        interactive_session(dry_run=args.dry_run)
    elif args.command == "skills":
        show_skills()
    elif args.command == "history":
        show_history()
    else:
        # Default: if no command is specified, launch interactive session
        interactive_session(dry_run=False)


if __name__ == "__main__":
    main()
