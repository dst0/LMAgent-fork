import tempfile
import unittest
from pathlib import Path
import sys


# test_payload_compat installs lightweight stubs in sys.modules; drop them here so
# this file exercises the real runtime modules even when the suite runs in one process.
for module_name in ("agent_core", "agent_tools"):
    sys.modules.pop(module_name, None)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LMAgent"))

from agent_core import PlanManager, SessionManager, TaskStateManager, TodoManager
from agent_tools import TOOL_SCHEMAS


class CodeManagedStateTests(unittest.TestCase):
    def test_todo_progress_is_advanced_by_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            mgr = TodoManager(workspace, "session")
            mgr.add("first")
            mgr.add("second")

            sync = mgr.advance_after_progress("server progress")
            todos = mgr.list_all()["todos"]

            self.assertEqual(sync["completed"], 1)
            self.assertEqual(sync["activated"], 2)
            self.assertEqual(todos[0]["status"], "completed")
            self.assertEqual(todos[1]["status"], "in_progress")

    def test_plan_and_task_are_finalized_by_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            plan_mgr = PlanManager(workspace, "session")
            plan_mgr.create({
                "title": "Plan",
                "steps": [
                    {"id": "step-1", "description": "One", "status": "pending"},
                    {"id": "step-2", "description": "Two", "status": "pending"},
                ],
            })
            plan_sync = plan_mgr.complete_remaining()

            self.assertTrue(plan_sync["success"])
            self.assertEqual(plan_mgr.plan["status"], "completed")
            self.assertTrue(all(step["status"] == "completed" for step in plan_mgr.plan["steps"]))

            task_mgr = TaskStateManager(workspace, "session")
            task_mgr.sync(
                objective="Ship it",
                total_count=2,
                processed_count=1,
                next_action="Continue",
                remaining_queue=["b.py"],
            )
            task_mgr.mark_complete()

            self.assertEqual(task_mgr.current_state.processed_count, 2)
            self.assertEqual(task_mgr.current_state.remaining_queue, [])
            self.assertEqual(task_mgr.current_state.next_action, "Completed")

    def test_plan_only_keeps_one_active_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            plan_mgr = PlanManager(workspace, "session")
            plan_mgr.create({
                "title": "Plan",
                "steps": [
                    {"id": "step-1", "description": "One", "status": "pending"},
                    {"id": "step-2", "description": "Two", "status": "pending"},
                ],
            })

            plan_mgr.start_step("step-1")
            plan_mgr.start_step("step-2")

            statuses = {step["id"]: step["status"] for step in plan_mgr.plan["steps"]}
            self.assertEqual(statuses["step-1"], "pending")
            self.assertEqual(statuses["step-2"], "in_progress")
            self.assertEqual(plan_mgr.current_step_id, "step-2")

    def test_ai_tool_list_excludes_status_mutation_tools(self):
        names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}

        self.assertNotIn("todo_complete", names)
        self.assertNotIn("todo_update", names)
        self.assertNotIn("plan_complete_step", names)
        self.assertNotIn("task_state_update", names)
        self.assertIn("todo_add", names)
        self.assertIn("todo_list", names)
        self.assertIn("task_state_get", names)

    def test_session_list_includes_persisted_plan_todo_and_task_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            session_mgr = SessionManager(workspace)
            sid = session_mgr.create("Large batch rename")

            todo_mgr = TodoManager(workspace, sid)
            todo_mgr.add("Scan files")
            todo_mgr.add("Rename files")
            todo_mgr.start_next_pending("Started by test")

            plan_mgr = PlanManager(workspace, sid)
            plan_mgr.create({
                "title": "Plan",
                "steps": [
                    {"id": "scan", "description": "Scan", "status": "completed"},
                    {"id": "rename", "description": "Rename", "status": "in_progress"},
                ],
            })
            plan_mgr.start_step("rename")

            task_mgr = TaskStateManager(workspace, sid)
            task_mgr.sync(
                objective="Rename all files",
                total_count=10,
                processed_count=4,
                next_action="Continue renaming",
                remaining_queue=["f5.py", "f6.py"],
            )

            recent = session_mgr.list_recent(1)
            self.assertEqual(len(recent), 1)
            summary = recent[0]

            self.assertEqual(summary["todos"]["total"], 2)
            self.assertEqual(summary["todos"]["completed"], 0)
            self.assertEqual(summary["plan"]["completed"], 1)
            self.assertEqual(summary["plan"]["total"], 2)
            self.assertEqual(summary["task_state"]["processed_count"], 4)
            self.assertEqual(summary["task_state"]["total_count"], 10)


if __name__ == "__main__":
    unittest.main()
