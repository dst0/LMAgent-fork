import tempfile
import unittest
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LMAgent"))

from agent_core import PlanManager, TaskStateManager, TodoManager
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

    def test_ai_tool_list_excludes_status_mutation_tools(self):
        names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}

        self.assertNotIn("todo_complete", names)
        self.assertNotIn("todo_update", names)
        self.assertNotIn("plan_complete_step", names)
        self.assertNotIn("task_state_update", names)
        self.assertIn("todo_add", names)
        self.assertIn("todo_list", names)
        self.assertIn("task_state_get", names)


if __name__ == "__main__":
    unittest.main()
