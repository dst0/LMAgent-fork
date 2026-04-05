import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LMAgent"))

from session_inspect import (
    build_readable_history,
    build_versioned_status_payload,
    compute_status_version,
    load_session_inspect_payload,
)


class _TaskState:
    def __init__(self, value):
        self._value = value

    def to_dict(self):
        return self._value


class _TodoManager:
    def __init__(self, workspace, sid):
        self.workspace = workspace
        self.sid = sid

    def list_all(self):
        return {"todos": [{"id": 1, "description": "Inspect", "status": "completed"}], "completed": 1, "total": 1}


class _TaskStateManager:
    def __init__(self, workspace, sid):
        self.workspace = workspace
        self.sid = sid

    def load(self):
        return _TaskState({"objective": "Ship", "processed_count": 1, "total_count": 1})


class _PlanManager:
    def __init__(self, workspace, sid):
        self.workspace = workspace
        self.sid = sid
        self.plan = {"title": "Plan", "steps": [{"id": "step-1", "status": "completed"}]}
        self.current_step_id = "step-1"


class _SessionManager:
    def __init__(self, workspace):
        self.workspace = workspace

    def load(self, sid):
        return (
            [
                {"role": "user", "content": "Fix it"},
                {"role": "assistant", "content": "<thinking>hidden</thinking>Done"},
                {"role": "tool", "name": "bash", "content": "{\"success\": false}"},
            ],
            {"name": sid},
        )


class _MissingSessionManager(_SessionManager):
    def load(self, sid):
        return None


class SessionInspectTests(unittest.TestCase):
    def test_build_readable_history_filters_roles_and_trims_hidden_content(self):
        readable = build_readable_history(
            [
                {"role": "user", "content": "User request"},
                {"role": "assistant", "content": "<thinking>secret</thinking>Visible answer"},
                {"role": "tool", "name": "bash", "content": "{\"success\": false}"},
                {"role": "assistant", "content": "   "},
            ],
            lambda content: (content.replace("<thinking>secret</thinking>", ""), ""),
        )

        self.assertEqual(
            readable,
            [
                {"role": "user", "content": "User request"},
                {"role": "assistant", "content": "Visible answer"},
                {"role": "tool", "name": "bash", "success": False},
            ],
        )

    def test_load_session_inspect_payload_combines_all_sections(self):
        payload = load_session_inspect_payload(
            workspace=Path("/tmp/workspace"),
            sid="session-1",
            todo_manager_cls=_TodoManager,
            task_state_manager_cls=_TaskStateManager,
            plan_manager_cls=_PlanManager,
            session_manager_cls=_SessionManager,
            strip_thinking_func=lambda content: (content.replace("<thinking>hidden</thinking>", ""), ""),
        )

        self.assertEqual(payload["todos"]["total"], 1)
        self.assertEqual(payload["tasks"]["state"]["objective"], "Ship")
        self.assertEqual(payload["plan"]["current_step_id"], "step-1")
        self.assertEqual(payload["history"]["metadata"]["name"], "session-1")
        self.assertEqual(payload["history"]["messages"][1]["content"], "Done")
        self.assertFalse(payload["history"]["messages"][2]["success"])

    def test_load_session_inspect_payload_raises_when_session_is_missing(self):
        with self.assertRaises(LookupError):
            load_session_inspect_payload(
                workspace=Path("/tmp/workspace"),
                sid="session-1",
                todo_manager_cls=_TodoManager,
                task_state_manager_cls=_TaskStateManager,
                plan_manager_cls=_PlanManager,
                session_manager_cls=_MissingSessionManager,
                strip_thinking_func=lambda content: (content, ""),
            )

    def test_build_versioned_status_payload_returns_no_changes_for_same_version(self):
        payload = {"todos": {"total": 1}, "history": {"messages": []}}
        version = compute_status_version(payload)

        versioned = build_versioned_status_payload(payload, version, sid="session-1")

        self.assertEqual(
            versioned,
            {
                "status": "no_changes",
                "changed": False,
                "version": version,
                "session_id": "session-1",
            },
        )

    def test_build_versioned_status_payload_includes_full_payload_for_new_version(self):
        payload = {"todos": {"total": 1}, "history": {"messages": []}}

        versioned = build_versioned_status_payload(payload, "-1", sid="session-1")

        self.assertTrue(versioned["changed"])
        self.assertEqual(versioned["status"], "ok")
        self.assertEqual(versioned["session_id"], "session-1")
        self.assertEqual(versioned["todos"]["total"], 1)
        self.assertIn("version", versioned)


if __name__ == "__main__":
    unittest.main()
