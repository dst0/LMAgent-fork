import sys
import types
import unittest
from pathlib import Path


if "agent_core" not in sys.modules:
    stub = types.ModuleType("agent_core")

    class _Config:
        TEMPERATURE = 0.65
        LLM_MODEL = ""
        MAX_TOKENS = -1
        THINKING_MODEL = False
        THINKING_MAX_TOKENS = 16000

    class _Log:
        @staticmethod
        def warning(*args, **kwargs):
            return None

    stub.Config = _Config
    stub.Colors = type("Colors", (), {})
    stub.Log = _Log
    stub.PermissionMode = type("PermissionMode", (), {})
    stub.truncate_output = lambda text, *args, **kwargs: text
    stub.strip_thinking = lambda content: (content, "")
    stub.get_current_context = lambda: {}
    stub.set_current_context = lambda *args, **kwargs: None
    stub._get_ctx = lambda: {}
    stub.TodoManager = type("TodoManager", (), {})
    stub.PlanManager = type("PlanManager", (), {})
    stub.TaskStateManager = type("TaskStateManager", (), {})
    stub.SessionManager = type("SessionManager", (), {})
    stub.LoopDetector = type("LoopDetector", (), {})
    stub.colored = lambda text, *args, **kwargs: text
    sys.modules["agent_core"] = stub

if "agent_tools" not in sys.modules:
    stub = types.ModuleType("agent_tools")
    stub.TOOL_SCHEMAS = []
    stub.TOOL_HANDLERS = {}
    stub.get_available_tools = lambda *args, **kwargs: []
    stub._REQUIRED_ARG_TOOLS = set()
    stub._unpack_tc = lambda *args, **kwargs: None
    stub._parse_tool_args = lambda *args, **kwargs: {}
    stub.set_tool_context = lambda *args, **kwargs: None
    sys.modules["agent_tools"] = stub

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LMAgent"))

from agent_llm import LLMClient, _prepare_messages_for_payload


class PayloadCompatTests(unittest.TestCase):
    def test_keeps_plain_user_task_at_top_when_already_visible(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Rename the files"},
            {"role": "assistant", "content": "Working on it"},
        ]

        prepared = _prepare_messages_for_payload(messages)

        self.assertEqual(prepared, messages)

    def test_inserts_active_request_after_leading_system_messages(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "system", "content": "PROJECT CONFIG:\n\nx"},
            {"role": "system", "content": "[TASK STATE - DO NOT SUMMARIZE]\nOBJECTIVE: Fix login flow\n[END TASK STATE]"},
            {"role": "assistant", "content": "Calling tool"},
            {"role": "tool", "name": "read", "tool_call_id": "1", "content": '{"success": true}'},
        ]

        prepared = _prepare_messages_for_payload(messages)

        self.assertEqual(prepared[3]["role"], "user")
        self.assertIn("Fix login flow", prepared[3]["content"])
        self.assertEqual(prepared[4:], messages[3:])

    def test_build_payload_surfaces_latest_real_user_request_not_agent_nudge(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "system", "content": "YOUR TODO LIST:\n\nx"},
            {"role": "user", "content": "Implement export endpoint"},
            {"role": "assistant", "content": "Done one step"},
            {"role": "user", "content": "⚠️ HARD STOP: If verified output TASK_COMPLETE"},
            {"role": "assistant", "content": "More tool work"},
        ]

        payload = LLMClient._build_payload(messages, [])

        self.assertEqual(payload["messages"][2]["role"], "user")
        self.assertIn("Implement export endpoint", payload["messages"][2]["content"])
        self.assertNotIn("HARD STOP", payload["messages"][2]["content"])
        self.assertEqual(payload["messages"][-2], messages[-2])
        self.assertEqual(payload["messages"][-1], messages[-1])


if __name__ == "__main__":
    unittest.main()
