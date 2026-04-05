import ast
import threading
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
AGENT_WEB = ROOT / "LMAgent" / "agent_web.py"
AGENT_WEB_UI = ROOT / "LMAgent" / "agent_web_ui.html"


class SessionSwitchDuringRunTests(unittest.TestCase):
    def _compile_agent_web_functions(self, *names, extra_globals=None):
        source = AGENT_WEB.read_text()
        tree = ast.parse(source)
        wanted = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        module = ast.Module(body=wanted, type_ignores=[])
        namespace = {}
        if extra_globals:
            namespace.update(extra_globals)
        exec(compile(module, str(AGENT_WEB), "exec"), namespace)
        return namespace

    def test_backend_tracks_active_session_separately_from_viewed_session(self):
        content = AGENT_WEB.read_text()

        self.assertIn("_active_session_id       = None", content)
        self.assertIn('active_sid = _active_session_id', content)
        self.assertIn('s["id"] == active_sid', content)

    def test_backend_chatlog_appends_to_active_run_session_even_after_view_switch(self):
        event_writes = []
        tl = SimpleNamespace(chatlog_session_id="active-session")
        extra_globals = {
            "_current_session_id": "viewed-session",
            "_tl": tl,
            "_NO_SESSION_KEY": "_none_",
            "_REPLAY_KINDS": {"status"},
            "_chat_logs": defaultdict(list),
            "_chat_log_lock": threading.Lock(),
            "_eventlog_write": lambda sid, kind, payload: event_writes.append((sid, kind, payload)),
        }
        namespace = self._compile_agent_web_functions(
            "_clog_key",
            "_get_chatlog_session_id",
            "_chatlog_append",
            extra_globals=extra_globals,
        )
        chatlog_append = namespace["_chatlog_append"]

        chatlog_append(("status", "still running"))

        self.assertEqual(extra_globals["_chat_logs"]["active-session"], [("status", "still running")])
        self.assertEqual(event_writes, [("active-session", "status", "still running")])

    def test_backend_session_switch_endpoint_no_longer_rejects_busy_agent(self):
        content = AGENT_WEB.read_text()
        start = content.index("def session_switch():")
        end = content.index('@app.route("/upload"', start)
        body = content[start:end]
        self.assertNotIn('agent is busy', body)
        self.assertNotIn('_get_agent_state() != "idle"', body)
        self.assertIn('"""Switch the currently viewed session."""', body)

    def test_frontend_switch_flow_keeps_live_run_separate_from_selected_session(self):
        content = AGENT_WEB_UI.read_text()

        self.assertNotIn("Cannot switch session while agent is running", content)
        self.assertIn("activeSessionId = null", content)
        self.assertIn("shouldSuppressLiveRender", content)
        self.assertIn("suppressLiveRender", content)
        self.assertIn("keepLiveState", content)
        self.assertIn("if(suppressLiveRender) break;", content)


if __name__ == "__main__":
    unittest.main()
