import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_WEB = ROOT / "LMAgent" / "agent_web.py"
AGENT_WEB_UI = ROOT / "LMAgent" / "agent_web_ui.html"


class SessionSwitchDuringRunTests(unittest.TestCase):
    def test_backend_tracks_active_session_separately_from_viewed_session(self):
        content = AGENT_WEB.read_text()

        self.assertIn("_active_session_id       = None", content)
        self.assertIn('sid = _get_chatlog_session_id()', content)
        self.assertIn('active_sid = _active_session_id', content)
        self.assertIn('s["id"] == active_sid', content)

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
        self.assertIn("suppressLiveRender", content)
        self.assertIn("keepLiveState", content)
        self.assertIn("id !== activeSessionId", content)
        self.assertIn("if(suppressLiveRender) break;", content)


if __name__ == "__main__":
    unittest.main()
