import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_CORE = ROOT / "LMAgent" / "agent_core.py"
AGENT_WEB_UI = ROOT / "LMAgent" / "agent_web_ui.html"


class SlowWarningDefaultsTests(unittest.TestCase):
    def test_backend_slow_warning_defaults_fit_slower_local_models(self):
        content = AGENT_CORE.read_text()

        self.assertRegex(
            content,
            r'SLOW_LLM_CALL_SECONDS\s*=\s*float\(os\.getenv\("SLOW_LLM_CALL_SECONDS",\s*"30"\)\)',
        )
        self.assertRegex(
            content,
            r'SLOW_ITERATION_SECONDS\s*=\s*float\(os\.getenv\("SLOW_ITERATION_SECONDS",\s*"60"\)\)',
        )

    def test_live_activity_clears_slow_alert_after_iteration_advances(self):
        content = AGENT_WEB_UI.read_text()

        self.assertIn("alertIteration: 0", content)
        self.assertIn("state.alertIteration = Number(data.iteration) || 0;", content)
        self.assertIn("if(state.alertIteration !== 0 && nextIteration > state.alertIteration){", content)
        self.assertIn("state.alert = '';", content)

    def test_idle_nudge_waits_longer_before_warning(self):
        content = AGENT_WEB_UI.read_text()

        self.assertRegex(content, r"var ACTIVITY_IDLE_ALERT_MS = 30000;")


if __name__ == "__main__":
    unittest.main()
