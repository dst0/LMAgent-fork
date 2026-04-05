import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_WEB = ROOT / "LMAgent" / "agent_web.py"
AGENT_WEB_UI = ROOT / "LMAgent" / "agent_web_ui.html"


class DoneMessageRenderingTests(unittest.TestCase):
    def test_backend_done_event_keeps_full_final_answer(self):
        content = AGENT_WEB.read_text()

        self.assertIn('reason = (result.final_answer or "").strip()', content)
        self.assertIn('_broadcast(("done", reason or result.status))', content)
        self.assertNotIn('" ".join(words[:8])', content)

    def test_frontend_system_messages_can_expand_when_long(self):
        content = AGENT_WEB_UI.read_text()

        self.assertIn("const _SYSMSG_PREVIEW_LEN = 200;", content)
        self.assertIn("function _makeSystemMessageCollapsible(wrap, raw){", content)
        self.assertIn("wrap.classList.add('sys-collapsible', 'sys-collapsed');", content)
        self.assertIn("function sysMsg(text, v){ Messages.sys(text, v); }", content)
        self.assertIn("_makeSystemMessageCollapsible(body && body.closest('.msg'), text);", content)


if __name__ == "__main__":
    unittest.main()
