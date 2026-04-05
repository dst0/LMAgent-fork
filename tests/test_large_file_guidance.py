import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LMAgent"))

from agent_core import AgentEvent, AgentResult, Config
from agent_tools import tool_outline, tool_read


class LargeFileGuidanceTests(unittest.TestCase):
    def _set_workspace(self, workspace):
        Config.WORKSPACE = str(workspace)
        Config._resolved_workspace = None

    def test_read_returns_metadata_and_targeted_guidance_for_large_files(self):
        original_threshold = Config.LARGE_FILE_THRESHOLD
        original_read_lines = Config.DEFAULT_READ_LINES
        original_workspace = Config.WORKSPACE
        original_resolved = Config._resolved_workspace
        try:
            Config.LARGE_FILE_THRESHOLD = 10
            Config.DEFAULT_READ_LINES = 3
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                self._set_workspace(workspace)
                fp = workspace / "big.py"
                fp.write_text("a\nb\nc\nd\ne\nf\n", encoding="utf-8")

                result = tool_read(workspace, "big.py")

                self.assertTrue(result["success"])
                self.assertTrue(result["truncated"])
                self.assertEqual(result["file_size"], fp.stat().st_size)
                self.assertEqual(result["line_start"], 1)
                self.assertEqual(result["line_end"], 3)
                self.assertEqual(result["returned_lines"], 3)
                self.assertIn("file_info/outline", result["advice"])
        finally:
            Config.LARGE_FILE_THRESHOLD = original_threshold
            Config.DEFAULT_READ_LINES = original_read_lines
            Config.WORKSPACE = original_workspace
            Config._resolved_workspace = original_resolved

    def test_read_can_fetch_targeted_line_ranges(self):
        original_workspace = Config.WORKSPACE
        original_resolved = Config._resolved_workspace
        try:
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                self._set_workspace(workspace)
                (workspace / "sample.txt").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

                result = tool_read(workspace, "sample.txt", start_line=2, max_lines=2)

                self.assertTrue(result["success"])
                self.assertEqual(result["content"], "two\nthree")
                self.assertEqual(result["line_start"], 2)
                self.assertEqual(result["line_end"], 3)
                self.assertFalse(result["truncated"])
        finally:
            Config.WORKSPACE = original_workspace
            Config._resolved_workspace = original_resolved

    def test_outline_returns_python_symbol_ranges(self):
        original_workspace = Config.WORKSPACE
        original_resolved = Config._resolved_workspace
        try:
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                self._set_workspace(workspace)
                (workspace / "mod.py").write_text(
                    "class Thing:\n"
                    "    def method(self):\n"
                    "        return 1\n\n"
                    "def helper():\n"
                    "    return 2\n",
                    encoding="utf-8",
                )

                result = tool_outline(workspace, "mod.py")

                self.assertTrue(result["success"])
                self.assertEqual(result["parser"], "python-ast")
                names = {(item["kind"], item["name"]) for item in result["symbols"]}
                self.assertIn(("class", "Thing"), names)
                self.assertIn(("function", "method"), names)
                self.assertIn(("function", "helper"), names)
        finally:
            Config.WORKSPACE = original_workspace
            Config._resolved_workspace = original_resolved

    def test_agent_result_to_dict_includes_timing(self):
        result = AgentResult(
            status="completed",
            final_answer="done",
            events=[AgentEvent("log", {"message": "ok"})],
            session_id="sid",
            iterations=2,
            timing={"elapsed_seconds": 1.23},
        )

        payload = result.to_dict()

        self.assertEqual(payload["timing"]["elapsed_seconds"], 1.23)


if __name__ == "__main__":
    unittest.main()
