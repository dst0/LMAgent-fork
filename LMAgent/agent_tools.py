#!/usr/bin/env python3
"""
agent_tools.py — Tool layer for LMAgent.

Contains: tool handlers, schemas, registry, vision support, git helpers.
LLM client, agent loops, and system prompts live in agent_llm.py.

Sub-agent delegation is handled by the Brief-Contract Architecture (BCA)
in agent_bca.py. The old tool_task / run_sub_agent / _build_parent_context
are replaced; the 'task' tool entry-point is kept for backward compatibility
but now routes through BCA. New tools: delegate, decompose, report_result.
"""

import base64
import json
import os
import re
import shlex
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from agent_core import (
    Config, Colors, Log, PermissionMode, _IS_WINDOWS,
    truncate_output, strip_thinking,
    get_current_context, set_current_context, _get_ctx,
    Safety, FileEditor,
    get_shell_session,
    TodoManager, PlanManager, TaskStateManager, TaskState,
    SessionManager, LoopDetector,
    colored,
)
from sandboxed_shell import run_sandboxed, sandbox_info

# BCA sub-agent system — brief/contract delegation
from agent_bca import (
    tool_report_result,
    tool_delegate,
    tool_decompose,
    tool_task_bca,
    _get_bca_tool_schemas,
    MAX_DEPTH,
)

# Backward-compat alias
get_powershell_session = get_shell_session


# =============================================================================
# SHELL QUOTING UTILITIES
# =============================================================================

def _shell_quote(s: str) -> str:
    if _IS_WINDOWS:
        escaped = s.replace("`", "``").replace('"', '`"').replace("$", "`$")
        return f'"{escaped}"'
    return shlex.quote(s)


_GIT_REF_RE = re.compile(r'^[A-Za-z0-9_.\-/]+$')


def _validate_git_ref(name: str) -> Tuple[bool, str]:
    if not name:
        return False, "Empty ref name"
    if not _GIT_REF_RE.match(name):
        bad = sorted({c for c in name if not re.match(r'[A-Za-z0-9_.\-/]', c)})
        return False, (
            f"Ref name '{name}' contains disallowed characters: {bad!r}. "
            "Only alphanumerics, hyphens, underscores, dots, and slashes are permitted."
        )
    if ".." in name:
        return False, f"Ref name '{name}' contains '..' (path-traversal sequence)"
    return True, ""


def _git_safe(workspace: Path, cmd: str) -> Tuple[str, int]:
    ok, reason = Safety.validate_command(cmd, workspace)
    if not ok:
        raise ValueError(reason)
    return run_sandboxed(cmd=cmd, workspace=workspace)


# =============================================================================
# VISION SUPPORT
# =============================================================================

_vision_cache: Optional[bool] = None
_vision_lock = threading.Lock()

_VISION_MIME_MAP: Dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}


def _detect_vision_support() -> bool:
    global _vision_cache
    with _vision_lock:
        if _vision_cache is not None:
            return _vision_cache

        mode = getattr(Config, "VISION_ENABLED", "auto").lower()
        if mode == "true":
            _vision_cache = True
            return True
        if mode == "false":
            _vision_cache = False
            return False

        base = getattr(Config, "LLM_BASE_URL", "http://localhost:1234")
        try:
            resp = requests.get(
                f"{base}/api/v1/models",
                timeout=5,
                headers={"Authorization": f"Bearer {Config.LLM_API_KEY}"},
            )
            resp.raise_for_status()
            models = resp.json().get("models", [])
            target = (Config.LLM_MODEL or "").lower()

            for m in models:
                key = (m.get("key") or m.get("id") or "").lower()
                if target and target not in key:
                    continue
                if m.get("capabilities", {}).get("vision", False):
                    Log.info(f"Vision capability detected: {key}")
                    _vision_cache = True
                    return True

            for m in models:
                if m.get("capabilities", {}).get("vision", False):
                    key = m.get("key") or m.get("id") or "unknown"
                    Log.info(f"Vision fallback model: {key}")
                    _vision_cache = True
                    return True

            Log.info("No vision-capable model found — vision tool disabled")
            _vision_cache = False

        except Exception as e:
            Log.warning(f"Vision probe failed ({e}) — disabling vision for this session")
            _vision_cache = False

        return _vision_cache


def vision_cache_invalidate() -> None:
    global _vision_cache
    with _vision_lock:
        _vision_cache = None
    Log.info("Vision cache invalidated — will re-probe on next tool call")


def tool_vision(
    workspace: Path,
    path: str,
    prompt: str = "Describe this image in detail.",
) -> Dict[str, Any]:
    if not _detect_vision_support():
        return {
            "success": False,
            "error": (
                "No vision-capable model is currently loaded in LM Studio. "
                "Load a VLM (e.g. LLaVA, Qwen-VL, Pixtral) and try again."
            ),
        }

    ok, err, fp = Safety.validate_path(workspace, path, must_exist=True)
    if not ok:
        return {"success": False, "error": err}
    if not fp.is_file():
        return {"success": False, "error": f"Not a file: {path}"}

    mime = _VISION_MIME_MAP.get(fp.suffix.lower())
    if not mime:
        supported = ", ".join(sorted(_VISION_MIME_MAP.keys()))
        return {
            "success": False,
            "error": f"Unsupported image type '{fp.suffix}'. Supported: {supported}",
        }

    try:
        b64 = base64.b64encode(fp.read_bytes()).decode("utf-8")
    except Exception as e:
        return {"success": False, "error": f"Could not read image: {e}"}

    payload: Dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": Config.TEMPERATURE,
    }
    if Config.LLM_MODEL:
        payload["model"] = Config.LLM_MODEL

    try:
        resp = requests.post(
            Config.LLM_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {Config.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )
        resp.raise_for_status()
        description = resp.json()["choices"][0]["message"]["content"]
        return {
            "success":     True,
            "path":        str(fp.relative_to(workspace)).replace("\\", "/"),
            "description": description,
        }
    except Exception as e:
        return {"success": False, "error": f"Vision request failed: {e}"}


# =============================================================================
# SHARED TOOL-CALL HELPERS
# =============================================================================

def _unpack_tc(tc: Dict[str, Any], fallback_id: str) -> Tuple[str, str, str]:
    fn = tc.get("function") or {}
    return (
        fn.get("name", "").strip(),
        fn.get("arguments", ""),
        tc.get("id") or fallback_id,
    )


def _parse_tool_args(fn_name: str, args_raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not args_raw or not args_raw.strip():
        return {}, None
    try:
        return json.loads(args_raw), None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON in tool arguments for '{fn_name}': {e}"


# =============================================================================
# TOOL HANDLERS — FILE OPERATIONS
# =============================================================================

def tool_get_time(workspace: Path) -> Dict[str, Any]:
    now = datetime.now()
    return {
        "success":   True,
        "datetime":  now.isoformat(),
        "date":      now.strftime("%Y-%m-%d"),
        "time":      now.strftime("%H:%M:%S"),
        "weekday":   now.strftime("%A"),
        "timestamp": int(now.timestamp()),
    }


def tool_read(workspace: Path, path: str) -> Dict[str, Any]:
    ok, err, fp = Safety.validate_path(workspace, path, must_exist=True)
    if not ok:
        return {"success": False, "error": err}
    if not fp.is_file():
        return {"success": False, "error": "Not a file"}
    if fp.suffix.lower() in Config.BINARY_EXTS:
        return {"success": False, "error": "Cannot read binary file"}
    try:
        content = fp.read_text(encoding="utf-8", errors="replace")
        return {"success": True,
                "content": truncate_output(content, Config.MAX_FILE_READ, path),
                "lines":   content.count("\n") + 1}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_write(workspace: Path, path: str, content: str) -> Dict[str, Any]:
    ok, err, fp = Safety.validate_path(workspace, path)
    if not ok:
        return {"success": False, "error": err}
    if fp.exists() and fp.suffix.lower() in Config.BINARY_EXTS:
        return {"success": False, "error": "Cannot overwrite binary file"}
    try:
        existed = fp.exists()
        fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = fp.parent / (fp.name + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(str(tmp), str(fp))
        return {"success": True,
                "path":    str(fp.relative_to(workspace)).replace("\\", "/"),
                "action":  "modified" if existed else "created",
                "size":    len(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_edit(workspace: Path, path: str, search: str, replace: str) -> Dict[str, Any]:
    ok, err, fp = Safety.validate_path(workspace, path, must_exist=True)
    if not ok:
        return {"success": False, "error": err}
    try:
        content = fp.read_text(encoding="utf-8", errors="replace")
        success, new_content, msg = FileEditor.search_replace(content, search, replace)
        if not success:
            return {"success": False, "error": truncate_output(msg, 500)}
        tmp = fp.parent / (fp.name + ".tmp")
        tmp.write_text(new_content, encoding="utf-8")
        os.replace(str(tmp), str(fp))
        return {"success": True,
                "path":    str(fp.relative_to(workspace)).replace("\\", "/"),
                "method":  msg}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_glob(workspace: Path, pattern: str) -> Dict[str, Any]:
    try:
        matches = list(workspace.glob(pattern))
        files   = []
        for m in matches[:Config.MAX_LS_ENTRIES]:
            if m.is_file() and not any(p.startswith(".") for p in m.parts):
                try:
                    files.append(str(m.relative_to(workspace)).replace("\\", "/"))
                except Exception:
                    files.append(str(m).replace("\\", "/"))
        return {"success": True, "files": files, "count": len(files),
                "total_matches": len(matches),
                "truncated":     len(matches) > Config.MAX_LS_ENTRIES}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_grep(workspace: Path, pattern: str,
              paths: Optional[List[str]] = None) -> Dict[str, Any]:
    try:
        if paths:
            search_files: List[Path] = []
            for p in paths:
                if "*" in p:
                    search_files.extend(workspace.glob(p))
                else:
                    fp = workspace / p
                    if fp.exists():
                        search_files.append(fp)
        else:
            search_files = [
                f for f in workspace.rglob("*")
                if f.is_file()
                and not any(p.startswith(".") for p in f.relative_to(workspace).parts)
            ]
        matches: List[Dict[str, Any]] = []
        for fp in search_files[:100]:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if pattern in line:
                            matches.append({
                                "file":    str(fp.relative_to(workspace)).replace("\\", "/"),
                                "line":    i,
                                "content": line.rstrip()[:200],
                            })
                        if len(matches) >= Config.MAX_GREP_RESULTS:
                            break
                if len(matches) >= Config.MAX_GREP_RESULTS:
                    break
            except Exception:
                continue
        return {"success":   True,
                "matches":   matches,
                "truncated": len(matches) >= Config.MAX_GREP_RESULTS}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_ls(workspace: Path, path: str = ".") -> Dict[str, Any]:
    ok, err, dp = Safety.validate_path(workspace, path)
    if not ok:
        return {"success": False, "error": err}
    if not dp.exists():
        return {"success": False, "error": "Path does not exist"}
    if not dp.is_dir():
        return {"success": False, "error": "Not a directory"}
    try:
        all_entries = list(dp.iterdir())
        entries     = []
        for item in sorted(all_entries)[:Config.MAX_LS_ENTRIES]:
            if item.name.startswith("."):
                continue
            stat = item.stat()
            entries.append({"name": item.name,
                             "type": "dir" if item.is_dir() else "file",
                             "size": stat.st_size if item.is_file() else 0})
        return {"success": True, "entries": entries,
                "total":     len(all_entries),
                "truncated": len(all_entries) > Config.MAX_LS_ENTRIES}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_mkdir(workspace: Path, path: str) -> Dict[str, Any]:
    ok, err, dp = Safety.validate_path(workspace, path)
    if not ok:
        return {"success": False, "error": err}
    try:
        if dp.exists():
            return {"success": True, "message": "Already exists"}
        dp.mkdir(parents=True, exist_ok=True)
        return {"success": True, "message": "Created"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# TOOL HANDLERS — SANDBOXED SHELL
# =============================================================================

def tool_shell(
    workspace: Path,
    command: str,
    timeout: int = 30,
    max_memory_mb: int = 512,
) -> Dict[str, Any]:
    ok, reason = Safety.validate_command(command, workspace)
    if not ok:
        return {"success": False, "error": f"Command blocked by safety check: {reason}"}

    timeout       = max(1, min(timeout, 120))
    max_memory_mb = max(64, min(max_memory_mb, 2048))

    try:
        output, exit_code = run_sandboxed(
            cmd=command,
            workspace=workspace,
            timeout=timeout,
            max_output_bytes=32_768,
            max_memory_mb=max_memory_mb,
        )
        return {
            "success":   exit_code == 0,
            "exit_code": exit_code,
            "output":    output,
            "sandbox":   sandbox_info()["backend"],
        }
    except Exception as e:
        return {"success": False, "error": f"Sandbox error: {e}"}


# =============================================================================
# TOOL HANDLERS — GIT
# =============================================================================

def tool_git_status(workspace: Path) -> Dict[str, Any]:
    try:
        output, code = _git_safe(workspace, "git status --porcelain")
        if code != 0:
            return {"success": False, "error": "Not a git repository"}
        files: Dict[str, List[str]] = {
            "modified": [], "added": [], "deleted": [], "untracked": []
        }
        for line in output.splitlines()[:Config.MAX_LS_ENTRIES]:
            if len(line) < 3:
                continue
            status, path = line[:2], line[3:]
            if   "M" in status: files["modified"].append(path)
            elif "A" in status: files["added"].append(path)
            elif "D" in status: files["deleted"].append(path)
            elif "?" in status: files["untracked"].append(path)
        branch, _ = _git_safe(workspace, "git branch --show-current")
        total      = sum(len(v) for v in files.values())
        return {"success": True, "branch": branch.strip(), **files,
                "total_changes": total,
                "truncated":     total > Config.MAX_LS_ENTRIES}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_git_diff(workspace: Path, path: str = "", staged: bool = False) -> Dict[str, Any]:
    try:
        resolved_path = ""
        if path:
            ok, err, resolved = Safety.validate_path(workspace, path)
            if not ok:
                return {"success": False, "error": f"Invalid diff path: {err}"}
            resolved_path = str(resolved.relative_to(workspace))
        cmd = "git diff"
        if staged:
            cmd += " --staged"
        if resolved_path:
            cmd += f" -- {_shell_quote(resolved_path)}"
        output, code = _git_safe(workspace, cmd)
        if code != 0:
            return {"success": False, "error": "Git diff failed"}
        return {"success":     True,
                "diff":        truncate_output(output, Config.MAX_TOOL_OUTPUT),
                "has_changes": bool(output.strip())}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_git_add(workspace: Path, paths: List[str]) -> Dict[str, Any]:
    try:
        quoted: List[str] = []
        for p in paths:
            ok, err, _ = Safety.validate_path(workspace, p)
            if not ok:
                return {"success": False, "error": f"{p}: {err}"}
            quoted.append(_shell_quote(p))
        cmd = f"git add {' '.join(quoted)}"
        output, code = _git_safe(workspace, cmd)
        if code != 0:
            return {"success": False, "error": truncate_output(output, 500)}
        return {"success": True, "staged": len(paths)}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_git_commit(workspace: Path, message: str, allow_empty: bool = False) -> Dict[str, Any]:
    try:
        cmd = f"git commit -m {_shell_quote(message)}"
        if allow_empty:
            cmd += " --allow-empty"
        output, code = _git_safe(workspace, cmd)
        if code != 0:
            return {"success": False, "error": truncate_output(output, 500)}
        rev, _ = _git_safe(workspace, "git rev-parse HEAD")
        return {"success": True, "commit": rev.strip()[:8]}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_git_branch(workspace: Path, name: str = "",
                    create: bool = False, switch: bool = False) -> Dict[str, Any]:
    try:
        if not name:
            output, code = _git_safe(workspace, "git branch")
            if code != 0:
                return {"success": False, "error": "Failed to list branches"}
            branches, current = [], None
            for line in output.splitlines()[:20]:
                line = line.strip()
                if line.startswith("* "):
                    current = line[2:]
                elif line:
                    branches.append(line)
            return {"success": True, "action": "list",
                    "branches": branches, "current": current}
        ok, err = _validate_git_ref(name)
        if not ok:
            return {"success": False, "error": err}
        git_cmd = ("git checkout -b" if create else
                   "git checkout"    if switch  else "git branch")
        action  = "created" if create else ("switched" if switch else "created_local")
        output, code = _git_safe(workspace, f"{git_cmd} {_shell_quote(name)}")
        if code != 0:
            return {"success": False, "error": truncate_output(output, 500)}
        return {"success": True, "action": action, "branch": name}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# TOOL HANDLERS — TODO & PLAN
# =============================================================================

def set_tool_context(
    workspace: Path,
    session_id: str,
    todo_manager: TodoManager,
    plan_manager: PlanManager,
    task_state_manager: TaskStateManager,
    messages: Optional[List[Dict[str, Any]]] = None,
    mode: str = "interactive",
    stream_callback: Optional[Callable] = None,
):
    set_current_context({
        "workspace":          workspace,
        "session_id":         session_id,
        "todo_manager":       todo_manager,
        "plan_manager":       plan_manager,
        "task_state_manager": task_state_manager,
        "messages":           messages,
        "mode":               mode,
        "stream_callback":    stream_callback,
    })


def tool_todo_add(workspace: Path, description: str, notes: str = "") -> Dict[str, Any]:
    mgr, err = _get_ctx("todo_manager", "Todo manager")
    if err: return err
    return mgr.add(description, notes)


def tool_todo_complete(workspace: Path, todo_id: int) -> Dict[str, Any]:
    mgr, err = _get_ctx("todo_manager", "Todo manager")
    if err: return err
    existing = next((t for t in mgr.todos if t.id == todo_id), None)
    if existing and existing.status == "completed":
        summary  = mgr.list_all()
        total    = summary.get("total", 0)
        done     = summary.get("completed", 0)
        all_done = total > 0 and done >= total
        return {
            "success":           False,
            "error":             f"Todo #{todo_id} is already completed.",
            "already_completed": True,
            "all_complete":      all_done,
            "completed_count":   done,
            "total_count":       total,
            "message": ("ALL TODOS DONE. Output TASK_COMPLETE now."
                        if all_done else f"{done}/{total} todos complete."),
        }
    result = mgr.complete(todo_id)
    if result.get("success"):
        summary  = mgr.list_all()
        total    = summary.get("total", 0)
        done     = summary.get("completed", 0)
        all_done = total > 0 and done >= total
        result.update({
            "all_complete":    all_done,
            "completed_count": done,
            "total_count":     total,
            "message": (
                f"Todo #{todo_id} complete. ALL {total} TODOS DONE. "
                "YOUR TASK IS FINISHED. Output TASK_COMPLETE now. Do not call any more tools."
                if all_done else f"Todo #{todo_id} complete. {done}/{total} done."
            ),
        })
    return result


def tool_todo_update(workspace: Path, todo_id: int,
                     status: str, notes: str = "") -> Dict[str, Any]:
    mgr, err = _get_ctx("todo_manager", "Todo manager")
    if err: return err
    valid = {"pending", "in_progress", "completed", "blocked"}
    if status not in valid:
        return {"success": False,
                "error": f"Invalid status '{status}'. Must be one of: {sorted(valid)}"}
    return mgr.update_status(todo_id, status, notes)


def tool_todo_list(workspace: Path) -> Dict[str, Any]:
    mgr, err = _get_ctx("todo_manager", "Todo manager")
    if err: return err
    result = mgr.list_all()
    if result.get("success"):
        total    = result.get("total", 0)
        done     = result.get("completed", 0)
        all_done = total > 0 and done >= total
        result["all_complete"] = all_done
        if all_done:
            result["hint"] = ("ALL TODOS COMPLETE. If your task deliverable is verified, "
                              "output TASK_COMPLETE now.")
    return result


def tool_plan_complete_step(workspace: Path, step_id: str,
                             verification_notes: str = "") -> Dict[str, Any]:
    mgr, err = _get_ctx("plan_manager", "Plan manager")
    if err: return err
    if not mgr.plan:
        return {"success": False, "error": "No active plan"}
    step = next((s for s in mgr.plan["steps"] if s["id"] == step_id), None)
    if not step:
        return {"success": False, "error": f"Step '{step_id}' not found"}
    if step["status"] == "completed":
        return {"success": False, "error": f"Step '{step_id}' already completed"}
    if step["status"] == "pending":
        return {"success": False, "error": f"Step '{step_id}' not started yet"}
    mgr.complete_step(step_id)
    return {"success":            True,
            "step_id":            step_id,
            "description":        step.get("description", ""),
            "verification_notes": verification_notes,
            "message":            f"Step '{step_id}' marked complete"}


# =============================================================================
# TOOL HANDLERS — TASK STATE
# =============================================================================

def tool_task_state_update(
    workspace: Path,
    objective: str = "",
    completion_gate: str = "",
    total_count: int = -1,
    processed_count: int = -1,
    remaining_queue: Optional[List[str]] = None,
    rename_map: Optional[Dict[str, str]] = None,
    last_error: str = "",
    recovery_instruction: str = "",
    next_action: str = "",
) -> Dict[str, Any]:
    mgr, err = _get_ctx("task_state_manager", "Task state manager")
    if err: return err
    state = mgr.current_state
    if not state:
        if not objective:
            return {"success": False, "error": "objective required for new task state"}
        state = TaskState(
            objective=objective,
            completion_gate=completion_gate or "processed == total",
            inventory_hash="",
            total_count=max(total_count, 0),
            processed_count=max(processed_count, 0),
            remaining_queue=remaining_queue or [],
            rename_map=rename_map or {},
            last_error=last_error,
            recovery_instruction=recovery_instruction,
            next_action=next_action,
            last_updated="",
        )
    else:
        if objective:                   state.objective            = objective
        if completion_gate:             state.completion_gate      = completion_gate
        if total_count     >= 0:        state.total_count          = total_count
        if processed_count >= 0:        state.processed_count      = processed_count
        if remaining_queue is not None: state.remaining_queue      = remaining_queue
        if rename_map      is not None: state.rename_map.update(rename_map)
        if last_error:                  state.last_error           = last_error
        if recovery_instruction:        state.recovery_instruction = recovery_instruction
        if next_action:                 state.next_action          = next_action
    if remaining_queue:
        state.inventory_hash = TaskState.compute_inventory_hash(remaining_queue)
    mgr.checkpoint(state)
    return {
        "success": True,
        "state":   state.to_dict(),
        "message": f"Checkpointed: {state.processed_count}/{state.total_count} processed",
    }


def tool_task_state_get(workspace: Path) -> Dict[str, Any]:
    mgr, err = _get_ctx("task_state_manager", "Task state manager")
    if err: return err
    return {"success": True,
            "state": mgr.current_state.to_dict() if mgr.current_state else None}


def tool_task_reconcile(workspace: Path) -> Dict[str, Any]:
    mgr, err = _get_ctx("task_state_manager", "Task state manager")
    if err: return err
    if not mgr.current_state:
        return {"success": False, "error": "No task state to reconcile"}
    s = mgr.current_state
    return {
        "success": True,
        "message": (
            f"Reconciliation checkpoint. Current: {s.processed_count}/{s.total_count} processed. "
            f"You MUST: 1) Re-enumerate targets via ls/glob, "
            f"2) Update remaining_queue, 3) Continue from next unprocessed file."
        ),
        "current_state":   s.to_dict(),
        "action_required": "re_enumerate",
    }


# =============================================================================
# BACKWARD-COMPAT: tool_task → routes through BCA
#
# External code that calls tool_task() directly still works. The 'task' schema
# entry is kept so agents that learned the old tool name keep working. Both
# 'task' and the new 'delegate'/'decompose' tools are available.
# =============================================================================

def tool_task(workspace: Path, task_type: str,
              instructions: str, file_path: str) -> Dict[str, Any]:
    """Backward-compatible entry point — routes through BCA (tool_task_bca)."""
    return tool_task_bca(workspace, task_type, instructions, file_path)


# =============================================================================
# TOOL REGISTRY
# =============================================================================

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "get_time",
        "description": "Get the current date and time.",
        "parameters": {"type": "object", "properties": {}}}},

    {"type": "function", "function": {
        "name": "read", "description": "Read file contents (workspace only).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write", "description": "Create or overwrite a file (workspace only).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit", "description": "Edit file with fuzzy search/replace (workspace only).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "search": {"type": "string"},
                                      "replace": {"type": "string"}},
                       "required": ["path", "search", "replace"]}}},
    {"type": "function", "function": {
        "name": "glob", "description": "Find files matching a glob pattern (workspace only).",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search for a text pattern in files (workspace only).",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"},
                                      "paths": {"type": "array", "items": {"type": "string"}}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "ls", "description": "List directory contents (workspace only).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "mkdir", "description": "Create a directory (workspace only).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},

    {"type": "function", "function": {
        "name": "shell",
        "description": (
            "Run a shell command inside a platform sandbox (bash on macOS/Linux, "
            "cmd on Windows). stdout and stderr are merged. Output is capped at 32 KB. "
            "Timeout defaults to 30 s (max 120 s). Memory defaults to 512 MB (max 2 GB)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command":        {"type": "string"},
                "timeout":        {"type": "integer"},
                "max_memory_mb":  {"type": "integer"},
            },
            "required": ["command"],
        }}},

    {"type": "function", "function": {
        "name": "git_status", "description": "Get git status of the workspace.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "git_diff", "description": "Show git diff.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "staged": {"type": "boolean"}}}}},
    {"type": "function", "function": {
        "name": "git_add", "description": "Stage files for commit.",
        "parameters": {"type": "object",
                       "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
                       "required": ["paths"]}}},
    {"type": "function", "function": {
        "name": "git_commit", "description": "Commit staged changes.",
        "parameters": {"type": "object",
                       "properties": {"message": {"type": "string"},
                                      "allow_empty": {"type": "boolean"}},
                       "required": ["message"]}}},
    {"type": "function", "function": {
        "name": "git_branch",
        "description": "Manage git branches. No args → list. With name+flags → create/switch.",
        "parameters": {"type": "object",
                       "properties": {"name":   {"type": "string"},
                                      "create": {"type": "boolean"},
                                      "switch": {"type": "boolean"}}}}},

    {"type": "function", "function": {
        "name": "todo_add",
        "description": "Add a todo item. WARNING: Only add todos BEFORE execution begins.",
        "parameters": {"type": "object",
                       "properties": {"description": {"type": "string"},
                                      "notes": {"type": "string"}},
                       "required": ["description"]}}},
    {"type": "function", "function": {
        "name": "todo_complete",
        "description": "Mark a todo completed. Each todo_id must be completed ONCE. When all_complete=true, immediately output TASK_COMPLETE.",
        "parameters": {"type": "object",
                       "properties": {"todo_id": {"type": "integer"}},
                       "required": ["todo_id"]}}},
    {"type": "function", "function": {
        "name": "todo_update", "description": "Update todo status.",
        "parameters": {"type": "object",
                       "properties": {"todo_id": {"type": "integer"},
                                      "status": {"type": "string",
                                                 "enum": ["pending", "in_progress",
                                                          "completed", "blocked"]},
                                      "notes": {"type": "string"}},
                       "required": ["todo_id", "status"]}}},
    {"type": "function", "function": {
        "name": "todo_list",
        "description": "List todos. If all_complete=true is returned, output TASK_COMPLETE immediately.",
        "parameters": {"type": "object", "properties": {}}}},

    {"type": "function", "function": {
        "name": "plan_complete_step",
        "description": "Mark a plan step complete after verification.",
        "parameters": {"type": "object",
                       "properties": {"step_id": {"type": "string"},
                                      "verification_notes": {"type": "string"}},
                       "required": ["step_id"]}}},

    # ── Delegation ────────────────────────────────────────────────────────────
    # 'task' kept for backward compat — routes to BCA internally.
    # Prefer 'delegate' for new usage.
    {"type": "function", "function": {
        "name": "task",
        "description": (
            "Delegate single-file creation to an isolated sub-agent (backward-compat). "
            "Prefer 'delegate' for new delegation — it gives the sub-agent a full contract "
            "and returns structured results."
        ),
        "parameters": {"type": "object",
                       "properties": {
                           "task_type":    {"type": "string",
                                           "description": "File type (html, css, js, python, etc.)"},
                           "instructions": {"type": "string",
                                           "description": "Full specification for the file"},
                           "file_path":    {"type": "string",
                                           "description": "Where to save the file"},
                       },
                       "required": ["task_type", "instructions", "file_path"]}}},

    {"type": "function", "function": {
        "name": "task_state_update",
        "description": "Checkpoint task progress. Call after each file processed.",
        "parameters": {"type": "object",
                       "properties": {
                           "objective":            {"type": "string"},
                           "completion_gate":      {"type": "string"},
                           "total_count":          {"type": "integer"},
                           "processed_count":      {"type": "integer"},
                           "remaining_queue":      {"type": "array", "items": {"type": "string"}},
                           "rename_map":           {"type": "object"},
                           "last_error":           {"type": "string"},
                           "recovery_instruction": {"type": "string"},
                           "next_action":          {"type": "string"},
                       }}}},
    {"type": "function", "function": {
        "name": "task_state_get",
        "description": "Retrieve current task state checkpoint.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "task_reconcile",
        "description": "Reconcile after rename/move operations. Re-enumerate from disk.",
        "parameters": {"type": "object", "properties": {}}}},

    # ── Vision (conditionally included by get_available_tools) ────────────────
    {"type": "function", "function": {
        "name": "vision",
        "description": (
            "Read and analyse an image file from the workspace using the loaded vision model. "
            "Only available when a vision-capable model (VLM) is loaded in LM Studio."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path":   {"type": "string",
                           "description": "Workspace-relative path to the image (jpg, png, gif, webp)"},
                "prompt": {"type": "string",
                           "description": "What to ask about the image."},
            },
            "required": ["path"],
        }}},
]

TOOL_HANDLERS: Dict[str, Callable] = {
    "get_time":           tool_get_time,
    "shell":              tool_shell,
    "read":               tool_read,
    "write":              tool_write,
    "edit":               tool_edit,
    "glob":               tool_glob,
    "grep":               tool_grep,
    "ls":                 tool_ls,
    "mkdir":              tool_mkdir,
    "git_status":         tool_git_status,
    "git_diff":           tool_git_diff,
    "git_add":            tool_git_add,
    "git_commit":         tool_git_commit,
    "git_branch":         tool_git_branch,
    "todo_add":           tool_todo_add,
    "todo_complete":      tool_todo_complete,
    "todo_update":        tool_todo_update,
    "todo_list":          tool_todo_list,
    "plan_complete_step": tool_plan_complete_step,
    # delegation — old compat + BCA
    "task":               tool_task,           # backward-compat wrapper → tool_task_bca
    "delegate":           tool_delegate,       # BCA: single focused sub-agent
    "decompose":          tool_decompose,      # BCA: sequential multi-task decomposition
    "report_result":      tool_report_result,  # BCA: sub-agent completion signal
    "task_state_update":  tool_task_state_update,
    "task_state_get":     tool_task_state_get,
    "task_reconcile":     tool_task_reconcile,
    # vision: conditionally available — registered so dispatch works
    "vision":             tool_vision,
}

_REQUIRED_ARG_TOOLS = frozenset({
    "shell",
    "write", "read", "edit", "glob", "grep", "ls", "mkdir",
    "todo_add", "todo_complete", "todo_update", "plan_complete_step",
    "git_add", "git_commit", "git_branch", "git_diff",
    "task", "delegate", "decompose", "report_result",
    "vision",
})


# =============================================================================
# DYNAMIC TOOL LIST
# =============================================================================

def get_available_tools(
    extra_schemas: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return the tool list to pass to LLMClient.call() for the current session.

    Injects BCA tools (delegate, decompose, report_result) for the root agent
    (depth=0). Vision is included only when a VLM is detected.
    """
    has_vision = _detect_vision_support()

    tools = [
        t for t in TOOL_SCHEMAS
        if t["function"]["name"] != "vision" or has_vision
    ]

    # BCA tools for root agent (depth=0 → all BCA tools available)
    bca_schemas = _get_bca_tool_schemas(depth=0, max_depth=MAX_DEPTH)
    existing    = {t["function"]["name"] for t in tools}
    for schema in bca_schemas:
        name = schema["function"]["name"]
        if name not in existing:
            tools.append(schema)
            existing.add(name)

    if extra_schemas:
        for schema in extra_schemas:
            name = (schema.get("function") or {}).get("name", "")
            if name and name not in existing:
                tools.append(schema)
                existing.add(name)

    return tools


# =============================================================================
# BACKWARD-COMPAT RE-EXPORTS
# =============================================================================

from agent_llm import (  # noqa: E402, F401
    LLMClient,
    detect_completion,
    should_ask_permission,
    ask_permission,
    SYSTEM_PROMPT,
    SUB_AGENT_SYSTEM_PROMPT,
    PLAN_MODE_PROMPT,
    run_sub_agent,
    run_plan_mode,
    _execute_tool,
    _process_tool_calls,
    _HeaderStreamCb,
)
