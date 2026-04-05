import json


def build_readable_history(messages, strip_thinking_func):
    readable = []
    for msg in messages or []:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if role == "user" and content:
            readable.append({"role": "user", "content": str(content)[:4000]})
        elif role == "assistant" and content:
            clean, _ = strip_thinking_func(str(content))
            if clean.strip():
                readable.append({"role": "assistant", "content": clean[:4000]})
        elif role == "tool":
            tool_name = msg.get("name", "tool")
            try:
                result = json.loads(str(content)) if content else {}
                success = bool(result.get("success", True))
            except Exception:
                success = True
            readable.append({"role": "tool", "name": tool_name, "success": success})
    return readable


def load_session_inspect_payload(
    workspace,
    sid,
    *,
    todo_manager_cls,
    task_state_manager_cls,
    plan_manager_cls,
    session_manager_cls,
    strip_thinking_func,
):
    todos = todo_manager_cls(workspace, sid).list_all()
    task_state = task_state_manager_cls(workspace, sid).load()
    plan_mgr = plan_manager_cls(workspace, sid)

    session_data = session_manager_cls(workspace).load(sid)
    if not session_data:
        raise LookupError(f"session not found: {sid}")

    messages, metadata = session_data
    return {
        "todos": todos,
        "tasks": {"state": task_state.to_dict() if task_state else None},
        "plan": {"plan": plan_mgr.plan, "current_step_id": plan_mgr.current_step_id},
        "history": {
            "messages": build_readable_history(messages, strip_thinking_func),
            "metadata": metadata,
        },
    }
