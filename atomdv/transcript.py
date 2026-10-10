"""Complete record of an agent run: every LLM call, tool call, EDA command and route decision.

Writes two files into the run directory, flushed after every event so a crash still leaves them usable:
  transcript.log    human-readable, in order
  transcript.jsonl  one JSON event per line, for analysis

LLM and tool calls are captured by TranscriptCallback, attached once at graph.invoke(); nodes
never log their own LLM traffic, so new LLM nodes are recorded automatically.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

RULE = "=" * 100


class Transcript:
    def __init__(self, run_dir: Path, echo: bool = False):
        """`echo` also prints each event's one-line header to the terminal (a live view)."""
        run_dir.mkdir(parents=True, exist_ok=True)
        self._echo = echo
        self._log = open(run_dir / "transcript.log", "a", encoding="utf-8")
        self._jsonl = open(run_dir / "transcript.jsonl", "a", encoding="utf-8")
        self._seq = 0

    def event(
        self,
        kind: str,
        node: str,
        title: str = "",
        sections: dict[str, str] | None = None,
        attempt: int | None = None,
        duration_s: float | None = None,
        **data: Any,
    ) -> None:
        """Record one event. `sections` are the full texts shown in the .log (prompt, response, output)."""
        self._seq += 1
        sections = sections or {}
        header = f"{RULE}\n[{self._seq:03d}] {node} ({kind})"
        if attempt is not None:
            header += f"  attempt {attempt}"
        if duration_s is not None:
            header += f"  {duration_s:.2f}s"
        if title:
            header += f"  {title}"
        parts = [header]
        for name, text in sections.items():
            parts.append(f"--- {name} ---\n{text.rstrip()}")
        self._log.write("\n".join(parts) + "\n\n")
        self._log.flush()
        if self._echo:
            print(header.splitlines()[1], flush=True)

        record = {
            "seq": self._seq,
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "node": node,
            "kind": kind,
            "attempt": attempt,
            "title": title,
            "duration_s": duration_s,
            "sections": sections,
            **data,
        }
        self._jsonl.write(json.dumps(record, default=str) + "\n")
        self._jsonl.flush()

    def close(self) -> None:
        self._log.close()
        self._jsonl.close()


def _format_messages(messages: list[BaseMessage]) -> dict[str, str]:
    sections = {}
    for i, m in enumerate(messages):
        sections[f"{m.type.upper()} [{i}]"] = m.content if isinstance(m.content, str) else json.dumps(m.content)
    return sections


class TranscriptCallback(BaseCallbackHandler):
    """Records every chat-model call and tool call made inside the graph."""

    def __init__(self, transcript: Transcript):
        self.transcript = transcript
        self._pending: dict[UUID, dict] = {}

    def on_chat_model_start(
        self, serialized: dict, messages: list[list[BaseMessage]], *, run_id: UUID, metadata: dict | None = None, **kw
    ) -> None:
        metadata = metadata or {}
        params = kw.get("invocation_params") or {}
        self._pending[run_id] = {
            "start": time.monotonic(),
            "messages": messages[0],
            "metadata": metadata,
            # LangChain puts the model's standard params in metadata as ls_*.
            "model": metadata.get("ls_model_name") or params.get("model"),
            "temperature": metadata.get("ls_temperature", params.get("temperature")),
        }

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kw) -> None:
        call = self._pending.pop(run_id, None)
        if call is None:
            return
        gen = response.generations[0][0]
        msg = getattr(gen, "message", None)
        usage = getattr(msg, "usage_metadata", None) or {}
        sections = _format_messages(call["messages"])
        sections["RESPONSE"] = gen.text
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            sections["RESPONSE TOOL CALLS"] = json.dumps(tool_calls, indent=2, default=str)
        meta = call["metadata"]
        self.transcript.event(
            "llm",
            meta.get("langgraph_node", "?"),
            title=(
                f"model={call['model']} temp={call['temperature']} "
                f"in={usage.get('input_tokens', '?')} out={usage.get('output_tokens', '?')} tokens"
            ),
            sections=sections,
            attempt=meta.get("attempt"),
            duration_s=time.monotonic() - call["start"],
            model=call["model"],
            temperature=call["temperature"],
            tokens=usage,
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kw) -> None:
        call = self._pending.pop(run_id, {})
        meta = call.get("metadata", {})
        sections = _format_messages(call.get("messages", []))
        sections["ERROR"] = repr(error)
        self.transcript.event("llm_error", meta.get("langgraph_node", "?"), sections=sections, attempt=meta.get("attempt"))

    def on_tool_start(
        self, serialized: dict, input_str: str, *, run_id: UUID, metadata: dict | None = None, inputs: dict | None = None, **kw
    ) -> None:
        self._pending[run_id] = {
            "start": time.monotonic(),
            "name": (serialized or {}).get("name", "?"),
            "input": json.dumps(inputs, default=str) if inputs else input_str,
            "metadata": metadata or {},
        }

    def on_tool_end(self, output: Any, *, run_id: UUID, **kw) -> None:
        call = self._pending.pop(run_id, None)
        if call is None:
            return
        meta = call["metadata"]
        self.transcript.event(
            "tool",
            meta.get("langgraph_node", "?"),
            title=f"tool={call['name']}",
            sections={"ARGS": call["input"], "RESULT": str(getattr(output, "content", output))},
            attempt=meta.get("attempt"),
            duration_s=time.monotonic() - call["start"],
            tool=call["name"],
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kw) -> None:
        call = self._pending.pop(run_id, {})
        meta = call.get("metadata", {})
        self.transcript.event(
            "tool_error",
            meta.get("langgraph_node", "?"),
            title=f"tool={call.get('name', '?')}",
            sections={"ARGS": call.get("input", ""), "ERROR": repr(error)},
            attempt=meta.get("attempt"),
        )
