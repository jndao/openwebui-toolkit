"""
title: Context Manager
id: context_manager
author: jndao
description: An intelligent context-layer for OpenWebUI that maintains a permanent, compressed archive of technical decisions and user preferences while preserving full UI fidelity and token efficiency.
version: 0.0.1
author_url: https://github.com/jndao
repository_url: https://github.com/jndao/openwebui-toolkit
funding_url: https://ko-fi.com/jndao
license: https://github.com/jndao/openwebui-toolkit/blob/main/LICENSE
"""

import asyncio
import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable, Set, Tuple

from fastapi.requests import Request
from pydantic import BaseModel, Field

from open_webui.models.users import Users
from open_webui.utils.chat import generate_chat_completion
from open_webui.models.chats import Chats
from open_webui.internal.db import get_db_context

try:
    import tiktoken

    ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    ENCODING = None

try:
    from open_webui.internal.db import Base as owui_Base
except ImportError:
    owui_Base = None

try:
    from sqlalchemy import Column, Integer, String, Text, DateTime
except ImportError:
    Column = None
    Integer = None
    String = None
    Text = None
    DateTime = None

logger = logging.getLogger(__name__)
SUMMARY_TAG = "context_summary"
SUMMARY_SOURCE = "context_manager"

TOOL_DETAILS_BLOCK_RE = re.compile(r'<details type="tool_calls"[\s\S]*?</details>')
TOOL_RESULT_ATTR_RE = re.compile(r'result="([^"]*)"')


def _discover_owui_schema() -> Optional[str]:
    try:
        from open_webui.config import DATABASE_SCHEMA

        schema = (
            DATABASE_SCHEMA.value
            if hasattr(DATABASE_SCHEMA, "value")
            else DATABASE_SCHEMA
        )
        return schema if schema else None
    except Exception:
        return None


_owui_schema = _discover_owui_schema()
if owui_Base is not None and Column is not None:

    class ChatManifest(owui_Base):
        """Chat Summary Storage Table - stores compression metadata separately."""

        __tablename__ = "chat_manifests"
        __table_args__ = (
            {"extend_existing": True, "schema": _owui_schema}
            if _owui_schema
            else {"extend_existing": True}
        )

        id = Column(Integer, primary_key=True, autoincrement=True)
        chat_id = Column(String(255), unique=True, nullable=False)
        summary_content = Column(Text, nullable=False)
        until_timestamp = Column(Integer, nullable=True)
        created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
        updated_at = Column(
            DateTime,
            default=lambda: datetime.now(timezone.utc),
            onupdate=lambda: datetime.now(timezone.utc),
        )

else:
    ChatManifest = None


@dataclass
class SummaryState:
    content: str
    until_ts: Optional[int]
    raw: Optional[Dict[str, Any]] = None


@dataclass
class MessagePools:
    total_msgs: int
    protected_start: int
    protected_end: int
    tail_start_idx: int
    start_end_idx: int
    summary_indices: Set[int]
    protected_indices: Set[int]
    compressible_indices: Set[int]


class SummaryStore:
    def __init__(self):
        self._initialized = False
        self._init_error = None

    def _ensure_table(self):
        if self._initialized:
            return self._init_error is None

        self._initialized = True
        try:
            if ChatManifest is None:
                raise RuntimeError("Database table dependencies are unavailable")
            with get_db_context() as db:
                ChatManifest.__table__.create(bind=db.bind, checkfirst=True)
                db.commit()
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                return True
            self._init_error = str(e)
            logger.error(f"[SummaryStore] Failed to ensure table: {e}")
            return False

    def get(self, chat_id: str) -> Optional[Dict[str, Any]]:
        if not self._ensure_table():
            return None
        try:
            with get_db_context() as db:
                record = db.query(ChatManifest).filter_by(chat_id=chat_id).first()
                if record:
                    return {
                        "content": record.summary_content,
                        "until_timestamp": record.until_timestamp,
                        "updated_at": record.updated_at,
                    }
                return None
        except Exception as e:
            logger.error(f"[SummaryStore] Failed to get summary: {e}")
            return None

    def save(
        self, chat_id: str, content: str, until_timestamp: Optional[int] = None
    ) -> bool:
        if not self._ensure_table():
            return False
        try:
            with get_db_context() as db:
                record = db.query(ChatManifest).filter_by(chat_id=chat_id).first()
                if record:
                    record.summary_content = content
                    record.until_timestamp = until_timestamp
                    record.updated_at = datetime.now(timezone.utc)
                else:
                    record = ChatManifest(
                        chat_id=chat_id,
                        summary_content=content,
                        until_timestamp=until_timestamp,
                    )
                    db.add(record)
                db.commit()
        except Exception as e:
            logger.error(f"[SummaryStore] Failed to save summary: {e}")
            return False

        return True

    def delete(self, chat_id: str) -> bool:
        if not self._ensure_table():
            return False
        try:
            with get_db_context() as db:
                db.query(ChatManifest).filter_by(chat_id=chat_id).delete()
                db.commit()
            return True
        except Exception as e:
            logger.error(f"[SummaryStore] Failed to delete summary: {e}")
            return False


_summary_store: Optional[SummaryStore] = None


def _get_store() -> Optional[SummaryStore]:
    global _summary_store
    if _summary_store is None:
        _summary_store = SummaryStore()
    return _summary_store


def get_summary_from_store(chat_id: str) -> Optional[Dict[str, Any]]:
    store = _get_store()
    if store is None:
        return None
    return store.get(chat_id)


class TokenCounter:
    @staticmethod
    def count(item: Any) -> int:
        if isinstance(item, str):
            return TokenCounter._count_text(item)
        if isinstance(item, dict):
            return TokenCounter._count_message(item)
        if isinstance(item, list):
            return sum(TokenCounter.count(m) for m in item)
        return 0

    @staticmethod
    def _count_text(text: str) -> int:
        if ENCODING:
            try:
                return len(ENCODING.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    @staticmethod
    def _count_message(msg: Dict[str, Any]) -> int:
        total = 0
        content = msg.get("content", "")
        if isinstance(content, str):
            total += TokenCounter._count_text(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += TokenCounter._count_text(part.get("text", ""))
                elif isinstance(part, str):
                    total += TokenCounter._count_text(part)
        total += 4
        return total

    @staticmethod
    def extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif isinstance(part, str):
                    texts.append(part)
            return " ".join(texts)
        return ""


class ContextReconstructor:
    @staticmethod
    def trim_messages(
        messages: List[Dict[str, Any]],
        max_tokens: int,
        keep_start_messages: int = 0,
        keep_last_messages: int = 4,
    ) -> List[Dict[str, Any]]:
        if not messages:
            return messages

        total_tokens = sum(TokenCounter.count(m) for m in messages)
        if total_tokens <= max_tokens:
            return messages
        if keep_last_messages >= len(messages):
            return messages

        front_messages = (
            messages[:keep_start_messages] if keep_start_messages > 0 else []
        )
        front_tokens = sum(TokenCounter.count(m) for m in front_messages)

        tail_messages = messages[-keep_last_messages:] if keep_last_messages > 0 else []
        tail_tokens = sum(TokenCounter.count(m) for m in tail_messages)

        remaining_budget = max_tokens - front_tokens - tail_tokens
        if remaining_budget <= 0:
            return front_messages + tail_messages

        middle_messages = messages[
            keep_start_messages : len(messages) - keep_last_messages
        ]
        added = []
        added_tokens = 0

        for msg in middle_messages:
            msg_tokens = TokenCounter.count(msg)
            if added_tokens + msg_tokens <= remaining_budget:
                added.append(msg)
                added_tokens += msg_tokens
            else:
                break

        return front_messages + added + tail_messages

    def safe_hard_limit(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        keep_start_messages: int = 0,
        keep_last_messages: int = 4,
    ) -> List[Dict[str, Any]]:
        return self.trim_messages(
            messages,
            max_tokens,
            keep_start_messages=keep_start_messages,
            keep_last_messages=keep_last_messages,
        )

    @staticmethod
    def collapsed_tool_text() -> str:
        return "[TOOL OUTPUT COLLAPSED]"

    @staticmethod
    def _shorten_tool_call_id(value: str, max_len: int = 64) -> str:
        if not isinstance(value, str):
            return value

        raw = value.strip()
        if len(raw) <= max_len:
            return raw

        prefix_len = max(16, max_len // 2)
        suffix_len = max(8, max_len - prefix_len - 3)
        if prefix_len + suffix_len + 3 > max_len:
            suffix_len = max(4, max_len - prefix_len - 3)

        return f"{raw[:prefix_len]}...{raw[-suffix_len:]}"

    def normalize_tool_call_ids(self, messages: List[Dict[str, Any]]) -> int:
        rewritten_ids: Dict[str, str] = {}

        for message in messages:
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue

            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue

                original_id = tool_call.get("id")
                if not isinstance(original_id, str) or not original_id.strip():
                    continue
                if len(original_id.strip()) <= 64:
                    continue

                normalized_id = rewritten_ids.get(original_id)
                if normalized_id is None:
                    normalized_id = self._shorten_tool_call_id(original_id)
                    rewritten_ids[original_id] = normalized_id

                tool_call["id"] = normalized_id

        if not rewritten_ids:
            return 0

        for message in messages:
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str):
                continue

            normalized_id = rewritten_ids.get(tool_call_id)
            if normalized_id and normalized_id != tool_call_id:
                message["tool_call_id"] = normalized_id

        return sum(1 for old_id, new_id in rewritten_ids.items() if old_id != new_id)

    def trim_tool_content(
        self,
        messages: List[Dict[str, Any]],
        threshold: int,
        target_indices: Optional[Set[int]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        trimmed: List[Dict[str, Any]] = [deepcopy(msg) for msg in messages]
        stats = {
            "trimmed_count": 0,
            "chars_removed": 0,
            "tool_messages_trimmed": 0,
            "detail_blocks_trimmed": 0,
            "tool_call_ids_normalized": 0,
        }

        stats["tool_call_ids_normalized"] = self.normalize_tool_call_ids(trimmed)
        collapsed_text = self.collapsed_tool_text()

        for i, msg_copy in enumerate(trimmed):
            if target_indices is not None and i not in target_indices:
                continue

            role = msg_copy.get("role")
            if role == "tool":
                content = msg_copy.get("content")
                content_text = TokenCounter.extract_text(content)
                if content_text and len(content_text) > threshold:
                    removed = max(0, len(content_text) - len(collapsed_text))
                    msg_copy["content"] = collapsed_text
                    stats["trimmed_count"] += 1
                    stats["tool_messages_trimmed"] += 1
                    stats["chars_removed"] += removed

            content = msg_copy.get("content")
            if (
                not isinstance(content, str)
                or '<details type="tool_calls"' not in content
            ):
                continue

            def _replace_tool_result(match: re.Match) -> str:
                block = match.group(0)
                result_match = TOOL_RESULT_ATTR_RE.search(block)
                if not result_match:
                    return block

                result_payload = result_match.group(1)
                if len(result_payload) <= threshold:
                    return block

                removed = max(0, len(result_payload) - len(collapsed_text))
                stats["trimmed_count"] += 1
                stats["detail_blocks_trimmed"] += 1
                stats["chars_removed"] += removed

                return TOOL_RESULT_ATTR_RE.sub(
                    f'result="{collapsed_text}"',
                    block,
                    count=1,
                )

            msg_copy["content"] = TOOL_DETAILS_BLOCK_RE.sub(
                _replace_tool_result, content
            )

        return trimmed, stats


class Filter:
    class Valves(BaseModel):
        compression_threshold_tokens: int = Field(default=30000)
        max_context_tokens: int = Field(default=120000)
        keep_start_messages: int = Field(default=0)
        keep_last_messages: int = Field(default=10)
        summary_model: Optional[str] = Field(default=None)
        include_protected_in_threshold: bool = Field(default=True)
        tool_trim_threshold: int = Field(default=1000)
        protected_tool_trim_mode: str = Field(
            default="never"
        )  # never|overflow_only|always
        debug_logging: bool = Field(default=False)

    def __init__(self):
        self.valves = self.Valves()
        self.reconstructor = ContextReconstructor()
        self._locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, chat_id: str) -> asyncio.Lock:
        if chat_id not in self._locks:
            self._locks[chat_id] = asyncio.Lock()
        return self._locks[chat_id]

    async def _emit_status(
        self, emitter: Optional[Callable], message: str, done: bool = True
    ):
        if emitter is None:
            return
        try:
            await emitter(
                {"type": "status", "data": {"description": message, "done": done}}
            )
        except Exception as e:
            logger.debug(f"[Status Emit] Failed: {e}")

    def _timestamp_of(self, msg: Dict[str, Any]) -> Optional[int]:
        if not isinstance(msg, dict):
            return None

        msg_ts = msg.get("timestamp")
        if msg_ts is None:
            msg_ts = msg.get("created_at")
        if msg_ts is None:
            return None

        if isinstance(msg_ts, datetime):
            try:
                return int(msg_ts.timestamp())
            except Exception:
                return None

        if isinstance(msg_ts, (int, float)):
            return int(msg_ts)

        if isinstance(msg_ts, str):
            raw = msg_ts.strip()
            if not raw:
                return None
            try:
                return int(float(raw))
            except (TypeError, ValueError):
                pass
            try:
                return int(
                    datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
                )
            except Exception:
                return None

        return None

    def _message_tokens(self, messages: List[Dict[str, Any]], indices: Set[int]) -> int:
        return sum(
            TokenCounter.count(messages[i]) for i in indices if 0 <= i < len(messages)
        )

    def _message_identity(self, msg: Dict[str, Any]) -> str:
        if not isinstance(msg, dict):
            return ""

        for key in ("id", "message_id", "uuid"):
            value = msg.get(key)
            if value is not None:
                value_str = str(value).strip()
                if value_str:
                    return f"id:{value_str}"

        role = str(msg.get("role", ""))
        ts = self._timestamp_of(msg)
        content = TokenCounter.extract_text(msg.get("content", ""))
        return f"fallback:{role}:{ts}:{content[:200]}"

    def _clean_message_history(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not messages:
            return []

        filtered = list(messages)
        if filtered and not filtered[-1].get("content"):
            filtered = filtered[:-1]
        return filtered

    def _unfold_messages(self, messages: Any) -> List[Dict[str, Any]]:
        if not messages:
            return []

        result = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            msg = deepcopy(msg)
            children = msg.pop("children", None)
            if children:
                if isinstance(children, list) and children:
                    child = children[0] if isinstance(children[0], dict) else {}
                    child_msg = {**msg, **child}
                    child_msg.pop("children", None)
                    result.append(child_msg)
                else:
                    result.append(msg)
            else:
                result.append(msg)

        return result

    def _normalize_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        normalized = self._unfold_messages(messages)

        deduped: List[Dict[str, Any]] = []
        seen = set()
        for msg in normalized:
            identity = self._message_identity(msg)
            if identity and identity in seen:
                continue
            if identity:
                seen.add(identity)
            deduped.append(msg)

        indexed = list(enumerate(deduped))
        indexed.sort(
            key=lambda item: (
                self._timestamp_of(item[1]) is None,
                (
                    self._timestamp_of(item[1])
                    if self._timestamp_of(item[1]) is not None
                    else 0
                ),
                item[0],
            )
        )
        return [msg for _, msg in indexed]

    def _load_chat_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        return self._normalize_messages(
            self._clean_message_history(self._load_db_messages_by_chat_id(chat_id))
        )

    def _merge_body_last_message(
        self, messages: List[Dict[str, Any]], body: dict
    ) -> List[Dict[str, Any]]:
        merged = list(messages)
        if not isinstance(body, dict):
            return merged

        body_messages = body.get("messages")
        if not isinstance(body_messages, list) or not body_messages:
            return merged

        candidate = body_messages[-1]
        if not isinstance(candidate, dict):
            return merged

        candidate_identity = self._message_identity(candidate)
        if candidate_identity and any(
            self._message_identity(msg) == candidate_identity for msg in merged
        ):
            return merged

        merged.append(deepcopy(candidate))
        return merged

    def _get_summary_state(self, chat_id: str) -> SummaryState:
        summary_data = get_summary_from_store(chat_id)
        if not summary_data:
            return SummaryState(content="", until_ts=None, raw=None)

        return SummaryState(
            content=summary_data.get("content", "") or "",
            until_ts=summary_data.get("until_timestamp"),
            raw=summary_data,
        )

    def _split_message_pools(
        self,
        messages: List[Dict[str, Any]],
        summary_time: Optional[int],
        protected_start: int,
        protected_end: int,
    ) -> MessagePools:
        total_msgs = len(messages)
        protected_start = max(0, min(protected_start, total_msgs))
        protected_end = max(0, min(protected_end, total_msgs))

        start_end_idx = protected_start
        tail_start_idx = max(0, total_msgs - protected_end)

        protected_indices = set(range(start_end_idx))
        protected_indices.update(range(tail_start_idx, total_msgs))

        summary_indices = set()
        if summary_time is not None:
            for i, msg in enumerate(messages):
                msg_ts = self._timestamp_of(msg)
                if msg_ts is not None and msg_ts <= summary_time:
                    summary_indices.add(i)

        summary_indices -= protected_indices

        compressible_start = min(start_end_idx, tail_start_idx)
        compressible_end = max(start_end_idx, tail_start_idx)
        compressible_indices = (
            set(range(compressible_start, compressible_end))
            - summary_indices
            - protected_indices
        )

        return MessagePools(
            total_msgs=total_msgs,
            protected_start=protected_start,
            protected_end=protected_end,
            tail_start_idx=tail_start_idx,
            start_end_idx=start_end_idx,
            summary_indices=summary_indices,
            protected_indices=protected_indices,
            compressible_indices=compressible_indices,
        )

    def _debug_pool_snapshot(
        self,
        chat_id: str,
        messages: List[Dict[str, Any]],
        pools: MessagePools,
        summary_time: Optional[int],
        label: str,
    ):
        if not self.valves.debug_logging:
            return

        try:
            compressible_indices = sorted(pools.compressible_indices)
            summary_indices = sorted(pools.summary_indices)
            protected_indices = sorted(pools.protected_indices)

            logger.debug(
                "[Context Manager][%s][%s] total=%s summary_time=%s summary=%s/%s(%s tok) compressible=%s(%s tok) protected=%s(%s tok)",
                label,
                chat_id,
                len(messages),
                summary_time,
                len(summary_indices),
                len(messages),
                self._message_tokens(messages, set(summary_indices)),
                len(compressible_indices),
                self._message_tokens(messages, set(compressible_indices)),
                len(protected_indices),
                self._message_tokens(messages, set(protected_indices)),
            )

            for idx in compressible_indices[:5]:
                msg = messages[idx]
                logger.debug(
                    "[Context Manager][%s][%s] compressible idx=%s ts=%s role=%s tokens=%s id=%s",
                    label,
                    chat_id,
                    idx,
                    self._timestamp_of(msg),
                    msg.get("role"),
                    TokenCounter.count(msg),
                    self._message_identity(msg)[:120],
                )
        except Exception as exc:
            logger.debug(f"[Context Manager][debug] snapshot failed: {exc}")

    def _build_stats_message(
        self,
        messages: List[Dict[str, Any]],
        summary_content: Optional[str],
        until_ts: Optional[int],
    ) -> str:
        return self._generate_stats_event(
            messages=messages,
            summary=summary_content,
            summary_time=until_ts,
            protected_start=self.valves.keep_start_messages,
            protected_end=self.valves.keep_last_messages,
        )

    def _generate_stats_event(
        self,
        messages: List[Dict[str, Any]],
        summary: Optional[str] = None,
        summary_time: Optional[int] = None,
        protected_start: int = 0,
        protected_end: int = 0,
    ) -> str:
        if not messages:
            return "🪙 0 │ 🛡️ 0 (0) · ⏳ 0 (0) · 📦 0 (0)"

        pools = self._split_message_pools(
            messages, summary_time, protected_start, protected_end
        )

        summary_tokens = TokenCounter.count(summary) if summary else 0
        summarised_msgs = len(pools.summary_indices)
        summarised_tokens = self._message_tokens(messages, pools.summary_indices)

        protected_msgs = len(pools.protected_indices)
        protected_tokens = self._message_tokens(messages, pools.protected_indices)

        compressible_msgs = len(pools.compressible_indices)
        compressible_tokens = self._message_tokens(messages, pools.compressible_indices)

        def fmt(n: int) -> str:
            if n >= 1000:
                return f"{n/1000:.1f}k"
            return str(int(n))

        efficiency_part = ""
        if summarised_tokens > 0 and summary:
            efficiency = (1 - (summary_tokens / summarised_tokens)) * 100
            efficiency_part = f" @ {round(efficiency, 1)}%"

        projected_tokens = summary_tokens + protected_tokens + compressible_tokens

        return (
            f"🪙 {fmt(projected_tokens)} │ "
            f"🛡️ {fmt(protected_tokens)} ({protected_msgs}) · "
            f"⏳ {fmt(compressible_tokens)} ({compressible_msgs}) · "
            f"📦 {fmt(summary_tokens)} ({summarised_msgs}{efficiency_part})"
        )

    def _is_message_after_timestamp(
        self, message: Dict[str, Any], timestamp: int
    ) -> bool:
        msg_ts = self._timestamp_of(message)
        if msg_ts is None:
            return False
        return msg_ts > timestamp

    def _trim_message_content(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        if self.valves.tool_trim_threshold > 0:
            return self.reconstructor.trim_tool_content(
                messages,
                self.valves.tool_trim_threshold,
            )
        return list(messages), {"trimmed_count": 0, "chars_removed": 0}

    def _normalize_trim_mode(self) -> str:
        raw = str(self.valves.protected_tool_trim_mode or "never").strip().lower()
        if raw not in {"never", "overflow_only", "always"}:
            return "never"
        return raw

    def _should_trim_protected_for_inlet(
        self,
        messages: List[Dict[str, Any]],
        summary_state: SummaryState,
    ) -> bool:
        mode = self._normalize_trim_mode()
        if mode == "never":
            return False
        if mode == "always":
            return True

        projected_tokens = TokenCounter.count(messages)
        if summary_state.raw:
            pools = self._split_message_pools(
                messages,
                summary_state.until_ts,
                self.valves.keep_start_messages,
                self.valves.keep_last_messages,
            )
            summary_tokens = TokenCounter.count(summary_state.content)
            summarised_tokens = self._message_tokens(messages, pools.summary_indices)
            projected_tokens = projected_tokens - summarised_tokens + summary_tokens

        return projected_tokens > self.valves.max_context_tokens

    def _build_inlet_trim_targets(
        self,
        messages: List[Dict[str, Any]],
        summary_state: SummaryState,
        pools: MessagePools,
    ) -> Tuple[Set[int], bool]:
        target_indices = set(pools.compressible_indices)
        trim_protected = self._should_trim_protected_for_inlet(messages, summary_state)
        if trim_protected:
            target_indices.update(pools.protected_indices)
        return target_indices, trim_protected

    def _build_inlet_messages(
        self,
        db_messages: List[Dict[str, Any]],
        summary_state: SummaryState,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        pools = self._split_message_pools(
            db_messages,
            summary_state.until_ts,
            self.valves.keep_start_messages,
            self.valves.keep_last_messages,
        )

        target_indices, trim_protected = self._build_inlet_trim_targets(
            db_messages, summary_state, pools
        )

        trimmed_messages, trim_stats = self.reconstructor.trim_tool_content(
            db_messages,
            self.valves.tool_trim_threshold,
            target_indices=target_indices,
        )
        trim_stats["protected_trim_applied"] = 1 if trim_protected else 0

        if summary_state.raw:
            summary_msg = {
                "role": "system",
                "content": f"<{SUMMARY_TAG}>\n{summary_state.content}\n</{SUMMARY_TAG}>",
            }
            active_messages = [summary_msg]
            if summary_state.until_ts is not None:
                for msg in trimmed_messages:
                    if self._is_message_after_timestamp(msg, summary_state.until_ts):
                        active_messages.append(msg)
            else:
                active_messages.extend(trimmed_messages)
        else:
            active_messages = trimmed_messages

        active_messages = self._normalize_messages(active_messages)
        active_messages = self.reconstructor.safe_hard_limit(
            active_messages,
            self.valves.max_context_tokens,
            keep_start_messages=self.valves.keep_start_messages,
            keep_last_messages=self.valves.keep_last_messages,
        )

        return active_messages, trim_stats

    def _compression_metrics(
        self,
        messages: List[Dict[str, Any]],
        pools: MessagePools,
    ) -> Tuple[int, int]:
        compressible_tokens = self._message_tokens(messages, pools.compressible_indices)
        protected_tokens = self._message_tokens(messages, pools.protected_indices)

        threshold_tokens = (
            compressible_tokens + protected_tokens
            if self.valves.include_protected_in_threshold
            else compressible_tokens
        )

        return compressible_tokens, threshold_tokens

    def _prepare_outlet_state(
        self,
        chat_id: str,
        body: dict,
    ) -> Dict[str, Any]:
        messages_raw = self._load_chat_messages(chat_id)
        messages_raw = self._normalize_messages(
            self._merge_body_last_message(messages_raw, body)
        )

        summary_state = self._get_summary_state(chat_id)

        raw_pools = self._split_message_pools(
            messages_raw,
            summary_state.until_ts,
            self.valves.keep_start_messages,
            self.valves.keep_last_messages,
        )
        raw_compressible_tokens, raw_threshold_tokens = self._compression_metrics(
            messages_raw, raw_pools
        )

        messages_working, trim_stats = self._trim_message_content(messages_raw)
        messages_working = self._normalize_messages(messages_working)

        working_pools = self._split_message_pools(
            messages_working,
            summary_state.until_ts,
            self.valves.keep_start_messages,
            self.valves.keep_last_messages,
        )
        working_compressible = [
            messages_working[i] for i in sorted(working_pools.compressible_indices)
        ]

        return {
            "messages_raw": messages_raw,
            "messages_working": messages_working,
            "summary_state": summary_state,
            "raw_pools": raw_pools,
            "working_pools": working_pools,
            "raw_compressible_tokens": raw_compressible_tokens,
            "raw_threshold_tokens": raw_threshold_tokens,
            "working_compressible": working_compressible,
            "trim_stats": trim_stats,
        }

    def _get_chat_id(self, body: dict, metadata: Optional[dict]) -> Optional[str]:
        if metadata and isinstance(metadata, dict):
            chat_id = metadata.get("chat_id") or metadata.get("chatId")
            if chat_id:
                return str(chat_id)

        if body and isinstance(body, dict):
            chat_id = body.get("chat_id") or body.get("chatId")
            if chat_id:
                return str(chat_id)

            meta = body.get("meta", {})
            if isinstance(meta, dict):
                chat_id = meta.get("chat_id") or meta.get("chatId")
                if chat_id:
                    return str(chat_id)

        return None

    def _load_db_messages_by_chat_id(self, chat_id: str) -> List[Dict[str, Any]]:
        if not chat_id or Chats is None:
            return []

        try:
            chat_record = Chats.get_chat_by_id(chat_id)
        except Exception as exc:
            logger.warning(f"[Chat Load] Failed to fetch chat {chat_id}: {exc}")
            return []

        chat_payload = getattr(chat_record, "chat", None)
        if not isinstance(chat_payload, dict):
            return []

        direct_messages = chat_payload.get("messages")
        if isinstance(direct_messages, list) and direct_messages:
            return self._unfold_messages(deepcopy(direct_messages))

        history = chat_payload.get("history")
        if not isinstance(history, dict):
            return []

        history_messages = history.get("messages")
        if not isinstance(history_messages, dict) or not history_messages:
            return []

        current_id = history.get("currentId") or history.get("current_id")
        return self._unfold_messages(
            self._reconstruct_active_history_branch(history_messages, current_id)
        )

    def _reconstruct_active_history_branch(
        self, history_messages: Any, current_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        if not isinstance(history_messages, dict) or not history_messages:
            return []

        if isinstance(current_id, str) and current_id in history_messages:
            ordered_messages: List[Dict[str, Any]] = []
            visited = set()
            cursor = current_id

            while isinstance(cursor, str) and cursor and cursor not in visited:
                visited.add(cursor)
                node = history_messages.get(cursor)
                if not isinstance(node, dict):
                    break
                ordered_messages.append(deepcopy(node))
                cursor = node.get("parentId") or node.get("parent_id")

            if ordered_messages:
                ordered_messages.reverse()
                return ordered_messages

        sortable_messages = []
        for index, node in enumerate(history_messages.values()):
            if not isinstance(node, dict):
                continue

            timestamp = self._timestamp_of(node)
            if timestamp is None:
                timestamp = index
            sortable_messages.append((float(timestamp), index, deepcopy(node)))

        sortable_messages.sort(key=lambda item: (item[0], item[1]))
        return [message for _, _, message in sortable_messages]

    async def inlet(
        self,
        body: dict,
        __user__: dict = None,
        __metadata__: dict = None,
        __event_emitter__: Callable = None,
        __event_call__: Callable = None,
        __request__: Request = None,
    ) -> dict:
        chat_id = self._get_chat_id(body, __metadata__)
        if not chat_id:
            return body

        messages_raw = self._load_chat_messages(chat_id)
        summary_state = self._get_summary_state(chat_id)

        active_messages, trim_stats = self._build_inlet_messages(
            messages_raw, summary_state
        )
        body["messages"] = active_messages

        inlet_pools = self._split_message_pools(
            messages_raw,
            summary_state.until_ts,
            self.valves.keep_start_messages,
            self.valves.keep_last_messages,
        )
        self._debug_pool_snapshot(
            chat_id, messages_raw, inlet_pools, summary_state.until_ts, "inlet"
        )

        if self.valves.debug_logging and trim_stats.get("trimmed_count"):
            protected_note = (
                " protected=on" if trim_stats.get("protected_trim_applied") else ""
            )
            logger.debug(
                "[Context Manager][inlet][%s] tool trim applied in-memory only: trimmed=%s removed_chars=%s%s",
                chat_id,
                trim_stats.get("trimmed_count", 0),
                trim_stats.get("chars_removed", 0),
                protected_note,
            )

        stats_message = self._build_stats_message(
            messages=messages_raw,
            summary_content=summary_state.content,
            until_ts=summary_state.until_ts,
        )
        await self._emit_status(__event_emitter__, f"🔄{stats_message}", done=True)
        return body

    async def outlet(
        self,
        body: dict,
        __user__: dict = None,
        __metadata__: dict = None,
        __event_emitter__: Callable = None,
        __event_call__: Callable = None,
        __request__: Request = None,
    ) -> dict:
        chat_id = self._get_chat_id(body, __metadata__)
        if not chat_id:
            return body

        outlet_state = self._prepare_outlet_state(chat_id, body)
        messages_raw = outlet_state["messages_raw"]
        messages_working = outlet_state["messages_working"]
        summary_state = outlet_state["summary_state"]
        working_pools = outlet_state["working_pools"]
        raw_compressible_tokens = outlet_state["raw_compressible_tokens"]
        raw_threshold_tokens = outlet_state["raw_threshold_tokens"]
        working_compressible = outlet_state["working_compressible"]
        trim_stats = outlet_state["trim_stats"]

        self._debug_pool_snapshot(
            chat_id,
            messages_working,
            working_pools,
            summary_state.until_ts,
            "outlet-before",
        )

        if self.valves.debug_logging and trim_stats.get("trimmed_count"):
            logger.debug(
                "[Context Manager][outlet][%s] tool trim applied to summarization snapshot only: trimmed=%s removed_chars=%s",
                chat_id,
                trim_stats.get("trimmed_count", 0),
                trim_stats.get("chars_removed", 0),
            )

        if (
            raw_threshold_tokens > self.valves.compression_threshold_tokens
            and working_compressible
        ):
            await self._emit_status(
                __event_emitter__,
                f"Summarizing {raw_compressible_tokens:,} new tokens...",
                done=False,
            )

            lock = self._lock_for(chat_id)
            if not lock.locked():
                await self._background_compress(
                    lock,
                    chat_id,
                    summary_state.content,
                    working_compressible,
                    self.valves.summary_model or body.get("model"),
                    __user__,
                    __event_emitter__,
                    __request__,
                )

                summary_state = self._get_summary_state(chat_id)
                refreshed_working_pools = self._split_message_pools(
                    messages_working,
                    summary_state.until_ts,
                    self.valves.keep_start_messages,
                    self.valves.keep_last_messages,
                )
                self._debug_pool_snapshot(
                    chat_id,
                    messages_working,
                    refreshed_working_pools,
                    summary_state.until_ts,
                    "outlet-after",
                )

        stats_message = self._build_stats_message(
            messages=messages_raw,
            summary_content=summary_state.content,
            until_ts=summary_state.until_ts,
        )
        await self._emit_status(__event_emitter__, f"✅{stats_message}", done=True)
        return body

    async def _background_compress(
        self,
        lock: asyncio.Lock,
        chat_id: str,
        old_summary_content: str,
        compressible: List[Dict[str, Any]],
        model_id: Optional[str],
        user_data: Optional[dict],
        emitter: Optional[Callable],
        request: Optional[Request],
    ):
        async with lock:
            try:
                if not model_id:
                    await self._emit_status(emitter, "⚠️ Summary failed: missing model")
                    return

                if not compressible:
                    return

                pool_text = "\n".join(
                    f"{m.get('role', 'user').upper()}: {TokenCounter.extract_text(m.get('content', ''))}"
                    for m in compressible
                    if m.get("role") != "system"
                ).strip()
                if not pool_text:
                    return

                prompt = f"""
You are the "Context Architect" responsible for maintaining a conversation's Permanent Archive.
### CURRENT ARCHIVE:
{old_summary_content or "No existing archive."}
### NEW EVENTS TO INTEGRATE:
{pool_text}
### YOUR TASK:
Integrate the NEW EVENTS into the CURRENT ARCHIVE to create a single, cohesive, and updated summary.
### GUIDELINES:
1. UPDATE FACTS: If new information contradicts or evolves the archive (e.g., a change in project tech stack or user preference), update the record accordingly.
2. CATEGORIZE: Maintain sections for [Technical Decisions], [User Preferences], and [Key Narrative/Context].
3. DEDUPLICATE: Do not repeat information already clearly stated in the archive.
4. BE ATOMIC: Use concise bullet points. Focus on "What was decided" and "What is the current state."
5. IGNORE: Discard greetings, meta-talk about the AI, and transient errors.
### OUTPUT:
Provide ONLY the updated archive text. No conversational filler, no markdown blocks, no "Here is the summary."
"""

                user = None
                if user_data and user_data.get("id"):
                    user = await asyncio.to_thread(
                        Users.get_user_by_id, user_data.get("id")
                    )
                if user is None:
                    await self._emit_status(
                        emitter, "⚠️ Summary failed: missing user context"
                    )
                    return

                payload = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": 0,
                }

                active_request = request or Request(scope={"type": "http"})
                response = await generate_chat_completion(active_request, payload, user)

                if hasattr(response, "body"):
                    response = json.loads(response.body.decode())

                if not (
                    response and isinstance(response, dict) and response.get("choices")
                ):
                    await self._emit_status(
                        emitter, "⚠️ Summary failed: invalid model response"
                    )
                    return

                new_summary_content = response["choices"][0]["message"][
                    "content"
                ].strip()
                if not new_summary_content:
                    await self._emit_status(emitter, "⚠️ Summary failed: empty summary")
                    return

                valid_ts = [self._timestamp_of(m) for m in compressible]
                valid_ts = [ts for ts in valid_ts if ts is not None]
                until_ts = (
                    max(valid_ts)
                    if valid_ts
                    else int(datetime.now(timezone.utc).timestamp())
                )

                store = _get_store()
                if store is None:
                    await self._emit_status(
                        emitter, "⚠️ Summary failed: storage not available"
                    )
                    return

                saved = store.save(
                    chat_id=chat_id,
                    content=new_summary_content,
                    until_timestamp=until_ts,
                )
                if not saved:
                    await self._emit_status(
                        emitter, "⚠️ Summary generated but save failed"
                    )
                    return

                pool_tokens = sum(TokenCounter.count(m) for m in compressible)
                summary_tokens = TokenCounter.count(new_summary_content)

                efficiency = None
                if pool_tokens > 0:
                    efficiency = max(
                        0.0, min(100.0, (1.0 - (summary_tokens / pool_tokens)) * 100.0)
                    )

                eff_str = (
                    f" {efficiency:.2f}% efficiency" if efficiency is not None else ""
                )
                await self._emit_status(emitter, f"💾 Summary saved!{eff_str}")

            except Exception as e:
                logger.error(
                    f"[Context Manager] Background compression failed: {e}",
                    exc_info=True,
                )
                await self._emit_status(emitter, f"⚠️ Summary failed: {str(e)[:80]}")
