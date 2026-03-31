"""
title: Context Manager
id: context_manager
author: jndao
description: An intelligent context-layer for OpenWebUI that preserves multimodal inputs while maintaining a permanent compressed archive and token efficiency.
version: 0.1.0-preview.4
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
    """Message pools returned as actual message lists."""

    protected_start: List[Dict[str, Any]]
    summarized: List[Dict[str, Any]]
    compressible: List[Dict[str, Any]]
    protected_end: List[Dict[str, Any]]


@dataclass
class RuntimeSegments:
    """Runtime message segments in deterministic merge order."""

    protected_start: List[Dict[str, Any]]
    summary_message: Optional[Dict[str, Any]]
    media_messages: List[Dict[str, Any]]
    uncompressed: List[Dict[str, Any]]
    protected_end: List[Dict[str, Any]]

    @property
    def final_messages(self) -> List[Dict[str, Any]]:
        """Build final message list in runtime order:
        protected_start → summary → media → uncompressed → protected_end
        """
        merged: List[Dict[str, Any]] = []
        merged.extend(self.protected_start)
        if self.summary_message:
            merged.append(self.summary_message)
        merged.extend(self.uncompressed)
        merged.extend(self.media_messages)
        merged.extend(self.protected_end)
        return merged


@dataclass
class RuntimeView:
    """Runtime view with explicit segments and accurate stats."""

    final_messages: List[Dict[str, Any]]
    stats_message: str
    segments: RuntimeSegments
    total_tokens: int
    protected_tokens: int
    uncompressed_tokens: int
    summary_tokens: int
    media_tokens: int


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
        elif isinstance(content, dict):
            total += TokenCounter._count_text(TokenCounter.extract_text(content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    part_type = str(part.get("type", "")).strip().lower()
                    if part_type in {"text", "input_text"}:
                        total += TokenCounter._count_text(
                            part.get("text", "") or part.get("content", "")
                        )
                elif isinstance(part, str):
                    total += TokenCounter._count_text(part)

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_id = tool_call.get("id")
                if isinstance(tool_call_id, str):
                    total += TokenCounter._count_text(tool_call_id)
                tool_call_type = tool_call.get("type")
                if isinstance(tool_call_type, str):
                    total += TokenCounter._count_text(tool_call_type)
                function_payload = tool_call.get("function")
                if isinstance(function_payload, dict):
                    function_name = function_payload.get("name")
                    if isinstance(function_name, str):
                        total += TokenCounter._count_text(function_name)
                    arguments = function_payload.get("arguments")
                    if isinstance(arguments, str):
                        total += TokenCounter._count_text(arguments)

        tool_call_id = msg.get("tool_call_id")
        if isinstance(tool_call_id, str):
            total += TokenCounter._count_text(tool_call_id)

        name = msg.get("name")
        if isinstance(name, str):
            total += TokenCounter._count_text(name)

        total += 4
        return total

    @staticmethod
    def extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            content_type = str(content.get("type", "")).strip().lower()
            if content_type in {"text", "input_text"}:
                text_value = content.get("text")
                if isinstance(text_value, str):
                    return text_value
                nested_content = content.get("content")
                if isinstance(nested_content, str):
                    return nested_content
            return ""
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict):
                    part_type = str(part.get("type", "")).strip().lower()
                    if part_type in {"text", "input_text"}:
                        text_value = part.get("text")
                        if isinstance(text_value, str) and text_value:
                            texts.append(text_value)
                            continue
                        nested_content = part.get("content")
                        if isinstance(nested_content, str) and nested_content:
                            texts.append(nested_content)
                elif isinstance(part, str):
                    texts.append(part)
            return " ".join(t for t in texts if t)
        return ""


class ContextReconstructor:
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
            "tool_arguments_trimmed": 0,
            "detail_blocks_trimmed": 0,
            "tool_call_ids_normalized": 0,
        }

        stats["tool_call_ids_normalized"] = self.normalize_tool_call_ids(trimmed)
        collapsed_text = self.collapsed_tool_text()

        for i, msg_copy in enumerate(trimmed):
            if target_indices is not None and i not in target_indices:
                continue

            if msg_copy.get("role") == "tool":
                content = msg_copy.get("content")
                content_text = TokenCounter.extract_text(content)
                if content_text and TokenCounter._count_text(content_text) > threshold:
                    removed = max(0, len(content_text) - len(collapsed_text))
                    msg_copy["content"] = collapsed_text
                    stats["trimmed_count"] += 1
                    stats["tool_messages_trimmed"] += 1
                    stats["chars_removed"] += removed

            tool_calls = msg_copy.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    function_payload = tool_call.get("function")
                    if not isinstance(function_payload, dict):
                        continue
                    arguments = function_payload.get("arguments")
                    if (
                        isinstance(arguments, str)
                        and TokenCounter._count_text(arguments) > threshold
                    ):
                        removed = max(0, len(arguments) - len(collapsed_text))
                        function_payload["arguments"] = collapsed_text
                        stats["trimmed_count"] += 1
                        stats["tool_arguments_trimmed"] += 1
                        stats["chars_removed"] += removed

            content = msg_copy.get("content")
            if not isinstance(content, str) or '<details type="tool_calls"' not in content:
                continue

            def _replace_tool_result(match: re.Match) -> str:
                block = match.group(0)
                result_match = TOOL_RESULT_ATTR_RE.search(block)
                if not result_match:
                    return block
                result_payload = result_match.group(1)

                if TokenCounter._count_text(result_payload) <= threshold:
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

            msg_copy["content"] = TOOL_DETAILS_BLOCK_RE.sub(_replace_tool_result, content)

        return trimmed, stats


class Filter:
    class Valves(BaseModel):
        emit_status_events: bool = Field(
            default=True,
            description="Toggle whether users should see Context Manager events in OWUI",
        )
        compression_threshold_tokens: int = Field(
            default=40000,
            description="Trigger archival when the compressible zone exceeds this token count.",
        )
        max_context_tokens: int = Field(
            default=120000,
            description="Hard limit for the model context window. Oldest non-protected messages are shed if exceeded.",
        )
        keep_start_messages: int = Field(
            default=0,
            description="Number of messages at the start of the chat to protect.",
        )
        keep_last_messages: int = Field(
            default=10,
            description="Number of recent messages to protect at the end of the chat.",
        )
        summary_model: Optional[str] = Field(
            default=None,
            description="Model ID to use for background summarization.",
        )
        include_protected_in_threshold: bool = Field(
            default=True,
            description="If true, protected messages count toward the compression threshold.",
        )
        tool_trim_threshold: int = Field(
            default=1000,
            description="Tool outputs, tool-call arguments, or <details> result blocks larger than this token count are eligible for trimming.",
        )
        trim_protected_messages: bool = Field(
            default=False,
            description="Apply tool content trimming to protected messages (protected_start and protected_end).",
        )
        debug_logging: bool = Field(
            default=False,
            description="Enable detailed console logging.",
        )

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
        if emitter is None or not self.valves.emit_status_events:
            return
        try:
            await emitter(
                {"type": "status", "data": {"description": message, "done": done}}
            )
        except Exception as e:
            logger.debug(f"[Status Emit] Failed: {e}")

    def _normalize_epoch_timestamp(self, value: Any) -> Optional[int]:
        try:
            if isinstance(value, datetime):
                return int(value.timestamp())
            if isinstance(value, str):
                raw = value.strip()
                if not raw:
                    return None
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    try:
                        return int(
                            datetime.fromisoformat(
                                raw.replace("Z", "+00:00")
                            ).timestamp()
                        )
                    except Exception:
                        return None
            if isinstance(value, (int, float)):
                numeric = float(value)
                if numeric <= 0:
                    return None
                if numeric > 1e18:
                    numeric /= 1_000_000_000
                elif numeric > 1e15:
                    numeric /= 1_000_000
                elif numeric > 1e12:
                    numeric /= 1000
                ts = int(numeric)
                return ts
        except Exception:
            return None
        return None

    def _timestamp_of(self, msg: Dict[str, Any]) -> Optional[int]:
        if not isinstance(msg, dict):
            return None
        msg_ts = msg.get("timestamp")
        if msg_ts is None:
            msg_ts = msg.get("created_at")
        return self._normalize_epoch_timestamp(msg_ts)

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

    # Fields to keep for LLM context (scrub everything else)
    KEEP_FIELDS = frozenset({
        "id", "parentId", "role", "content", "timestamp"
    })

    def _scrub_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Remove unnecessary fields from a message, keeping only essential fields."""
        if not isinstance(msg, dict):
            return {}
        return {k: v for k, v in msg.items() if k in self.KEEP_FIELDS}

    def _prepare_db_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare DB messages for LLM context.
        
        - Unfolds children structure
        - Deduplicates by identity
        - Preserves original order (no timestamp sorting)
        - Scrubs unnecessary fields
        """
        if not messages:
            return []
        # Step 1: Unfold children structure
        unfolded = self._unfold_messages(messages)
        
        # Step 2: Remove any empty messages
        cleaned = [x for x in unfolded if x['content']]
        
        # Step 4: Scrub unnecessary fields
        return [self._scrub_message(msg) for msg in cleaned]

    def _load_chat_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        return self._prepare_db_messages(
            self._load_db_messages_by_chat_id(chat_id)
        )

    def _extract_media_messages_from_body(self, body: dict) -> List[Dict[str, Any]]:
        """Extract media messages from body without normalization.
        
        Body messages are already in correct order - just filter and dedupe.
        No need to normalize or scrub - preserve original structure.
        """
        if not isinstance(body, dict):
            return []
        body_messages = body.get("messages")
        if not isinstance(body_messages, list):
            return []
        
        media_messages: List[Dict[str, Any]] = []
        seen = set()
        for msg in body_messages:
            if not isinstance(msg, dict):
                continue
            if not self._message_has_passthrough_media(msg):
                continue
            
            identity = self._message_identity(msg)
            if identity and identity in seen:
                continue
            if identity:
                seen.add(identity)
            
            media_messages.append(deepcopy(msg))
            
        # Return as-is - no normalization, no scrubbing
        return media_messages

    def _get_summary_state(self, chat_id: str) -> SummaryState:
        summary_data = get_summary_from_store(chat_id)
        if not summary_data:
            return SummaryState(content="", until_ts=None, raw=None)
        return SummaryState(
            content=summary_data.get("content", "") or "",
            until_ts=summary_data.get("until_timestamp"),
            raw=summary_data,
        )

    def _build_summary_message(
        self, summary_state: SummaryState
    ) -> Optional[Dict[str, Any]]:
        """Build the summary message to inject into the runtime payload."""
        if not summary_state or not summary_state.content:
            return None
        return {
            "role": "system",
            "content": f"<{SUMMARY_TAG}>\n{summary_state.content}\n</{SUMMARY_TAG}>",
        }

    def _count_tokens_in_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count total tokens in a list of messages."""
        if not messages:
            return 0
        return sum(
            TokenCounter.count(msg)
            for msg in messages
            if isinstance(msg, dict)
        )

    def _format_token_count(self, n: int) -> str:
        """Format token count for display."""
        if n >= 1000:
            return f"{n / 1000:.1f}k"
        return str(int(n))

    def _build_runtime_stats_message(
        self,
        summarized_messages: List[Dict[str, Any]],
        protected_messages: List[Dict[str, Any]],
        summary_message: Optional[Dict[str, Any]],
        uncompressed_messages: List[Dict[str, Any]],
        media_messages: List[Dict[str, Any]],
        was_shed: bool = False,
    ) -> Tuple[str, int, int, int, int, int]:
        """Build runtime stats message from explicit segments.
        
        Returns tuple of (stats_message, total_tokens, protected_tokens,
                         uncompressed_tokens, summary_tokens, media_tokens)
        """
        protected_count = len(protected_messages)
        uncompressed_count = len(uncompressed_messages)

        protected_tokens = self._count_tokens_in_messages(protected_messages)
        summarized_tokens = self._count_tokens_in_messages(summarized_messages)
        summary_tokens = (
            self._count_tokens_in_messages([summary_message])
            if summary_message else 0
        )
        uncompressed_tokens = self._count_tokens_in_messages(uncompressed_messages)
        media_tokens = self._count_tokens_in_messages(media_messages)

        total_tokens = (
            protected_tokens
            + summary_tokens
            + uncompressed_tokens
            + media_tokens
        )

        stats = (
            f"🪙 {self._format_token_count(total_tokens)} │ "
            f"🛡️ {self._format_token_count(protected_tokens)} ({protected_count}) · "
            f"⏳ {self._format_token_count(uncompressed_tokens)} ({uncompressed_count}) · "
            f"📦 {self._format_token_count(summary_tokens)} ({len(summarized_messages)}{ f' @ {round((summarized_tokens - summary_tokens)/summarized_tokens * 100, 2)}%' if summarized_tokens > 0 else ''})"
        )

        if was_shed:
            stats = f"⚠️ Limit Reached │ {stats}"

        return (
            stats,
            total_tokens,
            protected_tokens,
            uncompressed_tokens,
            summary_tokens,
            media_tokens,
        )

    def _split_message_pools(
        self,
        messages: List[Dict[str, Any]],
        summary_time: Optional[int],
        keep_start: int,
        keep_end: int,
    ) -> MessagePools:
        """Split messages into pools as actual message lists."""
        total = len(messages)
        start_cut = min(max(keep_start, 0), total)
        end_count = min(max(keep_end, 0), max(0, total - start_cut))
        end_start = total - end_count

        protected_start = list(messages[:start_cut])
        protected_end = list(messages[end_start:]) if end_count > 0 else []
        middle = list(messages[start_cut:end_start])

        summarized: List[Dict[str, Any]] = []
        compressible: List[Dict[str, Any]] = []

        for msg in middle:
            ts = self._timestamp_of(msg)
            if summary_time is not None and ts is not None and ts <= summary_time:
                summarized.append(msg)
            else:
                compressible.append(msg)

        return MessagePools(
            protected_start=protected_start,
            summarized=summarized,
            compressible=compressible,
            protected_end=protected_end,
        )

    def _message_has_passthrough_media(self, message: Dict[str, Any]) -> bool:
        if not isinstance(message, dict):
            return False
        content = message.get("content")
        media_types = {"image_url", "file", "input_image", "input_file"}
        if isinstance(content, dict):
            return str(content.get("type", "")).strip().lower() in media_types
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if str(part.get("type", "")).strip().lower() in media_types:
                    return True
        return False

    def _build_text_only_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(message, dict):
            return None
        text_content = TokenCounter.extract_text(message.get("content", ""))
        if not text_content:
            return None

        text_message = {
            "role": message.get("role", "user"),
            "content": text_content,
        }

        for key in ("timestamp", "created_at", "id", "message_id", "uuid"):
            if key in message:
                text_message[key] = message[key]

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            text_tool_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_copy = deepcopy(tool_call)
                text_tool_calls.append(tool_call_copy)
            if text_tool_calls:
                text_message["tool_calls"] = text_tool_calls

        if "tool_call_id" in message:
            text_message["tool_call_id"] = message["tool_call_id"]
        if "name" in message:
            text_message["name"] = message["name"]

        return text_message

    def _build_text_only_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        text_messages: List[Dict[str, Any]] = []
        for msg in messages:
            text_msg = self._build_text_only_message(msg)
            if text_msg is not None:
                text_messages.append(text_msg)
        return text_messages

    def _package_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        packaged: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            msg_copy = deepcopy(msg)
            msg_copy.pop("children", None)
            packaged.append(msg_copy)
        return packaged

    def _enforce_context_limits(
        self,
        protected_start: List[Dict[str, Any]],
        summary_message: Optional[Dict[str, Any]],
        media_messages: List[Dict[str, Any]],
        uncompressed: List[Dict[str, Any]],
        protected_end: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], bool]:
        """Sheds oldest messages to ensure the payload fits within max_context_tokens."""
        max_tokens = self.valves.max_context_tokens
        if max_tokens <= 0:
            return media_messages, uncompressed, protected_end, False

        p_start_tok = self._count_tokens_in_messages(protected_start)
        sum_tok = self._count_tokens_in_messages([summary_message]) if summary_message else 0
        media_tok = self._count_tokens_in_messages(media_messages)
        uncomp_tok = self._count_tokens_in_messages(uncompressed)
        p_end_tok = self._count_tokens_in_messages(protected_end)

        total = p_start_tok + sum_tok + media_tok + uncomp_tok + p_end_tok
        was_shed = False

        # Shedding priority: 1. Oldest Uncompressed -> 2. Oldest Media -> 3. Oldest Protected End
        while total > max_tokens:
            was_shed = True
            if uncompressed:
                dropped = uncompressed.pop(0)
                dropped_tok = TokenCounter.count(dropped)
                uncomp_tok -= dropped_tok
                total -= dropped_tok
            elif media_messages:
                dropped = media_messages.pop(0)
                dropped_tok = TokenCounter.count(dropped)
                media_tok -= dropped_tok
                total -= dropped_tok
            elif len(protected_end) > 1: # Always keep at least the very last message
                dropped = protected_end.pop(0)
                dropped_tok = TokenCounter.count(dropped)
                p_end_tok -= dropped_tok
                total -= dropped_tok
            else:
                break # Cannot safely shed anymore

        return media_messages, uncompressed, protected_end, was_shed

    def _build_runtime_view(
        self,
        db_messages: List[Dict[str, Any]],
        media_messages: List[Dict[str, Any]],
        summary_state: SummaryState,
    ) -> RuntimeView:
        """Build the runtime message view using deterministic segments."""
        message_count = len(db_messages)
        keep_start = min(self.valves.keep_start_messages, message_count)
        keep_end = min(
            self.valves.keep_last_messages,
            max(0, message_count - keep_start)
        )

        pools = self._split_message_pools(
            db_messages,
            summary_state.until_ts,
            keep_start,
            keep_end,
        )

        # Apply tool trimming to the compressible (uncompressed) messages
        trim_targets = set(range(len(pools.compressible)))
        trimmed_compressible, trim_stats = self.reconstructor.trim_tool_content(
            pools.compressible,
            self.valves.tool_trim_threshold,
            target_indices=trim_targets,
        )

        # Package the segments (optionally trim protected messages)
        if self.valves.trim_protected_messages and pools.protected_start:
            trimmed_protected_start, _ = self.reconstructor.trim_tool_content(
                pools.protected_start,
                self.valves.tool_trim_threshold,
                target_indices=set(range(len(pools.protected_start))),
            )
            protected_start = self._package_messages(trimmed_protected_start)
        else:
            protected_start = self._package_messages(pools.protected_start)
        
        if self.valves.trim_protected_messages and pools.protected_end:
            trimmed_protected_end, _ = self.reconstructor.trim_tool_content(
                pools.protected_end,
                self.valves.tool_trim_threshold,
                target_indices=set(range(len(pools.protected_end))),
            )
            protected_end = self._package_messages(trimmed_protected_end)
        else:
            protected_end = self._package_messages(pools.protected_end)
        
        uncompressed = self._package_messages(trimmed_compressible)
        
        # Build summary message
        summary_message = self._build_summary_message(summary_state)

        # --- Enforce max context limits ---
        media_messages, uncompressed, protected_end, was_shed = self._enforce_context_limits(
            protected_start,
            summary_message,
            media_messages,
            uncompressed,
            protected_end
        )

        # Build segments
        segments = RuntimeSegments(
            protected_start=protected_start,
            summary_message=summary_message,
            media_messages=media_messages,
            uncompressed=uncompressed,
            protected_end=protected_end,
        )

        # Build stats from explicit segments
        protected_all = protected_start + protected_end
        (
            stats_message,
            total_tokens,
            protected_tokens,
            uncompressed_tokens,
            summary_tokens,
            media_tokens,
        ) = self._build_runtime_stats_message(
            summarized_messages=pools.summarized,
            protected_messages=protected_all,
            summary_message=summary_message,
            uncompressed_messages=uncompressed,
            media_messages=media_messages,
            was_shed=was_shed,
        )

        return RuntimeView(
            final_messages=segments.final_messages,
            stats_message=stats_message,
            segments=segments,
            total_tokens=total_tokens,
            protected_tokens=protected_tokens,
            uncompressed_tokens=uncompressed_tokens,
            summary_tokens=summary_tokens,
            media_tokens=media_tokens,
        )

    def _debug_runtime_view(
        self,
        chat_id: str,
        label: str,
        view: RuntimeView,
        summary_state: SummaryState,
    ):
        if not self.valves.debug_logging:
            return
        try:
            logger.debug(
                "[Context Manager][%s][%s] total=%s protected=%s uncompressed=%s summary=%s media=%s final=%s summary_until=%s",
                label,
                chat_id,
                view.total_tokens,
                view.protected_tokens,
                view.uncompressed_tokens,
                view.summary_tokens,
                view.media_tokens,
                len(view.final_messages),
                summary_state.until_ts,
            )
        except Exception as exc:
            logger.debug(f"[Context Manager][debug] runtime view failed: {exc}")

    def _get_compressible_text_messages(
        self,
        db_messages: List[Dict[str, Any]],
        summary_state: SummaryState,
    ) -> List[Dict[str, Any]]:
        text_only_db_messages = self._build_text_only_messages(db_messages)
        text_only_db_messages = self._package_messages(
            self._prepare_db_messages(text_only_db_messages)
        )

        pools = self._split_message_pools(
            text_only_db_messages,
            summary_state.until_ts,
            self.valves.keep_start_messages,
            self.valves.keep_last_messages,
        )

        # Apply trimming to the compressible messages
        if pools.compressible:
            trim_targets = set(range(len(pools.compressible)))
            trimmed_compressible, _ = self.reconstructor.trim_tool_content(
                pools.compressible,
                self.valves.tool_trim_threshold,
                target_indices=trim_targets,
            )
            return trimmed_compressible

        return []

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

        summary_state = self._get_summary_state(chat_id)
        db_messages = self._load_chat_messages(chat_id)
        media_messages = self._extract_media_messages_from_body(body)

        view = self._build_runtime_view(
            db_messages,
            media_messages,
            summary_state,
        )
        
        body["messages"] = view.final_messages
        self._debug_runtime_view(chat_id, "inlet", view, summary_state)

        await self._emit_status(
            __event_emitter__,
            f"💭{view.stats_message}",
            done=True,
        )

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

        summary_state = self._get_summary_state(chat_id)
        db_messages = self._load_chat_messages(chat_id)
        
        # inject assistant message as it has yet to be written to DB
        db_messages.append(body["messages"][-1])
        media_messages = self._extract_media_messages_from_body(body)

        view = self._build_runtime_view(
            db_messages,
            media_messages,
            summary_state,
        )
        
        compressible_text_messages = self._get_compressible_text_messages(
            db_messages, summary_state
        )
        
        # Count the actual tokens sitting uncompressed in the DB
        db_uncompressed_tokens = self._count_tokens_in_messages(compressible_text_messages)
        
        self._debug_runtime_view(chat_id, "outlet-before", view, summary_state)

        # Trigger based on db_uncompressed_tokens, not view.uncompressed_tokens
        if (
            db_uncompressed_tokens > self.valves.compression_threshold_tokens
            and compressible_text_messages
        ):
            await self._emit_status(
                __event_emitter__,
                f"Summarizing {db_uncompressed_tokens:,} new tokens...",
                done=False,
            )

            lock = self._lock_for(chat_id)
            if not lock.locked():
                await self._background_compress(
                    lock=lock,
                    chat_id=chat_id,
                    old_summary_content=summary_state.content,
                    compressible_messages=compressible_text_messages,
                    model_id=self.valves.summary_model or body.get("model"),
                    user_data=__user__,
                    emitter=__event_emitter__,
                    request=__request__,
                )

                summary_state = self._get_summary_state(chat_id)
                view = self._build_runtime_view(
                    db_messages,
                    media_messages,
                    summary_state,
                )
                self._debug_runtime_view(chat_id, "outlet-after", view, summary_state)

        await self._emit_status(
            __event_emitter__,
            f"☑️{view.stats_message}",
            done=True,
        )

        return body

    async def _background_compress(
        self,
        lock: asyncio.Lock,
        chat_id: str,
        old_summary_content: str,
        compressible_messages: List[Dict[str, Any]],
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

                if not compressible_messages:
                    return

                # --- NEW: Summarizer Circuit Breaker (Chunking) ---
                # Reserve 6k tokens for the system prompt and the generated output
                budget = max(10000, self.valves.max_context_tokens - 6000)
                
                batch_to_compress = []
                current_tokens = 0
                pool_text = ""

                for m in compressible_messages:
                    msg_text = f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                    msg_tokens = TokenCounter.count(msg_text)
                    
                    if current_tokens + msg_tokens > budget:
                        if not batch_to_compress:
                            # Edge Case: The very first message is larger than the entire budget!
                            # We MUST process it to unblock the queue, so we truncate the text.
                            # 1 token ~= 4 chars roughly.
                            allowed_chars = budget * 3 
                            pool_text = msg_text[:allowed_chars] + "\n...[CONTENT TRUNCATED FOR SUMMARY]..."
                            batch_to_compress = [m]
                            logger.warning(f"[Context Manager] Single message exceeded budget ({msg_tokens} > {budget}). Truncating for summary.")
                        else:
                            logger.info(f"[Context Manager] Chunking summary: taking {len(batch_to_compress)} of {len(compressible_messages)} messages to stay under {budget} tokens.")
                        break # Stop adding messages to this batch
                        
                    batch_to_compress.append(m)
                    current_tokens += msg_tokens

                if not batch_to_compress:
                    return

                # If we didn't hit the massive single-message edge case, build the text normally
                if not pool_text:
                    pool_text = "\n".join(
                        f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                        for m in batch_to_compress
                        if m.get("content")
                    ).strip()

                if not pool_text:
                    return
                # --------------------------------------------------

                prompt = f"""
You are the "Context Architect". Update the conversation archive using the new events. Replace the old archive entirely.

### STRUCTURE (Keep exact order. Include all headers even if empty)
## Current State
Active facts, preferences, and state. Include confidence %:
- 90-100%: Verified/Implemented
- 70-89%: Strongly implied
- 50-69%: Tentative/Discussed
- <50%: Omit entirely

## Decisions
What was chosen and why. Replace superseded decisions.

## Resolutions
Resolved problems/errors. Remove obsolete ones.

## Final Code/Config
Verbatim final code, commands, or text. Replace older versions. If none, omit section.

## Open Items
Pending actions/questions. Remove when resolved.

### RULES
1. PRECEDENCE: New events overwrite the old archive.
2. NO HALLUCINATION: Only use provided text.
3. CONCISE: Bullet points only. Strip filler.
4. TERMINOLOGY: Preserve user's exact terms.
5. OMIT: Small talk, greetings, AI meta-talk.
6. FORMAT: Do not wrap the entire response in markdown fences. Start directly with "## Current State".

### CURRENT ARCHIVE:
{old_summary_content or "No existing archive."}

### NEW EVENTS:
{pool_text}

### OUTPUT:
Provide ONLY the updated archive text. Start directly with "## Current State".
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

                # --- NEW: Only use timestamps from the batch we actually compressed ---
                valid_ts = [self._timestamp_of(m) for m in batch_to_compress]
                valid_ts = [ts for ts in valid_ts if ts is not None]
                until_ts = (
                    max(valid_ts)
                    if valid_ts
                    else int(datetime.now(timezone.utc).timestamp())
                )
                # ----------------------------------------------------------------------

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

                # Stats based on the batch
                pool_tokens = sum(TokenCounter.count(m) for m in batch_to_compress)
                summary_tokens = TokenCounter.count(new_summary_content)

                efficiency = None
                if pool_tokens > 0:
                    efficiency = max(
                        0.0, min(100.0, (1.0 - (summary_tokens / pool_tokens)) * 100.0)
                    )

                eff_str = (
                    f" {efficiency:.2f}% efficiency" if efficiency is not None else ""
                )
                
                # If we chunked, let the user know there is more to go
                chunk_str = f" (Chunk {len(batch_to_compress)}/{len(compressible_messages)})" if len(batch_to_compress) < len(compressible_messages) else ""
                
                await self._emit_status(emitter, f"💾 Summary saved!{eff_str}{chunk_str}")

            except Exception as e:
                logger.error(
                    f"[Context Manager] Background compression failed: {e}",
                    exc_info=True,
                )
                await self._emit_status(emitter, f"⚠️ Summary failed: {str(e)[:80]}")