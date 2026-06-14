"""
title: Context Manager
id: context_manager
author: jndao
description: An intelligent context-layer for OpenWebUI that preserves multimodal inputs while maintaining a permanent compressed archive and token efficiency. Includes native dimension-based image optimization.
version: 0.4.0-dev.1
author_url: https://github.com/jndao
repository_url: https://github.com/jndao/openwebui-toolkit
funding_url: https://ko-fi.com/jndao
license: https://github.com/jndao/openwebui-toolkit/blob/main/LICENSE
"""

import asyncio
import json
import logging
import re
import base64
import io
import os
import math
import mimetypes
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable, Set, Tuple

from fastapi.requests import Request
from pydantic import BaseModel, Field
from sqlalchemy import select

# OpenWebUI Internal Imports
from open_webui.models.users import Users
from open_webui.utils.chat import generate_chat_completion
from open_webui.models.chats import Chats
from open_webui.internal.db import get_async_db_context

try:
    from open_webui.models.files import Files
except ImportError:
    Files = None

try:
    import tiktoken

    ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    ENCODING = None

try:
    from open_webui.internal.db import Base as owui_Base
    from sqlalchemy import Column, Integer, String, Text, DateTime
except ImportError:
    owui_Base = None
    Column = Integer = String = Text = DateTime = None

try:
    from PIL import Image

    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

SUMMARY_TAG = "context_summary"
SUMMARY_SOURCE = "context_manager"
TOOL_DETAILS_BLOCK_RE = re.compile(r'<details type="tool_calls"[\s\S]*?</details>')
TOOL_RESULT_ATTR_RE = re.compile(r'result="([^"]*)"')

# Images smaller than this (decoded bytes) are left untouched to avoid
# re-opening trivial assets (avatars, icons) on every turn.
IMG_PROCESS_MIN_BYTES = 65536

# =============================================================================
# IMAGE OPTIMIZATION CONSTANTS & HELPERS
# =============================================================================
IMAGE_PREFIXES = {
    b"/9j/": "jpeg",
    b"iVBORw0KGgo": "png",
    b"R0lGOD": "gif",
    b"UklGR": "webp",
    b"Qk0": "bmp",
}
MIME_TYPES = {
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
}


def detect_image_format(base64_data: str) -> Optional[str]:
    if "base64," in base64_data:
        base64_data = base64_data.split("base64,")[1]
    try:
        sample = base64.b64decode(base64_data[:32])
        for prefix, fmt in IMAGE_PREFIXES.items():
            if sample.startswith(prefix):
                return fmt
    except Exception:
        pass
    return None


def extract_base64_data(image_url: str) -> Tuple[Optional[str], Optional[str], str]:
    if not image_url:
        return None, None, image_url
    if image_url.startswith("data:image/"):
        match = re.match(r"data:image/([^;]+);base64,(.+)", image_url)
        if match:
            return (
                match.group(2),
                match.group(1).lower().replace("jpg", "jpeg"),
                image_url,
            )
    # Raw base64 detection: only sniff a prefix instead of scanning the whole
    # (potentially multi-MB) string. Base64 images are long; short tokens like
    # file paths ("/api/v1/files/.../content") are excluded by the length gate.
    stripped = image_url.strip()
    if len(stripped) > 256:
        prefix = stripped[:64]
        if re.match(r"^[A-Za-z0-9+/]+$", prefix):
            return stripped, detect_image_format(stripped), image_url
    return None, None, image_url


def calculate_base64_size(base64_data: str) -> int:
    clean_data = base64_data.replace("\n", "").replace("\r", "").strip()
    return (len(clean_data) * 3) // 4 - clean_data.count("=")


class _ByteBoundedLRU:
    """Async-safe LRU cache bounded by total stored bytes (not item count),
    since cached images vary wildly in size."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._store: "OrderedDict[str, str]" = OrderedDict()
        self._total = 0
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
            return None

    async def put(self, key: str, value: str) -> None:
        size = len(value)
        async with self._lock:
            if key in self._store:
                self._total -= len(self._store[key])
                self._store.move_to_end(key)
            self._store[key] = value
            self._total += size
            # Always keep at least one entry even if it exceeds the cap.
            while self._total > self.max_bytes and len(self._store) > 1:
                _, old = self._store.popitem(last=False)
                self._total -= len(old)


# Files are immutable after upload, so caching their base64 is safe.
_file_b64_cache = _ByteBoundedLRU(max_bytes=256 * 1024 * 1024)


async def get_file_base64(file_id: str) -> Optional[str]:
    """
    Fetches file from DB, reads from disk, and caches the base64 string in a
    byte-bounded LRU.
    """
    if not file_id or Files is None:
        return None

    cached = await _file_b64_cache.get(file_id)
    if cached is not None:
        return cached

    try:
        file_record = await Files.get_file_by_id(file_id)
        if (
            not file_record
            or not file_record.path
            or not os.path.exists(file_record.path)
        ):
            return None

        mime_type = None
        if file_record.meta and isinstance(file_record.meta, dict):
            mime_type = file_record.meta.get("content_type")
        if not mime_type and file_record.filename:
            mime_type, _ = mimetypes.guess_type(file_record.filename)

        if not mime_type or not mime_type.startswith("image/"):
            return None

        def read_file():
            with open(file_record.path, "rb") as f:
                return f.read()

        file_bytes = await asyncio.to_thread(read_file)
        b64_data = base64.b64encode(file_bytes).decode("utf-8")
        res = f"data:{mime_type};base64,{b64_data}"
        await _file_b64_cache.put(file_id, res)
        return res
    except Exception as e:
        logger.debug(f"Failed to load file {file_id} from disk: {e}")
        return None


def format_tokens(token_count: int) -> str:
    if token_count >= 1_000_000:
        return f"{token_count/1_000_000:.1f}M"
    if token_count >= 1000:
        return f"{token_count/1000:.1f}k"
    return str(int(token_count))


def model_supports_vision(model: Optional[Dict[str, Any]]) -> bool:
    if not model:
        return True
    return bool(
        model.get("info", {})
        .get("meta", {})
        .get("capabilities", {})
        .get("vision", True)
    )


def estimate_image_tokens_from_dimensions(
    width: int, height: int, detail: str = "auto"
) -> int:
    if width <= 0 or height <= 0:
        return 0
    if detail == "low":
        return 85
    max_dim = max(width, height)
    scale = 2048 / max_dim if max_dim > 2048 else 1.0
    scaled_w, scaled_h = width * scale, height * scale
    min_dim = min(scaled_w, scaled_h)
    if min_dim > 768:
        scale2 = 768 / min_dim
        scaled_w, scaled_h = scaled_w * scale2, scaled_h * scale2
    tiles_w = max(1, math.ceil(scaled_w / 512))
    tiles_h = max(1, math.ceil(scaled_h / 512))
    return 85 + 170 * (tiles_w * tiles_h)


class ImageProcessor:
    """
    Format-preserving image optimizer.

    Two decoupled levers:
      1. Dimension downscale  -> reduces vision TOKEN cost (tile count).
      2. Byte cap             -> satisfies provider MAX-MB payload limits.

    Format is never converted (PNG stays PNG, JPEG stays JPEG, WebP stays WebP),
    so transparency is preserved and no provider-incompatible WebP is introduced.
    """

    def __init__(self, max_size_bytes: int):
        self.max_size_bytes = max_size_bytes

    def process(
        self,
        base64_data: str,
        original_format: Optional[str],
        max_dim: Optional[int],
        quality: int,
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not PILLOW_AVAILABLE:
            raise RuntimeError("Pillow is not installed.")

        image_bytes = base64.b64decode(base64_data)
        original_size = len(image_bytes)
        image = Image.open(io.BytesIO(image_bytes))
        fmt = original_format or (image.format.lower() if image.format else "png")
        if fmt == "jpg":
            fmt = "jpeg"

        no_op = {
            "changed": False,
            "original_size": original_size,
            "compressed_size": original_size,
        }

        # Only touch formats we can safely re-encode. GIF (possibly animated),
        # BMP, etc. pass through untouched.
        if fmt not in ("jpeg", "png", "webp"):
            return base64_data, fmt, no_op

        resized = False
        # 1) Dimension downscale (token + byte reduction), format preserved.
        if max_dim and max(image.size) > max_dim:
            scale = max_dim / float(max(image.size))
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.LANCZOS,
            )
            resized = True

        data = self._encode(image, fmt, quality) if resized else None
        current_size = len(data) if data is not None else original_size

        # 2) Byte cap (provider payload limits).
        if self.max_size_bytes and current_size > self.max_size_bytes:
            if data is None:
                data = self._encode(image, fmt, quality)
            data, image = self._enforce_byte_cap(image, fmt, quality, data)

        if data is None:
            return base64_data, fmt, no_op

        # If we only re-encoded (no resize) and it didn't help, keep the original.
        if not resized and len(data) >= original_size:
            return base64_data, fmt, no_op

        return (
            base64.b64encode(data).decode("utf-8"),
            fmt,
            {
                "changed": True,
                "original_size": original_size,
                "compressed_size": len(data),
            },
        )

    def _encode(self, image: "Image.Image", fmt: str, quality: int) -> bytes:
        buffer = io.BytesIO()
        if fmt == "jpeg":
            img = image if image.mode in ("RGB", "L") else image.convert("RGB")
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
        elif fmt == "webp":
            img = image
            if img.mode not in ("RGB", "RGBA", "L"):
                img = img.convert("RGBA" if "A" in img.mode else "RGB")
            img.save(buffer, format="WEBP", quality=quality, method=4)
        else:  # png — transparency preserved, lossless. No quality knob.
            image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    def _enforce_byte_cap(
        self, image: "Image.Image", fmt: str, quality: int, data: bytes
    ) -> Tuple[bytes, "Image.Image"]:
        q = quality
        # Lossy formats: step quality down first.
        if fmt in ("jpeg", "webp"):
            while len(data) > self.max_size_bytes and q > 20:
                q = max(20, q - 15)
                data = self._encode(image, fmt, q)
        # Still over (or PNG, which has no quality knob): downscale iteratively.
        while len(data) > self.max_size_bytes and max(image.size) > 256:
            image = image.resize(
                (max(1, int(image.width * 0.8)), max(1, int(image.height * 0.8))),
                Image.LANCZOS,
            )
            data = self._encode(image, fmt, q)
        return data, image


# =============================================================================
# CONTEXT MANAGER CORE & DB
# =============================================================================
def _discover_owui_schema() -> Optional[str]:
    try:
        from open_webui.config import DATABASE_SCHEMA

        return (
            DATABASE_SCHEMA.value
            if hasattr(DATABASE_SCHEMA, "value")
            else DATABASE_SCHEMA
        )
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
    protected_start: List[Dict[str, Any]]
    summarized: List[Dict[str, Any]]
    compressible: List[Dict[str, Any]]
    protected_end: List[Dict[str, Any]]


@dataclass
class RuntimeSegments:
    protected_start: List[Dict[str, Any]]
    summary_message: Optional[Dict[str, Any]]
    summarized_media: List[Dict[str, Any]]
    uncompressed: List[Dict[str, Any]]
    protected_end: List[Dict[str, Any]]

    @property
    def final_messages(self) -> List[Dict[str, Any]]:
        merged = list(self.protected_start)
        if self.summary_message:
            merged.append(self.summary_message)
        merged.extend(self.summarized_media)
        merged.extend(self.uncompressed)
        merged.extend(self.protected_end)
        return merged


@dataclass
class RuntimeView:
    final_messages: List[Dict[str, Any]]
    stats_message: str
    segments: RuntimeSegments
    total_tokens: int
    protected_tokens: int
    uncompressed_tokens: int
    summary_tokens: int
    summarized_media_tokens: int


class SummaryStore:
    def __init__(self):
        self._initialized = False
        self._init_error = None

    async def _ensure_table(self):
        if self._initialized:
            return self._init_error is None
        self._initialized = True
        try:
            if ChatManifest is None:
                raise RuntimeError("DB dependencies unavailable")
            async with get_async_db_context() as db:
                conn = await db.connection()
                await conn.run_sync(ChatManifest.__table__.create, checkfirst=True)
                await db.commit()
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                return True
            self._init_error = str(e)
            return False

    async def get(self, chat_id: str) -> Optional[Dict[str, Any]]:
        if not await self._ensure_table():
            return None
        try:
            async with get_async_db_context() as db:
                result = await db.execute(
                    select(ChatManifest).filter_by(chat_id=chat_id)
                )
                record = result.scalars().first()
                return (
                    {
                        "content": record.summary_content,
                        "until_timestamp": record.until_timestamp,
                    }
                    if record
                    else None
                )
        except Exception:
            return None

    async def save(
        self, chat_id: str, content: str, until_timestamp: Optional[int] = None
    ) -> bool:
        if not await self._ensure_table():
            return False
        try:
            async with get_async_db_context() as db:
                result = await db.execute(
                    select(ChatManifest).filter_by(chat_id=chat_id)
                )
                record = result.scalars().first()
                if record:
                    record.summary_content, record.until_timestamp = (
                        content,
                        until_timestamp,
                    )
                    record.updated_at = datetime.now(timezone.utc)
                else:
                    db.add(
                        ChatManifest(
                            chat_id=chat_id,
                            summary_content=content,
                            until_timestamp=until_timestamp,
                        )
                    )
                await db.commit()
            return True
        except Exception:
            return False


_summary_store: Optional[SummaryStore] = None


def _get_store() -> Optional[SummaryStore]:
    global _summary_store
    if _summary_store is None:
        _summary_store = SummaryStore()
    return _summary_store


async def get_summary_from_store(chat_id: str) -> Optional[Dict[str, Any]]:
    store = _get_store()
    return await store.get(chat_id) if store else None


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
                if isinstance(part, dict) and str(
                    part.get("type", "")
                ).strip().lower() in {"text", "input_text"}:
                    total += TokenCounter._count_text(
                        part.get("text", "") or part.get("content", "")
                    )
                elif isinstance(part, str):
                    total += TokenCounter._count_text(part)

        for tc in (
            msg.get("tool_calls", []) if isinstance(msg.get("tool_calls"), list) else []
        ):
            if not isinstance(tc, dict):
                continue
            total += TokenCounter._count_text(
                tc.get("id", "")
            ) + TokenCounter._count_text(tc.get("type", ""))
            if isinstance(func := tc.get("function", {}), dict):
                total += TokenCounter._count_text(
                    func.get("name", "")
                ) + TokenCounter._count_text(func.get("arguments", ""))

        total += TokenCounter._count_text(
            msg.get("tool_call_id", "")
        ) + TokenCounter._count_text(msg.get("name", ""))
        return total + 4

    @staticmethod
    def extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return (
                str(content.get("text") or content.get("content") or "")
                if str(content.get("type", "")).strip().lower()
                in {"text", "input_text"}
                else ""
            )
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and str(p.get("type", "")).strip().lower() in {
                    "text",
                    "input_text",
                }:
                    parts.append(str(p.get("text") or p.get("content") or ""))
            return " ".join(parts).strip()
        return ""


class ContextReconstructor:
    @staticmethod
    def collapsed_tool_text() -> str:
        return "[TOOL OUTPUT COLLAPSED]"

    def trim_tool_content(
        self,
        messages: List[Dict[str, Any]],
        threshold: int,
        target_indices: Optional[Set[int]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        trimmed = [deepcopy(msg) for msg in messages]
        stats = {"trimmed_count": 0}
        collapsed = self.collapsed_tool_text()

        for i, msg in enumerate(trimmed):
            if target_indices is not None and i not in target_indices:
                continue
            if (
                msg.get("role") == "tool"
                and TokenCounter._count_text(
                    TokenCounter.extract_text(msg.get("content"))
                )
                > threshold
            ):
                msg["content"] = collapsed
                stats["trimmed_count"] += 1

            for tc in (
                msg.get("tool_calls", [])
                if isinstance(msg.get("tool_calls"), list)
                else []
            ):
                if isinstance(tc, dict) and isinstance(
                    func := tc.get("function"), dict
                ):
                    if (
                        isinstance(args := func.get("arguments"), str)
                        and TokenCounter._count_text(args) > threshold
                    ):
                        func["arguments"] = collapsed
                        stats["trimmed_count"] += 1

            if (
                isinstance(content := msg.get("content"), str)
                and '<details type="tool_calls"' in content
            ):

                def _replace(match):
                    block = match.group(0)
                    if (
                        res := TOOL_RESULT_ATTR_RE.search(block)
                    ) and TokenCounter._count_text(res.group(1)) > threshold:
                        stats["trimmed_count"] += 1
                        return TOOL_RESULT_ATTR_RE.sub(
                            f'result="{collapsed}"', block, count=1
                        )
                    return block

                msg["content"] = TOOL_DETAILS_BLOCK_RE.sub(_replace, content)

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
            default=None, description="Model ID to use for background summarization."
        )
        include_protected_in_threshold: bool = Field(
            default=True,
            description="If true, protected messages count toward the compression threshold.",
        )
        tool_trim_threshold: int = Field(
            default=1000,
            description="Tool outputs larger than this token count are eligible for trimming.",
        )
        trim_protected_messages: bool = Field(
            default=False,
            description="Apply tool content trimming to protected messages.",
        )
        debug_logging: bool = Field(
            default=False, description="Enable detailed console logging."
        )

        # Image Optimization Settings
        enable_image_compression: bool = Field(
            default=False,
            description="Opt-in to native image optimization (dimension downscaling + byte cap).",
        )
        image_max_dim_protected: int = Field(
            default=2048,
            ge=64,
            description="Max pixel dimension for images in protected zones (High Fidelity).",
        )
        image_max_dim_uncompressed: int = Field(
            default=1024,
            ge=64,
            description="Max pixel dimension for images in the uncompressed zone (Medium Fidelity).",
        )
        image_max_dim_summarized: int = Field(
            default=768,
            ge=64,
            description="Max pixel dimension for images in the summarized zone (Low Fidelity).",
        )
        image_quality_protected: int = Field(
            default=85,
            ge=1,
            le=100,
            description="JPEG/WebP quality used when byte-capping protected-zone images.",
        )
        image_quality_uncompressed: int = Field(
            default=60,
            ge=1,
            le=100,
            description="JPEG/WebP quality used when byte-capping uncompressed-zone images.",
        )
        image_quality_summarized: int = Field(
            default=40,
            ge=1,
            le=100,
            description="JPEG/WebP quality used when byte-capping summarized-zone images.",
        )
        max_image_size_bytes: int = Field(
            default=1048576,
            ge=1024,
            description="Provider payload ceiling. Images larger than this are re-encoded/downscaled to fit.",
        )
        enable_vision_detection: bool = Field(
            default=True, description="Check if the model supports vision."
        )
        drop_images_for_non_vision: bool = Field(
            default=True, description="Drop images if the model doesn't support vision."
        )
        image_token_detail: str = Field(
            default="auto", description="Token estimation detail level (auto/low/high)."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.reconstructor = ContextReconstructor()
        self._locks: Dict[str, asyncio.Lock] = {}
        # Failure backoff tracking for background summarization.
        self._compress_failures: Dict[str, int] = {}
        self._compress_cooldown_until: Dict[str, float] = {}
        self._image_stats = {
            "compressed": 0,
            "saved_bytes": 0,
            "original_bytes": 0,
            "tokens": 0,
            "count": 0,
        }

    def _lock_for(self, chat_id: str) -> asyncio.Lock:
        if chat_id not in self._locks:
            self._locks[chat_id] = asyncio.Lock()
        return self._locks[chat_id]

    async def _emit_status(
        self, emitter: Optional[Callable], message: str, done: bool = True
    ):
        if emitter and self.valves.emit_status_events:
            try:
                await emitter(
                    {"type": "status", "data": {"description": message, "done": done}}
                )
            except Exception:
                pass

    def _get_chat_id(self, body: dict, metadata: dict) -> Optional[str]:
        meta = metadata or {}
        return (
            meta.get("chat_id")
            or body.get("chat_id")
            or body.get("meta", {}).get("chat_id")
        )

    def _timestamp_of(self, msg: Dict[str, Any]) -> Optional[int]:
        val = msg.get("timestamp") or msg.get("created_at")
        try:
            if isinstance(val, (int, float)):
                return int(val) if val < 1e12 else int(val / 1000)
            if isinstance(val, str):
                return int(
                    datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
                )
        except Exception:
            pass
        return None

    def _unfold_messages(self, messages: Any) -> List[Dict[str, Any]]:
        if not messages or not isinstance(messages, list):
            return []
        result = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            msg = deepcopy(msg)
            children = msg.pop("children", None)
            if (
                children
                and isinstance(children, list)
                and isinstance(children[0], dict)
            ):
                child_msg = {**msg, **children[0]}
                child_msg.pop("children", None)
                result.append(child_msg)
            else:
                result.append(msg)
        return result

    async def _scrub_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Safely extracts and rebuilds message content, fetching images from disk if needed."""
        if not isinstance(msg, dict):
            return {"role": "user", "content": ""}

        scrubbed = {"role": msg.get("role", "user"), "content": msg.get("content", "")}

        # Keep essential metadata
        for k in ["id", "parentId", "timestamp", "created_at"]:
            if k in msg:
                scrubbed[k] = msg[k]

        files, images = msg.get("files", []), msg.get("images", [])
        if not files and not images:
            return scrubbed

        new_content = (
            [{"type": "text", "text": scrubbed["content"]}]
            if scrubbed["content"]
            else []
        )

        for f in files:
            if not isinstance(f, dict):
                continue
            file_id = f.get("id")
            url = f.get("url")

            b64_url = await get_file_base64(file_id) if file_id else None
            if b64_url:
                url = b64_url
            elif not url and file_id:
                url = f"/api/v1/files/{file_id}/content"

            if not url:
                continue

            if (
                f.get("type") == "image"
                or "image/" in f.get("meta", {}).get("content_type", "")
                or url.startswith("data:image/")
            ):
                new_content.append({"type": "image_url", "image_url": {"url": url}})

        for img in images:
            if isinstance(img, str):
                if not img.startswith("data:") and not img.startswith("http"):
                    b64_url = await get_file_base64(img)
                    if b64_url:
                        img = b64_url
                new_content.append({"type": "image_url", "image_url": {"url": img}})

        if len(new_content) > (1 if scrubbed["content"] else 0):
            scrubbed["content"] = new_content

        return scrubbed

    async def _load_chat_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        if not chat_id or Chats is None:
            return []
        try:
            chat_record = await Chats.get_chat_by_id(chat_id)
        except Exception:
            return []

        chat_payload = getattr(chat_record, "chat", {})
        if not isinstance(chat_payload, dict):
            return []

        history = chat_payload.get("history", {})
        history_msgs = history.get("messages", {})
        current_id = history.get("currentId") or history.get("current_id")

        if isinstance(current_id, str) and current_id in history_msgs:
            ordered, cursor, visited = [], current_id, set()
            while isinstance(cursor, str) and cursor and cursor not in visited:
                visited.add(cursor)
                node = history_msgs.get(cursor)
                if not isinstance(node, dict):
                    break
                ordered.append(deepcopy(node))
                cursor = node.get("parentId") or node.get("parent_id")
            ordered.reverse()

            res = []
            for m in self._unfold_messages(ordered):
                if m.get("content"):
                    res.append(await self._scrub_message(m))
            return res

        if isinstance(chat_payload.get("messages"), list):
            res = []
            for m in self._unfold_messages(deepcopy(chat_payload["messages"])):
                if m.get("content"):
                    res.append(await self._scrub_message(m))
            return res
        return []

    async def _get_summary_state(self, chat_id: str) -> SummaryState:
        data = await get_summary_from_store(chat_id)
        return (
            SummaryState(
                content=data["content"], until_ts=data["until_timestamp"], raw=data
            )
            if data
            else SummaryState("", None)
        )

    def _split_message_pools(
        self,
        messages: List[Dict[str, Any]],
        summary_time: Optional[int],
        keep_start: int,
        keep_end: int,
    ) -> MessagePools:
        total = len(messages)
        start_cut = min(max(keep_start, 0), total)
        end_count = min(max(keep_end, 0), max(0, total - start_cut))
        end_start = total - end_count

        protected_start = list(messages[:start_cut])
        protected_end = list(messages[end_start:]) if end_count > 0 else []
        middle = list(messages[start_cut:end_start])

        summarized, compressible = [], []
        for msg in middle:
            ts = self._timestamp_of(msg)
            if summary_time is not None and ts is not None and ts <= summary_time:
                summarized.append(msg)
            else:
                compressible.append(msg)
        return MessagePools(protected_start, summarized, compressible, protected_end)

    def _select_summary_batch(
        self, msgs: List[Dict[str, Any]], budget: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Selects the batch of messages to summarize this round.

        - Always includes at least the first message (so a single oversized
          message can never stall the queue forever).
        - Greedily fills up to `budget`.
        - CRITICAL: never cuts in the middle of a group of messages sharing the
          same timestamp. Otherwise the trailing tied messages would satisfy
          `ts <= until_ts` and be classified as 'summarized' (stripped to
          media-only) without ever actually being summarized -> silent gap.
          If retracting the tie-tail would make zero progress, we instead
          force-extend to swallow the whole tie group (accepting budget
          overflow) to guarantee forward movement.
        """
        batch, cur_tok = [], 0
        for m in msgs:
            txt = f"{str(m.get('role', 'user')).upper()}: {m.get('content', '')}"
            tok = TokenCounter.count(txt)
            if batch and cur_tok + tok > budget:
                break
            batch.append(m)
            cur_tok += tok

        if len(batch) < len(msgs):
            boundary_ts = self._timestamp_of(batch[-1])
            next_ts = self._timestamp_of(msgs[len(batch)])
            if boundary_ts is not None and boundary_ts == next_ts:
                retracted = list(batch)
                while retracted and self._timestamp_of(retracted[-1]) == boundary_ts:
                    retracted.pop()
                if retracted:
                    # Safe: defer the tied tail to the next round.
                    batch = retracted
                else:
                    # Whole batch shares boundary_ts with the following messages.
                    # Force-extend to include the entire tie group.
                    i = len(batch)
                    while i < len(msgs) and self._timestamp_of(msgs[i]) == boundary_ts:
                        batch.append(msgs[i])
                        i += 1

        cur_tok = sum(
            TokenCounter.count(
                f"{str(m.get('role', 'user')).upper()}: {m.get('content', '')}"
            )
            for m in batch
        )
        return batch, cur_tok

    def _message_has_passthrough_media(self, message: Dict[str, Any]) -> bool:
        content = message.get("content")
        media_types = {"image_url", "file", "input_image", "input_file"}
        if isinstance(content, dict):
            return str(content.get("type", "")).strip().lower() in media_types
        if isinstance(content, list):
            return any(
                isinstance(p, dict)
                and str(p.get("type", "")).strip().lower() in media_types
                for p in content
            )
        return False

    def _align_messages(
        self, db_msgs: List[Dict[str, Any]], body_msgs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        aligned = []
        # 1. System prompts from frontend
        for b in body_msgs:
            if isinstance(b, dict) and b.get("role") == "system":
                aligned.append(deepcopy(b))

        # 2. DB messages
        for d in db_msgs:
            if isinstance(d, dict):
                aligned.append(deepcopy(d))

        if not body_msgs:
            return aligned

        # 3. Append the final user/assistant message if it's new
        last_b = body_msgs[-1]
        if isinstance(last_b, dict) and last_b.get("role") != "system":
            b_text = TokenCounter.extract_text(last_b.get("content", "")).strip()
            is_new = True
            if aligned:
                last_aligned = aligned[-1]
                if isinstance(last_aligned, dict) and last_aligned.get(
                    "role", "user"
                ) == last_b.get("role", "user"):
                    a_text = TokenCounter.extract_text(
                        last_aligned.get("content", "")
                    ).strip()
                    if b_text == a_text:
                        is_new = False
                        frontend_content = last_b.get("content")
                        if isinstance(frontend_content, list):
                            has_image = any(
                                isinstance(p, dict)
                                and str(p.get("type", "")).strip().lower()
                                in {"image_url", "image"}
                                for p in frontend_content
                            )
                            if has_image:
                                aligned[-1]["content"] = deepcopy(frontend_content)
            if is_new:
                aligned.append(deepcopy(last_b))
        return aligned

    def _build_media_only_message(
        self, msg: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        content = msg.get("content")
        if not isinstance(content, list):
            return None
        media_parts = [
            p
            for p in content
            if isinstance(p, dict)
            and str(p.get("type", "")).strip().lower()
            in {"image_url", "image", "file", "input_image", "input_file"}
        ]
        return (
            {"role": msg.get("role", "user"), "content": media_parts}
            if media_parts
            else None
        )

    async def _process_pool_images(
        self,
        pool: List[Dict[str, Any]],
        max_dim: int,
        quality: int,
        processor: Optional[ImageProcessor],
        supports_vision: bool,
    ) -> List[Dict[str, Any]]:
        if not self.valves.enable_image_compression or not pool:
            return pool

        processed_pool = []
        for msg in pool:
            if not isinstance(msg, dict) or not msg.get("content"):
                processed_pool.append(msg)
                continue

            msg_copy = deepcopy(msg)
            content = msg_copy["content"]

            if not supports_vision and self.valves.drop_images_for_non_vision:
                if isinstance(content, list):
                    new_content = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            new_content.append(
                                {
                                    "type": "text",
                                    "text": "[Image dropped - model doesn't support vision]",
                                }
                            )
                        else:
                            new_content.append(part)
                    msg_copy["content"] = new_content
                processed_pool.append(msg_copy)
                continue

            if processor and PILLOW_AVAILABLE and isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        img_url = part.get("image_url", {})
                        url = (
                            img_url.get("url", "")
                            if isinstance(img_url, dict)
                            else str(img_url)
                        )
                        b64, fmt, _ = extract_base64_data(url)

                        if b64 and calculate_base64_size(b64) > IMG_PROCESS_MIN_BYTES:
                            try:

                                def _proc():
                                    return processor.process(b64, fmt, max_dim, quality)

                                new_b64, new_fmt, stats = await asyncio.to_thread(_proc)
                                if stats.get("changed"):
                                    new_url = f"data:{MIME_TYPES.get(new_fmt, 'image/jpeg')};base64,{new_b64}"
                                    if isinstance(part["image_url"], dict):
                                        part["image_url"]["url"] = new_url
                                    else:
                                        part["image_url"] = new_url

                                    self._image_stats["compressed"] += 1
                                    self._image_stats["saved_bytes"] += (
                                        stats["original_size"]
                                        - stats["compressed_size"]
                                    )
                                    self._image_stats["original_bytes"] += stats[
                                        "original_size"
                                    ]
                            except Exception as e:
                                logger.debug(f"Image processing failed: {e}")
            processed_pool.append(msg_copy)
        return processed_pool

    def _calculate_image_tokens(self, messages: List[Dict[str, Any]]) -> None:
        if not self.valves.enable_image_compression:
            return
        for msg in messages:
            if isinstance(content := msg.get("content"), list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        self._image_stats["count"] += 1
                        img_url = part.get("image_url", {})
                        url = (
                            img_url.get("url", "")
                            if isinstance(img_url, dict)
                            else str(img_url)
                        )
                        b64, _, _ = extract_base64_data(url)
                        if b64 and PILLOW_AVAILABLE:
                            try:
                                img = Image.open(io.BytesIO(base64.b64decode(b64)))
                                self._image_stats[
                                    "tokens"
                                ] += estimate_image_tokens_from_dimensions(
                                    img.width,
                                    img.height,
                                    self.valves.image_token_detail,
                                )
                            except Exception:
                                pass
                        else:
                            self._image_stats["tokens"] += 85

    async def _build_runtime_view(
        self,
        aligned_messages: List[Dict[str, Any]],
        summary_state: SummaryState,
        model: Optional[Dict[str, Any]] = None,
    ) -> RuntimeView:
        self._image_stats = {
            "compressed": 0,
            "saved_bytes": 0,
            "original_bytes": 0,
            "tokens": 0,
            "count": 0,
        }
        pools = self._split_message_pools(
            aligned_messages,
            summary_state.until_ts,
            min(self.valves.keep_start_messages, len(aligned_messages)),
            min(
                self.valves.keep_last_messages,
                max(0, len(aligned_messages) - self.valves.keep_start_messages),
            ),
        )

        summarized_media = [
            m
            for p in pools.summarized
            if (m := self._build_media_only_message(p))
            and self._message_has_passthrough_media(p)
        ]
        trimmed_compressible, _ = self.reconstructor.trim_tool_content(
            pools.compressible,
            self.valves.tool_trim_threshold,
            set(range(len(pools.compressible))),
        )

        protected_start = (
            self.reconstructor.trim_tool_content(
                pools.protected_start, self.valves.tool_trim_threshold
            )[0]
            if self.valves.trim_protected_messages
            else pools.protected_start
        )
        protected_end = (
            self.reconstructor.trim_tool_content(
                pools.protected_end, self.valves.tool_trim_threshold
            )[0]
            if self.valves.trim_protected_messages
            else pools.protected_end
        )

        if self.valves.enable_image_compression:
            processor = (
                ImageProcessor(self.valves.max_image_size_bytes)
                if PILLOW_AVAILABLE
                else None
            )
            supports_vision = (
                model_supports_vision(model)
                if self.valves.enable_vision_detection
                else True
            )

            protected_start = await self._process_pool_images(
                protected_start,
                self.valves.image_max_dim_protected,
                self.valves.image_quality_protected,
                processor,
                supports_vision,
            )
            protected_end = await self._process_pool_images(
                protected_end,
                self.valves.image_max_dim_protected,
                self.valves.image_quality_protected,
                processor,
                supports_vision,
            )
            trimmed_compressible = await self._process_pool_images(
                trimmed_compressible,
                self.valves.image_max_dim_uncompressed,
                self.valves.image_quality_uncompressed,
                processor,
                supports_vision,
            )
            summarized_media = await self._process_pool_images(
                summarized_media,
                self.valves.image_max_dim_summarized,
                self.valves.image_quality_summarized,
                processor,
                supports_vision,
            )

        protected_start = [
            {k: v for k, v in m.items() if k != "children"} for m in protected_start
        ]
        protected_end = [
            {k: v for k, v in m.items() if k != "children"} for m in protected_end
        ]
        uncompressed = [
            {k: v for k, v in m.items() if k != "children"}
            for m in trimmed_compressible
        ]

        summary_message = (
            {
                "role": "system",
                "content": f"<{SUMMARY_TAG}>\n{summary_state.content}\n</{SUMMARY_TAG}>",
            }
            if summary_state.content
            else None
        )

        max_tok = self.valves.max_context_tokens
        total_tok = sum(
            TokenCounter.count(m)
            for pool in [
                protected_start,
                [summary_message] if summary_message else [],
                summarized_media,
                uncompressed,
                protected_end,
            ]
            for m in pool
        )

        # Shedding policy: drop content whose substance is ALREADY preserved
        # before dropping un-archived live content.
        #   1. summarized_media -> its TEXT is already in the archive; safe-ish.
        #   2. uncompressed     -> NOT yet summarized; lives nowhere if dropped
        #                          -> raise a loud warning.
        #   3. protected_end    -> last resort, keep at least one.
        was_shed = False
        shed_unsummarized = False
        while total_tok > max_tok and max_tok > 0:
            was_shed = True
            if summarized_media:
                total_tok -= TokenCounter.count(summarized_media.pop(0))
            elif uncompressed:
                shed_unsummarized = True
                total_tok -= TokenCounter.count(uncompressed.pop(0))
            elif len(protected_end) > 1:
                total_tok -= TokenCounter.count(protected_end.pop(0))
            else:
                break

        segments = RuntimeSegments(
            protected_start,
            summary_message,
            summarized_media,
            uncompressed,
            protected_end,
        )
        self._calculate_image_tokens(segments.final_messages)

        p_tok = sum(TokenCounter.count(m) for m in protected_start + protected_end)
        u_tok = sum(TokenCounter.count(m) for m in uncompressed)
        s_tok = TokenCounter.count(summary_message) if summary_message else 0
        sm_tok = sum(TokenCounter.count(m) for m in summarized_media)
        raw_s_tok = sum(TokenCounter.count(m) for m in pools.summarized)

        eff_str = (
            f" @ {round((raw_s_tok - s_tok)/raw_s_tok * 100, 2)}%"
            if raw_s_tok > 0
            else ""
        )
        stats = f"🪙 {format_tokens(p_tok + u_tok + s_tok + sm_tok)} │ 🛡️ {format_tokens(p_tok)} ({len(protected_start)+len(protected_end)}) · ⏳ {format_tokens(u_tok)} ({len(uncompressed)}) · 📦 {format_tokens(s_tok)} ({len(pools.summarized)}{eff_str})"

        if was_shed:
            if shed_unsummarized:
                stats = (
                    f"⚠️ Context limit hit — dropped un-archived messages "
                    f"(raise max_context_tokens) │ {stats}"
                )
            else:
                stats = f"⚠️ Limit Reached │ {stats}"
        if self.valves.enable_image_compression and self._image_stats["count"] > 0:
            img_tok, img_cnt, orig_b, saved_b = (
                self._image_stats["tokens"],
                self._image_stats["count"],
                self._image_stats["original_bytes"],
                self._image_stats["saved_bytes"],
            )
            img_eff = f" @ {round((saved_b / orig_b) * 100)}%" if orig_b > 0 else ""
            stats += f" │ 🖼️ {format_tokens(img_tok)} ({img_cnt}{img_eff})"

        return RuntimeView(
            segments.final_messages,
            stats,
            segments,
            p_tok + u_tok + s_tok + sm_tok,
            p_tok,
            u_tok,
            s_tok,
            sm_tok,
        )

    async def inlet(
        self,
        body: dict,
        __user__: dict = None,
        __metadata__: dict = None,
        __event_emitter__: Callable = None,
        __event_call__: Callable = None,
        __request__: Request = None,
        __model__: dict = None,
    ) -> dict:
        if not (chat_id := self._get_chat_id(body, __metadata__)):
            return body

        state = await self._get_summary_state(chat_id)
        db_msgs = await self._load_chat_messages(chat_id)
        aligned = self._align_messages(db_msgs, body.get("messages", []))
        view = await self._build_runtime_view(aligned, state, __model__)

        body["messages"] = view.final_messages
        await self._emit_status(__event_emitter__, f"💭{view.stats_message}")
        return body

    async def outlet(
        self,
        body: dict,
        __user__: dict = None,
        __metadata__: dict = None,
        __event_emitter__: Callable = None,
        __event_call__: Callable = None,
        __request__: Request = None,
        __model__: dict = None,
    ) -> dict:
        if not (chat_id := self._get_chat_id(body, __metadata__)):
            return body

        state = await self._get_summary_state(chat_id)
        db_msgs = await self._load_chat_messages(chat_id)
        aligned = self._align_messages(db_msgs, body.get("messages", []))
        view = await self._build_runtime_view(aligned, state, __model__)

        text_msgs = []
        for m in aligned:
            text_msg = {
                "role": m.get("role", "user"),
                "content": TokenCounter.extract_text(m.get("content", "")),
            }
            text_msg.update(
                {k: m[k] for k in ("timestamp", "created_at", "id") if k in m}
            )
            if text_msg["content"]:
                text_msgs.append(text_msg)

        pools = self._split_message_pools(
            text_msgs,
            state.until_ts,
            self.valves.keep_start_messages,
            self.valves.keep_last_messages,
        )
        comp_text = (
            self.reconstructor.trim_tool_content(
                pools.compressible, self.valves.tool_trim_threshold
            )[0]
            if pools.compressible
            else []
        )

        db_u_tok = sum(TokenCounter.count(m) for m in comp_text)
        trigger = db_u_tok + (
            view.protected_tokens if self.valves.include_protected_in_threshold else 0
        )

        if trigger > self.valves.compression_threshold_tokens and comp_text:
            now_ts = datetime.now(timezone.utc).timestamp()
            cooldown_until = self._compress_cooldown_until.get(chat_id, 0)
            lock = self._lock_for(chat_id)

            if now_ts < cooldown_until:
                # In failure backoff: keep summarization paused but stay loud so
                # the degraded state is never silent.
                fails = self._compress_failures.get(chat_id, 0)
                await self._emit_status(
                    __event_emitter__,
                    f"⚠️ Summarization paused after {fails} failure(s); will retry │ ☑️{view.stats_message}",
                )
                return body

            if not lock.locked():
                await self._emit_status(
                    __event_emitter__, f"Summarizing {db_u_tok:,} new tokens...", False
                )
                asyncio.create_task(
                    self._background_compress(
                        lock,
                        chat_id,
                        state.content,
                        comp_text,
                        self.valves.summary_model or body.get("model"),
                        __user__,
                        __event_emitter__,
                        __request__,
                    )
                )

        await self._emit_status(__event_emitter__, f"☑️{view.stats_message}")
        return body

    async def _background_compress(
        self,
        lock: asyncio.Lock,
        chat_id: str,
        old_summary: str,
        msgs: List[Dict[str, Any]],
        model_id: str,
        user_data: dict,
        emitter: Callable,
        request: Request,
    ):
        async with lock:
            try:
                if not model_id or not msgs:
                    return

                budget = max(10000, self.valves.max_context_tokens - 6000)
                batch, cur_tok = self._select_summary_batch(msgs, budget)

                pool_txt = "\n".join(
                    f"{str(m.get('role', 'user')).upper()}: {m.get('content', '')}"
                    for m in batch
                ).strip()

                prompt = f"""You are the "Context Architect". Update the conversation archive using the new events. Output the complete updated archive.
### STRUCTURE (Keep exact order. Include all headers even if empty)
## Current State
Active facts, preferences, project constraints, and state. Include confidence %:
- 90-100%: Verified/Implemented/Purchased
- 70-89%: Strongly implied/Planned
- 50-69%: Tentative/Discussed
- <50%: Retain but tag as (low-confidence). DO NOT delete low-confidence items.
## Decisions
What was chosen and why (e.g., architecture, purchases, methodologies). Replace superseded decisions.
## Resolutions
Resolved problems, fixed errors, or completed tasks. Remove obsolete ones.
## Working States & Code
Preserve VERBATIM code blocks, configurations, terminal commands, and finalized lists. 
- Code: DO NOT summarize working code. Keep the exact syntax and language fences. Replace older versions with the latest working state.
- General: Track the latest working state of architectures, itineraries, or configurations.
If none, omit section.
## Open Items
Pending actions, blockers, or unanswered questions. Remove when resolved.
### RULES
1. PRESERVATION (CRITICAL): Reproduce EVERY entry from the CURRENT ARCHIVE verbatim unless it is explicitly superseded or resolved by a new event. Never drop, merge, or condense existing entries to save space. The archive is append-and-revise, not re-summarize.
2. PRECEDENCE: When a new event contradicts the archive, the new event wins and replaces the old entry.
3. NO HALLUCINATION: Only use provided text.
4. CONCISE (for NEW content only): Bullet points. Strip filler from new events; do not strip existing entries.
5. TERMINOLOGY: Preserve user's exact terms (e.g., specific brand names, technical jargon).
6. OMIT: Small talk, greetings, AI meta-talk.
7. FORMAT: Do not wrap the entire response in markdown fences. Start directly with "## Current State".
### CURRENT ARCHIVE:
{old_summary or "No existing archive."}
### NEW EVENTS:
{pool_txt}
### OUTPUT:
Provide ONLY the updated archive text. Start directly with "## Current State"."""

                user = (
                    await Users.get_user_by_id(user_data["id"])
                    if user_data and user_data.get("id")
                    else None
                )
                if not user:
                    return

                res = await generate_chat_completion(
                    request or Request(scope={"type": "http"}),
                    {
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "temperature": 0,
                    },
                    user,
                )
                res = json.loads(res.body.decode()) if hasattr(res, "body") else res
                new_sum = res["choices"][0]["message"]["content"].strip()

                valid_ts = [
                    ts for m in batch if (ts := self._timestamp_of(m)) is not None
                ]
                until_ts = (
                    max(valid_ts)
                    if valid_ts
                    else int(datetime.now(timezone.utc).timestamp())
                )

                if await _get_store().save(chat_id, new_sum, until_ts):
                    # Success: clear any failure backoff state.
                    self._compress_failures.pop(chat_id, None)
                    self._compress_cooldown_until.pop(chat_id, None)
                    eff = (
                        max(
                            0.0,
                            min(
                                100.0,
                                (1.0 - (TokenCounter.count(new_sum) / cur_tok)) * 100.0,
                            ),
                        )
                        if cur_tok > 0
                        else 0
                    )
                    await self._emit_status(
                        emitter, f"💾 Summary saved! {eff:.2f}% efficiency"
                    )
            except Exception as e:
                # Failure backoff: exponential with a 10-minute cap, so a broken
                # summarizer doesn't hammer the model every turn — but the outlet
                # surfaces a persistent warning so degradation is never silent.
                fails = self._compress_failures.get(chat_id, 0) + 1
                self._compress_failures[chat_id] = fails
                backoff = min(600, 15 * (2 ** min(fails - 1, 6)))
                self._compress_cooldown_until[chat_id] = (
                    datetime.now(timezone.utc).timestamp() + backoff
                )
                await self._emit_status(
                    emitter,
                    f"⚠️ Summary failed (attempt {fails}); retry in {int(backoff)}s: {str(e)[:60]}",
                )
