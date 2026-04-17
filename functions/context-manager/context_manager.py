"""
title: Context Manager
id: context_manager
author: jndao
description: An intelligent context-layer for OpenWebUI that preserves multimodal inputs while maintaining a permanent compressed archive and token efficiency. Includes native semantic image compression.
version: 0.2.0-dev.24
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

from functools import lru_cache
from datetime import datetime, timezone
import os
import mimetypes
import base64

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

try:
    from rapidocr_onnxruntime import RapidOCR

    RAPIDOCR_AVAILABLE = True
    _ocr_engine = RapidOCR()
except ImportError:
    RAPIDOCR_AVAILABLE = False
    _ocr_engine = None

logger = logging.getLogger(__name__)

SUMMARY_TAG = "context_summary"
SUMMARY_SOURCE = "context_manager"
TOOL_DETAILS_BLOCK_RE = re.compile(r'<details type="tool_calls"[\s\S]*?</details>')
TOOL_RESULT_ATTR_RE = re.compile(r'result="([^"]*)"')

# =============================================================================
# IMAGE COMPRESSION CONSTANTS & HELPERS
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
    if re.match(r"^[A-Za-z0-9+/=]+$", image_url.strip()):
        return image_url.strip(), detect_image_format(image_url), image_url
    return None, None, image_url


def calculate_base64_size(base64_data: str) -> int:
    clean_data = base64_data.replace("\n", "").replace("\r", "").strip()
    return (len(clean_data) * 3) // 4 - clean_data.count("=")


@lru_cache(maxsize=256)
def get_cached_ocr_description(file_id: str) -> str:
    """
    Performs OCR and caches the result based on the immutable File ID.
    This avoids hashing massive base64 strings.
    """
    b64_uri = get_file_base64(file_id)
    if not b64_uri:
        return "[Image content not available]"
    
    # Extract the raw b64 part from the data URI
    _, _, raw_url = extract_base64_data(b64_uri)
    b64_data, _, _ = extract_base64_data(raw_url)
    
    if text := extract_text_from_image(b64_data):
        return f"[OCR Text]: {text}"
    return "[Image content available but no text detected]"

@lru_cache(maxsize=256)
def get_file_base64(file_id: str) -> Optional[str]:
    """
    Fetches file from DB, reads from disk, and caches the base64 string.
    Files are assumed to be immutable after upload to OWUI thus making caching
    deterministic.
    """
    if not file_id or Files is None:
        return None
    try:
        file_record = Files.get_file_by_id(file_id)
        if not file_record or not file_record.path:
            return None
        if not os.path.exists(file_record.path):
            return None

        # 1. Get the real mime type from the DB metadata, fallback to filename extension
        mime_type = None
        if file_record.meta and isinstance(file_record.meta, dict):
            mime_type = file_record.meta.get("content_type")

        if not mime_type and file_record.filename:
            mime_type, _ = mimetypes.guess_type(file_record.filename)

        # 2. If it's STILL unknown, or explicitly NOT an image, abort.
        if not mime_type or not mime_type.startswith("image/"):
            return None

        with open(file_record.path, "rb") as f:
            file_bytes = f.read()

        b64_data = base64.b64encode(file_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{b64_data}"
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


def extract_text_from_image(base64_data: str) -> Optional[str]:
    if not RAPIDOCR_AVAILABLE or _ocr_engine is None:
        return None
    try:
        # Extract data from URI if present
        if "base64," in base64_data:
            base64_data = base64_data.split("base64,")[1]
            
        result, _ = _ocr_engine(base64.b64decode(base64_data))
        if not result:
            return None
        return " ".join([line[1] for line in result if len(line) >= 2 and line[1]]).strip()
    except Exception:
        return None


@lru_cache(maxsize=256)
def generate_smart_image_description(base64_data: str, use_ocr: bool = True) -> str:
    if use_ocr and RAPIDOCR_AVAILABLE:
        if ocr_text := extract_text_from_image(base64_data):
            return f"[OCR Text]: {ocr_text}"
    return "[Image content not available - could not extract description]"


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


class ImageCompressor:
    def __init__(
        self,
        max_size_bytes: int,
        convert_png_to_jpeg: bool,
        preserve_transparency: bool,
    ):
        self.max_size_bytes = max_size_bytes
        self.convert_png_to_jpeg = convert_png_to_jpeg
        self.preserve_transparency = preserve_transparency

    def compress_image(
        self, base64_data: str, original_format: Optional[str], quality: int
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not PILLOW_AVAILABLE:
            raise RuntimeError("Pillow is not installed.")
        image_bytes = base64.b64decode(base64_data)
        original_size = len(image_bytes)
        image = Image.open(io.BytesIO(image_bytes))
        original_format = original_format or (
            image.format.lower() if image.format else "png"
        )
        has_transparency = image.mode in ("RGBA", "LA", "P")

        target_format = self._determine_target_format(original_format, has_transparency)
        processed_image = self._prepare_image_for_save(image, target_format)
        compressed_data = self._compress_at_quality(
            processed_image, target_format, quality
        )

        stats = {
            "original_size": original_size,
            "compressed_size": len(compressed_data),
            "original_format": original_format,
            "new_format": target_format,
            "quality": quality,
        }
        return base64.b64encode(compressed_data).decode("utf-8"), target_format, stats

    def _determine_target_format(
        self, original_format: str, has_transparency: bool
    ) -> str:
        if original_format in ("jpeg", "jpg", "webp", "gif"):
            return original_format
        if self.convert_png_to_jpeg:
            return (
                "webp" if (has_transparency and self.preserve_transparency) else "jpeg"
            )
        return original_format

    def _prepare_image_for_save(
        self, image: Image.Image, target_format: str
    ) -> Image.Image:
        if target_format == "jpeg":
            if image.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode == "P":
                    image = image.convert("RGBA")
                if image.mode in ("RGBA", "LA"):
                    background.paste(image, mask=image.split()[-1])
                    return background
            return image.convert("RGB") if image.mode != "RGB" else image
        elif target_format == "webp":
            if image.mode == "P":
                return image.convert(
                    "RGBA" if image.info.get("transparency") else "RGB"
                )
            if image.mode not in ("RGB", "RGBA", "L"):
                return image.convert("RGB")
        return image

    def _compress_at_quality(
        self, image: Image.Image, target_format: str, quality: int
    ) -> bytes:
        buffer = io.BytesIO()
        kwargs = (
            {"quality": quality, "optimize": True}
            if target_format in ("jpeg", "webp")
            else {"optimize": True}
        )
        if target_format == "webp":
            kwargs["method"] = 4
        image.save(buffer, format=target_format.upper(), **kwargs)
        return buffer.getvalue()


# =============================================================================
# CONTEXT MANAGER CORE
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

    def _ensure_table(self):
        if self._initialized:
            return self._init_error is None
        self._initialized = True
        try:
            if ChatManifest is None:
                raise RuntimeError("DB dependencies unavailable")
            with get_db_context() as db:
                ChatManifest.__table__.create(bind=db.bind, checkfirst=True)
                db.commit()
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                return True
            self._init_error = str(e)
            return False

    def get(self, chat_id: str) -> Optional[Dict[str, Any]]:
        if not self._ensure_table():
            return None
        try:
            with get_db_context() as db:
                record = db.query(ChatManifest).filter_by(chat_id=chat_id).first()
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

    def save(
        self, chat_id: str, content: str, until_timestamp: Optional[int] = None
    ) -> bool:
        if not self._ensure_table():
            return False
        try:
            with get_db_context() as db:
                record = db.query(ChatManifest).filter_by(chat_id=chat_id).first()
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
                db.commit()
            return True
        except Exception:
            return False


_summary_store: Optional[SummaryStore] = None


def _get_store() -> Optional[SummaryStore]:
    global _summary_store
    if _summary_store is None:
        _summary_store = SummaryStore()
    return _summary_store


def get_summary_from_store(chat_id: str) -> Optional[Dict[str, Any]]:
    store = _get_store()
    return store.get(chat_id) if store else None


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
                elif isinstance(p, dict):
                    if str(p.get("type", "")).strip().lower() in {"text", "input_text"}:
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
        enable_image_compression: bool = Field(
            default=False, description="Opt-in to native semantic image compression."
        )
        image_quality_protected: int = Field(
            default=85,
            ge=1,
            le=100,
            description="Quality for images in protected zones (High Fidelity).",
        )
        image_quality_uncompressed: int = Field(
            default=60,
            ge=1,
            le=100,
            description="Quality for images in uncompressed zone (Medium Fidelity).",
        )
        image_quality_summarized: int = Field(
            default=20,
            ge=1,
            le=100,
            description="Quality for images in summarized zone (Low Fidelity).",
        )
        max_image_size_bytes: int = Field(
            default=1048576,
            ge=1024,
            description="Maximum image size in bytes before compression triggers.",
        )
        convert_png_to_jpeg: bool = Field(
            default=True,
            description="Convert PNG images to JPEG for better compression.",
        )
        preserve_transparency: bool = Field(
            default=True,
            description="Convert transparent PNGs to WebP instead of JPEG.",
        )
        enable_vision_detection: bool = Field(
            default=True, description="Check if the model supports vision."
        )
        drop_images_for_non_vision: bool = Field(
            default=True, description="Drop images if the model doesn't support vision."
        )
        enable_smart_drop: bool = Field(
            default=True, description="Generate OCR descriptions for dropped images."
        )
        use_ocr: bool = Field(default=True, description="Use RapidOCR to extract text.")
        image_token_detail: str = Field(
            default="auto", description="Token estimation detail level (auto/low/high)."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.reconstructor = ContextReconstructor()
        self._locks: Dict[str, asyncio.Lock] = {}
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
        return (
            (metadata or {}).get("chat_id")
            or body.get("chat_id")
            or body.get("meta", {}).get("chat_id")
        )

    def _timestamp_of(self, msg: Dict[str, Any]) -> Optional[int]:
        if not isinstance(msg, dict):
            return None
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
        if not messages:
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

    def _scrub_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        scrubbed = {
            k: v
            for k, v in msg.items()
            if k in {"id", "parentId", "role", "content", "timestamp"}
        }
        if not isinstance(scrubbed.get("content"), str):
            return scrubbed

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

            # Fetch base64 directly from disk if possible!
            b64_url = get_file_base64(file_id) if file_id else None

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
                # If it's a file ID, fetch from disk
                if not img.startswith("data:") and not img.startswith("http"):
                    b64_url = get_file_base64(img)
                    if b64_url:
                        img = b64_url
                new_content.append({"type": "image_url", "image_url": {"url": img}})

        if len(new_content) > (1 if scrubbed["content"] else 0):
            scrubbed["content"] = new_content

        return scrubbed

    def _load_chat_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        if not chat_id or Chats is None:
            return []
        try:
            chat_record = Chats.get_chat_by_id(chat_id)
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
            return [
                self._scrub_message(m)
                for m in self._unfold_messages(ordered)
                if m.get("content")
            ]

        if isinstance(chat_payload.get("messages"), list):
            return [
                self._scrub_message(m)
                for m in self._unfold_messages(deepcopy(chat_payload["messages"]))
                if m.get("content")
            ]
        return []

    def _get_summary_state(self, chat_id: str) -> SummaryState:
        data = get_summary_from_store(chat_id)
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
            if b.get("role") == "system":
                aligned.append(deepcopy(b))

        # 2. DB messages (now with base64 injected from disk!)
        for d in db_msgs:
            aligned.append(deepcopy(d))

        if not body_msgs:
            return aligned

        # 3. Append the final user/assistant message if it's new
        last_b = body_msgs[-1]
        if last_b.get("role") != "system":
            b_text = TokenCounter.extract_text(last_b.get("content", "")).strip()

            is_new = True
            if aligned:
                last_aligned = aligned[-1]
                if last_aligned.get("role") == last_b.get("role"):
                    a_text = TokenCounter.extract_text(
                        last_aligned.get("content", "")
                    ).strip()
                    if b_text == a_text:
                        is_new = False
                        # If it's not new, optionally upgrade content if frontend has richer media
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

    def _process_pool_images(
        self,
        pool: List[Dict[str, Any]],
        quality: int,
        compressor: Optional[ImageCompressor],
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
                            text = "[Image dropped - model doesn't support vision]"
                            # Inside _process_pool_images...
                            if self.valves.enable_smart_drop:
                                # Use the file_id directly for the cache key!
                                if file_id := part.get("id"): 
                                    text = get_cached_ocr_description(file_id)
                                else:
                                    # Fallback for images without IDs (pasted blobs)
                                    img_url = part.get("image_url", {})
                                    url = img_url.get("url", "") if isinstance(img_url, dict) else str(img_url)
                                    b64, _, _ = extract_base64_data(url)
                                    text = generate_smart_image_description(b64, self.valves.use_ocr)
                            new_content.append({"type": "text", "text": text})
                        else:
                            new_content.append(part)
                    msg_copy["content"] = new_content
                processed_pool.append(msg_copy)
                continue

            if compressor and PILLOW_AVAILABLE and isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        img_url = part.get("image_url", {})
                        url = (
                            img_url.get("url", "")
                            if isinstance(img_url, dict)
                            else str(img_url)
                        )
                        b64, fmt, _ = extract_base64_data(url)

                        if (
                            b64
                            and calculate_base64_size(b64)
                            > self.valves.max_image_size_bytes
                        ):
                            try:
                                new_b64, new_fmt, stats = compressor.compress_image(
                                    b64, fmt, quality
                                )
                                new_url = f"data:{MIME_TYPES.get(new_fmt, 'image/jpeg')};base64,{new_b64}"
                                if isinstance(part["image_url"], dict):
                                    part["image_url"]["url"] = new_url
                                else:
                                    part["image_url"] = new_url

                                self._image_stats["compressed"] += 1
                                self._image_stats["saved_bytes"] += (
                                    stats["original_size"] - stats["compressed_size"]
                                )
                                self._image_stats["original_bytes"] += stats[
                                    "original_size"
                                ]
                            except Exception as e:
                                logger.debug(f"Image compression failed: {e}")

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

    def _build_runtime_view(
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
            compressor = (
                ImageCompressor(
                    self.valves.max_image_size_bytes,
                    self.valves.convert_png_to_jpeg,
                    self.valves.preserve_transparency,
                )
                if PILLOW_AVAILABLE
                else None
            )
            supports_vision = (
                model_supports_vision(model)
                if self.valves.enable_vision_detection
                else True
            )

            protected_start = self._process_pool_images(
                protected_start,
                self.valves.image_quality_protected,
                compressor,
                supports_vision,
            )
            protected_end = self._process_pool_images(
                protected_end,
                self.valves.image_quality_protected,
                compressor,
                supports_vision,
            )
            trimmed_compressible = self._process_pool_images(
                trimmed_compressible,
                self.valves.image_quality_uncompressed,
                compressor,
                supports_vision,
            )
            summarized_media = self._process_pool_images(
                summarized_media,
                self.valves.image_quality_summarized,
                compressor,
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
        was_shed = False

        while total_tok > max_tok and max_tok > 0:
            was_shed = True
            if uncompressed:
                total_tok -= TokenCounter.count(uncompressed.pop(0))
            elif summarized_media:
                total_tok -= TokenCounter.count(summarized_media.pop(0))
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

        state = self._get_summary_state(chat_id)
        db_msgs = self._load_chat_messages(chat_id)
        aligned = self._align_messages(db_msgs, body.get("messages", []))

        view = self._build_runtime_view(aligned, state, __model__)
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

        state = self._get_summary_state(chat_id)
        db_msgs = self._load_chat_messages(chat_id)
        aligned = self._align_messages(db_msgs, body.get("messages", []))

        view = self._build_runtime_view(aligned, state, __model__)

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
            await self._emit_status(
                __event_emitter__, f"Summarizing {db_u_tok:,} new tokens...", False
            )
            lock = self._lock_for(chat_id)
            if not lock.locked():
                await self._background_compress(
                    lock,
                    chat_id,
                    state.content,
                    comp_text,
                    self.valves.summary_model or body.get("model"),
                    __user__,
                    __event_emitter__,
                    __request__,
                )
                view = self._build_runtime_view(
                    aligned, self._get_summary_state(chat_id), __model__
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
                batch, cur_tok = [], 0

                for m in msgs:
                    txt = f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                    tok = TokenCounter.count(txt)
                    if cur_tok + tok > budget:
                        if not batch:
                            batch = [m]
                        break
                    batch.append(m)
                    cur_tok += tok

                pool_txt = "\n".join(
                    f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                    for m in batch
                ).strip()
                prompt = f"""You are the "Context Architect". Update the conversation archive using the new events. Replace the old archive entirely.
### STRUCTURE (Keep exact order. Include all headers even if empty)
## Current State
Active facts, preferences, project constraints, and state. Include confidence %:
- 90-100%: Verified/Implemented/Purchased
- 70-89%: Strongly implied/Planned
- 50-69%: Tentative/Discussed
- <50%: Omit entirely
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
1. PRECEDENCE: New events overwrite the old archive.
2. NO HALLUCINATION: Only use provided text.
3. CONCISE: Bullet points only. Strip filler.
4. TERMINOLOGY: Preserve user's exact terms (e.g., specific brand names, technical jargon).
5. OMIT: Small talk, greetings, AI meta-talk.
6. FORMAT: Do not wrap the entire response in markdown fences. Start directly with "## Current State".
### CURRENT ARCHIVE:
{old_summary or "No existing archive."}
### NEW EVENTS:
{pool_txt}
### OUTPUT:
Provide ONLY the updated archive text. Start directly with "## Current State"."""

                user = (
                    await asyncio.to_thread(Users.get_user_by_id, user_data["id"])
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

                if _get_store().save(chat_id, new_sum, until_ts):
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
                await self._emit_status(emitter, f"⚠️ Summary failed: {str(e)[:80]}")
