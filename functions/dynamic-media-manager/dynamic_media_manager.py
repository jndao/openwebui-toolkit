"""
title: Dynamic Media Manager
id: dynamic_media_manager
description: Automatically manages large media (images, videos) in messages to prevent 413 Request Entity Too Large errors. Supports compression, size thresholds, quality gradients, vision model detection, and smart image dropping with OCR/VLM descriptions.
author: jndao
author_url: https://github.com/jndao
repository_url: https://github.com/jndao/openwebui-toolkit
version: 0.2.3
license: https://github.com/jndao/openwebui-toolkit/blob/main/LICENSE

Overview:
  Compresses images in chat messages to prevent 413 errors. Applies quality gradient
  (recent images = higher quality, older = lower). Supports PNG→JPEG conversion, vision
  model detection, and OCR-based image descriptions for non-vision models.

Configuration:
  priority: 15 (runs after context compression at 10)
  max_image_size_bytes: 1048576 (1MB) - images above this are compressed
  max_payload_size_bytes: 10485760 (10MB) - oldest images dropped if exceeded
  enable_quality_gradient: true - recent images get higher quality
  recent_image_quality: 85 (0-100)
  old_image_quality: 40 (0-100)
  convert_png_to_jpeg: true - better compression
  preserve_transparency: true - transparent PNGs → WebP
  enable_status_notifications: true - show compression status
  debug_mode: false
  enable_vision_detection: true - check model vision capabilities
  drop_images_for_non_vision: true - drop images for non-vision models
  enable_smart_drop: true - generate OCR descriptions for dropped images
  description_quality: medium (low/medium/high)
  use_ocr: true - extract text via RapidOCR
  image_token_detail: auto (auto/low/high)

Requirements: Pillow (PIL) - typically included in Open WebUI
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple, Callable
import base64
import io
import logging
import asyncio
import re
import json
import math

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    logger.warning("[Image Compressor] Pillow not installed. Image compression will be disabled.")


# =============================================================================
# CONSTANTS
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

FORMAT_EXTENSIONS = {
    "jpeg": ".jpg",
    "jpg": ".jpg",
    "png": ".png",
    "gif": ".gif",
    "webp": ".webp",
    "bmp": ".bmp",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_image_format(base64_data: str) -> Optional[str]:
    """Detect image format from base64 data prefix."""
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
    """
    Extract base64 data from various image URL formats.
    Returns: (base64_data, detected_format, original_url)
    """
    if not image_url:
        return None, None, image_url

    if image_url.startswith("data:image/"):
        match = re.match(r"data:image/([^;]+);base64,(.+)", image_url)
        if match:
            mime_type = match.group(1).lower()
            base64_data = match.group(2)
            fmt = mime_type.replace("jpg", "jpeg")
            return base64_data, fmt, image_url

    if re.match(r'^[A-Za-z0-9+/=]+$', image_url.strip()):
        fmt = detect_image_format(image_url)
        return image_url.strip(), fmt, image_url

    return None, None, image_url


def calculate_base64_size(base64_data: str) -> int:
    """Calculate the approximate byte size of base64-encoded data."""
    clean_data = base64_data.replace("\n", "").replace("\r", "").strip()
    padding = clean_data.count("=")
    return (len(clean_data) * 3) // 4 - padding


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def format_tokens(token_count: int) -> str:
    """Format tokens as human-readable string."""
    if token_count >= 1_000_000:
        return f"{token_count/1_000_000:.1f}M"
    if token_count >= 1000:
        return f"{token_count/1000:.1f}k"
    return str(int(token_count))


def model_supports_vision(model: Optional[Dict[str, Any]]) -> bool:
    """
    Check if a model supports vision capabilities.
    Checks explicit capabilities in model["info"]["meta"]["capabilities"]["vision"].
    """
    if not model:
        return True

    capabilities = model.get("info", {}).get("meta", {}).get("capabilities", {})
    if capabilities is not None and "vision" in capabilities:
        return bool(capabilities["vision"])

    return True


try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
    _ocr_engine = RapidOCR()
except ImportError:
    RAPIDOCR_AVAILABLE = False
    _ocr_engine = None
    logger.warning("[Dynamic Media Manager] rapidocr-onnxruntime not installed. OCR will be disabled.")


def extract_text_from_image(base64_data: str, debug: bool = False) -> Optional[str]:
    """Extract text from an image using RapidOCR."""
    if not RAPIDOCR_AVAILABLE or _ocr_engine is None:
        if debug:
            logger.info("[Dynamic Media Manager] RapidOCR not available for OCR")
        return None

    try:
        image_bytes = base64.b64decode(base64_data)
        result, elapsed = _ocr_engine(image_bytes)

        if result is None or len(result) == 0:
            return None

        text_lines = []
        for line in result:
            if len(line) >= 2 and line[1]:
                text_lines.append(line[1])

        text = " ".join(text_lines).strip()

        if text and debug:
            logger.info(f"[Dynamic Media Manager] RapidOCR extracted {len(text)} characters in {elapsed:.2f}s")

        return text if text else None
    except Exception as e:
        if debug:
            logger.info(f"[Dynamic Media Manager] RapidOCR failed: {e}")
        return None


def generate_smart_image_description(
    base64_data: str,
    use_ocr: bool = True,
    debug: bool = False,
) -> str:
    """Generate a smart description of an image using OCR and/or fallback."""
    description_parts = []

    if use_ocr and RAPIDOCR_AVAILABLE:
        ocr_text = extract_text_from_image(base64_data, debug)
        if ocr_text:
            description_parts.append(f"[OCR Text]: {ocr_text}")
            if debug:
                logger.info(f"[Dynamic Media Manager] OCR found text: {ocr_text[:100]}...")

    if description_parts:
        return " ".join(description_parts)

    return "[Image content not available - could not extract description]"


def estimate_image_tokens_from_dimensions(
    width: int,
    height: int,
    detail: str = "auto",
) -> int:
    """
    Approximate image token cost using a GPT-style tiling heuristic.

    low  -> ~85 tokens/image
    high -> 85 + 170 * number_of_512_tiles after normalization
    auto -> high
    """
    if width <= 0 or height <= 0:
        return 0

    if detail == "low":
        return 85

    max_dim = max(width, height)
    scale = 1.0
    if max_dim > 2048:
        scale = 2048 / max_dim

    scaled_w = width * scale
    scaled_h = height * scale

    min_dim = min(scaled_w, scaled_h)
    if min_dim > 768:
        scale2 = 768 / min_dim
        scaled_w *= scale2
        scaled_h *= scale2

    tiles_w = max(1, math.ceil(scaled_w / 512))
    tiles_h = max(1, math.ceil(scaled_h / 512))
    tiles = tiles_w * tiles_h

    return 85 + 170 * tiles


# =============================================================================
# IMAGE COMPRESSOR CLASS
# =============================================================================

class ImageCompressor:
    """Handles image compression operations - simplified version."""

    def __init__(
        self,
        max_size_bytes: int = 1 * 1024 * 1024,
        convert_png_to_jpeg: bool = True,
        preserve_transparency: bool = True,
        debug: bool = False,
    ):
        self.max_size_bytes = max_size_bytes
        self.convert_png_to_jpeg = convert_png_to_jpeg
        self.preserve_transparency = preserve_transparency
        self.debug = debug

    def _log(self, message: str):
        if self.debug:
            logger.info(f"[Image Compressor] {message}")

    def compress_image(
        self,
        base64_data: str,
        original_format: Optional[str] = None,
        quality: int = 85,
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not PILLOW_AVAILABLE:
            raise RuntimeError("Pillow is not installed. Cannot compress images.")

        try:
            image_bytes = base64.b64decode(base64_data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 data: {e}")

        original_size = len(image_bytes)

        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}")

        if not original_format:
            original_format = image.format.lower() if image.format else "png"

        original_mode = image.mode
        original_dimensions = image.size
        has_transparency = image.mode in ("RGBA", "LA", "P")

        self._log(
            f"Original: {original_format.upper()}, {original_dimensions[0]}x{original_dimensions[1]}, "
            f"{format_size(original_size)}, mode={original_mode}"
        )

        target_format = self._determine_target_format(
            original_format, has_transparency, original_size
        )

        processed_image = self._prepare_image_for_save(image, target_format)
        compressed_data = self._compress_at_quality(processed_image, target_format, quality)
        final_quality = quality

        stats = {
            "original_size": original_size,
            "compressed_size": len(compressed_data),
            "original_format": original_format,
            "new_format": target_format,
            "original_dimensions": original_dimensions,
            "new_dimensions": processed_image.size,
            "original_mode": original_mode,
            "new_mode": processed_image.mode,
            "quality": final_quality,
            "compression_ratio": len(compressed_data) / original_size if original_size > 0 else 1.0,
        }

        self._log(
            f"Compressed: {target_format.upper()}, {processed_image.size[0]}x{processed_image.size[1]}, "
            f"{format_size(len(compressed_data))}, quality={final_quality}, "
            f"ratio={stats['compression_ratio']:.2%}"
        )

        compressed_base64 = base64.b64encode(compressed_data).decode("utf-8")
        return compressed_base64, target_format, stats

    def _determine_target_format(
        self,
        original_format: str,
        has_transparency: bool,
        original_size: int,
    ) -> str:
        if original_format in ("jpeg", "jpg"):
            return "jpeg"
        if original_format == "webp":
            return "webp"
        if original_format == "gif":
            return "gif"

        if self.convert_png_to_jpeg:
            if has_transparency and self.preserve_transparency:
                return "webp"
            return "jpeg"

        return original_format

    def _prepare_image_for_save(self, image: Image.Image, target_format: str) -> Image.Image:
        if target_format == "jpeg":
            if image.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode == "P":
                    image = image.convert("RGBA")
                if image.mode in ("RGBA", "LA"):
                    background.paste(image, mask=image.split()[-1])
                    image = background
                else:
                    image = image.convert("RGB")
            elif image.mode != "RGB":
                image = image.convert("RGB")

        elif target_format == "webp":
            if image.mode == "P":
                image = image.convert("RGBA" if image.info.get("transparency") else "RGB")
            elif image.mode not in ("RGB", "RGBA", "L"):
                image = image.convert("RGB")

        return image

    def _compress_at_quality(
        self,
        image: Image.Image,
        target_format: str,
        quality: int,
    ) -> bytes:
        buffer = io.BytesIO()

        save_kwargs = {}
        if target_format == "jpeg":
            save_kwargs = {"quality": quality, "optimize": True}
        elif target_format == "webp":
            save_kwargs = {"quality": quality, "method": 4}
        elif target_format == "png":
            save_kwargs = {"optimize": True}
        elif target_format == "gif":
            save_kwargs = {"optimize": True}

        try:
            image.save(buffer, format=target_format.upper(), **save_kwargs)
            return buffer.getvalue()
        except Exception as e:
            self._log(f"Failed to compress at quality {quality}: {e}")
            raise


# =============================================================================
# FILTER CLASS
# =============================================================================

class Filter:
    """Open WebUI Filter for dynamic image compression."""

    def __init__(self):
        self.valves = self.Valves()

    class Valves(BaseModel):
        """Configuration valves for the filter."""

        priority: int = Field(
            default=15,
            description="Filter priority. Set HIGHER than context compression (10) to process the final message list after context compression has assembled it.",
        )

        max_image_size_bytes: int = Field(
            default=1 * 1024 * 1024,
            ge=1024,
            description="Maximum image size in bytes. Images larger than this will be compressed. Default: 1MB",
        )

        max_payload_size_bytes: int = Field(
            default=10 * 1024 * 1024,
            ge=1024,
            description="Maximum total payload size in bytes. If exceeded after compression, oldest images will be dropped. Default: 10MB",
        )

        enable_quality_gradient: bool = Field(
            default=True,
            description="Enable quality gradient - recent images get higher quality, older images get more compression.",
        )

        recent_image_quality: int = Field(
            default=85,
            ge=1,
            le=100,
            description="Compression quality for the most recent images (0-100). Higher = better quality, larger files.",
        )

        old_image_quality: int = Field(
            default=40,
            ge=1,
            le=100,
            description="Compression quality for the oldest images (0-100). Lower = more compression, smaller files.",
        )

        convert_png_to_jpeg: bool = Field(
            default=True,
            description="Convert PNG images to JPEG for better compression.",
        )

        preserve_transparency: bool = Field(
            default=True,
            description="When enabled, transparent PNGs will be converted to WebP instead of JPEG to preserve transparency.",
        )

        enable_status_notifications: bool = Field(
            default=True,
            description="Show status notifications when images are compressed.",
        )

        debug_mode: bool = Field(
            default=False,
            description="Enable detailed debug logging.",
        )

        enable_vision_detection: bool = Field(
            default=True,
            description="Enable detection of vision capabilities. When enabled, the filter will check if the model supports vision and can optionally drop images for non-vision models.",
        )

        drop_images_for_non_vision: bool = Field(
            default=True,
            description="If the model doesn't support vision and this is enabled, all images will be dropped and replaced with a text placeholder. Requires enable_vision_detection to be true.",
        )

        enable_smart_drop: bool = Field(
            default=True,
            description="When dropping images for non-vision models, attempt to generate text descriptions (OCR) instead of just placeholder text.",
        )

        description_quality: str = Field(
            default="medium",
            description="Quality of image descriptions. Options: low (brief), medium (standard), high (detailed). Higher quality takes longer to generate.",
        )

        use_ocr: bool = Field(
            default=True,
            description="Use OCR to extract text from images when generating descriptions. Uses RapidOCR (included in Open WebUI dependencies).",
        )

        image_token_detail: str = Field(
            default="auto",
            description="Token estimation detail level for images. Options: auto, low, high. 'auto' currently uses the high-detail tiling heuristic.",
        )

    def _log(self, message: str, level: str = "info"):
        if level == "debug" and not self.valves.debug_mode:
            return
        log_func = getattr(logger, level, logger.info)
        log_func(f"[Dynamic Image Compressor] {message}")

    async def _emit_status(
        self,
        message: str,
        event_emitter,
        done: bool = True,
    ):
        if not self.valves.enable_status_notifications or not event_emitter:
            return

        try:
            await event_emitter({
                "type": "status",
                "data": {
                    "description": message,
                    "done": done,
                },
            })
        except Exception as e:
            self._log(f"Failed to emit status: {e}", level="warning")

    async def _emit_debug_log(
        self,
        message: str,
        event_call,
    ):
        if not self.valves.debug_mode or not event_call:
            return

        try:
            js_code = f"""
                try {{
                    console.log("%c[Image Compressor] {message}", "color: #8b5cf6;");
                    return true;
                }} catch (e) {{
                    return false;
                }}
            """
            await asyncio.wait_for(
                event_call({"type": "execute", "data": {"code": js_code}}),
                timeout=2.0,
            )
        except Exception as e:
            self._log(f"Failed to emit debug log: {e}", level="debug")

    def _count_images_in_content(self, content: Any) -> int:
        count = 0

        if isinstance(content, str):
            if "data:image/" in content:
                count += content.count("data:image/")
            return count

        if isinstance(content, dict):
            if content.get("type") == "image_url":
                return 1
            return 0

        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    count += 1
            return count

        return 0

    def _estimate_tokens_for_image_part(self, image_part: Dict[str, Any]) -> int:
        """Estimate image token usage from dimensions."""
        image_url = image_part.get("image_url", {})
        if isinstance(image_url, dict):
            url = image_url.get("url", "")
        else:
            url = image_url

        base64_data, detected_format, _ = extract_base64_data(url)
        if not base64_data or not PILLOW_AVAILABLE:
            return 0

        try:
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
            detail = (self.valves.image_token_detail or "auto").lower()
            if detail not in {"auto", "low", "high"}:
                detail = "auto"
            return estimate_image_tokens_from_dimensions(width, height, detail=detail)
        except Exception:
            return 0

    def _calculate_image_token_usage(self, messages: List[Dict[str, Any]]) -> Dict[str, int]:
        """Estimate total image token usage across the current payload."""
        total_images = 0
        total_tokens = 0

        for message in messages:
            if not isinstance(message, dict):
                continue

            content = message.get("content")

            if isinstance(content, dict):
                if content.get("type") == "image_url":
                    total_images += 1
                    total_tokens += self._estimate_tokens_for_image_part(content)

            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        total_images += 1
                        total_tokens += self._estimate_tokens_for_image_part(part)

        return {
            "image_count": total_images,
            "image_tokens": total_tokens,
        }

    def _build_image_token_status(self, image_usage: Dict[str, int]) -> str:
        """Build final image token usage status string."""
        image_count = image_usage.get("image_count", 0)
        image_tokens = image_usage.get("image_tokens", 0)
        return f"🪙 Image tokens: {format_tokens(image_tokens)} across {image_count} image(s)"

    def _process_image_in_message(
        self,
        content: Any,
        compressor: ImageCompressor,
        quality: Optional[int | Callable[[int], int]] = None,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        compression_stats = []

        if isinstance(content, str):
            return content, compression_stats

        if isinstance(content, dict):
            if content.get("type") == "image_url":
                q = quality(0) if callable(quality) else quality
                processed, stats = self._process_image_url(content, compressor, q)
                if stats:
                    compression_stats.append(stats)
                return processed, compression_stats
            return content, compression_stats

        if isinstance(content, list):
            processed_content = []
            image_idx = 0
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    q = quality(image_idx) if callable(quality) else quality
                    processed_part, stats = self._process_image_url(part, compressor, q)
                    if stats:
                        compression_stats.append(stats)
                    processed_content.append(processed_part)
                    image_idx += 1
                else:
                    processed_content.append(part)
            return processed_content, compression_stats

        return content, compression_stats

    def _process_image_url(
        self,
        image_part: Dict[str, Any],
        compressor: ImageCompressor,
        quality: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        image_url = image_part.get("image_url", {})
        if isinstance(image_url, dict):
            url = image_url.get("url", "")
        else:
            url = image_url
            image_url = {"url": url}

        base64_data, detected_format, original_url = extract_base64_data(url)

        if not base64_data:
            self._log(f"Skipping non-base64 image: {url[:50]}...", level="debug")
            return image_part, None

        size_bytes = calculate_base64_size(base64_data)

        self._log(
            f"Found image: format={detected_format or 'unknown'}, "
            f"size={format_size(size_bytes)}, threshold={format_size(self.valves.max_image_size_bytes)}"
        )

        should_compress = (
            size_bytes > self.valves.max_image_size_bytes
            or self.valves.enable_quality_gradient
        )

        if not should_compress:
            self._log(f"Image under size threshold, skipping compression")
            return image_part, None

        if not PILLOW_AVAILABLE:
            self._log(
                "Pillow not installed, cannot compress image. "
                "Install with: pip install Pillow",
                level="warning",
            )
            return image_part, None

        try:
            quality_str = f" (quality={quality})" if quality else ""
            self._log(
                f"Compressing image ({format_size(size_bytes)} > {format_size(self.valves.max_image_size_bytes)}){quality_str}"
            )

            compressed_base64, new_format, stats = compressor.compress_image(
                base64_data, detected_format, quality
            )

            self._log(
                f"Compressed: {format_size(stats['original_size'])} → {format_size(stats['compressed_size'])} "
                f"({stats['compression_ratio']:.1%} of original)"
            )

            compressed_size = stats.get("compressed_size", 0)
            if compressed_size > self.valves.max_image_size_bytes and quality and quality > 20:
                self._log(
                    f"Image still too large ({format_size(compressed_size)}), retrying with quality=20"
                )
                compressed_base64, new_format, stats = compressor.compress_image(
                    base64_data, detected_format, 20
                )
                compressed_size = stats.get("compressed_size", 0)

                if compressed_size > self.valves.max_image_size_bytes:
                    self._log(
                        f"Image still too large after quality=20, marking for dropping"
                    )
                    return {"type": "image_url", "image_url": {"url": "__DROP_IMAGE__"}}, {
                        "was_compressed": True,
                        "was_dropped": True,
                        "original_size": stats.get("original_size"),
                        "compressed_size": 0,
                    }

            mime_type = MIME_TYPES.get(new_format, "image/jpeg")
            new_url = f"data:{mime_type};base64,{compressed_base64}"

            new_image_part = {
                "type": "image_url",
                "image_url": {"url": new_url},
            }

            stats["was_compressed"] = True
            stats["threshold_bytes"] = self.valves.max_image_size_bytes

            return new_image_part, stats

        except Exception as e:
            self._log(f"Failed to compress image: {e}", level="error")
            return image_part, None

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__: Optional[callable] = None,
        __event_call__: Optional[callable] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        """
        Process messages before sending to LLM.
        Compresses images that exceed the size threshold and reports final estimated image token usage.
        """
        if not PILLOW_AVAILABLE:
            self._log(
                "Pillow not installed. Image compression disabled. "
                "Install with: pip install Pillow",
                level="warning",
            )
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        if self.valves.enable_vision_detection:
            supports_vision = model_supports_vision(__model__)
            self._log(f"Model vision support: {supports_vision}")

            if not supports_vision and self.valves.drop_images_for_non_vision:
                self._log("Model doesn't support vision, dropping all images")
                dropped_count = 0
                descriptions_generated = 0

                for message in messages:
                    if not isinstance(message, dict):
                        continue

                    content = message.get("content")
                    if not content:
                        continue

                    if isinstance(content, list):
                        new_content = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "image_url":
                                placeholder_text = "[Image dropped - model doesn't support vision]"

                                if self.valves.enable_smart_drop:
                                    image_url = part.get("image_url", {})
                                    if isinstance(image_url, dict):
                                        url = image_url.get("url", "")
                                    else:
                                        url = image_url

                                    base64_data, _, _ = extract_base64_data(url)

                                    if base64_data:
                                        description = generate_smart_image_description(
                                            base64_data,
                                            use_ocr=self.valves.use_ocr,
                                            debug=self.valves.debug_mode,
                                        )
                                        if description and description != "[Image content not available - could not extract description]":
                                            placeholder_text = f"[Image: {description}]"
                                            descriptions_generated += 1
                                            self._log(f"Generated description for dropped image: {description[:100]}...")
                                        else:
                                            placeholder_text = "[Image dropped - model doesn't support vision]"
                                    else:
                                        self._log("Could not extract base64 data for smart description")

                                new_content.append({
                                    "type": "text",
                                    "text": placeholder_text
                                })
                                dropped_count += 1
                            else:
                                new_content.append(part)
                        message["content"] = new_content

                    elif isinstance(content, dict) and content.get("type") == "image_url":
                        placeholder_text = "[Image dropped - model doesn't support vision]"

                        if self.valves.enable_smart_drop:
                            image_url = content.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                            else:
                                url = image_url

                            base64_data, _, _ = extract_base64_data(url)

                            if base64_data:
                                description = generate_smart_image_description(
                                    base64_data,
                                    use_ocr=self.valves.use_ocr,
                                    debug=self.valves.debug_mode,
                                )
                                if description and description != "[Image content not available - could not extract description]":
                                    placeholder_text = f"[Image: {description}]"
                                    descriptions_generated += 1
                                    self._log(f"Generated description for dropped image: {description[:100]}...")

                        message["content"] = [
                            {"type": "text", "text": placeholder_text}
                        ]
                        dropped_count += 1

                if dropped_count > 0:
                    status_msg = f"🖼️ Dropped {dropped_count} image(s) - model doesn't support vision"
                    if descriptions_generated > 0:
                        status_msg += f" ({descriptions_generated} with descriptions)"
                    await self._emit_status(status_msg, __event_emitter__)
                    self._log(
                        f"Dropped {dropped_count} image(s) due to no vision support ({descriptions_generated} with descriptions)"
                    )

                return body

        total_images = sum(
            self._count_images_in_content(msg.get("content"))
            for msg in messages
            if isinstance(msg, dict)
        )
        self._log(f"Processing {len(messages)} messages with {total_images} image(s)")

        initial_payload_size = len(json.dumps(body, default=str))
        self._log(f"Initial payload size: {format_size(initial_payload_size)}")

        await self._emit_debug_log(
            f"Inlet: {len(messages)} messages, {total_images} images, payload: {format_size(initial_payload_size)}",
            __event_call__,
        )

        compressor = ImageCompressor(
            max_size_bytes=self.valves.max_image_size_bytes,
            convert_png_to_jpeg=self.valves.convert_png_to_jpeg,
            preserve_transparency=self.valves.preserve_transparency,
            debug=self.valves.debug_mode,
        )

        total_compressed = 0
        total_saved = 0
        compression_details = []

        after_image_usage = self._calculate_image_token_usage(messages)

        image_positions = []
        for msg_idx, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not content:
                continue
            if isinstance(content, list):
                for part_idx, part in enumerate(content):
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_positions.append((msg_idx, part_idx))
            elif isinstance(content, dict) and content.get("type") == "image_url":
                image_positions.append((msg_idx, 0))

        total_images = len(image_positions)
        self._log(f"Found {total_images} images to process")

        new_messages = []
        current_image_idx = 0
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                new_messages.append(message)
                continue

            new_message = dict(message)
            content = new_message.get("content")
            if not content:
                new_messages.append(new_message)
                continue

            image_count_in_message = 0
            if isinstance(content, list):
                image_count_in_message = sum(
                    1 for part in content
                    if isinstance(part, dict) and part.get("type") == "image_url"
                )
            elif isinstance(content, dict) and content.get("type") == "image_url":
                image_count_in_message = 1

            def get_image_quality(relative_idx: int) -> int:
                if not self.valves.enable_quality_gradient or total_images <= 1:
                    return self.valves.recent_image_quality
                absolute_idx = current_image_idx + relative_idx
                ratio = absolute_idx / max(1, total_images - 1)
                quality = int(
                    self.valves.old_image_quality +
                    ratio * (self.valves.recent_image_quality - self.valves.old_image_quality)
                )
                return quality

            processed_content, stats_list = self._process_image_in_message(
                content, compressor, get_image_quality
            )

            dropped_in_message = 0
            if isinstance(processed_content, list):
                new_content = []
                for part in processed_content:
                    if isinstance(part, dict):
                        img_url = part.get("image_url", {})
                        if isinstance(img_url, dict):
                            url = img_url.get("url", "")
                        else:
                            url = img_url
                        if url == "__DROP_IMAGE__":
                            new_content.append({
                                "type": "text",
                                "text": "[Image dropped due to size constraints]"
                            })
                            dropped_in_message += 1
                        else:
                            new_content.append(part)
                    else:
                        new_content.append(part)
                processed_content = new_content

            elif isinstance(processed_content, dict):
                img_url = processed_content.get("image_url", {})
                if isinstance(img_url, dict):
                    url = img_url.get("url", "")
                else:
                    url = img_url
                if url == "__DROP_IMAGE__":
                    processed_content = [
                        {"type": "text", "text": "[Image dropped due to size constraints]"}
                    ]
                    dropped_in_message = 1

            current_image_idx += image_count_in_message

            if stats_list:
                new_message["content"] = processed_content
                for stats in stats_list:
                    if stats.get("was_compressed"):
                        total_compressed += 1
                        if stats.get("was_dropped"):
                            self._log(f"Dropped image in message {i}")
                        else:
                            saved = stats.get("original_size", 0) - stats.get("compressed_size", 0)
                            total_saved += saved
                            compression_details.append({
                                "message_index": i,
                                "original_format": stats.get("original_format"),
                                "new_format": stats.get("new_format"),
                                "original_size": stats.get("original_size"),
                                "compressed_size": stats.get("compressed_size"),
                                "saved_bytes": saved,
                            })

            new_messages.append(new_message)

        body["messages"] = new_messages

        final_payload_size = len(json.dumps(body, default=str))

        if final_payload_size > self.valves.max_payload_size_bytes:
            self._log(
                f"Payload too large ({format_size(final_payload_size)} > {format_size(self.valves.max_payload_size_bytes)}), "
                f"dropping oldest images..."
            )

            messages = body.get("messages", [])
            dropped_count = 0

            for msg_idx, message in enumerate(messages):
                if not isinstance(message, dict):
                    continue

                content = message.get("content")
                if not content:
                    continue

                has_images = False
                if isinstance(content, list):
                    has_images = any(
                        isinstance(part, dict) and part.get("type") == "image_url"
                        for part in content
                    )
                elif isinstance(content, dict) and content.get("type") == "image_url":
                    has_images = True

                if has_images:
                    if isinstance(content, list):
                        new_content = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "image_url":
                                new_content.append({
                                    "type": "text",
                                    "text": "[Image dropped due to size constraints]"
                                })
                                dropped_count += 1
                            else:
                                new_content.append(part)
                        message["content"] = new_content
                    elif isinstance(content, dict) and content.get("type") == "image_url":
                        message["content"] = [
                            {"type": "text", "text": "[Image dropped due to size constraints]"}
                        ]
                        dropped_count += 1

                    final_payload_size = len(json.dumps(body, default=str))
                    if final_payload_size <= self.valves.max_payload_size_bytes:
                        self._log(
                            f"Payload now {format_size(final_payload_size)} after dropping {dropped_count} image(s)"
                        )
                        break

            if dropped_count > 0:
                await self._emit_status(
                    f"🖼️ Dropped {dropped_count} image(s) due to size constraints",
                    __event_emitter__,
                )

        final_image_usage = self._calculate_image_token_usage(body.get("messages", []))

        if total_compressed > 0:
            self._log(
                f"Compressed {total_compressed} image(s), saved {format_size(total_saved)}"
            )

            status_msg = (
                f"🖼️ Compressed {total_compressed} image(s) - saved {format_size(total_saved)}"
            )
            status_msg += f" · {self._build_image_token_status(final_image_usage)}"

            await self._emit_status(status_msg, __event_emitter__)

            await self._emit_debug_log(
                f"Compressed {total_compressed} image(s), saved {format_size(total_saved)}, final image tokens: {final_image_usage.get('image_tokens', 0)}",
                __event_call__,
            )

            if self.valves.debug_mode:
                for detail in compression_details:
                    self._log(
                        f"  Message {detail['message_index']}: "
                        f"{detail['original_format']}→{detail['new_format']}, "
                        f"{format_size(detail['original_size'])}→{format_size(detail['compressed_size'])}",
                        level="debug",
                    )
        else:
            self._log(f"No images needed compression (checked {total_images} images)")
            await self._emit_debug_log(
                f"No compression needed (checked {total_images} images), final image tokens: {final_image_usage.get('image_tokens', 0)} across {final_image_usage.get('image_count', 0)} image(s)",
                __event_call__,
            )

            if final_image_usage.get("image_count", 0) > 0:
                await self._emit_status(
                    self._build_image_token_status(final_image_usage),
                    __event_emitter__,
                )

        self._log(
            f"Final payload size: {format_size(final_payload_size)} "
            f"(saved {format_size(initial_payload_size - final_payload_size)})"
        )

        await self._emit_debug_log(
            f"Final payload: {format_size(final_payload_size)}, final image tokens: {final_image_usage.get('image_tokens', 0)}",
            __event_call__,
        )

        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__: Optional[callable] = None,
    ) -> dict:
        """Process messages after receiving LLM response."""
        return body

    async def stream(
        self,
        event: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
    ) -> dict:
        """Process streaming events."""
        return event