"""
title: Scrubber
id: scrubber
description: Advanced content scrubbing to prevent rendering of potentially malicious content and fix malformed tool calls.
version: 0.2.0
author_url: https://github.com/jndao
repository_url: https://github.com/jndao/openwebui-toolkit
funding_url: https://ko-fi.com/jndao
license: https://github.com/jndao/openwebui-toolkit/blob/main/LICENSE
Overview:
  Filters and scrubs potentially malicious content from LLM responses. Validates image URLs,
  removes dangerous HTML tags/scripts, and sanitizes JSON/SSE streams.
  Includes an aggressive ToolScrubber that runs AFTER context compression to ensure
  no malformed IDs survive summarization or injection.
Configuration:
  priority: 15 - filter execution order (Runs AFTER Context Compression default at 10)
  enable_html_scrubbing: true - remove dangerous HTML tags
  enable_json_scrubbing: true - sanitize JSON output
  enable_image_validation: true - validate image URLs
  allowed_html_tags: ["b", "i", "em", "strong", "code", "pre", "br", "p", "ul", "ol", "li", "a", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "span"]
  debug_mode: false
Requirements: None (pure Python)
"""

from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Tuple
import re
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS: Image Magic Bytes (Base64-encoded)
# =============================================================================
IMAGE_MAGIC_BYTES: Dict[str, Tuple[str, ...]] = {
    "png": ("iVBORw0KGgo",),
    "jpeg": ("/9j/", "/9j/2", "/9j/4"),
    "gif": ("R0lGOD",),
    "webp": ("UklGR",),
    "bmp": ("Qk0",),
    "ico": ("AAABAA",),
}
VALID_IMAGE_PREFIXES = tuple(
    prefix for prefixes in IMAGE_MAGIC_BYTES.values() for prefix in prefixes
)


# =============================================================================
# VALIDATORS & HELPERS
# =============================================================================
def is_valid_image_url(url: str) -> bool:
    """Check if a URL contains valid image data."""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    if url.startswith(("http://", "https://")):
        return True
    if url.startswith("data:image/svg+xml"):
        return True
    if url.startswith("data:image/") and ";base64," in url:
        base64_data = url.split(";base64,", 1)[1]
        return base64_data.startswith(VALID_IMAGE_PREFIXES)
    return url.startswith(VALID_IMAGE_PREFIXES)


def extract_image_url(image_data: Any) -> Optional[str]:
    """Extract URL from various image data formats."""
    if isinstance(image_data, str):
        return image_data
    if isinstance(image_data, dict):
        if "image_url" in image_data:
            return image_data["image_url"].get("url")
        if "url" in image_data:
            return image_data.get("url")
    return None


# =============================================================================
# BASE SCRUBBER
# =============================================================================
class Scrubber:
    """Base class for scrubbing invalid data from streams."""

    def should_scrub(self, data: Any) -> bool:
        return False

    def scrub(self, data: Any) -> Any:
        return data

    def scrub_message(self, message: dict) -> dict:
        return message


# =============================================================================
# TEXT SCRUBBER (Base for PII/Credentials)
# =============================================================================
class TextScrubber(Scrubber):
    """Base for scrubbers that modify text strings within complex objects."""

    def scrub_text(self, text: str) -> str:
        raise NotImplementedError

    def scrub(self, data: Any) -> Any:
        if isinstance(data, dict):
            if "choices" in data:
                for choice in data["choices"]:
                    delta = choice.get("delta", {})
                    if "content" in delta and delta["content"]:
                        delta["content"] = self.scrub_text(delta["content"])
            if "content" in data:
                if isinstance(data["content"], str):
                    data["content"] = self.scrub_text(data["content"])
                elif isinstance(data["content"], list):
                    for item in data["content"]:
                        if isinstance(item, dict) and "text" in item:
                            item["text"] = self.scrub_text(item["text"])
        return data

    def scrub_message(self, message: dict) -> dict:
        if "content" in message:
            if isinstance(message["content"], str):
                message["content"] = self.scrub_text(message["content"])
            elif isinstance(message["content"], list):
                for item in message["content"]:
                    if isinstance(item, dict) and "text" in item:
                        item["text"] = self.scrub_text(item["text"])
        return message


# =============================================================================
# IMAGE SCRUBBER
# =============================================================================
class ImageScrubber(Scrubber):
    """Scrubs invalid/phantom images from stream events."""

    def should_scrub(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        if "choices" not in data:
            return False
        for choice in data.get("choices", []):
            delta = choice.get("delta", {})
            if any(key in delta for key in ["images", "image", "image_url"]):
                return True
        return False

    def should_scrub_image(self, image_data: Any) -> bool:
        if image_data is None:
            return True
        url = extract_image_url(image_data)
        return not is_valid_image_url(url) if url else True

    def scrub(self, event: dict) -> dict:
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            for key in ["images", "image", "image_url"]:
                if key in delta:
                    if key == "images":
                        valid = [
                            img
                            for img in delta[key]
                            if not self.should_scrub_image(img)
                        ]
                        if valid:
                            delta[key] = valid
                        else:
                            del delta[key]
                    elif self.should_scrub_image(delta[key]):
                        del delta[key]
        return event

    def scrub_message(self, message: dict) -> dict:
        if "files" in message:
            del message["files"]
        if isinstance(message.get("content"), list):
            valid_content = []
            for item in message["content"]:
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if is_valid_image_url(url):
                        valid_content.append(item)
                else:
                    valid_content.append(item)
            message["content"] = valid_content
        return message


# =============================================================================
# PII SCRUBBER
# =============================================================================
class PIIScrubber(TextScrubber):
    """Scrubs personally identifiable information from text content."""

    patterns = {
        "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    }
    quick_check = re.compile(r"@|\d{3}[-.]?\d{3}")

    def should_scrub(self, data: Any) -> bool:
        if isinstance(data, dict):
            content = data.get("content", "")
            if isinstance(content, str) and content.strip():
                return bool(self.quick_check.search(content))
            if "choices" in data:
                for choice in data.get("choices", []):
                    if choice.get("delta", {}).get("content"):
                        return bool(self.quick_check.search(choice["delta"]["content"]))
        return False

    def scrub_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return text
        scrubbed = text
        for pii_type, pattern in self.patterns.items():
            if pii_type == "email":
                scrubbed = re.sub(
                    pattern,
                    lambda m: (
                        f"{m.group(0)[0]}***@{m.group(0).split('@')[1]}"
                        if "@" in m.group(0)
                        else "***@***"
                    ),
                    scrubbed,
                )
            else:
                scrubbed = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", scrubbed)
        return scrubbed


# =============================================================================
# CREDENTIAL SCRUBBER
# =============================================================================
class CredentialScrubber(TextScrubber):
    """Scrubs various types of credentials and secrets from text content."""

    patterns = {
        "api_key": re.compile(
            r"(?i)(api[_-]?key|token|secret|credential)[\s:=]+[a-zA-Z0-9_-]{20,}"
        ),
        "openai_api_key": re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b"),
        "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "github_token": re.compile(
            r"\b(ghp_|gho_|ghu_|ghs_|github_pat_)[a-zA-Z0-9]{20,}\b"
        ),
        "private_key": re.compile(
            r"\b-----BEGIN (?:RSA )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA )?PRIVATE KEY-----\b"
        ),
        "jwt_token": re.compile(
            r"\beyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b"
        ),
    }
    quick_check = re.compile(
        r"(api[_-]?key|token|secret|sk-|AKIA|github_pat_)", re.IGNORECASE
    )

    def should_scrub(self, data: Any) -> bool:
        if isinstance(data, dict):
            content = data.get("content", "")
            if isinstance(content, str):
                return bool(self.quick_check.search(content))
        return False

    def scrub_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return text
        for cred_type, pattern in self.patterns.items():
            text = pattern.sub(f"[REDACTED_{cred_type.upper()}]", text)
        return text


# =============================================================================
# TOOL CALL SCRUBBER (Fixes MiniMax/Provider ID mismatches)
# =============================================================================
class ToolScrubber(Scrubber):
    """
    Aggressively scrubs malformed tool calls from history to prevent 400 errors.
    Targets the 'call_function_...' pattern used by MiniMax/OpenRouter.
    """

    # Use .search() with this pattern to catch non-standard IDs
    MINIMAX_ID_PATTERN = re.compile(r"call_function_[a-zA-Z0-9]+_\d+")

    def scrub_message_list(
        self, messages: List[Dict], debug: bool = False
    ) -> List[Dict]:
        if not messages:
            return messages

        # 1. Identify all malformed IDs in the entire history
        # We use a set for O(1) lookups
        malformed_ids = set()
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            # Check standard tool_calls
            for tc in msg.get("tool_calls", []):
                tc_id = str(tc.get("id", ""))
                if self.MINIMAX_ID_PATTERN.search(tc_id):
                    malformed_ids.add(tc_id)

            # Check tool responses
            if msg.get("role") == "tool":
                tc_id = str(msg.get("tool_call_id", ""))
                if self.MINIMAX_ID_PATTERN.search(tc_id):
                    malformed_ids.add(tc_id)

            # Check OpenWebUI 'output' metadata (hidden field)
            output_field = msg.get("output")
            if isinstance(output_field, list):
                for item in output_field:
                    # Check both possible ID keys in metadata
                    for key in ["id", "tool_call_id"]:
                        val = str(item.get(key, ""))
                        if self.MINIMAX_ID_PATTERN.search(val):
                            malformed_ids.add(val)

        if not malformed_ids:
            return messages

        if debug:
            logger.warning(
                f"[ToolScrubber] Scrubbing malformed IDs: {list(malformed_ids)}"
            )

        # 2. Reconstruct the message list, stripping all traces of these IDs
        cleaned_messages = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                cleaned_messages.append(msg)
                continue

            role = msg.get("role")

            # Scrub IDs from text content (Summaries/system prompts)
            content = msg.get("content", "")
            if isinstance(content, str) and self.MINIMAX_ID_PATTERN.search(content):
                msg["content"] = self.MINIMAX_ID_PATTERN.sub("[REDACTED_ID]", content)

            if role == "assistant":
                # Remove specific tool calls that match malformed IDs
                if "tool_calls" in msg:
                    msg["tool_calls"] = [
                        tc
                        for tc in msg["tool_calls"]
                        if str(tc.get("id")) not in malformed_ids
                    ]
                    if not msg["tool_calls"]:
                        msg.pop("tool_calls")

                # Remove from OpenWebUI metadata
                if "output" in msg and isinstance(msg["output"], list):
                    msg["output"] = [
                        item
                        for item in msg["output"]
                        if str(item.get("id") or item.get("tool_call_id"))
                        not in malformed_ids
                    ]
                    if not msg["output"]:
                        msg.pop("output")

                # Drop assistant message if it's now completely empty
                if (
                    not msg.get("content")
                    and not msg.get("tool_calls")
                    and not msg.get("output")
                ):
                    if debug:
                        logger.info(
                            f"[ToolScrubber] Dropping empty assistant message {i}"
                        )
                    continue

            elif role == "tool":
                # Drop the tool response entirely if it matches a malformed ID
                if str(msg.get("tool_call_id")) in malformed_ids:
                    if debug:
                        logger.info(
                            f"[ToolScrubber] Dropping tool response {i} for malformed ID"
                        )
                    continue

            cleaned_messages.append(msg)

        if debug:
            logger.info(
                f"[ToolScrubber] OUTLET: Returning {len(cleaned_messages)} messages."
            )
        return cleaned_messages

    def scrub_body(self, body: Dict) -> Dict:
        if "messages" in body:  # Body from OpenWebUI inlet
            body["messages"] = self.scrub_message_list(body["messages"])
        return body


# =============================================================================
# FILTER
# =============================================================================
class Filter:
    class Valves(BaseModel):
        # Set priority to 15 so it runs AFTER Async Context Compression (priority 10)
        priority: int = 15
        debug_mode: bool = False
        enable_html_scrubbing: bool = True
        enable_json_scrubbing: bool = True
        enable_image_validation: bool = True

    def __init__(self):
        self.valves = self.Valves()
        self.scrubbers = [
            ImageScrubber(),
            PIIScrubber(),
            CredentialScrubber(),
            ToolScrubber(),
        ]

    def inlet(self, body: Dict, __user__: Optional[Dict] = None) -> Dict:
        """Process inlet messages (Request to LLM)."""
        if "messages" in body:
            # 1. ToolScrubber handles the entire history list
            tool_scrubber = next(
                (s for s in self.scrubbers if isinstance(s, ToolScrubber)), None
            )
            if tool_scrubber:
                # Pass debug_mode to the scrubber
                body["messages"] = tool_scrubber.scrub_message_list(
                    body["messages"], debug=self.valves.debug_mode
                )

            # 2. Other scrubbers handle individual messages
            for msg in body["messages"]:
                for scrubber in self.scrubbers:
                    if not isinstance(scrubber, ToolScrubber) and hasattr(
                        scrubber, "scrub_message"
                    ):
                        if scrubber.should_scrub(msg):
                            scrubber.scrub_message(msg)
        return body

    def stream(self, event: dict) -> dict:
        """Process output stream events."""
        for scrubber in self.scrubbers:
            if scrubber.should_scrub(event):
                event = scrubber.scrub(event)
        return event

    def outlet(self, body: Dict, __user__: Dict) -> Dict:
        """Process final outlet response."""
        if "messages" in body:
            for msg in body["messages"]:
                for scrubber in self.scrubbers:
                    if not isinstance(scrubber, ToolScrubber):
                        scrubber.scrub_message(msg)
        return body
