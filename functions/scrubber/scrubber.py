"""
title: Scrubber
id: scrubber
description: Advanced content scrubbing to prevent rendering of potentially malicious content and fix malformed tool calls.
version: 0.1.4-dev.1
author_url: https://github.com/jndao
repository_url: https://github.com/jndao/openwebui-toolkit
license: https://github.com/jndao/openwebui-toolkit/blob/main/LICENSE
Overview:
  Filters and scrubs potentially malicious content from LLM responses. Validates image URLs,
  removes dangerous HTML tags/scripts, and sanitizes JSON/SSE streams. 
  Now includes a ToolScrubber to fix "No tool output found" errors caused by MiniMax/OpenRouter.
Configuration:
  priority: 90 - filter execution order
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
# VALIDATOR
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
        """Check if this scrubber should process this data."""
        return False
    def scrub(self, data: Any) -> Any:
        """Perform the scrubbing action."""
        return data
    def scrub_message(self, message: dict) -> dict:
        """Scrub a message object (optional override)."""
        return message

# =============================================================================
# TEXT SCRUBBER
# =============================================================================
class TextScrubber(Scrubber):
    """Base for scrubbers that modify text strings within complex objects."""
    
    def scrub_text(self, text: str) -> str:
        raise NotImplementedError
    def scrub(self, data: Any) -> Any:
        if isinstance(data, dict):
            # Handle stream choices
            if "choices" in data:
                for choice in data["choices"]:
                    delta = choice.get("delta", {})
                    if "content" in delta and delta["content"]:
                        delta["content"] = self.scrub_text(delta["content"])
            
            # Handle direct content
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
        if url is None:
            return True
        return not is_valid_image_url(url)
    def scrub(self, event: dict) -> dict:
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if "images" in delta:
                valid = [
                    img for img in delta["images"] if not self.should_scrub_image(img)
                ]
                if valid:
                    delta["images"] = valid
                else:
                    del delta["images"]
            if "image" in delta and self.should_scrub_image(delta["image"]):
                del delta["image"]
            if "image_url" in delta and self.should_scrub_image(delta["image_url"]):
                del delta["image_url"]
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
    quick_check = re.compile(r'@|\d{3}[-.]?\d{3}')
    def should_scrub(self, data: Any) -> bool:
        if isinstance(data, dict):
            if "content" in data:
                content = data["content"]
                if isinstance(content, str) and content.strip():
                    return bool(self.quick_check.search(content))
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("text", "").strip():
                            return bool(self.quick_check.search(item.get("text", "")))
            if "choices" in data:
                for choice in data.get("choices", []):
                    delta = choice.get("delta", {})
                    if "content" in delta and delta["content"]:
                        return bool(self.quick_check.search(delta["content"]))
        return False
    def scrub_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return text
        scrubbed_text = text
        for pii_type, pattern in self.patterns.items():
            if pii_type == "email":
                def anonymize_email(match):
                    email = match.group(0)
                    if "@" in email:
                        local, domain = email.split("@", 1)
                        if len(local) > 0:
                            return f"{local[0]}***@{domain}"
                    return "***@***"
                scrubbed_text = re.sub(pattern, anonymize_email, scrubbed_text)
            elif pii_type == "phone":
                def anonymize_phone(match):
                    phone = match.group(0)
                    digits = ''.join(c for c in phone if c.isdigit())
                    if len(digits) >= 4:
                        return f"***-***-{digits[-4:]}"
                    return "***-***-****"
                scrubbed_text = re.sub(pattern, anonymize_phone, scrubbed_text)
            elif pii_type == "ssn":
                def anonymize_ssn(match):
                    ssn = match.group(0)
                    digits = ''.join(c for c in ssn if c.isdigit())
                    if len(digits) == 9:
                        return f"***-**-{digits[-4:]}"
                    return "***-**-****"
                scrubbed_text = re.sub(pattern, anonymize_ssn, scrubbed_text)
            elif pii_type == "credit_card":
                def anonymize_cc(match):
                    cc = match.group(0)
                    digits = ''.join(c for c in cc if c.isdigit())
                    if len(digits) >= 4:
                        return f"****-****-****-{digits[-4:]}"
                    return "****-****-****-****"
                scrubbed_text = re.sub(pattern, anonymize_cc, scrubbed_text)
            else:
                scrubbed_text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", scrubbed_text)
        return scrubbed_text

# =============================================================================
# CREDENTIAL SCRUBBER
# =============================================================================
class CredentialScrubber(TextScrubber):
    """Scrubs various types of credentials and secrets from text content."""
    patterns = {
        "api_key": re.compile(r"(?i)(api[_-]?key|token|secret|credential)[\s:=]+[a-zA-Z0-9_-]{20,}"),
        "openai_api_key": re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b"),
        "google_api_key": re.compile(r"\bAIza[0-9A-Za-z-_]{30,}\b"),
        "stripe_key": re.compile(r"\b(sk|pk)_(test|live)_[a-zA-Z0-9]{20,}\b"),
        "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "aws_secret_key": re.compile(r"\b[a-zA-Z0-9/+=]{40}\b"),
        "gcp_service_account": re.compile(r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b"),
        "github_token": re.compile(r"\b(ghp_|gho_|ghu_|ghs_|github_pat_)[a-zA-Z0-9]{20,}\b"),
        "slack_token": re.compile(r"\b(xox[baprs]-[a-zA-Z0-9]{10,})\b"),
        "discord_token": re.compile(r"\b[a-zA-Z0-9_-]{24}\.[a-zA-Z0-9_-]{6}\.[a-zA-Z0-9_-]{27}\b"),
        "private_key": re.compile(r"\b-----BEGIN (?:RSA )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA )?PRIVATE KEY-----\b"),
        "jwt_token": re.compile(r"\beyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b"),
        "authorization_header": re.compile(r"\bBearer [a-zA-Z0-9-_=]+\b"), 
        "session_id": re.compile(r"\b[A-Fa-f0-9]{32}\b"),
        "password_in_url": re.compile(r"\b[a-zA-Z]+://[^/\s]+:[^@\s]+@[^/\s]+\b"),
    }
    quick_check = re.compile(r'(api[_-]?key|token|secret|sk-|AKIA|Bearer|github_pat_)', re.IGNORECASE)
    def should_scrub(self, data: Any) -> bool:
        if isinstance(data, dict):
            if "content" in data:
                content = data["content"]
                if isinstance(content, str) and content.strip():
                    return bool(self.quick_check.search(content))
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("text", "").strip():
                            text = item.get("text", "")
                            if isinstance(text, str) and bool(self.quick_check.search(text)):
                                return True
            if "choices" in data:
                for choice in data.get("choices", []):
                    delta = choice.get("delta", {})
                    if "content" in delta and delta["content"]:
                        content = delta["content"]
                        if isinstance(content, str) and bool(self.quick_check.search(content)):
                            return True
        return False
    def scrub_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return text
        scrubbed_text = text
        for cred_type, pattern in self.patterns.items():
            scrubbed_text = pattern.sub(f"[REDACTED_{cred_type.upper()}]", scrubbed_text)
        return scrubbed_text

# =============================================================================
# TOOL CALL SCRUBBER (Fixes MiniMax/Provider ID mismatches)
# =============================================================================
class ToolScrubber(Scrubber):
    """
    Scrubs malformed or problematic tool calls from the message history.
    Specifically targets the 'call_function_...' pattern used by MiniMax.
    """
    
    # Pattern for MiniMax style IDs: call_function_[alphanumeric]_[index]
    MINIMAX_ID_PATTERN = re.compile(r"call_function_[a-zA-Z0-9]+_\d+")

    def scrub_message_list(self, messages: List[Dict]) -> List[Dict]:
        """
        Filters the message list to remove pairs of malformed tool calls 
        and their corresponding outputs.
        """
        if not messages:
            return messages

        cleaned_messages = []
        # Track IDs we've decided to scrub to ensure we remove the 'tool' role response too
        scrubbed_ids = set()

        for msg in messages:
            role = msg.get("role")
            
            # 1. Check Assistant Messages for malformed tool_calls
            if role == "assistant" and "tool_calls" in msg:
                original_tool_calls = msg["tool_calls"]
                # Filter out calls matching the bad pattern
                valid_tool_calls = []
                for tool_call in original_tool_calls:
                    tc_id = tool_call.get("id", "")
                    if self.MINIMAX_ID_PATTERN.match(tc_id):
                        scrubbed_ids.add(tc_id)
                    else:
                        valid_tool_calls.append(tool_call)
                
                if not valid_tool_calls:
                    # If all tool calls were bad, remove the tool_calls key entirely
                    msg.pop("tool_calls")
                else:
                    msg["tool_calls"] = valid_tool_calls

            # 2. Check Tool Messages for matching scrubbed IDs
            if role == "tool":
                tc_id = msg.get("tool_call_id", "")
                if tc_id in scrubbed_ids or self.MINIMAX_ID_PATTERN.match(tc_id):
                    continue # Skip this message (scrub it)

            cleaned_messages.append(msg)
            
        return cleaned_messages

    def scrub_body(self, body: Dict) -> Dict:
        """Process the entire request body."""
        if "messages" in body:
            body["messages"] = self.scrub_message_list(body["messages"])
        return body

# =============================================================================
# FILTER
# =============================================================================
class Filter:
    """
    OpenWebUI Filter that orchestrates scrubbing.
    """
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        # Simple list of all scrubbers to apply
        self.scrubbers = [
            ImageScrubber(),
            PIIScrubber(),
            CredentialScrubber(),
            ToolScrubber()
        ]

    def inlet(self, body: Dict, __user__: Optional[Dict] = None) -> Dict:
        """Process inlet messages through all scrubbers in sequence."""
        if "messages" in body:
            # 1. Handle ToolScrubber (needs whole list context)
            tool_scrubber = next((s for s in self.scrubbers if isinstance(s, ToolScrubber)), None)
            if tool_scrubber:
                body = tool_scrubber.scrub_body(body)
            
            # 2. Run other scrubbers on individual messages
            for msg in body["messages"]:
                for scrubber in self.scrubbers:
                    if not isinstance(scrubber, ToolScrubber) and hasattr(scrubber, 'scrub_message'):
                        # Check should_scrub for performance before applying
                        if scrubber.should_scrub(msg):
                            scrubber.scrub_message(msg)
        return body

    def stream(self, event: dict) -> dict:
        """Process stream event through all scrubbers in sequence."""
        for scrubber in self.scrubbers:
            if scrubber.should_scrub(event):
                event = scrubber.scrub(event)
        return event

    def outlet(self, body: Dict, __user__: Dict) -> Dict:
        """Process outlet messages through all scrubbers in sequence."""
        if "messages" in body:
            for msg in body["messages"]:
                for scrubber in self.scrubbers:
                    # ToolScrubber is inlet-only as it fixes history; others apply to output
                    if not isinstance(scrubber, ToolScrubber):
                        scrubber.scrub_message(msg)
        return body
