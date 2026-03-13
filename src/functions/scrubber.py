"""
title: Scrubber (Filter)
description: Performs "scrubbing" of chat outputs to ensure they are valid. Image scrubbing is currently supported.
version: 0.0.1-alpha
"""

from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Tuple

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
        """
        Check if this scrubber should process this data.
        Override in subclass.
        """
        return False

    def scrub(self, data: Any) -> Any:
        """
        Perform the scrubbing action.
        Override in subclass.
        """
        return data


# =============================================================================
# IMAGE SCRUBBER
# =============================================================================


class PIIScrubber(Scrubber):
    """Scrubs personally identifiable information from text content."""
    
    patterns = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "api_key": r"(?i)(api[_-]?key|token|secret)[\s:=]+[a-zA-Z0-9_-]{20,}",
    }

    def should_scrub(self, data: Any) -> bool:
        """
        Check if data contains text that might need PII scrubbing.
        """
        if isinstance(data, dict):
            # Check for text content in various formats
            if "content" in data:
                content = data["content"]
                if isinstance(content, str) and content.strip():
                    return True
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("text", "").strip():
                            return True

            # Check for choices with delta content
            if "choices" in data:
                for choice in data.get("choices", []):
                    delta = choice.get("delta", {})
                    if "content" in delta and delta["content"]:
                        return True

        return False

    def scrub_text(self, text: str) -> str:
        """Scrub PII patterns from text content."""
        if not text or not isinstance(text, str):
            return text

        import re
        scrubbed_text = text
        
        # Apply each pattern
        for pii_type, pattern in self.patterns.items():
            if pii_type == "api_key":
                # For API keys, replace with [REDACTED_API_KEY]
                scrubbed_text = re.sub(pattern, "[REDACTED_API_KEY]", scrubbed_text)
            elif pii_type == "email":
                # For emails, show first letter and domain
                def anonymize_email(match):
                    email = match.group(0)
                    if "@" in email:
                        local, domain = email.split("@", 1)
                        if len(local) > 0:
                            return f"{local[0]}***@{domain}"
                    return "***@***"
                scrubbed_text = re.sub(pattern, anonymize_email, scrubbed_text)
            elif pii_type == "phone":
                # For phone numbers, show last 4 digits only
                def anonymize_phone(match):
                    phone = match.group(0)
                    # Keep only last 4 digits
                    digits = ''.join(c for c in phone if c.isdigit())
                    if len(digits) >= 4:
                        return f"***-***-{digits[-4:]}"
                    return "***-***-****"
                scrubbed_text = re.sub(pattern, anonymize_phone, scrubbed_text)
            elif pii_type == "ssn":
                # For SSN, show only last 4 digits
                def anonymize_ssn(match):
                    ssn = match.group(0)
                    digits = ''.join(c for c in ssn if c.isdigit())
                    if len(digits) == 9:
                        return f"***-**-{digits[-4:]}"
                    return "***-**-****"
                scrubbed_text = re.sub(pattern, anonymize_ssn, scrubbed_text)
            elif pii_type == "credit_card":
                # For credit cards, show only last 4 digits
                def anonymize_cc(match):
                    cc = match.group(0)
                    digits = ''.join(c for c in cc if c.isdigit())
                    if len(digits) >= 4:
                        return f"****-****-****-{digits[-4:]}"
                    return "****-****-****-****"
                scrubbed_text = re.sub(pattern, anonymize_cc, scrubbed_text)

        return scrubbed_text

    def scrub(self, data: Any) -> Any:
        """Scrub PII from the data object."""
        if isinstance(data, dict):
            result = data.copy()
            
            # Scrub content in choices
            if "choices" in result:
                for choice in result["choices"]:
                    delta = choice.get("delta", {})
                    if "content" in delta and delta["content"]:
                        delta["content"] = self.scrub_text(delta["content"])
            
            # Scrub direct content
            if "content" in result:
                if isinstance(result["content"], str):
                    result["content"] = self.scrub_text(result["content"])
                elif isinstance(result["content"], list):
                    for item in result["content"]:
                        if isinstance(item, dict) and "text" in item:
                            item["text"] = self.scrub_text(item["text"])
            
            return result
        
        return data

    def scrub_message(self, message: dict) -> dict:
        """Scrub PII from a message object."""
        if "content" in message:
            if isinstance(message["content"], str):
                message["content"] = self.scrub_text(message["content"])
            elif isinstance(message["content"], list):
                for item in message["content"]:
                    if isinstance(item, dict) and "text" in item:
                        item["text"] = self.scrub_text(item["text"])
        
        # Also scrub other fields that might contain PII
        for key in ["role", "name", "content"]:
            if key in message and isinstance(message[key], str):
                message[key] = self.scrub_text(message[key])
        
        return message


class ImageScrubber(Scrubber):
    """Scrubs invalid/phantom images from stream events."""

    def should_scrub(self, data: Any) -> bool:
        """
        Check if data contains image fields that need validation.
        Quick check - doesn't validate, just detects presence.
        """
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
        """Check if a specific image is invalid and should be removed."""
        if image_data is None:
            return True

        url = extract_image_url(image_data)

        if url is None:
            return True

        return not is_valid_image_url(url)

    def scrub(self, event: dict) -> dict:
        """Remove invalid images from the event."""
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})

            # Scrub 'images' array
            if "images" in delta:
                valid = [
                    img for img in delta["images"] if not self.should_scrub_image(img)
                ]
                if valid:
                    delta["images"] = valid
                else:
                    del delta["images"]

            # Scrub 'image' field
            if "image" in delta and self.should_scrub_image(delta["image"]):
                del delta["image"]

            # Scrub 'image_url' field
            if "image_url" in delta and self.should_scrub_image(delta["image_url"]):
                del delta["image_url"]

        return event

    def scrub_message(self, message: dict) -> dict:
        """Remove invalid images from a message object."""
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
# FILTER
# =============================================================================


class Filter:
    """
    OpenWebUI Filter that orchestrates scrubbing.
    Determines which scrubber to use, checks if scrubbing is needed,
    and invokes the scrub action.
    """

    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.scrubbers = [
            ImageScrubber(),
            PIIScrubber(),
        ]

    def stream(self, event: dict) -> dict:
        """Process stream event through all scrubbers."""
        for scrubber in self.scrubbers:
            if scrubber.should_scrub(event):
                event = scrubber.scrub(event)
        return event

    def outlet(self, body: Dict, __user__: Dict) -> Dict:
        """Process outlet messages through all scrubbers."""
        if "messages" in body:
            for msg in body["messages"]:
                for scrubber in self.scrubbers:
                    if hasattr(scrubber, "scrub_message"):
                        scrubber.scrub_message(msg)
        return body
