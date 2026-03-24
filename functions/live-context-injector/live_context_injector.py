"""
title: Live Context Injector
id: live_context_injector
description: Injects structured live context (time, user, location, metadata) into the system prompt.
version: 0.0.5
author_url: https://github.com/jndao
repository_url: https://github.com/jndao/openwebui-toolkit
funding_url: https://ko-fi.com/jndao
license: https://github.com/jndao/openwebui-toolkit/blob/main/LICENSE
"""

import logging
import re
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)


def get_chat_metadata(chat_id: str, debug_mode: bool = False) -> dict:
    """Retrieve chat metadata from the Open WebUI database."""
    result = {
        "title": "New Conversation",
        "message_count": 0,
        "time_since_created": "0s ago",
        "created_at_utc": None,
    }
    try:
        from open_webui.models.chats import Chats
        from open_webui.utils.misc import get_message_list

        chat = Chats.get_chat_by_id(chat_id)
        if not chat:
            return result

        now_utc = datetime.now(timezone.utc)

        if chat.chat and isinstance(chat.chat, dict):
            history = chat.chat.get("history", {})
            messages_map = history.get("messages", {})
            current_id = history.get("currentId")
            if messages_map and current_id:
                message_list = get_message_list(messages_map, current_id)
                if message_list:
                    result["message_count"] = len(message_list)

        if chat.created_at:
            created_utc = datetime.fromtimestamp(chat.created_at, tz=timezone.utc)
            result["created_at_utc"] = created_utc
            diff = now_utc - created_utc
            total_seconds = int(diff.total_seconds())
            if total_seconds < 60:
                result["time_since_created"] = f"{total_seconds}s ago"
            elif total_seconds < 3600:
                result["time_since_created"] = f"{total_seconds // 60}m ago"
            elif total_seconds < 86400:
                result["time_since_created"] = f"{total_seconds // 3600}h ago"
            else:
                result["time_since_created"] = f"{total_seconds // 86400}d ago"

        if chat.title:
            result["title"] = chat.title

    except Exception as e:
        if debug_mode:
            logger.error(f"[Live Context] Metadata Error: {e}")
    return result


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=5, description="Filter execution order.")
        debug_mode: bool = Field(default=False, description="Enable debug logging")

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(
        self,
        body: dict,
        __user__: dict = None,
        __metadata__: dict = None,
        __request__=None,
        **kwargs,
    ):
        messages = body.get("messages", [])
        if not messages:
            return body

        # 1. Variables Extraction
        variables = {}
        if __request__ is not None:
            try:
                variables = (
                    getattr(__request__, "scope", {})
                    .get("state", {})
                    .get("metadata", {})
                    .get("variables", {})
                )
            except Exception:
                pass

        user_name = __user__.get("name", variables.get("{{USER_NAME}}", "unknown"))
        user_role = __user__.get("role", "user")
        current_datetime_str = variables.get(
            "{{CURRENT_DATETIME}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        current_timezone = variables.get("{{CURRENT_TIMEZONE}}", "UTC")
        user_location = variables.get("{{USER_LOCATION}}", "unknown")

        if user_location in ("unknown", "Unknown", "", None):
            if current_timezone and current_timezone != "unknown":
                user_location = f"{current_timezone} (Timezone Fallback)"

        # 2. Metadata & Time Normalization
        chat_metadata = {
            "time_since_created": "0s ago",
            "title": "New Conversation",
            "message_count": len(messages),
            "created_at_utc": None,
        }
        if __metadata__ and __metadata__.get("chat_id"):
            chat_id = __metadata__.get("chat_id")
            if not chat_id.startswith("local:"):
                chat_metadata = get_chat_metadata(chat_id, self.valves.debug_mode)

        # Normalize current_datetime to UTC for accurate math
        # Since CURRENT_DATETIME is local, we treat it as naive and compare it to UTC now
        # Or more accurately, we use the system's UTC time directly for calculations.
        now_utc = datetime.now(timezone.utc)
        fmt = "%Y-%m-%d %H:%M:%S"

        # 3. Calculate Last Interaction Age
        last_interaction_age = "New Session"
        first_msg_content = messages[0].get("content", "")
        old_time_match = re.search(
            r"<current_time>(.*?)</current_time>", first_msg_content
        )

        diff = None
        if old_time_match:
            try:
                # The old_time in the block was also 'local' time.
                # Comparing two local times from the same source is safe even if naive.
                old_t = datetime.strptime(old_time_match.group(1), fmt)
                new_t = datetime.strptime(current_datetime_str, fmt)
                diff = new_t - old_t
            except Exception:
                pass
        elif chat_metadata["created_at_utc"]:
            # Compare UTC to UTC
            diff = now_utc - chat_metadata["created_at_utc"]

        if diff:
            total_sec = max(0, int(diff.total_seconds()))
            if total_sec < 60:
                last_interaction_age = f"{total_sec}s ago"
            elif total_sec < 3600:
                last_interaction_age = f"{total_sec // 60}m ago"
            elif total_sec < 86400:
                last_interaction_age = f"{total_sec // 3600}h ago"
            else:
                last_interaction_age = f"{total_sec // 86400}d ago"

        # 4. Construct Block
        context_block = f"""<live_context>
  <temporal>
    <current_time>{current_datetime_str}</current_time>
    <timezone>{current_timezone}</timezone>
    <chat_age>{chat_metadata['time_since_created']}</chat_age>
    <last_interaction_age>{last_interaction_age}</last_interaction_age>
  </temporal>
  <user_profile>
    <name>{user_name}</name>
    <role>{user_role}</role>
    <location>{user_location}</location>
  </user_profile>
  <session_info>
    <chat_title>{chat_metadata['title']}</chat_title>
    <message_count>{chat_metadata['message_count']}</message_count>
  </session_info>
</live_context>"""

        # 5. Injection
        live_context_pattern = r"<live_context>[\s\S]*?</live_context>\n?\n?"
        if messages[0].get("role") == "system":
            if "<live_context>" in first_msg_content:
                messages[0]["content"] = re.sub(
                    live_context_pattern, context_block + "\n\n", first_msg_content
                )
            else:
                messages[0]["content"] = context_block + "\n\n" + first_msg_content
        else:
            found = False
            for i, msg in enumerate(messages):
                if msg.get("role") == "system" and "<live_context>" in msg.get(
                    "content", ""
                ):
                    messages[i]["content"] = re.sub(
                        live_context_pattern,
                        context_block + "\n\n",
                        msg.get("content", ""),
                    )
                    found = True
                    break
            if not found:
                messages.insert(0, {"role": "system", "content": context_block})

        body["messages"] = messages
        return body
