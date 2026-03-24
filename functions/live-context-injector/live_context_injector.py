"""
title: Live Context Injector
id: live_context_injector
description: Advanced environmental awareness (files, model, device, channel) for Open WebUI.
version: 0.1.0-dev
author_url: https://github.com/jndao
repository_url: https://github.com/jndao/openwebui-toolkit
funding_url: https://ko-fi.com/jndao
license: https://github.com/jndao/openwebui-toolkit/blob/main/LICENSE
"""

import logging
import re
import os
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field
from typing import Optional, List

logger = logging.getLogger(__name__)


def get_chat_metadata(chat_id: str, debug_mode: bool = False) -> dict:
    """Retrieve chat metadata and the timestamp of the PREVIOUS assistant message."""
    result = {
        "title": "New Conversation",
        "message_count": 0,
        "time_since_created": "0s ago",
        "created_at_utc": None,
        "last_interaction_at_utc": None,
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

                    # Find the PREVIOUS assistant message
                    assistant_messages = [
                        m for m in message_list if m.get("role") == "assistant"
                    ]
                    if assistant_messages:
                        last_msg = assistant_messages[-1]
                        ts = last_msg.get("timestamp") or last_msg.get("updated_at")
                        if ts:
                            last_ts_utc = datetime.fromtimestamp(
                                float(ts), tz=timezone.utc
                            )
                            # If the last message is "brand new" (current turn), take the one before it
                            if (now_utc - last_ts_utc).total_seconds() < 10 and len(
                                assistant_messages
                            ) > 1:
                                prev_msg = assistant_messages[-2]
                                prev_ts = prev_msg.get("timestamp") or prev_msg.get(
                                    "updated_at"
                                )
                                if prev_ts:
                                    result["last_interaction_at_utc"] = (
                                        datetime.fromtimestamp(
                                            float(prev_ts), tz=timezone.utc
                                        )
                                    )
                            else:
                                result["last_interaction_at_utc"] = last_ts_utc

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
        show_file_list: bool = Field(
            default=True, description="List files in /mnt/uploads/"
        )

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

        # 1. Variables & Environment
        variables = {}
        user_agent = ""
        if __request__ is not None:
            try:
                variables = (
                    getattr(__request__, "scope", {})
                    .get("state", {})
                    .get("metadata", {})
                    .get("variables", {})
                )
                user_agent = __request__.headers.get("user-agent", "").lower()
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

        # 2. Model & Device
        raw_model = __metadata__.get("model", "unknown") if __metadata__ else "unknown"
        active_model = (
            raw_model.get("id", raw_model.get("name", "unknown"))
            if isinstance(raw_model, dict)
            else raw_model
        )
        channel_id = (
            __metadata__.get("channel_id", "private_chat")
            if __metadata__
            else "private_chat"
        )
        device_type = (
            "mobile"
            if any(x in user_agent for x in ["mobi", "android", "iphone", "ipad"])
            else "desktop"
        )

        # 3. File Awareness
        available_files = []
        if self.valves.show_file_list:
            try:
                available_files = [
                    f for f in os.listdir("/mnt/uploads/") if not f.startswith(".")
                ]
            except Exception:
                pass

        # 4. Interaction Logic
        chat_id = __metadata__.get("chat_id", "unknown") if __metadata__ else "unknown"
        last_interaction_age = "New Session"
        now_utc = datetime.now(timezone.utc)

        chat_metadata = {
            "time_since_created": "0s ago",
            "title": "New Conversation",
            "message_count": len(messages),
        }
        diff = None

        if chat_id and not chat_id.startswith("local:"):
            db_meta = get_chat_metadata(chat_id, self.valves.debug_mode)
            chat_metadata.update(db_meta)

            if db_meta.get("last_interaction_at_utc"):
                diff = now_utc - db_meta["last_interaction_at_utc"]
            elif db_meta.get("created_at_utc"):
                diff = now_utc - db_meta["created_at_utc"]

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

        # 5. Construct XML Block
        files_xml = ""
        if available_files:
            files_xml = (
                "\n    <active_files>"
                + "".join([f"\n      <file>{f}</file>" for f in available_files])
                + "\n    </active_files>"
            )

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
    <active_model>{active_model}</active_model>
    <device_type>{device_type}</device_type>
    <channel_id>{channel_id}</channel_id>{files_xml}
  </session_info>
</live_context>"""

        # 6. Injection Logic
        live_context_pattern = r"<live_context>[\s\S]*?</live_context>\n?\n?"
        if (
            messages
            and isinstance(messages[0], dict)
            and messages[0].get("role") == "system"
        ):
            content = messages[0].get("content", "")
            messages[0]["content"] = (
                re.sub(live_context_pattern, context_block + "\n\n", content)
                if "<live_context>" in content
                else context_block + "\n\n" + content
            )
        else:
            found = False
            for i, msg in enumerate(messages):
                if (
                    isinstance(msg, dict)
                    and msg.get("role") == "system"
                    and "<live_context>" in (msg.get("content") or "")
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
