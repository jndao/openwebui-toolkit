"""
title: Live Context Injector
id: live_context_injector
description: Injects dynamic temporal, user, and system runtime context with truncated module lists and environment facts.
version: 0.1.0-dev.3
author_url: https://github.com/jndao
repository_url: https://github.com/jndao/openwebui-toolkit
"""

import logging
import re
import os
import shutil
import sys
from datetime import datetime, timezone
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def format_duration(seconds: int) -> str:
    """Converts seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds}s ago"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s ago"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m ago"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h ago"


def get_chat_data(chat_id: str, debug: bool = False) -> dict:
    """Fetches chat metadata and calculates interaction age from the database."""
    data = {
        "title": "New Chat",
        "msg_count": 0,
        "chat_age": "Just now",
        "interaction_age": "N/A",
    }
    try:
        from open_webui.models.chats import Chats
        from open_webui.utils.misc import get_message_list

        chat = Chats.get_chat_by_id(chat_id)
        if not chat:
            return data

        now = datetime.now(timezone.utc)
        if chat.created_at:
            created_dt = datetime.fromtimestamp(chat.created_at, tz=timezone.utc)
            data["chat_age"] = format_duration(int((now - created_dt).total_seconds()))

        if chat.chat and isinstance(chat.chat, dict):
            history = chat.chat.get("history", {})
            messages_map = history.get("messages", {})
            current_id = history.get("currentId")
            if messages_map and current_id:
                msg_list = get_message_list(messages_map, current_id)
                data["msg_count"] = len(msg_list)
                user_msgs = [m for m in msg_list if m.get("role") == "user"]
                if len(user_msgs) > 1:
                    prev_msg = user_msgs[-2]
                    prev_ts = prev_msg.get("timestamp")
                    if prev_ts:
                        prev_dt = datetime.fromtimestamp(prev_ts, tz=timezone.utc)
                        data["interaction_age"] = format_duration(
                            int((now - prev_dt).total_seconds())
                        )
        data["title"] = chat.title or "New Chat"
    except Exception as e:
        if debug:
            logger.error(f"[Live Context] DB Error: {e}")
    return data


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=5, description="Filter execution order.")
        debug_mode: bool = Field(default=False, description="Enable debug logging.")
        max_modules: int = Field(
            default=50,
            description="Maximum number of modules to list before truncating.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(
        self, body: dict, __user__: dict = None, __metadata__: dict = None, **kwargs
    ):
        messages = body.get("messages", [])
        if not messages:
            return body

        now = datetime.now(timezone.utc)
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")

        chat_id = (__metadata__ or {}).get("chat_id", "")
        chat_info = {
            "title": "Local Session",
            "msg_count": len(messages),
            "chat_age": "New",
            "interaction_age": "N/A",
        }
        if chat_id and not chat_id.startswith("local:"):
            chat_info = get_chat_data(chat_id, self.valves.debug_mode)

        # Location Fallback Logic
        user_location = __user__.get("location")
        user_tz = __user__.get("timezone", "Australia/Sydney")
        if not user_location or user_location.lower() == "unknown":
            user_location = f"{user_tz} (Timezone Fallback)"

        # Filesystem Discovery
        files_str = "None"
        storage_str = "Unknown"
        try:
            upload_dir = "/mnt/uploads/"
            file_list = []
            for f in os.listdir(upload_dir):
                f_path = os.path.join(upload_dir, f)
                size = os.path.getsize(f_path)
                size_str = (
                    f"{size/1024:.1f} KB"
                    if size < 1024**2
                    else f"{size/1024**2:.1f} MB"
                )
                file_list.append(f"- {f} ({size_str})")
            if file_list:
                files_str = "\n        ".join(file_list)

            usage = shutil.disk_usage("/")
            storage_str = f"{usage.free / (1024**3):.1f} GB free / {usage.total / (1024**3):.1f} GB total"
        except:
            pass

        # Dynamic Module Detection with Truncation
        all_libs = sorted(
            list(
                set(
                    [
                        m.split(".")[0]
                        for m in sys.modules.keys()
                        if not m.startswith("_")
                    ]
                )
            )
        )
        if len(all_libs) > self.valves.max_modules:
            libs_display = (
                ", ".join(all_libs[: self.valves.max_modules])
                + f" ... (+{len(all_libs) - self.valves.max_modules} more)"
            )
        else:
            libs_display = ", ".join(all_libs)

        context_xml = f"""<live_context>
  <temporal>
    <current_time>{current_time} (UTC)</current_time>
    <timezone>{user_tz}</timezone>
    <chat_age>{chat_info['chat_age']}</chat_age>
    <last_interaction_age>{chat_info['interaction_age']}</last_interaction_age>
  </temporal>

  <user_profile>
    <name>{__user__.get('name', 'Unknown')}</name>
    <role>{__user__.get('role', 'admin')}</role>
    <location>{user_location}</location>
  </user_profile>

  <session_info>
    <chat_title>{chat_info['title']}</chat_title>
    <message_count>{chat_info['msg_count']}</message_count>
    <active_model>{body.get('model', 'Unknown')}</active_model>
    <channel_id>{(__metadata__ or {}).get('channel_id', 'private_chat')}</channel_id>
  </session_info>

  <system_runtime>
    <runtime_type>Python {sys.version.split()[0]} (Pyodide/WASM)</runtime_type>
    <constraints>
      - Environment: Browser-based (Pyodide).
      - Package Management: pip, subprocess, and micropip.install() are unavailable.
    </constraints>
    <capabilities>
      <loaded_modules>{libs_display}</loaded_modules>
    </capabilities>
    <filesystem>
      <storage_status>{storage_str}</storage_status>
      <mount_point>/mnt/uploads/</mount_point>
      <persistence>Files at /mnt/uploads/ persist across executions in this session. Output files written here are accessible to the user.</persistence>
      <active_files>
        {files_str}
      </active_files>
    </filesystem>
  </system_runtime>
</live_context>"""

        pattern = r"<live_context>[\s\S]*?</live_context>\n?\n?"
        if messages[0].get("role") == "system":
            content = messages[0].get("content", "")
            messages[0]["content"] = re.sub(pattern, "", content).strip()
            messages[0]["content"] = context_xml + "\n\n" + messages[0]["content"]
        else:
            messages.insert(0, {"role": "system", "content": context_xml})

        body["messages"] = messages
        return body
