"""
title: Live Context Injector
id: live_context_injector
description: Injects dynamic temporal, user, interface, and system runtime context for Open WebUI.
version: 0.1.0
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


def parse_ua(ua_string: str) -> dict:
    """Simple regex-based User-Agent parser for OS and Browser detection."""
    ua = ua_string.lower()
    res = {"os": "Unknown", "browser": "Unknown", "device": "Desktop"}

    if "windows" in ua:
        res["os"] = "Windows"
    elif "macintosh" in ua or "mac os" in ua:
        res["os"] = "macOS"
    elif "android" in ua:
        res["os"], res["device"] = "Android", "Mobile"
    elif "iphone" in ua or "ipad" in ua:
        res["os"], res["device"] = "iOS", "Mobile"
    elif "linux" in ua:
        res["os"] = "Linux"

    if "edg/" in ua:
        res["browser"] = "Edge"
    elif "chrome/" in ua and "safari/" in ua:
        res["browser"] = "Chrome"
    elif "firefox/" in ua:
        res["browser"] = "Firefox"
    elif "safari/" in ua:
        res["browser"] = "Safari"

    return res


def get_chat_data(chat_id: str, debug: bool = False) -> dict:
    """Fetches chat metadata and calculates interaction metrics from the database."""
    data = {
        "title": "New Chat",
        "msg_count": 0,
        "chat_age": "Just now",
        "interaction_age": "N/A",
        "velocity": "N/A",
    }
    try:
        from open_webui.models.chats import Chats
        from open_webui.utils.misc import get_message_list

        chat = Chats.get_chat_by_id(chat_id)
        if not chat:
            return data

        now = datetime.now(timezone.utc)
        total_seconds = 0
        if chat.created_at:
            created_dt = datetime.fromtimestamp(chat.created_at, tz=timezone.utc)
            total_seconds = int((now - created_dt).total_seconds())
            data["chat_age"] = format_duration(total_seconds)

        if chat.chat and isinstance(chat.chat, dict):
            history = chat.chat.get("history", {})
            messages_map = history.get("messages", {})
            current_id = history.get("currentId")
            if messages_map and current_id:
                msg_list = get_message_list(messages_map, current_id)
                data["msg_count"] = len(msg_list)

                if data["msg_count"] > 1:
                    avg_seconds = total_seconds // (data["msg_count"] // 2)
                    data["velocity"] = (
                        f"Avg {format_duration(avg_seconds).replace(' ago', '')} per turn"
                    )

                user_msgs = [m for m in msg_list if m.get("role") == "user"]
                if len(user_msgs) > 1:
                    prev_ts = user_msgs[-2].get("timestamp")
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

        now = datetime.now(timezone.utc)
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")

        chat_id = (__metadata__ or {}).get("chat_id", "")
        chat_info = {
            "title": "Local Session",
            "msg_count": len(messages),
            "chat_age": "New",
            "interaction_age": "N/A",
            "velocity": "N/A",
        }
        if chat_id and not chat_id.startswith("local:"):
            chat_info = get_chat_data(chat_id, self.valves.debug_mode)

        user_tz = __user__.get("timezone", "UTC")
        user_location = __user__.get("location")
        if not user_location or user_location.lower() == "unknown":
            user_location = f"{user_tz} (Timezone Fallback)"

        ua_data = {"os": "Unknown", "browser": "Unknown", "device": "Desktop"}
        if __request__:
            ua_string = __request__.headers.get("user-agent", "")
            ua_data = parse_ua(ua_string)

        files_str, storage_str = "None", "Unknown"
        try:
            upload_dir = "/mnt/uploads/"
            file_list = [
                f"- {f} ({os.path.getsize(os.path.join(upload_dir, f))/1024:.1f} KB)"
                for f in os.listdir(upload_dir)
            ]
            if file_list:
                files_str = "\n        ".join(file_list)
            usage = shutil.disk_usage("/")
            storage_str = f"{usage.free / (1024**3):.1f} GB free / {usage.total / (1024**3):.1f} GB total"
        except:
            pass

        context_xml = f"""<live_context>
  <temporal>
    <current_time>{current_time} (UTC)</current_time>
    <timezone>{user_tz}</timezone>
    <chat_age>{chat_info['chat_age']}</chat_age>
    <last_interaction_age>{chat_info['interaction_age']}</last_interaction_age>
    <interaction_velocity>{chat_info['velocity']}</interaction_velocity>
  </temporal>

  <user_profile>
    <name>{__user__.get('name', 'Unknown')}</name>
    <role>{__user__.get('role', 'user')}</role>
    <location>{user_location}</location>
    <interface>
      <os>{ua_data['os']}</os>
      <browser>{ua_data['browser']}</browser>
      <device_type>{ua_data['device']}</device_type>
    </interface>
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
      - Do not attempt to install packages. If a library is missing, use an alternative approach.
    </constraints>
    <filesystem>
      <storage_status>{storage_str}</storage_status>
      <mount_point>/mnt/uploads/</mount_point>
      <persistence>Files at /mnt/uploads/ persist across executions. Output files written here are accessible to the user.</persistence>
      <active_files>
        {files_str}
      </active_files>
    </filesystem>
  </system_runtime>
</live_context>"""

        pattern = r"<live_context>[\s\S]*?</live_context>\n?\n?"
        if messages[0].get("role") == "system":
            content = messages[0].get("content", "")
            messages[0]["content"] = (
                context_xml + "\n\n" + re.sub(pattern, "", content).strip()
            )
        else:
            messages.insert(0, {"role": "system", "content": context_xml})

        body["messages"] = messages
        return body
