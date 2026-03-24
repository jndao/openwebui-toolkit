"""
title: Live Context Injector
id: live_context_injector
description: Advanced environmental awareness (files, model, device, channel) for Open WebUI.
version: 0.1.0
author_url: https://github.com/jndao
repository_url: https://github.com/jndao/openwebui-toolkit
"""

import logging
import re
import os
from datetime import datetime, timezone
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def format_dt(seconds: int) -> str:
    """Formats seconds into a human-readable duration."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s ago"
    if seconds < 86400:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m ago"
    return f"{seconds // 86400}d {(seconds % 86400) // 3600}h ago"


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

        # 1. Environment & User Data
        vars = (
            getattr(__request__, "scope", {})
            .get("state", {})
            .get("metadata", {})
            .get("variables", {})
            if __request__
            else {}
        )
        ua = __request__.headers.get("user-agent", "").lower() if __request__ else ""

        now = datetime.now(timezone.utc)
        cur_time = vars.get("{{CURRENT_DATETIME}}", now.strftime("%Y-%m-%d %H:%M:%S"))
        tz = vars.get("{{CURRENT_TIMEZONE}}", "UTC")
        loc = vars.get("{{USER_LOCATION}}") or (
            f"{tz} (Timezone Fallback)" if tz else "unknown"
        )

        # 2. Session & Model Info
        raw_m = __metadata__.get("model", "unknown") if __metadata__ else "unknown"
        model = (
            raw_m.get("id", raw_m.get("name", "unknown"))
            if isinstance(raw_m, dict)
            else raw_m
        )
        chan = (
            __metadata__.get("channel_id", "private_chat")
            if __metadata__
            else "private_chat"
        )
        device = (
            "mobile"
            if any(x in ua for x in ["mobi", "android", "iphone", "ipad"])
            else "desktop"
        )

        # 3. Chat Metadata & Interaction Age
        chat_id = __metadata__.get("chat_id", "") if __metadata__ else ""
        meta = {
            "age": "0s ago",
            "title": "New Chat",
            "count": len(messages),
            "last": "New Session",
        }

        if chat_id and not chat_id.startswith("local:"):
            try:
                from open_webui.models.chats import Chats
                from open_webui.utils.misc import get_message_list

                chat = Chats.get_chat_by_id(chat_id)
                if chat:
                    meta["title"] = chat.title or "Untitled"
                    if chat.created_at:
                        meta["age"] = format_dt(
                            (
                                now
                                - datetime.fromtimestamp(
                                    chat.created_at, tz=timezone.utc
                                )
                            ).total_seconds()
                        )

                    # Calculate last interaction from previous user message
                    hist = chat.chat.get("history", {}) if chat.chat else {}
                    msgs = (
                        get_message_list(
                            hist.get("messages", {}), hist.get("currentId")
                        )
                        if hist
                        else []
                    )
                    if msgs:
                        meta["count"] = len(msgs)
                        u_msgs = [m for m in msgs if m.get("role") == "user"]
                        if len(u_msgs) > 1:
                            ts = u_msgs[-2].get("timestamp") or u_msgs[-2].get(
                                "updated_at"
                            )
                            if ts:
                                meta["last"] = format_dt(
                                    (
                                        now
                                        - datetime.fromtimestamp(
                                            float(ts), tz=timezone.utc
                                        )
                                    ).total_seconds()
                                )
            except Exception as e:
                if self.valves.debug_mode:
                    logger.error(f"Metadata Error: {e}")

        # 4. File Discovery
        files = []
        try:
            if os.path.exists("/mnt/uploads/"):
                files = [
                    f for f in os.listdir("/mnt/uploads/") if not f.startswith(".")
                ]
        except:
            pass
        files_xml = (
            "\n    <active_files>"
            + "".join([f"\n      <file>{f}</file>" for f in files])
            + "\n    </active_files>"
        )

        # 5. Build XML
        context = f"""<live_context>
  <temporal>
    <current_time>{cur_time}</current_time>
    <timezone>{tz}</timezone>
    <chat_age>{meta['age']}</chat_age>
    <last_interaction_age>{meta['last']}</last_interaction_age>
  </temporal>
  <user_profile>
    <name>{__user__.get('name', 'unknown')}</name>
    <role>{__user__.get('role', 'user')}</role>
    <location>{loc}</location>
  </user_profile>
  <session_info>
    <chat_title>{meta['title']}</chat_title>
    <message_count>{meta['count']}</message_count>
    <active_model>{model}</active_model>
    <device_type>{device}</device_type>
    <channel_id>{chan}</channel_id>{files_xml}
  </session_info>
</live_context>"""

        # 6. Inject/Update
        pattern = r"<live_context>[\s\S]*?</live_context>"

        def inject(text):
            if "<live_context>" in text:
                res = re.sub(pattern, context, text)
                return (
                    res if res != text else context + text.split("</live_context>")[-1]
                )
            return f"{context}\n\n{text}"

        if messages[0].get("role") == "system":
            messages[0]["content"] = inject(messages[0].get("content", ""))
        else:
            for m in messages:
                if m.get("role") == "system" and "<live_context>" in (
                    m.get("content") or ""
                ):
                    m["content"] = inject(m["content"])
                    break
            else:
                messages.insert(0, {"role": "system", "content": context})

        body["messages"] = messages
        return body
