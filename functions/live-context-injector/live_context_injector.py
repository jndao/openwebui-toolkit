"""
title: Live Context Injector
id: live_context_injector
description: Injects relevant live information to allow models to be more aware of the live context of a chat.
version: 0.0.4
author_url: https://github.com/jndao
repository_url: https://github.com/jndao/openwebui-toolkit
funding_url: https://ko-fi.com/jndao
license: https://github.com/jndao/openwebui-toolkit/blob/main/LICENSE

Overview:
  Injects live user/context information (datetime, timezone, user details) into system messages
  so models can be aware of the current context. Updates existing live_context blocks.
  Also adds chat metadata: Time Since Chat Created, Chat Title, Message Count.

Configuration:
  priority: 100 - filter execution order
  debug_mode: false

Requirements: Open WebUI variables (USER_NAME, USER_EMAIL, CURRENT_DATETIME, etc.)
"""
import logging
import re
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)

def get_chat_metadata(chat_id: str, debug_mode: bool = False) -> dict:
    """Get chat metadata including time since created, title, and message count."""
    result = {"title": None, "message_count": 0, "time_since_created": None}
    try:
        from open_webui.models.chats import Chats
        from open_webui.utils.misc import get_message_list
        
        chat = Chats.get_chat_by_id(chat_id)
        if not chat:
            return result
        
        now = datetime.now(timezone.utc)
        if chat.chat and isinstance(chat.chat, dict):
            history = chat.chat.get("history", {})
            messages_map = history.get("messages", {})
            current_id = history.get("currentId")
            if messages_map and current_id:
                message_list = get_message_list(messages_map, current_id)
                if message_list:
                    result["message_count"] = len(message_list)
        
        if chat.created_at:
            created_time = datetime.fromtimestamp(chat.created_at, tz=timezone.utc)
            diff = now - created_time
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
        priority: int = Field(default=5, description="Defaults to priority 5. Recommended to have a lower priority than any context altering functions.")
        debug_mode: bool = Field(default=False, description="Enable debug logging")
    
    def __init__(self):
        self.valves = self.Valves()
    
    async def inlet(self, body: dict, __user__: dict = None, __metadata__: dict = None, __request__ = None, **kwargs):
        messages = body.get("messages", [])
        if not messages:
            return body

        # 1. Extract Variables from Request
        variables = {}
        if __request__ is not None:
            try:
                scope = getattr(__request__, 'scope', {})
                state = scope.get('state', {})
                metadata = state.get('metadata', {})
                variables = metadata.get('variables', {})
            except Exception:
                pass
        
        user_name = variables.get('{{USER_NAME}}', 'unknown')
        user_email = variables.get('{{USER_EMAIL}}', 'unknown')
        current_datetime = variables.get('{{CURRENT_DATETIME}}', 'unknown')
        current_timezone = variables.get('{{CURRENT_TIMEZONE}}', 'unknown')
        user_location = variables.get('{{USER_LOCATION}}', 'unknown')
        user_language = variables.get('{{USER_LANGUAGE}}', 'unknown')
        
        # Original Location Fallback Logic
        if user_location in ('unknown', 'Unknown', ''):
            if current_timezone and current_timezone != 'unknown':
                user_location = f"{current_timezone} (Timezone Fallback)"
        
        # 2. Get Chat Metadata
        chat_metadata = {"time_since_created": None, "title": None, "message_count": 0}
        if __metadata__ and __metadata__.get("chat_id"):
            chat_id = __metadata__.get("chat_id")
            if not chat_id.startswith("local:"):
                chat_metadata = get_chat_metadata(chat_id, self.valves.debug_mode)
        
        # 3. Construct the Context Template
        meta_parts = []
        if chat_metadata.get("time_since_created"): meta_parts.append(f"Chat Age: {chat_metadata['time_since_created']}")
        if chat_metadata.get("title"): meta_parts.append(f"Chat Title: {chat_metadata['title']}")
        if chat_metadata.get("message_count"): meta_parts.append(f"Total Messages: {chat_metadata['message_count']}")
        meta_str = "\n".join(meta_parts)

        context_block = f"""<live_context>
Current Time: {current_datetime}
User: {user_name}
Location: {user_location}
{meta_str}
</live_context>"""

        # 4. Optimized Injection Logic
        # We target the first message to avoid creating a "stack" of system messages in history
        live_context_pattern = r"<live_context>[\s\S]*?</live_context>\n?\n?"
        
        # Check if the first message is a system message (common for System Prompts)
        if messages[0].get("role") == "system":
            content = messages[0].get("content", "")
            if "<live_context>" in content:
                # Update existing block in place
                messages[0]["content"] = re.sub(live_context_pattern, context_block + "\n\n", content)
            else:
                # Prepend to the existing system prompt
                messages[0]["content"] = context_block + "\n\n" + content
        else:
            # First message isn't system (could be a Summary or User msg)
            # Search for an existing live_context block elsewhere to update it
            found_and_updated = False
            for i, msg in enumerate(messages):
                if msg.get("role") == "system" and "<live_context>" in msg.get("content", ""):
                    messages[i]["content"] = re.sub(live_context_pattern, context_block + "\n\n", msg.get("content", ""))
                    found_and_updated = True
                    break
            
            if not found_and_updated:
                # If no existing block was found, insert a fresh one at the very top
                messages.insert(0, {"role": "system", "content": context_block})

        if self.valves.debug_mode:
            logger.info(f"[Live Context] Injected into {messages[0]['role']}. Total: {len(messages)} msgs.")

        body["messages"] = messages
        return body