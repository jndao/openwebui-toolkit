"""
title: Live context injector
description: Injects relevant live information to allow models to be more aware of the live context of a chat.
version: 0.0.1
"""

import logging
import re
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)

class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=100, description="Filter execution order")
        debug_mode: bool = Field(default=False, description="Enable debug logging")
    
    def __init__(self):
        self.valves = self.Valves()
    
    async def inlet(self, body: dict, __user__: dict = None, __metadata__: dict = None, __request__ = None, **kwargs):
        if self.valves.debug_mode:
            logger.info("[Live Context Injector] Starting inlet")
            logger.info(f"[Live Context Injector] Initial messages count: {len(body.get('messages', []))}")
            
            # Debug: log __request__.scope['state']['metadata']['variables']
            if __request__ is not None:
                try:
                    scope = getattr(__request__, 'scope', {})
                    state = scope.get('state', {}) if isinstance(scope, dict) else {}
                    metadata = state.get('metadata', {}) if isinstance(state, dict) else {}
                    variables = metadata.get('variables', 'N/A')
                    logger.info(f"[Live Context Injector] __request__.scope['state']['metadata']['variables']: {variables}")
                except Exception as e:
                    logger.info(f"[Live Context Injector] Error accessing scope: {e}")
            else:
                logger.info("[Live Context Injector] __request__ is None")
        
        # Get variables from __request__.scope['state']['metadata']['variables']
        variables = {}
        if __request__ is not None:
            try:
                scope = getattr(__request__, 'scope', {})
                state = scope.get('state', {}) if isinstance(scope, dict) else {}
                metadata = state.get('metadata', {}) if isinstance(state, dict) else {}
                variables = metadata.get('variables', {}) if isinstance(metadata, dict) else {}
            except Exception:
                pass
        
        # Extract values from variables (they come as {{VARIABLE_NAME}} format)
        user_name = variables.get('{{USER_NAME}}', 'unknown')
        user_email = variables.get('{{USER_EMAIL}}', 'unknown')
        user_location = variables.get('{{USER_LOCATION}}', 'unknown')
        current_datetime = variables.get('{{CURRENT_DATETIME}}', 'unknown')
        current_date = variables.get('{{CURRENT_DATE}}', 'unknown')
        current_time = variables.get('{{CURRENT_TIME}}', 'unknown')
        current_weekday = variables.get('{{CURRENT_WEEKDAY}}', 'unknown')
        current_timezone = variables.get('{{CURRENT_TIMEZONE}}', 'unknown')
        user_language = variables.get('{{USER_LANGUAGE}}', 'unknown')
        
        if self.valves.debug_mode:
            logger.info(f"[Live Context Injector] Using variables - User: {user_name}, Timezone: {current_timezone}")
        
        # Build context template with variables from request
        context_template = f"""<live_context>
Current Datetime: {current_datetime}
Current Date: {current_date}
Current Time: {current_time}
Current Weekday: {current_weekday}
Current Timezone: {current_timezone}
User: {user_name}
User Email: {user_email}
User Location: {user_location}
User Language: {user_language}
</live_context>
"""
        
        messages = body.get("messages", [])
        
        # Try to find existing live context block in any system message and replace it
        live_context_pattern = r"<live_context>[\s\S]*?</live_context>\n*\n*"
        new_context = context_template.strip() + "\n\n"
        
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                # Check if live context already exists
                if "<live_context>" in content:
                    # Replace existing live context block
                    updated_content = re.sub(live_context_pattern, new_context, content)
                    messages[i]["content"] = updated_content
                    if self.valves.debug_mode:
                        logger.info("[Live Context Injector] Replaced existing live context block")
                        logger.info(f"[Live Context Injector] System message content:\n{updated_content}")
                    body["messages"] = messages
                    return body
        
        # No existing live context found - create new system message at the START
        # This ensures it exists before OpenWebUI adds the model's system prompt
        messages.insert(0, {
            "role": "system",
            "content": new_context.strip()
        })
        if self.valves.debug_mode:
            logger.info("[Live Context Injector] Created new system message with live context")
            logger.info(f"[Live Context Injector] System message content:\n{new_context.strip()}")
        
        body["messages"] = messages
        
        if self.valves.debug_mode:
            logger.info(f"[Live Context Injector] Final messages count: {len(messages)}")
            # Log first message to verify
            if messages:
                logger.info(f"[Live Context Injector] First message role: {messages[0].get('role')}")
        
        return body