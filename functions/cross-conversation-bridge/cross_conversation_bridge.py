"""
title: Cross-Conversation Bridge
id: cross_conversation_bridge
description: Search and retrieve relevant context from your past conversations to enrich the current chat.
version: 0.0.1
author_url: https://github.com/jndao
repository_url: https://github.com/jndao/openwebui-toolkit
license: https://github.com/jndao/openwebui-toolkit/blob/main/LICENSE

Overview:
  Searches past conversations and injects relevant context into current chats. Uses AI to
  extract keywords and rank conversations by relevance. Toggleable filter for on-demand use.

Configuration:
  priority: 100 - filter execution order
  max_results: 5 (1-20) - past conversations to retrieve
  max_interactions_per_chat: 3 (1-5) - user-AI pairs per chat (~6 messages)
  keyword_model: "" - model for AI operations (defaults to current model)
  debug_mode: false
  lookback_days: 30 (0-365) - search within N days, 0 = all time
  max_candidates: 50 (10-200) - candidate chats before AI ranking
  min_relevance_threshold: 0.75 (0-1) - minimum relevance score to include

Requirements: Open WebUI models (Chats, Users)
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import json
import re

logger = logging.getLogger(__name__)

# Import OpenWebUI components like async_context_compression does
try:
    from open_webui.models.chats import Chats
except ModuleNotFoundError:
    Chats = None


class Filter:
    """
    Cross-Conversation Bridge Filter (Alpha)
    
    A toggleable filter that searches all of a user's past conversations
    and injects relevant context into the current chat when enabled.
    
    Use cases:
    - "Where did I discuss X before?"
    - "Find that conversation about Y"
    - Continuity across unrelated chats
    
    Uses AI to:
    1. Extract search keywords from the user's message
    2. Rank candidate chats by relevance
    """

    class Valves(BaseModel):
        priority: int = Field(
            default=100,
            description="Filter execution order. Lower values run first."
        )
        max_results: int = Field(
            default=5,
            ge=1,
            le=20,
            description="Maximum number of past conversations to retrieve"
        )
        max_interactions_per_chat: int = Field(
            default=3,
            ge=1,
            le=5,
            description="Maximum user-AI interactions (pairs) to include per chat. 3 = ~6 messages."
        )
        keyword_model: str = Field(
            default="",
            description="Model to use for AI operations. If empty, uses the current conversation model."
        )
        debug_mode: bool = Field(
            default=False,
            description="Enable debug logging"
        )
        lookback_days: int = Field(
            default=30,
            ge=0,
            le=365,
            description="Only search chats from the last N days. Set to 0 to disable time filtering."
        )
        max_candidates: int = Field(
            default=50,
            ge=10,
            le=200,
            description="Maximum number of candidate chats to fetch before AI ranking. Lower values are faster."
        )
        min_relevance_threshold: float = Field(
            default=0.75,
            ge=0.0,
            le=1.0,
            description="Minimum relevance score (0-1) required to include a chat. Only chats with relevance >= threshold will be injected."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True  # Makes this a toggleable filter
        self.icon = "https://cdn.jsdelivr.net/npm/lucide-static@0.469.0/icons/git-compare.svg"

    async def _call_ai(self, prompt: str, user_dict: dict) -> Optional[Any]:
        """
        Call AI and parse JSON response.
        
        Args:
            prompt: The prompt to send to AI
            user_dict: User context including __request__ if available
            
        Returns:
            Parsed JSON response (list or dict) or None if parsing fails
        """
        from open_webui.utils.chat import generate_chat_completion
        from open_webui.models.users import Users
        
        model = self.valves.keyword_model or user_dict.get('__model_name__', '')
        if not model:
            if self.valves.debug_mode:
                logger.info("[Cross-Conv Bridge] No model available for AI call")
            return None
        
        user_id = user_dict.get('id')
        if not user_id:
            return None
        
        user_obj = Users.get_user_by_id(user_id)
        if not user_obj:
            return None
        
        request = user_dict.get("__request__")
        
        try:
            response = await generate_chat_completion(request, {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.3,
            }, user_obj)
        except Exception as e:
            if self.valves.debug_mode:
                logger.info(f"[Cross-Conv Bridge] AI call failed: {e}")
            return None
        
        # Handle JSONResponse objects
        if hasattr(response, 'body'):
            try:
                response = json.loads(response.body.decode('utf-8'))
            except:
                if self.valves.debug_mode:
                    logger.info("[Cross-Conv Bridge] Failed to decode JSONResponse")
                return None
        
        if not response or not isinstance(response, dict):
            return None
        
        choices = response.get("choices", [])
        if not choices:
            return None
        
        content = choices[0].get("message", {}).get("content", "")
        
        for block in re.findall(r'```json\s*([\s\S]*?)\s*```', content):
            parsed = json.loads(block.strip())
            if isinstance(parsed, (list, dict)):
                return parsed
    
        if self.valves.debug_mode:
            logger.info("[Cross-Conv Bridge] Could not parse JSON from response")
        return None

    async def _select_relevant_chats(
        self,
        candidates: List[Dict[str, Any]],
        user_message: str,
        user_dict: dict
    ) -> List[tuple]:
        """
        Single AI call to select and rank relevant chats with relevance scores.
        Returns list of (chat_id, relevance_score) tuples sorted by relevance (descending).
        Only returns chats with relevance >= min_relevance_threshold.
        """
        if not candidates:
            return []
        
        # Build metadata JSON for each candidate
        chats_data = []
        for chat in candidates:
            # Get first user message preview (text only, no images)
            first_msg = ""
            for msg in chat.get('messages', []):
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    # Extract text only, skip images
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        text = ""
                        for part in content:
                            if isinstance(part, dict) and part.get('type') == 'text':
                                text = part.get('text', '')
                                break
                            elif isinstance(part, str):
                                text = part
                                break
                    else:
                        text = str(content) if content else ""
                    
                    first_msg = text[:150] + "..." if len(text) > 150 else text
                    break
            
            chats_data.append({
                "chat_id": str(chat.get('chat_id', '')),
                "title": chat.get('title', 'Untitled'),
                "message_count": chat.get('message_count', 0),
                "first_message": first_msg
            })
        
        k = self.valves.max_results
        threshold = self.valves.min_relevance_threshold
        
        prompt = f"""You are a search assistant. Given the user's current message, select the most relevant past conversations from the list below.

User's current message:
"{user_message}"

Available conversations:
{json.dumps(chats_data, indent=2)}

Instructions:
1. Analyze the user's message to understand what topics they're asking about
2. Review each conversation's title and first message
3. Select up to {k} conversations that are most relevant to the user's current question
4. For each selected conversation, assign a relevance score from 0.0 to 1.0:
   - 1.0: Directly addresses the user's question/topic
   - 0.8-0.9: Highly relevant, covers related concepts
   - 0.6-0.7: Somewhat relevant, tangentially related
   - Below 0.6: Not relevant (do not include these)
5. Be objective and unbiased - only include conversations that genuinely help answer the user's question

Return ONLY a JSON array of objects with chat_id and relevance, sorted by relevance (highest first):
```json
[{{"chat_id": "id1", "relevance": 0.95}}, {{"chat_id": "id2", "relevance": 0.75}}]
```

If no conversations are relevant (all would score below 0.6), return an empty array: []

Do not include any explanation, only the JSON array.
"""
        
        result = await self._call_ai(prompt, user_dict)
        
        if self.valves.debug_mode:
            logger.info(f"[Cross-Conv Bridge] AI raw result: {result}")
        
        if isinstance(result, list):
            # Result is a list of chat_id strings (original format)
            # Assign a default relevance of 0.8 to all AI-selected chats
            # (if AI selected them, they're considered relevant)
            ranked = []
            for item in result:
                if isinstance(item, str):
                    # It's a chat_id string
                    ranked.append((str(item), 0.8))  # Default 0.8 relevance
                elif isinstance(item, dict):
                    # It's a dict with chat_id and relevance
                    chat_id = str(item.get('chat_id', ''))
                    relevance = item.get('relevance', 0.8)
                    try:
                        relevance = float(relevance)
                    except (TypeError, ValueError):
                        relevance = 0.8
                    
                    if chat_id and relevance >= threshold:
                        ranked.append((chat_id, relevance))
            
            # Sort by relevance descending
            ranked.sort(key=lambda x: x[1], reverse=True)
            
            if self.valves.debug_mode:
                logger.info(f"[Cross-Conv Bridge] AI ranked {len(ranked)} chats")
            
            return ranked
        
        # Fallback: if AI fails, return empty list (don't inject irrelevant chats)
        # This ensures we only inject chats that the AI actually rated as relevant
        if self.valves.debug_mode:
            logger.info(f"[Cross-Conv Bridge] AI ranking failed, returning empty list")
        
        return []

    def _get_all_user_chats(self, user_id: str) -> List:
        """Get ONLY the current user's chats using the Chats model.
        
        Uses the proper user-filtered method from Open WebUI's Chats model.
        Throws error if method is not available.
        """
        user_chats = Chats.get_chats_by_user_id_and_search_text(
            user_id=user_id,
            search_text="",
            include_archived=False,
            skip=0,
            limit=self.valves.max_candidates * 3,
        )
        
        if self.valves.debug_mode:
            logger.info(f"[Cross-Conv Bridge] Found {len(user_chats)} chats for user {user_id}")
        
        return list(user_chats)
    
    def _get_chat_messages(self, chat, max_interactions: int) -> List[Dict[str, Any]]:
        """Extract message pairs from a chat for context. Only extracts text content."""
        chat_data = getattr(chat, 'chat', None)
        
        # Get messages list
        messages_list = []
        history = chat_data.get('history', {}) if chat_data else {}
        msg_dict = history.get('messages', {})
        
        current_msg_id = history.get('currentId')
        visited = set()
        while current_msg_id and current_msg_id not in visited:
            visited.add(current_msg_id)
            msg = msg_dict.get(current_msg_id)
            if msg:
                messages_list.append(msg)
            current_msg_id = msg.get('parentId') if msg else None
        
        # Get up to max_interactions * 2 messages
        limit = min(len(messages_list), max_interactions * 2)
        relevant_messages = []
        for msg in messages_list[:limit]:
            text_content = self._extract_text_content(msg.get('content', ''))
            if text_content:
                truncated = text_content[:200] + "..." if len(text_content) > 200 else text_content
                relevant_messages.append({'role': msg.get('role', 'unknown'), 'content': truncated})
        
        return relevant_messages

    def _get_candidates(self, user_id: str, current_chat_id: str = None) -> List[Dict[str, Any]]:
        """Get candidate chats using simple filters (recency, message count, time window)."""
        user_chats = self._get_all_user_chats(user_id)
        
        if not user_chats:
            return []
        
        # Calculate time cutoff for lookback_days
        import time
        time_cutoff = 0
        if self.valves.lookback_days > 0:
            seconds_per_day = 86400
            time_cutoff = time.time() - (self.valves.lookback_days * seconds_per_day)
        
        candidates = []
        
        for chat in user_chats:
            # Skip current chat
            if current_chat_id and hasattr(chat, 'id') and str(chat.id) == str(current_chat_id):
                continue
            
            # Get timestamp for recency filtering and scoring
            updated_at = getattr(chat, 'updated_at', None)
            timestamp = 0
            if updated_at:
                if hasattr(updated_at, 'timestamp'):
                    timestamp = updated_at.timestamp()
                else:
                    timestamp = updated_at
            
            # Apply lookback_days filter
            if self.valves.lookback_days > 0 and timestamp < time_cutoff:
                continue
            
            # Get chat metadata
            chat_title = ""
            if hasattr(chat, 'title') and chat.title:
                chat_title = str(chat.title)
            
            chat_data = getattr(chat, 'chat', None)
            message_count = 0
            
            if isinstance(chat_data, dict):
                history = chat_data.get('history', {})
                msg_dict = history.get('messages', {})
                message_count = len(msg_dict)
            
            # Skip very short chats
            if message_count < 2:
                continue
            
            # Score: recency + message count
            # Using timestamp as primary sort, message count as tiebreaker
            score = timestamp + (message_count * 0.001)
            
            candidates.append({
                'chat': chat,
                'chat_id': str(getattr(chat, 'id', '')),
                'title': chat_title,
                'message_count': message_count,
                'score': score,
                'updated_at': timestamp
            })
        
        # Sort by score descending (most recent + longer chats first)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top candidates limited by max_candidates valve
        return candidates[:self.valves.max_candidates]

    def _extract_text_content(self, content: Any) -> str:
        """
        Extract text from message content, stripping images/vision content.
        Handles string, list (mixed content), and dict formats.
        """
        if not content:
            return ""
        
        # If content is a string, use it directly (text-only message)
        if isinstance(content, str) and content.strip():
            return content
        
        # If content is a list (mixed content with images), extract only text parts
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get('type', '')
                    if part_type == 'text':
                        text = part.get('text', '')
                        if isinstance(text, str) and text.strip():
                            text_parts.append(text)
                    elif 'text' in part:
                        text = part.get('text', '')
                        if isinstance(text, str) and text.strip():
                            text_parts.append(text)
                elif isinstance(part, str) and part.strip():
                    text_parts.append(part)
            
            return ' '.join(text_parts)
        
        # If content is a dict, check for text field
        if isinstance(content, dict):
            text = content.get('text', '')
            if isinstance(text, str) and text.strip():
                return text
        
        return ""

    def _build_context_message(self, search_results: List[Dict[str, Any]]) -> str:
        """Build a context message from search results."""
        if not search_results:
            return ""

        # Embedded style with full message content
        lines = [
            "📚 **Relevant context from your past conversations:**",
            ""
        ]
        
        for i, result in enumerate(search_results, 1):
            title = result.get('title', 'Untitled')
            chat_id = result.get('chat_id', '')[:8]
            message_count = result.get('message_count', 0)
            relevance = result.get('relevance', 0)
            relevance_pct = int(relevance * 100) if relevance else 0
            
            lines.append(f"### {i}. {title} (ID: {chat_id}... • {message_count} messages, relevance: {relevance_pct}%)")
            
            # Add message previews (text only, no images)
            messages = result.get('messages', [])
            for msg in messages[:2]:
                role = msg.get('role', '?')
                content = msg.get('content', '')
                # Extract text only, stripping images
                text = self._extract_text_content(content)
                if text:
                    display = text[:150] + "..." if len(text) > 150 else text
                    lines.append(f"   **{role}**: {display}")
            
            lines.append("")
        
        return "\n".join(lines)

    def _emit_status(self, __event_emitter__, description: str, done: bool = True):
        """Emit status update to the UI."""
        if not __event_emitter__:
            return
        
        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(__event_emitter__({
                        "type": "status",
                        "data": {"description": description, "done": done}
                    }))
                else:
                    __event_emitter__({
                        "type": "status",
                        "data": {"description": description, "done": done}
                    })
            except RuntimeError:
                __event_emitter__({
                    "type": "status",
                    "data": {"description": description, "done": done}
                })
        except Exception:
            pass

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__=None,
        __metadata__: Optional[dict] = None,
        __model__: Optional[dict] = None,
        __request__=None
    ) -> dict:
        """Main inlet function - searches past conversations and injects relevant context."""
        # Only process if user is provided
        if not __user__:
            return body
        
        user_id = __user__.get('id')
        if not user_id:
            return body
        
        messages = body.get('messages', [])
        if not messages:
            return body
        
        # Get the user's latest message
        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get('role') == 'user':
                user_message = msg.get('content', '')
                break
        
        if not user_message:
            return body
        
        # Get current chat ID to exclude from results
        current_chat_id = None
        if __metadata__:
            current_chat_id = __metadata__.get('chat_id')
        
        # Prepare user context with model name for _call_ai
        user_context = dict(__user__) if __user__ else {}
        user_context["__request__"] = __request__
        user_context["__model_name__"] = body.get('model', '')
        
        # Step 1: Get candidate chats using simple filters (recency, lookback window)
        if __event_emitter__:
            self._emit_status(__event_emitter__, "🔍 Finding relevant past conversations...", done=False)
        
        candidates = self._get_candidates(user_id, current_chat_id)
        
        if not candidates:
            if __event_emitter__:
                self._emit_status(__event_emitter__, "No past conversations found", done=True)
            return body
        
        # Step 2: Use AI to select and rank relevant chats (single AI call)
        if __event_emitter__:
            count_msg = f"🤔 Analyzing {len(candidates)} conversations with AI..."
            self._emit_status(__event_emitter__, count_msg, done=False)
        
        ranked_chat_ids = await self._select_relevant_chats(candidates, user_message, user_context)
        
        if self.valves.debug_mode:
            logger.info(f"[Cross-Conv Bridge] AI-selected chat IDs: {ranked_chat_ids}")
        
        if not ranked_chat_ids:
            if __event_emitter__:
                self._emit_status(__event_emitter__, "No relevant conversations found", done=True)
            return body
        
        # Step 3: Build final results from AI-selected chats
        # Create a lookup for candidates
        candidates_by_id = {c['chat_id']: c for c in candidates}
        
        search_results = []
        for chat_id, relevance in ranked_chat_ids[:self.valves.max_results]:
            candidate = candidates_by_id.get(chat_id)
            if candidate:
                chat = candidate['chat']
                messages_context = self._get_chat_messages(chat, self.valves.max_interactions_per_chat)
                
                search_results.append({
                    'chat_id': chat_id,
                    'title': candidate['title'],
                    'message_count': candidate['message_count'],
                    'messages': messages_context,
                    'relevance': relevance
                })
        
        if not search_results:
            if __event_emitter__:
                self._emit_status(__event_emitter__, "No relevant past conversations found", done=True)
            return body
        
        # Step 4: Build and inject context message
        context_message = self._build_context_message(search_results)
        
        if context_message:
            context_msg = {"role": "system", "content": context_message}
            messages.append(context_msg)
            body['messages'] = messages
            
            if __event_emitter__:
                for result in search_results:
                    title = result.get('title', 'Untitled')[:50]
                    self._emit_status(
                        __event_emitter__,
                        f"📚 {title}",
                        done=False
                    )
                self._emit_status(
                    __event_emitter__,
                    f"✅ Injected context from {len(search_results)} conversations",
                    done=True
                )
        
        if self.valves.debug_mode:
            logger.info(f"[Cross-Conv Bridge] Injected context from {len(search_results)} conversations")
        
        return body
