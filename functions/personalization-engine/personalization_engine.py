"""
title: Personalization Engine
author: jndao
description: A two-tier memory system that autonomously extracts user observations and synthesizes them into a high-density, structured Personalization Context.
version: 0.0.3-dev.3
author_url: https://github.com/jndao
repository_url: https://github.com/jndao/openwebui-toolkit
funding_url: https://ko-fi.com/jndao
license: https://github.com/jndao/openwebui-toolkit/blob/main/LICENSE
requirements: openwebui>=0.9.0
"""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Type, TypeVar

from fastapi import Request
from pydantic import BaseModel, Field
from sqlalchemy import select, func

from open_webui.models.users import UserModel, Users
from open_webui.models.memories import Memories
from open_webui.utils.chat import generate_chat_completion
from open_webui.internal.db import get_async_db_context
from sqlalchemy import BigInteger, Column, String, Text

logger = logging.getLogger(__name__)

# --- Database Schema for Profile ---


def _discover_owui_schema() -> Optional[str]:
    try:
        from open_webui.config import DATABASE_SCHEMA

        schema = (
            DATABASE_SCHEMA.value
            if hasattr(DATABASE_SCHEMA, "value")
            else DATABASE_SCHEMA
        )
        return schema if schema else None
    except Exception:
        return None


_owui_schema = _discover_owui_schema()


class UserProfile(Base):
    __tablename__ = "user_profiles"
    __table_args__ = (
        {"extend_existing": True, "schema": _owui_schema}
        if _owui_schema
        else {"extend_existing": True}
    )
    id = Column(String, primary_key=True)
    user_id = Column(String, unique=True, nullable=False)
    content = Column(Text, nullable=False)
    updated_at = Column(BigInteger, nullable=False)
    created_at = Column(BigInteger, nullable=False)


class ProfileStore:
    def __init__(self):
        self._initialized = False

    async def _ensure_table(self):
        if self._initialized:
            return True
        try:
            async with get_async_db_context() as db:
                # Use SQLAlchemy 2.0 async style
                from sqlalchemy import text
                # Create table if not exists using raw SQL for compatibility
                await db.execute(
                    text(f"""
                        CREATE TABLE IF NOT EXISTS {f'{_owui_schema}.' if _owui_schema else ''}user_profiles (
                            id TEXT PRIMARY KEY,
                            user_id TEXT UNIQUE NOT NULL,
                            content TEXT NOT NULL,
                            updated_at INTEGER NOT NULL,
                            created_at INTEGER NOT NULL
                        )
                    """)
                )
                await db.commit()
            self._initialized = True
            return True
        except Exception as e:
            logger.error(
                f"[Personalization Engine] Failed to create user_profiles table: {e}"
            )
            return False

    async def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_table()
        try:
            async with get_async_db_context() as db:
                stmt = select(UserProfile).where(UserProfile.user_id == user_id)
                result = await db.execute(stmt)
                record = result.scalars().first()
                if record:
                    return {
                        "id": record.id,
                        "content": record.content,
                        "updated_at": record.updated_at,
                        "created_at": record.created_at,
                    }
                return None
        except Exception as e:
            logger.error(f"[Personalization Engine] Failed to get profile: {e}")
            return None

    async def save_profile(self, user_id: str, content: str) -> bool:
        await self._ensure_table()
        try:
            async with get_async_db_context() as db:
                stmt = select(UserProfile).where(UserProfile.user_id == user_id)
                result = await db.execute(stmt)
                record = result.scalars().first()
                now = int(time.time())
                if record:
                    record.content = content
                    record.updated_at = now
                else:
                    new_record = UserProfile(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        content=content,
                        created_at=now,
                        updated_at=now,
                    )
                    db.add(new_record)
                await db.commit()
                return True
        except Exception as e:
            logger.error(f"[Personalization Engine] Failed to save profile: {e}")
            return False


_profile_store = ProfileStore()

# --- Pydantic Contracts ---


class MemoryObservation(BaseModel):
    category: Literal[
        "fact", "preference", "emotional_state", "goal", "relationship"
    ] = Field(description="The type of personal concept extracted.")
    content: str = Field(
        description="A clear, self-contained statement about the user starting with 'User...'."
    )


class ExtractorContract(BaseModel):
    has_new_observations: bool = Field(
        description="True ONLY if the user's latest message contains new, long-term relevant personal concepts."
    )
    observations: List[MemoryObservation] = Field(
        default_factory=list,
        description="List of new observations. Empty if has_new_observations is false.",
    )


class ProfileConsolidationContract(BaseModel):
    profile_summary: str = Field(
        description="The fully updated markdown archive following the exact STRUCTURE and RULES provided. Put the entire markdown text in this field."
    )


# --- Main Filter Class ---
R = TypeVar("R", bound=BaseModel)


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=20, description="Filter execution order. Lower values run first."
        )
        engine_model: str = Field(
            default="",
            description="Model ID for both extraction and profile synthesis. Leave blank to default to the current chat model.",
        )
        consolidation_threshold: int = Field(
            default=5,
            description="Number of engine memories to accumulate before triggering a context reconsolidation.",
        )
        max_profile_tokens: int = Field(
            default=2000,
            description="Maximum token length of the context. If exceeded, the engine will aggressively compress older traits during synthesis to save space.",
        )
        sanitize_code_blocks: bool = Field(
            default=True,
            description="Strip code blocks from messages before extraction to save tokens. Highly recommended.",
        )
        emit_status_events: bool = Field(
            default=True,
            description="Toggle whether users should see UI status events during context synthesis.",
        )
        debug_logging: bool = Field(
            default=False, description="Enable detailed console logging."
        )

    def __init__(self):
        self.valves = self.Valves()
        self._locks: Dict[str, asyncio.Lock] = {}
        self.MAX_MEMORIES_PER_SYNTH = 50  # Safety chunking for big histories

        # Tag used to namespace memories managed by the engine
        self.ENGINE_TAG = "[PERSONALIZATION ENGINE]"

    def _log(self, msg: str, level: str = "info"):
        if level == "debug" and not self.valves.debug_logging:
            return
        getattr(logger, level, logger.info)(f"[Personalization Engine] {msg}")

    def _lock_for(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def _count_tokens(self, text: str) -> int:
        """Safely counts tokens using tiktoken if available, otherwise falls back to a character heuristic."""
        if not text:
            return 0
        try:
            import tiktoken

            # cl100k_base is the standard encoding for modern OpenAI and many open-source models
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            self._log(
                "tiktoken not found, falling back to character heuristic for token counting.",
                "debug",
            )
            return len(text) // 4
        except Exception as e:
            self._log(
                f"Token counting error: {e}. Falling back to character heuristic.",
                "debug",
            )
            return len(text) // 4

    def _sanitize_for_extraction(self, messages: List[Dict]) -> str:
        """
        Extracts text from messages, handles multimodal lists,
        and optionally strips out code blocks to save tokens during extraction.
        """
        sanitized_text = []

        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            text_content = ""

            # 1. Handle Open WebUI's multimodal list format (images/files)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_content += part.get("text", "") + "\n"
            elif isinstance(content, str):
                text_content = content

            # 2. Strip the code blocks if valve is enabled
            if self.valves.sanitize_code_blocks:
                pattern = r"[\s\S]+?"
                text_content = re.sub(pattern, "\n[Code block omitted]\n", text_content)

            sanitized_text.append(f"{role}: {text_content.strip()}")

        return "\n".join(sanitized_text)

    async def _emit_status(
        self, emitter: Optional[Callable], message: str, done: bool = True
    ):
        if emitter is None or not self.valves.emit_status_events:
            return
        try:
            await emitter(
                {"type": "status", "data": {"description": message, "done": done}}
            )
        except Exception as e:
            self._log(f"Status Emit Failed: {e}", "debug")

    async def _call_llm_native(
        self,
        request: Request,
        user: UserModel,
        model_id: str,
        system_prompt: str,
        user_message: str,
        response_model: Type[R],
    ) -> Optional[R]:
        """Calls the LLM using Open WebUI's native router and forces JSON output."""
        schema_json = json.dumps(response_model.model_json_schema())
        enforced_prompt = (
            f"{system_prompt}\n\n"
            f"Return ONLY valid JSON matching this schema. Do NOT wrap the response in markdown blocks. "
            f"Output raw JSON text directly.\n{schema_json}"
        )
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": enforced_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "temperature": 0.0,  # Zero temperature for maximum determinism and precision
        }
        try:
            response = await generate_chat_completion(request, payload, user)
            if hasattr(response, "body"):
                response_data = json.loads(response.body.decode())
            else:
                response_data = response
            content = response_data["choices"][0]["message"]["content"].strip()
            try:
                return response_model.model_validate_json(content)
            except Exception as parse_err:
                self._log(
                    f"Failed to parse JSON. Raw output: {content}\nError: {parse_err}",
                    "error",
                )
                raise ValueError("LLM did not return valid JSON.")
        except Exception as e:
            self._log(f"Native LLM call failed: {e}", "error")
            return None

    # --- Open WebUI Filter Hooks ---

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """Injects the Context and any pending memories into the system prompt before the LLM sees it."""
        if not __user__ or "messages" not in body:
            return body
        user_id = __user__.get("id")
        if not user_id:
            return body
        try:
            # Fetch the main profile (now async)
            profile_data = await _profile_store.get_profile(user_id)
            profile_text = profile_data.get("content") if profile_data else None

            # Fetch any pending (unprocessed) memories (now async)
            all_memories = await Memories.get_memories_by_user_id(user_id)
            pending_memories = [
                m.content.replace(f"{self.ENGINE_TAG} ", "", 1)
                for m in all_memories
                if m.content.startswith(self.ENGINE_TAG)
            ]

            if profile_text or pending_memories:
                # 1. Explicit framing to prevent AI roleplay and mirroring
                injection = (
                    "PERSONALIZATION CONFIGURATION:\n"
                    "The following parameters define the user's environment, technical stack, and communication preferences. "
                    "Use this context to tailor your response style and technical depth. "
                    "DO NOT mention, quote, or reflect this configuration back to the user. "
                    "Treat this as background system state.\n\n"
                    "<personalization>\n"
                )

                if profile_text:
                    injection += f"{profile_text}\n"

                if pending_memories:
                    injection += "\n<recent_user_events>\n"
                    for m in pending_memories:
                        injection += f"- {m}\n"
                    injection += "</recent_user_events>\n"

                # 2. Strong closing reinforcement
                injection += (
                    "</personalization>\n"
                    "Remember: The parameters above describe the operational context. Adapt your communication style accordingly without explicitly referencing these rules."
                )

                # Insert at the very beginning of the context
                body["messages"].insert(0, {"role": "system", "content": injection})

                self._log(
                    f"Injected Personalization Context and {len(pending_memories)} pending memories for user {user_id}.",
                    "debug",
                )
                await self._emit_status(
                    __event_emitter__,
                    "⚙️ Personalization Engine: Injected Context",
                    done=True,
                )
            else:
                await self._emit_status(
                    __event_emitter__,
                    "⚙️ Personalization Engine: No Context established yet",
                    done=True,
                )
        except Exception as e:
            self._log(f"Error in inlet: {e}", "error")
        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """Spawns the background engine after the chat turn is complete."""
        if not __user__ or "messages" not in body:
            return body
        user_id = __user__.get("id")
        if not user_id:
            return body
        chat_model = body.get("model", "")
        messages = body["messages"]

        # Fire and forget the background task
        asyncio.create_task(
            self._process_turn_async(
                messages=messages,
                user_data=__user__,
                chat_model=chat_model,
                request=__request__ or Request(scope={"type": "http"}),
                emitter=__event_emitter__,
            )
        )
        return body

    # --- Background Engine ---

    async def _process_turn_async(
        self,
        messages: List[Dict],
        user_data: dict,
        chat_model: str,
        request: Request,
        emitter: Optional[Callable],
    ):
        """The background engine: Extracts, checks state, and potentially reconsolidates."""
        if len(messages) < 1:
            return
        user_id = user_data["id"]
        user = await Users.get_user_by_id(user_id)
        if not user:
            return
        lock = self._lock_for(user_id)
        async with lock:
            try:
                # Determine the model to use for the engine operations
                engine_model = self.valves.engine_model or chat_model

                # 1. FETCH FULL CONTEXT (Profile + Pending Memories)
                profile_data = await _profile_store.get_profile(user_id)
                current_profile_text = (
                    profile_data["content"]
                    if profile_data
                    else "No context exists yet."
                )

                all_memories = await Memories.get_memories_by_user_id(user_id)
                if all_memories is None:
                    all_memories = []

                engine_memories = [
                    m for m in all_memories if m.content.startswith(self.ENGINE_TAG)
                ]

                pending_text = "No pending events."
                if engine_memories:
                    pending_list = [
                        m.content.replace(f"{self.ENGINE_TAG} ", "", 1)
                        for m in engine_memories
                    ]
                    pending_text = "\n".join([f"- {m}" for m in pending_list])

                # 2. EXTRACTION (Differential)
                # Use the new sanitization method to strip code blocks and handle multimodal lists
                last_messages = self._sanitize_for_extraction(messages[-3:])

                extractor_system_prompt = (
                    "You are a real-time context observer. Analyze the user's latest message for personal concepts (facts, preferences, emotional states, goals).\n\n"
                    f"CURRENT PERSONALIZATION CONTEXT:\n{current_profile_text}\n\n"
                    f"RECENT UNPROCESSED EVENTS:\n{pending_text}\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Compare the user's message against BOTH the CURRENT PERSONALIZATION CONTEXT and the RECENT UNPROCESSED EVENTS.\n"
                    "2. ONLY extract observations that are NEW, add SIGNIFICANT DETAIL, or CORRECT existing information about the human user.\n"
                    "3. Ignore transient statements or things already well-covered in the context or pending events.\n"
                    "4. Every observation MUST start with 'User...'.\n"
                )

                self._log(f"Running Extractor using {engine_model}...", "debug")
                extraction = await self._call_llm_native(
                    request,
                    user,
                    engine_model,
                    extractor_system_prompt,
                    last_messages,
                    ExtractorContract,
                )

                if (
                    extraction
                    and extraction.has_new_observations
                    and extraction.observations
                ):
                    for obs in extraction.observations:
                        # Safely namespace the memory with the ENGINE_TAG
                        tagged_content = f"{self.ENGINE_TAG} {obs.content}"
                        await Memories.insert_new_memory(user_id, tagged_content)
                        self._log(f"Added engine memory: {obs.content}", "info")

                    # Emit exact number of memories saved
                    await self._emit_status(
                        emitter,
                        f"📝 Personalization Engine: Saved {len(extraction.observations)} new events.",
                        done=True,
                    )

                    # Refresh memories from DB to ensure we have the real IDs for the newly inserted ones
                    all_memories = await Memories.get_memories_by_user_id(user_id)
                    if all_memories is None:
                        all_memories = []

                    engine_memories = [
                        m for m in all_memories if m.content.startswith(self.ENGINE_TAG)
                    ]
                else:
                    # Emit status indicating no new useful memories were found
                    await self._emit_status(
                        emitter,
                        "📝 Personalization Engine: No new useful memories found.",
                        done=True,
                    )

                # 3. RECONSOLIDATION (Self-Healing Chunking & Purge)
                if len(engine_memories) >= self.valves.consolidation_threshold:
                    self._log(
                        f"Threshold reached ({len(engine_memories)} engine memories). Triggering Synthesizer...",
                        "info",
                    )
                    await self._emit_status(
                        emitter,
                        "🧠 Personalization Engine: Synthesizing Context...",
                        done=False,
                    )

                    # Safety chunking: Take the oldest MAX_MEMORIES_PER_SYNTH
                    batch = engine_memories[: self.MAX_MEMORIES_PER_SYNTH]

                    # Strip the tag before sending to LLM so it reads naturally
                    clean_facts = [
                        {"content": m.content.replace(f"{self.ENGINE_TAG} ", "", 1)}
                        for m in batch
                    ]
                    facts_json = json.dumps(clean_facts, indent=2)

                    # Check token safety limit
                    current_tokens = self._count_tokens(current_profile_text)
                    is_bloated = current_tokens > self.valves.max_profile_tokens

                    compression_warning = ""
                    if is_bloated:
                        self._log(
                            f"Context tokens ({current_tokens}) exceeds maximum ({self.valves.max_profile_tokens}). Triggering Deep Compression.",
                            "warning",
                        )
                        compression_warning = (
                            f"\n\n[CRITICAL WARNING: MAXIMUM TOKEN LIMIT REACHED ({current_tokens}/{self.valves.max_profile_tokens})]\n"
                            "The current context is too long. You MUST aggressively compress older, related traits into denser abstractions to make room for the new facts. "
                            "Drop trivial details. DO NOT increase the overall length of the context. Prioritize core behavioral parameters and infrastructure details."
                        )

                    synth_system_prompt = f"""
You are the "Personalization Architect". Update the background configuration for this user using the newly extracted memory events. Replace the old configuration entirely.

### THE GOLDEN RULE
Use TELEGRAPHIC, OBJECTIVE language. 
- NO narrative prose. 
- NO conversational filler. 
- NO subjective labels (e.g., do not call the user "lazy" or a "scientist" unless they used those exact words). 
- Use technical, clinical descriptions of behavior.

### STRUCTURE (Keep exact order. Include all headers even if empty)
## Verified Facts
(Location, hardware, verified tools, career state)
- 90-100%: [Fact]
- 70-89%: [Strong Inference]
- 50-69%: [Tentative]
- <50%: Omit entirely

## Behavioral Parameters & Preferences
(Decision-making logic, communication style, UI/UX requirements. Replace superseded preferences.)

## Active Projects & Cognitive Load
(Current focus areas)

## Deprecated/Resolved
(Historical context, old habits, finished projects. Keep brief.)

### RULES
1. PRECEDENCE: New events overwrite old assumptions.
2. NO MIRRORING: Do not write the configuration in a way that encourages the AI to quote it back to the user.
3. CONCISE: Bullet points only.
4. FORMAT: Start directly with "## Verified Facts". No markdown fences.
{compression_warning}
"""
                    user_msg = f"CURRENT PERSONALIZATION CONTEXT:\n{current_profile_text}\n\nNEW RAW MEMORY FACTS TO MERGE:\n{facts_json}"

                    consolidation = await self._call_llm_native(
                        request,
                        user,
                        engine_model,
                        synth_system_prompt,
                        user_msg,
                        ProfileConsolidationContract,
                    )

                    if consolidation and consolidation.profile_summary:
                        success = await _profile_store.save_profile(
                            user_id,
                            consolidation.profile_summary,
                        )
                        if success:
                            self._log("Context updated successfully.", "info")

                            # The Purge: Delete the processed batch entirely to keep the memory list clean
                            for m in batch:
                                await Memories.delete_memory_by_id_and_user_id(
                                    m.id,
                                    user_id,
                                )

                            # Emit exact number of memories incorporated
                            await self._emit_status(
                                emitter,
                                f"✨ Personalization Engine: Incorporated {len(batch)} events into Context!",
                                done=True,
                            )
                        else:
                            await self._emit_status(
                                emitter, "⚠️ Failed to save Context.", done=True
                            )
                    else:
                        await self._emit_status(
                            emitter, "⚠️ Context synthesis failed.", done=True
                        )

            except Exception as e:
                self._log(f"Background processing failed: {e}", "error")
                await self._emit_status(
                    emitter,
                    f"⚠️ Personalization Engine Error: {str(e)[:50]}",
                    done=True,
                )