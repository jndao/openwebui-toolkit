"""
title: Persona Engine
author: YourName
description: A two-tier memory system. Extracts real-time observations, safely tags them, and periodically synthesizes a cohesive User Persona natively within Open WebUI.
version: 0.0.1-dev.4
required_open_webui_version: >= 0.5.0
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Type, TypeVar

from fastapi import Request
from pydantic import BaseModel, Field

from open_webui.models.users import UserModel, Users
from open_webui.models.memories import Memories
from open_webui.utils.chat import generate_chat_completion
from open_webui.internal.db import Base, get_db_context
from sqlalchemy import BigInteger, Column, String, Text

logger = logging.getLogger(__name__)

# --- Database Schema for Persona ---

def _discover_owui_schema() -> Optional[str]:
    try:
        from open_webui.config import DATABASE_SCHEMA
        schema = DATABASE_SCHEMA.value if hasattr(DATABASE_SCHEMA, "value") else DATABASE_SCHEMA
        return schema if schema else None
    except Exception:
        return None

_owui_schema = _discover_owui_schema()

class UserPersona(Base):
    __tablename__ = "user_personas"
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

class PersonaStore:
    def __init__(self):
        self._initialized = False

    def _ensure_table(self):
        if self._initialized:
            return True
        try:
            with get_db_context() as db:
                UserPersona.__table__.create(bind=db.bind, checkfirst=True)
                db.commit()
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"[Persona Engine] Failed to create user_personas table: {e}")
            return False

    def get_persona(self, user_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_table()
        try:
            with get_db_context() as db:
                record = db.query(UserPersona).filter_by(user_id=user_id).first()
                if record:
                    return {
                        "id": record.id,
                        "content": record.content,
                        "updated_at": record.updated_at,
                        "created_at": record.created_at
                    }
                return None
        except Exception as e:
            logger.error(f"[Persona Engine] Failed to get persona: {e}")
            return None

    def save_persona(self, user_id: str, content: str) -> bool:
        self._ensure_table()
        try:
            with get_db_context() as db:
                record = db.query(UserPersona).filter_by(user_id=user_id).first()
                now = int(time.time())
                if record:
                    record.content = content
                    record.updated_at = now
                else:
                    record = UserPersona(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        content=content,
                        created_at=now,
                        updated_at=now
                    )
                    db.add(record)
                db.commit()
                return True
        except Exception as e:
            logger.error(f"[Persona Engine] Failed to save persona: {e}")
            return False

_persona_store = PersonaStore()

# --- Pydantic Contracts ---

class MemoryObservation(BaseModel):
    category: Literal["fact", "preference", "emotional_state", "goal", "relationship"] = Field(
        description="The type of personal concept extracted."
    )
    content: str = Field(
        description="A clear, self-contained statement about the user starting with 'User...'."
    )

class ExtractorContract(BaseModel):
    has_new_observations: bool = Field(
        description="True ONLY if the user's latest message contains new, long-term relevant personal concepts."
    )
    observations: List[MemoryObservation] = Field(
        default_factory=list,
        description="List of new observations. Empty if has_new_observations is false."
    )

class PersonaConsolidationContract(BaseModel):
    persona_summary: str = Field(
        description="A cohesive 1-2 paragraph summary of the user, their current state, and preferences incorporating all new facts."
    )

# --- Main Filter Class ---

R = TypeVar("R", bound=BaseModel)

class Filter:
    class Valves(BaseModel):
        extractor_model: str = Field(
            default="", 
            description="Model ID for extraction. Leave blank to default to the current chat model."
        )
        synthesizer_model: str = Field(
            default="", 
            description="Model ID for persona synthesis. Leave blank to default to the current chat model."
        )
        consolidation_threshold: int = Field(
            default=5, 
            description="Number of engine memories to accumulate before triggering a persona reconsolidation."
        )
        emit_status_events: bool = Field(
            default=True,
            description="Toggle whether users should see UI status events during persona synthesis."
        )
        debug_logging: bool = Field(
            default=False, 
            description="Enable detailed console logging."
        )

    def __init__(self):
        self.valves = self.Valves()
        self._locks: Dict[str, asyncio.Lock] = {}
        self.MAX_MEMORIES_PER_SYNTH = 50  # Safety chunking for big histories
        
        # Tag used to namespace memories managed by the engine
        self.ENGINE_TAG = "[PERSONA ENGINE]"

    def _log(self, msg: str, level: str = "info"):
        if level == "debug" and not self.valves.debug_logging:
            return
        getattr(logger, level, logger.info)(f"[Persona Engine] {msg}")

    def _lock_for(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    async def _emit_status(self, emitter: Optional[Callable], message: str, done: bool = True):
        if emitter is None or not self.valves.emit_status_events:
            return
        try:
            await emitter({"type": "status", "data": {"description": message, "done": done}})
        except Exception as e:
            self._log(f"Status Emit Failed: {e}", "debug")

    async def _call_llm_native(
        self, 
        request: Request, 
        user: UserModel, 
        model_id: str, 
        system_prompt: str, 
        user_message: str, 
        response_model: Type[R]
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
            "temperature": 0.1,
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
                self._log(f"Failed to parse JSON. Raw output: {content}\nError: {parse_err}", "error")
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
        __event_emitter__: Optional[Callable] = None
    ) -> dict:
        """Injects the Persona into the system prompt before the LLM sees it."""
        if not __user__ or "messages" not in body:
            return body

        user_id = __user__.get("id")
        if not user_id:
            return body

        try:
            persona_data = await asyncio.to_thread(_persona_store.get_persona, user_id)
            if persona_data and persona_data.get("content"):
                persona_msg = {
                    "role": "system",
                    "content": f"<user_persona>\n{persona_data['content']}\n</user_persona>\nKeep this persona in mind when responding."
                }
                body["messages"].insert(0, persona_msg)
                self._log(f"Injected Persona for user {user_id}.", "debug")
                await self._emit_status(__event_emitter__, "👤 Persona Engine: Injected Persona", done=True)
            else:
                await self._emit_status(__event_emitter__, "👤 Persona Engine: No Persona established yet", done=True)
                
        except Exception as e:
            self._log(f"Error in inlet: {e}", "error")

        return body

    async def outlet(
        self, 
        body: dict, 
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __event_emitter__: Optional[Callable] = None
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
                emitter=__event_emitter__
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
        emitter: Optional[Callable]
    ):
        """The background engine: Extracts, checks state, and potentially reconsolidates."""
        if len(messages) < 1:
            return

        user_id = user_data["id"]
        user = await asyncio.to_thread(Users.get_user_by_id, user_id)
        if not user:
            return

        lock = self._lock_for(user_id)
        async with lock:
            try:
                # 1. EXTRACTION
                extractor_model = self.valves.extractor_model or chat_model
                last_messages = "\n".join([f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages[-3:]])
                
                extractor_system_prompt = (
                    "You are a real-time psychological observer. Analyze the user's latest message for new, "
                    "long-term relevant personal concepts (facts, preferences, emotional states, goals). "
                    "Ignore transient statements. Every observation MUST start with 'User...'. "
                    "Do not worry about contradictions with the past."
                )
                
                self._log(f"Running Extractor using {extractor_model}...", "debug")
                extraction = await self._call_llm_native(
                    request, user, extractor_model, extractor_system_prompt, last_messages, ExtractorContract
                )
                
                if extraction and extraction.has_new_observations and extraction.observations:
                    for obs in extraction.observations:
                        # Safely namespace the memory with the ENGINE_TAG
                        tagged_content = f"{self.ENGINE_TAG} {obs.content}"
                        await asyncio.to_thread(Memories.insert_new_memory, user_id, tagged_content)
                        self._log(f"Added engine memory: {obs.content}", "info")
                    
                    await self._emit_status(
                        emitter, 
                        f"📝 Persona Engine: Saved {len(extraction.observations)} new observations", 
                        done=True
                    )

                # 2. STATE CHECK (Filter for Sandboxed Memories)
                all_memories = await asyncio.to_thread(Memories.get_memories_by_user_id, user_id)
                if not all_memories:
                    return

                # Only look at memories we manage
                engine_memories = [m for m in all_memories if m.content.startswith(self.ENGINE_TAG)]

                # 3. RECONSOLIDATION (Self-Healing Chunking & Purge)
                if len(engine_memories) >= self.valves.consolidation_threshold:
                    self._log(f"Threshold reached ({len(engine_memories)} engine memories). Triggering Synthesizer...", "info")
                    await self._emit_status(emitter, "🧠 Persona Engine: Synthesizing Persona...", done=False)
                    
                    synthesizer_model = self.valves.synthesizer_model or chat_model
                    persona_data = await asyncio.to_thread(_persona_store.get_persona, user_id)
                    current_persona_text = persona_data["content"] if persona_data else "No persona exists yet."
                    
                    # Safety chunking: Take the oldest MAX_MEMORIES_PER_SYNTH
                    batch = engine_memories[:self.MAX_MEMORIES_PER_SYNTH]
                    
                    # Strip the tag before sending to LLM so it reads naturally
                    clean_facts = [{"content": m.content.replace(f"{self.ENGINE_TAG} ", "", 1)} for m in batch]
                    facts_json = json.dumps(clean_facts, indent=2)
                    
                    synth_system_prompt = (
                        "You are a master psychological profiler. You will be given the user's CURRENT PERSONA and a batch of new RAW MEMORY FACTS.\n"
                        "Update the CURRENT PERSONA to seamlessly incorporate ALL the new facts. "
                        "Return ONLY the updated 1-2 paragraph cohesive USER PERSONA."
                    )
                    
                    user_msg = f"CURRENT PERSONA:\n{current_persona_text}\n\nNEW RAW MEMORY FACTS TO MERGE:\n{facts_json}"
                    
                    consolidation = await self._call_llm_native(
                        request, user, synthesizer_model, synth_system_prompt, user_msg, PersonaConsolidationContract
                    )
                    
                    if consolidation and consolidation.persona_summary:
                        success = await asyncio.to_thread(_persona_store.save_persona, user_id, consolidation.persona_summary)
                        if success:
                            self._log("Persona updated successfully.", "info")
                            
                            # The Purge: Delete the processed batch entirely to keep the memory list clean
                            for m in batch:
                                await asyncio.to_thread(Memories.delete_memory_by_id_and_user_id, m.id, user_id)
                            
                            await self._emit_status(emitter, "✨ Persona updated successfully!", done=True)
                        else:
                            await self._emit_status(emitter, "⚠️ Failed to save Persona.", done=True)
                    else:
                        await self._emit_status(emitter, "⚠️ Persona synthesis failed.", done=True)

            except Exception as e:
                self._log(f"Background processing failed: {e}", "error")
                await self._emit_status(emitter, f"⚠️ Persona Engine Error: {str(e)[:50]}", done=True)