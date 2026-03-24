# title: Cross-Conversation Bridge
# id: cross_conversation_bridge
# description: Search and retrieve relevant context from your past conversations to enrich the current chat.
# version: 0.0.3
# author_url: https://github.com/jndao
# repository_url: https://github.com/jndao/openwebui-toolkit
# funding_url: https://ko-fi.com/jndao
# license: https://github.com/jndao/openwebui-toolkit/blob/main/LICENSE

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import json
import time

logger = logging.getLogger(__name__)

try:
    from open_webui.models.chats import Chats
except ModuleNotFoundError:
    Chats = None


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=100, description="Execution order.")
        toggleable: bool = Field(
            default=False,
            description="If True, a toggle button appears. If False, it runs automatically and the button is hidden.",
        )
        max_results: int = Field(
            default=5, ge=1, le=20, description="Max conversations to retrieve."
        )
        max_interactions_per_chat: int = Field(
            default=3, ge=1, le=5, description="User-AI pairs per chat."
        )
        keyword_model: str = Field(default="", description="Model for AI operations.")
        debug_mode: bool = Field(default=False, description="Enable debug logging.")
        lookback_days: int = Field(
            default=30, ge=0, le=365, description="Search window in days."
        )
        max_candidates: int = Field(
            default=50, ge=10, le=200, description="Candidate pool size."
        )
        min_relevance_threshold: float = Field(
            default=0.75, ge=0.0, le=1.0, description="Min relevance score."
        )

    def __init__(self):
        self.type = "filter"
        self.valves = self.Valves()
        self.icon = (
            "https://cdn.jsdelivr.net/npm/lucide-static@0.469.0/icons/git-compare.svg"
        )

    @property
    def toggle(self) -> bool:
        return self.valves.toggleable

    async def _call_ai(self, prompt: str, user_dict: dict) -> Optional[Any]:
        from open_webui.utils.chat import generate_chat_completion
        from open_webui.models.users import Users

        model = self.valves.keyword_model or user_dict.get("__model_name__", "")
        user_id = user_dict.get("id")
        user_obj = Users.get_user_by_id(user_id) if user_id else None

        if not model or not user_obj:
            return None

        try:
            response = await generate_chat_completion(
                user_dict.get("__request__"),
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": 0.1,
                },
                user_obj,
            )

            if hasattr(response, "body"):
                response = json.loads(response.body.decode("utf-8"))

            choices = response.get("choices", [])
            if not choices:
                return None

            content = choices[0].get("message", {}).get("content", "")
            content = content.strip()

            # Manual JSON extraction without Regex to avoid UnterminatedString errors
            if "json" in content:
                content = content.split("json")[1].split("")[0]
            elif "" in content:
                content = content.split("")[1].split("")[0]
            return json.loads(content.strip())
        except Exception:
            return None

    async def _select_relevant_chats(
        self, candidates: List[Dict[str, Any]], user_message: str, user_dict: dict
    ) -> List[tuple]:
        chats_data = []
        for c in candidates:
            chats_data.append(
                {
                    "chat_id": c["chat_id"],
                    "title": c["title"],
                    "preview": c["first_message"][:150],
                }
            )

        prompt = "User Message: " + str(user_message) + "\n"
        prompt += "Past Conversations: " + json.dumps(chats_data) + "\n\n"
        prompt += "Return ONLY a JSON array of objects with chat_id and relevance (0.0 to 1.0)."

        result = await self._call_ai(prompt, user_dict)
        ranked = []
        if isinstance(result, list):
            for i in result:
                if isinstance(i, dict) and "chat_id" in i:
                    cid = str(i["chat_id"])
                    rel = float(i.get("relevance", 0))
                    ranked.append((cid, rel))

        return sorted(
            [r for r in ranked if r[1] >= self.valves.min_relevance_threshold],
            key=lambda x: x[1],
            reverse=True,
        )

    def _get_candidates(
        self, user_id: str, current_chat_id: str
    ) -> List[Dict[str, Any]]:
        if not Chats:
            return []

        user_chats = Chats.get_chats_by_user_id_and_search_text(
            user_id, "", False, 0, self.valves.max_candidates * 2
        )
        cutoff = (
            time.time() - (self.valves.lookback_days * 86400)
            if self.valves.lookback_days > 0
            else 0
        )

        candidates = []
        for c in user_chats:
            cid = str(getattr(c, "id", ""))
            ts = getattr(c, "updated_at", 0)
            if hasattr(ts, "timestamp"):
                ts = ts.timestamp()

            if cid == current_chat_id or ts < cutoff:
                continue

            first_msg = ""
            chat_data = getattr(c, "chat", {})
            history = chat_data.get("history", {})
            msgs = history.get("messages", {})
            if msgs:
                first_msg_id = next(iter(msgs))
                first_msg = msgs.get(first_msg_id, {}).get("content", "")

            candidates.append(
                {
                    "chat": c,
                    "chat_id": cid,
                    "title": getattr(c, "title", "Untitled"),
                    "first_message": str(first_msg),
                    "score": ts,
                }
            )

        return sorted(candidates, key=lambda x: x["score"], reverse=True)[
            : self.valves.max_candidates
        ]

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__=None,
        __metadata__: Optional[dict] = None,
        __request__=None,
    ) -> dict:
        if self.valves.toggleable:
            meta = __metadata__ or {}
            if not meta.get("filter_enabled", False):
                return body

        user_id = __user__.get("id") if __user__ else None
        messages = body.get("messages", [])
        if not user_id or not messages:
            return body

        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        if not user_msg:
            return body

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "🔍 Searching history...", "done": False},
                }
            )

        current_chat_id = (__metadata__ or {}).get("chat_id")
        candidates = self._get_candidates(user_id, current_chat_id)

        if not candidates:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "No history.", "done": True},
                    }
                )
            return body

        user_context = dict(__user__) if __user__ else {}
        user_context["__request__"] = __request__
        user_context["__model_name__"] = body.get("model", "")

        ranked_ids = await self._select_relevant_chats(
            candidates, user_msg, user_context
        )

        if not ranked_ids:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "No relevant context.", "done": True},
                    }
                )
            return body

        context_blocks = ["📚 **Context from past conversations:**"]
        c_map = {c["chat_id"]: c for c in candidates}
        count = 0
        for rid, rel in ranked_ids[: self.valves.max_results]:
            cand = c_map.get(rid)
            if cand:
                title = cand.get("title", "Untitled")
                rel_pct = str(int(rel * 100))
                context_blocks.append("### " + title + " (" + rel_pct + "% relevance)")
                count += 1

        system_msg = {"role": "system", "content": "\n\n".join(context_blocks)}
        messages.insert(-1, system_msg)
        body["messages"] = messages

        if __event_emitter__:
            status_txt = "✅ Injected " + str(count) + " references"
            await __event_emitter__(
                {"type": "status", "data": {"description": status_txt, "done": True}}
            )

        return body
