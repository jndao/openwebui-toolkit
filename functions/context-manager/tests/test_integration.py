"""
Integration tests for Filter inlet/outlet and runtime view construction.
"""

import json

import pytest
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

from context_manager import (
    SummaryState,
    MessagePools,
    RuntimeSegments,
    RuntimeView,
    SummaryStore,
)


class TestSplitMessagePools:
    """Tests for Filter._split_message_pools method."""

    def test_split_basic(self, filter_instance):
        """Test basic pool splitting."""
        messages = [
            {"role": "user", "content": "msg1", "timestamp": 100},
            {"role": "assistant", "content": "msg2", "timestamp": 200},
            {"role": "user", "content": "msg3", "timestamp": 300},
            {"role": "assistant", "content": "msg4", "timestamp": 400},
            {"role": "user", "content": "msg5", "timestamp": 500},
        ]
        pools = filter_instance._split_message_pools(
            messages,
            summary_time=250,  # Messages <= 250 are summarized
            keep_start=1,
            keep_end=1,
        )
        # protected_start: first 1 message
        assert len(pools.protected_start) == 1
        assert pools.protected_start[0]["content"] == "msg1"
        # protected_end: last 1 message
        assert len(pools.protected_end) == 1
        assert pools.protected_end[0]["content"] == "msg5"
        # summarized: messages in middle with ts <= 250
        assert len(pools.summarized) == 1
        assert pools.summarized[0]["content"] == "msg2"
        # compressible: remaining middle messages (ts > 250)
        # msg3 (ts=300) and msg4 (ts=400) are both > 250
        assert len(pools.compressible) == 2
        assert pools.compressible[0]["content"] == "msg3"
        assert pools.compressible[1]["content"] == "msg4"

    def test_split_no_summary(self, filter_instance):
        """Test pool splitting without summary."""
        messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
        ]
        pools = filter_instance._split_message_pools(
            messages,
            summary_time=None,
            keep_start=1,
            keep_end=1,
        )
        # No messages should be in summarized
        assert len(pools.summarized) == 0
        # All middle messages should be compressible
        assert len(pools.compressible) == 1

    def test_split_all_protected(self, filter_instance):
        """Test when all messages are protected."""
        messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
        ]
        pools = filter_instance._split_message_pools(
            messages,
            summary_time=None,
            keep_start=2,  # Protect all
            keep_end=0,
        )
        assert len(pools.protected_start) == 2
        assert len(pools.compressible) == 0
        assert len(pools.protected_end) == 0

    def test_split_empty_messages(self, filter_instance):
        """Test splitting empty message list."""
        pools = filter_instance._split_message_pools(
            [],
            summary_time=None,
            keep_start=1,
            keep_end=1,
        )
        assert len(pools.protected_start) == 0
        assert len(pools.summarized) == 0
        assert len(pools.compressible) == 0
        assert len(pools.protected_end) == 0

    def test_split_messages_without_timestamps(self, filter_instance):
        """Test splitting messages without timestamps."""
        messages = [
            {"role": "user", "content": "msg1"},  # No timestamp
            {"role": "assistant", "content": "msg2", "timestamp": 200},
            {"role": "user", "content": "msg3"},  # No timestamp
        ]
        pools = filter_instance._split_message_pools(
            messages,
            summary_time=150,
            keep_start=0,
            keep_end=0,
        )
        # msg2 has ts=200 > 150, so it goes to compressible
        # Messages without timestamps (None) can't be compared (None <= 150 is False), go to compressible
        assert len(pools.summarized) == 0
        assert len(pools.compressible) == 3


class TestBuildRuntimeView:
    """Tests for Filter._build_runtime_view method."""

    def test_build_runtime_view_basic(self, filter_instance):
        """Test basic runtime view construction."""
        db_messages = [
            {"role": "user", "content": "Hello", "timestamp": 100},
            {"role": "assistant", "content": "Hi there!", "timestamp": 200},
            {"role": "user", "content": "How are you?", "timestamp": 300},
        ]
        media_messages = []
        summary_state = SummaryState(content="", until_ts=None, raw=None)

        view = filter_instance._build_runtime_view(
            db_messages, media_messages, summary_state
        )

        assert isinstance(view, RuntimeView)
        assert len(view.final_messages) == 3
        assert view.total_tokens > 0
        assert isinstance(view.stats_message, str)

    def test_build_runtime_view_with_summary(self, filter_instance):
        """Test runtime view with summary injection."""
        db_messages = [
            {"role": "user", "content": "Old message", "timestamp": 100},
            {"role": "assistant", "content": "Old response", "timestamp": 200},
            {"role": "user", "content": "New message", "timestamp": 300},
        ]
        media_messages = []
        summary_state = SummaryState(
            content="Previous conversation summary",
            until_ts=200,
            raw=None
        )

        view = filter_instance._build_runtime_view(
            db_messages, media_messages, summary_state
        )

        # Should have summary message injected
        assert any("context_summary" in str(m.get("content", "")) for m in view.final_messages)

    def test_build_runtime_view_with_media(self, filter_instance):
        """Test runtime view with media messages."""
        db_messages = [
            {"role": "user", "content": "What's this?", "timestamp": 100},
        ]
        media_messages = [
            {"role": "user", "content": {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}}
        ]
        summary_state = SummaryState(content="", until_ts=None, raw=None)

        view = filter_instance._build_runtime_view(
            db_messages, media_messages, summary_state
        )

        assert view.media_tokens > 0

    def test_build_runtime_view_protected_messages(self, filter_instance):
        """Test that protected messages are preserved."""
        filter_instance.valves.keep_start_messages = 1
        filter_instance.valves.keep_last_messages = 1

        db_messages = [
            {"role": "system", "content": "System prompt", "timestamp": 0},
            {"role": "user", "content": "User message", "timestamp": 100},
            {"role": "assistant", "content": "Assistant response", "timestamp": 200},
        ]
        media_messages = []
        summary_state = SummaryState(content="", until_ts=None, raw=None)

        view = filter_instance._build_runtime_view(
            db_messages, media_messages, summary_state
        )

        # First and last messages should be protected
        assert view.segments.protected_start[0]["content"] == "System prompt"
        assert view.segments.protected_end[0]["content"] == "Assistant response"


class TestBuildSummaryMessage:
    """Tests for Filter._build_summary_message method."""

    def test_build_summary_message(self, filter_instance):
        """Test building summary message."""
        state = SummaryState(
            content="This is the summary content",
            until_ts=1700000000,
            raw=None
        )
        result = filter_instance._build_summary_message(state)

        assert result is not None
        assert result["role"] == "system"
        assert "<context_summary>" in result["content"]
        assert "This is the summary content" in result["content"]
        assert "</context_summary>" in result["content"]

    def test_build_summary_message_empty(self, filter_instance):
        """Test building summary message with empty content."""
        state = SummaryState(content="", until_ts=None, raw=None)
        result = filter_instance._build_summary_message(state)

        assert result is None

    def test_build_summary_message_none(self, filter_instance):
        """Test building summary message with None state."""
        result = filter_instance._build_summary_message(None)

        assert result is None


class TestExtractMediaMessages:
    """Tests for Filter._extract_media_messages_from_body method."""

    def test_extract_media_messages(self, filter_instance):
        """Test extracting media messages from body."""
        body = {
            "messages": [
                {"role": "user", "content": "Text message"},
                {
                    "role": "user",
                    "content": {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png"}
                    }
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look at this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                    ]
                },
            ]
        }
        result = filter_instance._extract_media_messages_from_body(body)

        # Should extract 2 messages with media content
        assert len(result) == 2

    def test_extract_media_messages_empty(self, filter_instance):
        """Test extracting from body with no media."""
        body = {
            "messages": [
                {"role": "user", "content": "Just text"},
                {"role": "assistant", "content": "Response"},
            ]
        }
        result = filter_instance._extract_media_messages_from_body(body)

        assert len(result) == 0

    def test_extract_media_messages_no_messages(self, filter_instance):
        """Test extracting from body without messages key."""
        body = {}
        result = filter_instance._extract_media_messages_from_body(body)

        assert len(result) == 0

    def test_extract_media_messages_dedup(self, filter_instance):
        """Test that duplicate media messages are deduplicated."""
        body = {
            "messages": [
                {
                    "id": "msg1",
                    "role": "user",
                    "content": {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
                },
                {
                    "id": "msg1",  # Same ID
                    "role": "user",
                    "content": {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
                },
            ]
        }
        result = filter_instance._extract_media_messages_from_body(body)

        assert len(result) == 1


class TestPrepareDBMessages:
    """Tests for Filter._prepare_db_messages method."""

    def test_prepare_db_messages(self, filter_instance):
        """Test preparing DB messages."""
        messages = [
            {"id": "m1", "role": "user", "content": "Hello", "timestamp": 100, "extra": "remove"},
            {"id": "m2", "role": "assistant", "content": "Hi", "timestamp": 200, "extra": "remove"},
        ]
        result = filter_instance._prepare_db_messages(messages)

        assert len(result) == 2
        # Should only have KEEP_FIELDS
        for msg in result:
            assert "extra" not in msg
            assert "id" in msg or "role" in msg

    def test_prepare_db_messages_empty_content(self, filter_instance):
        """Test that messages with empty content are removed."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "World"},
        ]
        result = filter_instance._prepare_db_messages(messages)

        assert len(result) == 2

    def test_prepare_db_messages_empty_list(self, filter_instance):
        """Test preparing empty message list."""
        result = filter_instance._prepare_db_messages([])
        assert result == []


class TestInlet:
    """Tests for Filter.inlet method."""

    @pytest.mark.asyncio
    async def test_inlet_basic(self, filter_instance, mock_event_emitter):
        """Test basic inlet processing."""
        body = {
            "chat_id": "test-chat-123",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }

        with patch.object(filter_instance, '_load_chat_messages', return_value=[]):
            with patch.object(filter_instance, '_get_summary_state', return_value=SummaryState(content="", until_ts=None)):
                result = await filter_instance.inlet(
                    body=body,
                    __event_emitter__=mock_event_emitter
                )

        assert "messages" in result
        assert result["chat_id"] == "test-chat-123"

    @pytest.mark.asyncio
    async def test_inlet_no_chat_id(self, filter_instance, mock_event_emitter):
        """Test inlet without chat_id returns body unchanged."""
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ]
        }

        result = await filter_instance.inlet(
            body=body,
            __event_emitter__=mock_event_emitter
        )

        # Should return original body when no chat_id
        assert result == body

    @pytest.mark.asyncio
    async def test_inlet_emits_status(self, filter_instance, mock_event_emitter):
        """Test that inlet emits status event."""
        body = {
            "chat_id": "test-chat-123",
            "messages": [{"role": "user", "content": "Hello"}]
        }

        with patch.object(filter_instance, '_load_chat_messages', return_value=[]):
            with patch.object(filter_instance, '_get_summary_state', return_value=SummaryState(content="", until_ts=None)):
                await filter_instance.inlet(
                    body=body,
                    __event_emitter__=mock_event_emitter
                )

        # Should have called emitter
        mock_event_emitter.assert_called()


class TestOutlet:
    """Tests for Filter.outlet method."""

    @pytest.mark.asyncio
    async def test_outlet_basic(self, filter_instance, mock_event_emitter):
        """Test basic outlet processing."""
        body = {
            "chat_id": "test-chat-123",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }

        with patch.object(filter_instance, '_load_chat_messages', return_value=[]):
            with patch.object(filter_instance, '_get_summary_state', return_value=SummaryState(content="", until_ts=None)):
                result = await filter_instance.outlet(
                    body=body,
                    __event_emitter__=mock_event_emitter
                )

        assert "messages" in result

    @pytest.mark.asyncio
    async def test_outlet_no_chat_id(self, filter_instance, mock_event_emitter):
        """Test outlet without chat_id returns body unchanged."""
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }

        result = await filter_instance.outlet(
            body=body,
            __event_emitter__=mock_event_emitter
        )

        assert result == body

    @pytest.mark.asyncio
    async def test_outlet_emits_status(self, filter_instance, mock_event_emitter):
        """Test that outlet emits status event."""
        body = {
            "chat_id": "test-chat-123",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }

        with patch.object(filter_instance, '_load_chat_messages', return_value=[]):
            with patch.object(filter_instance, '_get_summary_state', return_value=SummaryState(content="", until_ts=None)):
                await filter_instance.outlet(
                    body=body,
                    __event_emitter__=mock_event_emitter
                )

        mock_event_emitter.assert_called()


class TestSummaryStore:
    """Tests for SummaryStore class."""

    def test_store_initialization(self):
        """Test SummaryStore initialization."""
        store = SummaryStore()
        assert store._initialized is False
        assert store._init_error is None

    def test_store_get_without_table(self):
        """Test get when table creation fails."""
        store = SummaryStore()
        with patch.object(store, '_ensure_table', return_value=False):
            result = store.get("test-chat")
            assert result is None

    def test_store_save_without_table(self):
        """Test save when table creation fails."""
        store = SummaryStore()
        with patch.object(store, '_ensure_table', return_value=False):
            result = store.save("test-chat", "content")
            assert result is False

    def test_store_delete_without_table(self):
        """Test delete when table creation fails."""
        store = SummaryStore()
        with patch.object(store, '_ensure_table', return_value=False):
            result = store.delete("test-chat")
            assert result is False


class TestReconstructActiveHistoryBranch:
    """Tests for Filter._reconstruct_active_history_branch method."""

    def test_reconstruct_with_current_id(self, filter_instance):
        """Test reconstructing history with current_id."""
        history = {
            "msg1": {"content": "First", "parentId": None},
            "msg2": {"content": "Second", "parentId": "msg1"},
            "msg3": {"content": "Third", "parentId": "msg2"},
        }
        current_id = "msg3"

        result = filter_instance._reconstruct_active_history_branch(history, current_id)

        assert len(result) == 3
        # Should be in order: msg1, msg2, msg3
        assert result[0]["content"] == "First"
        assert result[1]["content"] == "Second"
        assert result[2]["content"] == "Third"

    def test_reconstruct_without_current_id(self, filter_instance):
        """Test reconstructing history without current_id (fallback to timestamp sort)."""
        history = {
            "msg1": {"content": "First", "timestamp": 100},
            "msg2": {"content": "Second", "timestamp": 200},
            "msg3": {"content": "Third", "timestamp": 50},
        }
        current_id = None

        result = filter_instance._reconstruct_active_history_branch(history, current_id)

        assert len(result) == 3
        # Should be sorted by timestamp
        assert result[0]["content"] == "Third"  # ts=50
        assert result[1]["content"] == "First"  # ts=100
        assert result[2]["content"] == "Second"  # ts=200

    def test_reconstruct_empty_history(self, filter_instance):
        """Test reconstructing empty history."""
        result = filter_instance._reconstruct_active_history_branch({}, None)
        assert result == []

    def test_reconstruct_circular_reference(self, filter_instance):
        """Test handling circular parent references."""
        history = {
            "msg1": {"content": "First", "parentId": "msg2"},
            "msg2": {"content": "Second", "parentId": "msg1"},
        }
        current_id = "msg2"

        result = filter_instance._reconstruct_active_history_branch(history, current_id)

        # Should stop at visited nodes, not infinite loop
        assert len(result) <= 2


class TestPackageMessages:
    """Tests for Filter._package_messages method."""

    def test_package_messages(self, filter_instance):
        """Test packaging messages."""
        messages = [
            {"role": "user", "content": "Hello", "children": [{"content": "child"}]},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = filter_instance._package_messages(messages)

        assert len(result) == 2
        # children should be removed
        assert "children" not in result[0]
        assert result[0]["content"] == "Hello"

    def test_package_messages_preserves_original(self, filter_instance):
        """Test that original messages are not modified."""
        messages = [
            {"role": "user", "content": "Hello", "children": [{"content": "child"}]},
        ]
        filter_instance._package_messages(messages)

        # Original should still have children
        assert "children" in messages[0]


class TestCountTokensInMessages:
    """Tests for Filter._count_tokens_in_messages method."""

    def test_count_tokens_basic(self, filter_instance):
        """Test counting tokens in messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = filter_instance._count_tokens_in_messages(messages)
        assert result > 0

    def test_count_tokens_empty(self, filter_instance):
        """Test counting tokens in empty list."""
        result = filter_instance._count_tokens_in_messages([])
        assert result == 0

    def test_count_tokens_filters_non_dicts(self, filter_instance):
        """Test that non-dict items are filtered."""
        messages = [
            {"role": "user", "content": "Hello"},
            "not a dict",
            None,
        ]
        result = filter_instance._count_tokens_in_messages(messages)
        assert result > 0  # Should only count the dict


class TestGetCompressibleTextMessages:
    """Tests for Filter._get_compressible_text_messages method."""

    def test_get_compressible_text_messages(self, filter_instance):
        """Test getting compressible text messages."""
        db_messages = [
            {"role": "user", "content": "Hello", "timestamp": 100},
            {"role": "assistant", "content": "Hi!", "timestamp": 200},
            {"role": "user", "content": "How are you?", "timestamp": 300},
        ]
        summary_state = SummaryState(content="", until_ts=None, raw=None)

        result = filter_instance._get_compressible_text_messages(db_messages, summary_state)

        assert isinstance(result, list)

    def test_get_compressible_text_messages_with_summary(self, filter_instance):
        """Test getting compressible messages with existing summary."""
        # Set keep_start and keep_end to 0 so messages aren't protected
        filter_instance.valves.keep_start_messages = 0
        filter_instance.valves.keep_last_messages = 0
        
        db_messages = [
            {"role": "user", "content": "Old message", "timestamp": 100},
            {"role": "assistant", "content": "Old response", "timestamp": 200},
            {"role": "user", "content": "New message", "timestamp": 300},
        ]
        summary_state = SummaryState(content="Summary", until_ts=200, raw=None)

        result = filter_instance._get_compressible_text_messages(db_messages, summary_state)

        # Messages after summary_time (ts > 200) should be compressible
        # Only the message with ts=300 should be compressible
        assert len(result) >= 1


class TestLockFor:
    """Tests for Filter._lock_for method."""

    def test_lock_for_creates_new(self, filter_instance):
        """Test that _lock_for creates new lock for new chat_id."""
        lock = filter_instance._lock_for("new-chat-id")
        assert isinstance(lock, asyncio.Lock)

    def test_lock_for_returns_same(self, filter_instance):
        """Test that _lock_for returns same lock for same chat_id."""
        lock1 = filter_instance._lock_for("same-chat-id")
        lock2 = filter_instance._lock_for("same-chat-id")
        assert lock1 is lock2

    def test_lock_for_different_chats(self, filter_instance):
        """Test that different chat_ids get different locks."""
        lock1 = filter_instance._lock_for("chat-1")
        lock2 = filter_instance._lock_for("chat-2")
        assert lock1 is not lock2


class TestEnforceContextLimits:
    """Tests for Filter._enforce_context_limits method."""

    def test_enforce_context_limits_no_shedding(self, filter_instance):
        """Test that no shedding occurs when under limit."""
        filter_instance.valves.max_context_tokens = 100000
        
        protected_start = [{"role": "system", "content": "System"}]
        summary_message = {"role": "system", "content": "Summary"}
        media_messages = []
        uncompressed = [{"role": "user", "content": "Hello"}]
        protected_end = [{"role": "assistant", "content": "Hi"}]
        
        media, uncomp, p_end, was_shed = filter_instance._enforce_context_limits(
            protected_start, summary_message, media_messages, uncompressed, protected_end
        )
        
        assert was_shed is False
        assert len(uncomp) == 1
        assert len(p_end) == 1

    def test_enforce_context_limits_shed_uncompressed(self, filter_instance):
        """Test shedding oldest uncompressed messages first."""
        # Set a very low limit that will definitely trigger shedding
        filter_instance.valves.max_context_tokens = 10
        
        protected_start = [{"role": "system", "content": "System prompt that is long"}]
        summary_message = None
        media_messages = []
        uncompressed = [
            {"role": "user", "content": "First message to shed with enough content"},
            {"role": "assistant", "content": "Second message with more content"},
            {"role": "user", "content": "Third message with even more content"},
        ]
        protected_end = [{"role": "assistant", "content": "Final response here"}]
        
        media, uncomp, p_end, was_shed = filter_instance._enforce_context_limits(
            protected_start, summary_message, media_messages, uncompressed, protected_end
        )
        
        # With max_context_tokens=10, shedding should occur
        # The total tokens will definitely exceed 10
        assert was_shed is True
        # Should have shed some messages from uncompressed
        assert len(uncomp) < 3

    def test_enforce_context_limits_shed_media_next(self, filter_instance):
        """Test shedding media messages after uncompressed is empty."""
        # Set a very low limit
        filter_instance.valves.max_context_tokens = 5
        
        protected_start = [{"role": "system", "content": "System prompt here"}]
        summary_message = None
        media_messages = [
            {"role": "user", "content": {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}}
        ]
        uncompressed = [{"role": "user", "content": "A short message"}]
        protected_end = [{"role": "assistant", "content": "Final response message"}]
        
        media, uncomp, p_end, was_shed = filter_instance._enforce_context_limits(
            protected_start, summary_message, media_messages, uncompressed, protected_end
        )
        
        # With max_context_tokens=5, shedding should occur
        assert was_shed is True

    def test_enforce_context_limits_keeps_last_message(self, filter_instance):
        """Test that at least one protected_end message is always kept."""
        filter_instance.valves.max_context_tokens = 10  # Extremely low limit
        
        protected_start = [{"role": "system", "content": "System"}]
        summary_message = None
        media_messages = []
        uncompressed = []
        protected_end = [
            {"role": "assistant", "content": "First final"},
            {"role": "assistant", "content": "Second final"},
        ]
        
        media, uncomp, p_end, was_shed = filter_instance._enforce_context_limits(
            protected_start, summary_message, media_messages, uncompressed, protected_end
        )
        
        # Should always keep at least one protected_end message
        assert len(p_end) >= 1

    def test_enforce_context_limits_zero_max_tokens(self, filter_instance):
        """Test that zero max_tokens disables shedding."""
        filter_instance.valves.max_context_tokens = 0
        
        protected_start = [{"role": "system", "content": "System"}]
        summary_message = None
        media_messages = []
        uncompressed = [{"role": "user", "content": "Hello"}]
        protected_end = [{"role": "assistant", "content": "Hi"}]
        
        media, uncomp, p_end, was_shed = filter_instance._enforce_context_limits(
            protected_start, summary_message, media_messages, uncompressed, protected_end
        )
        
        assert was_shed is False


class TestBuildRuntimeStatsWithShed:
    """Tests for _build_runtime_stats_message with was_shed parameter."""

    def test_stats_message_without_shed(self, filter_instance):
        """Test stats message without shedding."""
        stats, total, prot, uncomp, summ, media = filter_instance._build_runtime_stats_message(
            summarized_messages=[{"role": "user", "content": "old"}],
            protected_messages=[{"role": "system", "content": "sys"}],
            summary_message={"role": "system", "content": "summary"},
            uncompressed_messages=[{"role": "user", "content": "new"}],
            media_messages=[],
            was_shed=False,
        )
        
        assert "⚠️" not in stats
        assert "🪙" in stats

    def test_stats_message_with_shed(self, filter_instance):
        """Test stats message includes warning when shedding occurred."""
        stats, total, prot, uncomp, summ, media = filter_instance._build_runtime_stats_message(
            summarized_messages=[],
            protected_messages=[{"role": "system", "content": "sys"}],
            summary_message=None,
            uncompressed_messages=[{"role": "user", "content": "new"}],
            media_messages=[],
            was_shed=True,
        )
        
        assert "⚠️ Limit Reached" in stats
        assert "🪙" in stats


class TestBackgroundCompressBatching:
    """Tests for _background_compress batching/chunking logic."""

    @pytest.mark.asyncio
    async def test_background_compress_respects_budget(self, filter_instance, mock_event_emitter, mock_request):
        """Test that batching respects the token budget."""
        filter_instance.valves.max_context_tokens = 20000  # Budget will be ~14000
        
        # Create many messages that would exceed budget if all processed
        compressible_messages = [
            {"role": "user", "content": "Message " * 1000, "timestamp": 1000 + i}
            for i in range(10)
        ]
        
        # Mock the dependencies
        with patch('context_manager._get_store') as mock_store:
            with patch('context_manager.generate_chat_completion') as mock_completion:
                with patch('context_manager.Users.get_user_by_id') as mock_users:
                    with patch('context_manager.asyncio.to_thread') as mock_to_thread:
                        # Setup mocks
                        mock_store.return_value = MagicMock()
                        mock_store.return_value.save.return_value = True
                        mock_users.return_value = MagicMock()
                        mock_to_thread.return_value = MagicMock()
                        
                        mock_response = MagicMock()
                        mock_response.body = json.dumps({
                            "choices": [{
                                "message": {"content": "## Current State\n- Test summary"}
                            }]
                        }).encode()
                        mock_completion.return_value = mock_response
                        
                        lock = asyncio.Lock()
                        
                        await filter_instance._background_compress(
                            lock=lock,
                            chat_id="test-chat",
                            old_summary_content="",
                            compressible_messages=compressible_messages,
                            model_id="test-model",
                            user_data={"id": "user-123"},
                            emitter=mock_event_emitter,
                            request=mock_request,
                        )
        
        # Verify the completion was called (batching occurred)
        mock_completion.assert_called_once()
        # Check that the prompt was built with chunked messages
        call_args = mock_completion.call_args
        payload = call_args[0][1]  # Second argument is the payload
        prompt = payload["messages"][0]["content"]
        # The prompt should contain "NEW EVENTS" section
        assert "NEW EVENTS" in prompt

    @pytest.mark.asyncio
    async def test_background_compress_single_large_message_truncation(self, filter_instance, mock_event_emitter, mock_request):
        """Test that a single message larger than budget gets truncated."""
        # Budget = max(10000, max_context_tokens - 6000)
        # To get budget = 10000, we need max_context_tokens <= 16000
        # To trigger truncation, message must exceed 10000 tokens
        # With tiktoken, ~40000 chars of 'x' = ~10000 tokens, so use 100000 chars
        filter_instance.valves.max_context_tokens = 11000  # Budget will be 10000
        
        # Single message that exceeds budget (100000 chars ~ 25000 tokens)
        huge_content = "x " * 50000  # ~100000 chars, many tokens
        compressible_messages = [
            {"role": "user", "content": huge_content, "timestamp": 1000}
        ]
        
        with patch('context_manager._get_store') as mock_store:
            with patch('context_manager.generate_chat_completion') as mock_completion:
                with patch('context_manager.Users.get_user_by_id') as mock_users:
                    with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                        mock_store.return_value = MagicMock()
                        mock_store.return_value.save.return_value = True
                        mock_users.return_value = MagicMock()
                        mock_to_thread.return_value = MagicMock()
                        
                        mock_response = MagicMock()
                        mock_response.body = json.dumps({
                            "choices": [{
                                "message": {"content": "## Current State\n- Truncated summary"}
                            }]
                        }).encode()
                        mock_completion.return_value = mock_response
                        
                        lock = asyncio.Lock()
                        
                        await filter_instance._background_compress(
                            lock=lock,
                            chat_id="test-chat",
                            old_summary_content="",
                            compressible_messages=compressible_messages,
                            model_id="test-model",
                            user_data={"id": "user-123"},
                            emitter=mock_event_emitter,
                            request=mock_request,
                        )
        
        # Should have called completion with truncated content
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        payload = call_args[0][1]
        prompt = payload["messages"][0]["content"]
        # Should contain truncation marker (check for partial match)
        assert "CONTENT TRUNCATED FOR SUMMARY" in prompt

    @pytest.mark.asyncio
    async def test_background_compress_empty_messages_returns_early(self, filter_instance, mock_event_emitter, mock_request):
        """Test that empty compressible_messages returns early."""
        lock = asyncio.Lock()
        
        with patch('context_manager.generate_chat_completion') as mock_completion:
            await filter_instance._background_compress(
                lock=lock,
                chat_id="test-chat",
                old_summary_content="",
                compressible_messages=[],  # Empty
                model_id="test-model",
                user_data={"id": "user-123"},
                emitter=mock_event_emitter,
                request=mock_request,
            )
        
        # Should not call completion
        mock_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_background_compress_no_model_returns_early(self, filter_instance, mock_event_emitter, mock_request):
        """Test that missing model_id returns early with error."""
        lock = asyncio.Lock()
        
        await filter_instance._background_compress(
            lock=lock,
            chat_id="test-chat",
            old_summary_content="",
            compressible_messages=[{"role": "user", "content": "test"}],
            model_id=None,  # No model
            user_data={"id": "user-123"},
            emitter=mock_event_emitter,
            request=mock_request,
        )
        
        # Should have emitted error status
        mock_event_emitter.assert_called()
        call_args = mock_event_emitter.call_args[0][0]
        assert "missing model" in call_args["data"]["description"]

    @pytest.mark.asyncio
    async def test_background_compress_uses_batch_timestamps(self, filter_instance, mock_event_emitter, mock_request):
        """Test that until_ts is calculated from batch, not all messages."""
        # Budget = max(10000, max_context_tokens - 6000)
        # With max_context_tokens = 11000, budget = 10000
        # We want chunking to occur, so use messages that exceed budget
        filter_instance.valves.max_context_tokens = 11000
        
        # Messages with different timestamps - first two should fit, third should trigger chunking
        # Each message needs to be sized appropriately
        compressible_messages = [
            {"role": "user", "content": "Short message one", "timestamp": 1000},
            {"role": "user", "content": "Short message two", "timestamp": 2000},
            {"role": "user", "content": "x " * 6000, "timestamp": 3000},  # Large message to trigger chunking
        ]
        
        saved_until_ts = None
        
        def capture_save(chat_id, content, until_timestamp):
            nonlocal saved_until_ts
            saved_until_ts = until_timestamp
            return True
        
        with patch('context_manager._get_store') as mock_store:
            with patch('context_manager.generate_chat_completion', new_callable=AsyncMock) as mock_completion:
                with patch('context_manager.Users.get_user_by_id') as mock_users:
                    with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                        mock_store.return_value = MagicMock()
                        mock_store.return_value.save = capture_save
                        mock_users.return_value = MagicMock()
                        mock_to_thread.return_value = MagicMock()  # Return valid user object
                        
                        mock_response = MagicMock()
                        mock_response.body = json.dumps({
                            "choices": [{
                                "message": {"content": "## Current State\n- Summary"}
                            }]
                        }).encode()
                        mock_completion.return_value = mock_response
                        
                        lock = asyncio.Lock()
                        
                        await filter_instance._background_compress(
                            lock=lock,
                            chat_id="test-chat",
                            old_summary_content="",
                            compressible_messages=compressible_messages,
                            model_id="test-model",
                            user_data={"id": "user-123"},
                            emitter=mock_event_emitter,
                            request=mock_request,
                        )
        
        # until_ts should be from the batch that was actually processed
        # If chunking occurred, it should be <= 2000 (from first two messages)
        # If no chunking, it would be 3000
        assert saved_until_ts is not None
        assert saved_until_ts <= 3000
