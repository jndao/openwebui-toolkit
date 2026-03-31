"""
Unit tests for TokenCounter, ContextReconstructor, and Filter utility methods.
"""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from context_manager import (
    TokenCounter,
    ContextReconstructor,
    SummaryState,
    MessagePools,
    RuntimeSegments,
    RuntimeView,
)


class TestTokenCounter:
    """Tests for TokenCounter class."""

    def test_count_string(self):
        """Test counting tokens in a plain string."""
        result = TokenCounter.count("Hello, world!")
        assert isinstance(result, int)
        assert result > 0

    def test_count_empty_string(self):
        """Test counting tokens in empty string."""
        result = TokenCounter.count("")
        assert result == 0

    def test_count_dict_message(self):
        """Test counting tokens in a message dict."""
        msg = {"role": "user", "content": "Hello, world!"}
        result = TokenCounter.count(msg)
        assert isinstance(result, int)
        assert result > 0

    def test_count_list_of_messages(self):
        """Test counting tokens in a list of messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = TokenCounter.count(messages)
        assert isinstance(result, int)
        assert result > 0

    def test_count_empty_list(self):
        """Test counting tokens in empty list."""
        result = TokenCounter.count([])
        assert result == 0

    def test_count_unknown_type(self):
        """Test counting tokens for unknown type returns 0."""
        result = TokenCounter.count(None)
        assert result == 0

    def test_count_text_with_tiktoken(self):
        """Test that tiktoken is used when available."""
        # "Hello" should be 1 token with cl100k_base
        result = TokenCounter._count_text("Hello")
        assert result >= 1

    def test_count_message_with_dict_content(self):
        """Test counting message with dict content."""
        msg = {
            "role": "user",
            "content": {"type": "text", "text": "Hello world"}
        }
        result = TokenCounter._count_message(msg)
        assert result > 0

    def test_count_message_with_list_content(self):
        """Test counting message with list content (multimodal)."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
            ]
        }
        result = TokenCounter._count_message(msg)
        assert result > 0

    def test_count_message_with_tool_calls(self):
        """Test counting message with tool calls."""
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Sydney"}'
                    }
                }
            ]
        }
        result = TokenCounter._count_message(msg)
        assert result > 0

    def test_count_message_with_tool_call_id(self):
        """Test counting tool response message."""
        msg = {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "get_weather",
            "content": '{"temp": 22}'
        }
        result = TokenCounter._count_message(msg)
        assert result > 0

    def test_extract_text_from_string(self):
        """Test extracting text from plain string."""
        result = TokenCounter.extract_text("Hello world")
        assert result == "Hello world"

    def test_extract_text_from_dict(self):
        """Test extracting text from dict content."""
        content = {"type": "text", "text": "Hello world"}
        result = TokenCounter.extract_text(content)
        assert result == "Hello world"

    def test_extract_text_from_input_text(self):
        """Test extracting text from input_text type."""
        content = {"type": "input_text", "text": "User input"}
        result = TokenCounter.extract_text(content)
        assert result == "User input"

    def test_extract_text_from_nested_content(self):
        """Test extracting text from nested content field."""
        content = {"type": "text", "content": "Nested text"}
        result = TokenCounter.extract_text(content)
        assert result == "Nested text"

    def test_extract_text_from_list(self):
        """Test extracting text from list content."""
        content = [
            {"type": "text", "text": "Part one"},
            {"type": "text", "text": "Part two"},
        ]
        result = TokenCounter.extract_text(content)
        assert "Part one" in result
        assert "Part two" in result

    def test_extract_text_ignores_non_text_types(self):
        """Test that non-text types are ignored."""
        content = [
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            {"type": "text", "text": "Only this"},
        ]
        result = TokenCounter.extract_text(content)
        assert result == "Only this"


class TestContextReconstructor:
    """Tests for ContextReconstructor class."""

    def test_collapsed_tool_text(self):
        """Test collapsed tool text constant."""
        result = ContextReconstructor.collapsed_tool_text()
        assert result == "[TOOL OUTPUT COLLAPSED]"

    def test_shorten_tool_call_id_short(self):
        """Test that short IDs are not modified."""
        recon = ContextReconstructor()
        short_id = "call_abc123"
        result = recon._shorten_tool_call_id(short_id, max_len=64)
        assert result == short_id

    def test_shorten_tool_call_id_long(self):
        """Test that long IDs are shortened."""
        recon = ContextReconstructor()
        long_id = "call_" + "x" * 100  # 105 chars
        result = recon._shorten_tool_call_id(long_id, max_len=64)
        assert len(result) <= 64
        assert result.startswith("call_")
        assert "..." in result

    def test_shorten_tool_call_id_preserves_prefix(self):
        """Test that prefix is preserved in shortened ID."""
        recon = ContextReconstructor()
        long_id = "call_function_test_" + "x" * 80
        result = recon._shorten_tool_call_id(long_id, max_len=64)
        assert result.startswith("call_function_test_")

    def test_shorten_tool_call_id_non_string(self):
        """Test handling of non-string input."""
        recon = ContextReconstructor()
        result = recon._shorten_tool_call_id(None)
        assert result is None

    def test_normalize_tool_call_ids_no_change(self):
        """Test that short IDs are not normalized."""
        recon = ContextReconstructor()
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_abc", "function": {"name": "test"}}]
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "result"}
        ]
        result = recon.normalize_tool_call_ids(messages)
        assert result == 0

    def test_normalize_tool_call_ids_long_ids(self):
        """Test that long IDs are normalized."""
        recon = ContextReconstructor()
        long_id = "call_" + "x" * 100
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": long_id, "function": {"name": "test"}}]
            },
            {"role": "tool", "tool_call_id": long_id, "content": "result"}
        ]
        result = recon.normalize_tool_call_ids(messages)
        assert result == 1
        # Check that both IDs were updated
        assert len(messages[0]["tool_calls"][0]["id"]) <= 64
        assert messages[1]["tool_call_id"] == messages[0]["tool_calls"][0]["id"]

    def test_trim_tool_content_tool_messages(self):
        """Test trimming of tool message content."""
        recon = ContextReconstructor()
        long_content = "x" * 5000
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": long_content}
        ]
        trimmed, stats = recon.trim_tool_content(messages, threshold=100)
        assert stats["tool_messages_trimmed"] == 1
        assert trimmed[0]["content"] == "[TOOL OUTPUT COLLAPSED]"

    def test_trim_tool_content_tool_arguments(self):
        """Test trimming of tool call arguments."""
        recon = ContextReconstructor()
        long_args = '{"data": "' + "x" * 5000 + '"}'
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "test", "arguments": long_args}
                }]
            }
        ]
        trimmed, stats = recon.trim_tool_content(messages, threshold=100)
        assert stats["tool_arguments_trimmed"] == 1

    def test_trim_tool_content_detail_blocks(self, detail_block_messages):
        """Test trimming of <details type="tool_calls"> blocks."""
        recon = ContextReconstructor()
        trimmed, stats = recon.trim_tool_content(detail_block_messages, threshold=10)
        assert stats["detail_blocks_trimmed"] >= 1

    def test_trim_tool_content_target_indices(self):
        """Test that target_indices limits which messages are trimmed."""
        recon = ContextReconstructor()
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "x" * 5000},
            {"role": "tool", "tool_call_id": "call_2", "content": "y" * 5000},
        ]
        # Only trim first message
        trimmed, stats = recon.trim_tool_content(
            messages, threshold=100, target_indices={0}
        )
        assert stats["trimmed_count"] == 1
        assert trimmed[0]["content"] == "[TOOL OUTPUT COLLAPSED]"
        assert trimmed[1]["content"] == "y" * 5000

    def test_trim_tool_content_preserves_original(self):
        """Test that original messages are not modified (deepcopy)."""
        recon = ContextReconstructor()
        original_content = "x" * 5000
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": original_content}
        ]
        recon.trim_tool_content(messages, threshold=100)
        assert messages[0]["content"] == original_content


class TestSummaryState:
    """Tests for SummaryState dataclass."""

    def test_create_empty(self):
        """Test creating empty SummaryState."""
        state = SummaryState(content="", until_ts=None, raw=None)
        assert state.content == ""
        assert state.until_ts is None
        assert state.raw is None

    def test_create_with_data(self):
        """Test creating SummaryState with data."""
        raw = {"content": "test", "until_timestamp": 123}
        state = SummaryState(content="test", until_ts=123, raw=raw)
        assert state.content == "test"
        assert state.until_ts == 123
        assert state.raw == raw


class TestMessagePools:
    """Tests for MessagePools dataclass."""

    def test_create_pools(self):
        """Test creating MessagePools."""
        pools = MessagePools(
            protected_start=[{"role": "system", "content": "sys"}],
            summarized=[{"role": "user", "content": "old"}],
            compressible=[{"role": "user", "content": "new"}],
            protected_end=[{"role": "user", "content": "recent"}],
        )
        assert len(pools.protected_start) == 1
        assert len(pools.summarized) == 1
        assert len(pools.compressible) == 1
        assert len(pools.protected_end) == 1


class TestRuntimeSegments:
    """Tests for RuntimeSegments dataclass."""

    def test_final_messages_order(self):
        """Test that final_messages merges in correct order."""
        segments = RuntimeSegments(
            protected_start=[{"role": "system", "content": "start"}],
            summary_message={"role": "system", "content": "summary"},
            media_messages=[{"role": "user", "content": "media"}],
            uncompressed=[{"role": "user", "content": "uncompressed"}],
            protected_end=[{"role": "user", "content": "end"}],
        )
        result = segments.final_messages
        assert len(result) == 5
        assert result[0]["content"] == "start"
        assert result[1]["content"] == "summary"
        assert result[2]["content"] == "uncompressed"
        assert result[3]["content"] == "media"
        assert result[4]["content"] == "end"

    def test_final_messages_no_summary(self):
        """Test final_messages without summary."""
        segments = RuntimeSegments(
            protected_start=[{"role": "system", "content": "start"}],
            summary_message=None,
            media_messages=[],
            uncompressed=[{"role": "user", "content": "msg"}],
            protected_end=[],
        )
        result = segments.final_messages
        assert len(result) == 2
        assert result[0]["content"] == "start"
        assert result[1]["content"] == "msg"


class TestRuntimeView:
    """Tests for RuntimeView dataclass."""

    def test_create_runtime_view(self):
        """Test creating RuntimeView."""
        segments = RuntimeSegments(
            protected_start=[],
            summary_message=None,
            media_messages=[],
            uncompressed=[],
            protected_end=[],
        )
        view = RuntimeView(
            final_messages=[],
            stats_message="🪙 1.0k",
            segments=segments,
            total_tokens=1000,
            protected_tokens=200,
            uncompressed_tokens=500,
            summary_tokens=100,
            media_tokens=200,
        )
        assert view.total_tokens == 1000
        assert view.stats_message == "🪙 1.0k"


class TestFilterUtilityMethods:
    """Tests for Filter utility methods."""

    def test_normalize_epoch_timestamp_int(self, filter_instance):
        """Test normalizing integer timestamp."""
        result = filter_instance._normalize_epoch_timestamp(1700000000)
        assert result == 1700000000

    def test_normalize_epoch_timestamp_string_int(self, filter_instance):
        """Test normalizing string integer timestamp."""
        result = filter_instance._normalize_epoch_timestamp("1700000000")
        assert result == 1700000000

    def test_normalize_epoch_timestamp_iso_string(self, filter_instance):
        """Test normalizing ISO format timestamp."""
        result = filter_instance._normalize_epoch_timestamp("2023-11-15T00:00:00Z")
        assert result == 1700006400

    def test_normalize_epoch_timestamp_datetime(self, filter_instance):
        """Test normalizing datetime object."""
        dt = datetime(2023, 11, 15, 0, 0, 0, tzinfo=timezone.utc)
        result = filter_instance._normalize_epoch_timestamp(dt)
        assert result == 1700006400

    def test_normalize_epoch_timestamp_milliseconds(self, filter_instance):
        """Test normalizing millisecond timestamp."""
        result = filter_instance._normalize_epoch_timestamp(1700000000000)
        assert result == 1700000000

    def test_normalize_epoch_timestamp_none(self, filter_instance):
        """Test normalizing None returns None."""
        result = filter_instance._normalize_epoch_timestamp(None)
        assert result is None

    def test_normalize_epoch_timestamp_empty_string(self, filter_instance):
        """Test normalizing empty string returns None."""
        result = filter_instance._normalize_epoch_timestamp("")
        assert result is None

    def test_normalize_epoch_timestamp_zero(self, filter_instance):
        """Test normalizing zero returns None."""
        result = filter_instance._normalize_epoch_timestamp(0)
        assert result is None

    def test_timestamp_of_message(self, filter_instance):
        """Test extracting timestamp from message."""
        msg = {"role": "user", "content": "test", "timestamp": 1700000000}
        result = filter_instance._timestamp_of(msg)
        assert result == 1700000000

    def test_timestamp_of_message_created_at(self, filter_instance):
        """Test extracting created_at from message."""
        msg = {"role": "user", "content": "test", "created_at": 1700000000}
        result = filter_instance._timestamp_of(msg)
        assert result == 1700000000

    def test_message_identity_with_id(self, filter_instance):
        """Test message identity with explicit ID."""
        msg = {"id": "msg123", "role": "user", "content": "test"}
        result = filter_instance._message_identity(msg)
        assert result == "id:msg123"

    def test_message_identity_with_message_id(self, filter_instance):
        """Test message identity with message_id field."""
        msg = {"message_id": "msg456", "role": "user", "content": "test"}
        result = filter_instance._message_identity(msg)
        assert result == "id:msg456"

    def test_message_identity_fallback(self, filter_instance):
        """Test message identity fallback without ID."""
        msg = {"role": "user", "content": "test", "timestamp": 1700000000}
        result = filter_instance._message_identity(msg)
        assert result.startswith("fallback:user:1700000000:")

    def test_message_has_passthrough_media_image_url(self, filter_instance):
        """Test media detection for image_url type."""
        msg = {"role": "user", "content": {"type": "image_url", "image_url": {}}}
        result = filter_instance._message_has_passthrough_media(msg)
        assert result is True

    def test_message_has_passthrough_media_file(self, filter_instance):
        """Test media detection for file type."""
        msg = {"role": "user", "content": {"type": "file", "file": {}}}
        result = filter_instance._message_has_passthrough_media(msg)
        assert result is True

    def test_message_has_passthrough_media_input_image(self, filter_instance):
        """Test media detection for input_image type."""
        msg = {"role": "user", "content": {"type": "input_image", "url": "data:image/png;base64,abc"}}
        result = filter_instance._message_has_passthrough_media(msg)
        assert result is True

    def test_message_has_passthrough_media_text(self, filter_instance):
        """Test media detection for text type."""
        msg = {"role": "user", "content": "Just text"}
        result = filter_instance._message_has_passthrough_media(msg)
        assert result is False

    def test_message_has_passthrough_media_list(self, filter_instance):
        """Test media detection for list content."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
            ]
        }
        result = filter_instance._message_has_passthrough_media(msg)
        assert result is True

    def test_unfold_messages_with_children(self, filter_instance, messages_with_children):
        """Test unfolding messages with children."""
        result = filter_instance._unfold_messages(messages_with_children)
        assert len(result) == 2
        # First message should have merged child content
        assert result[0]["content"] == "Child content"

    def test_unfold_messages_empty(self, filter_instance):
        """Test unfolding empty message list."""
        result = filter_instance._unfold_messages([])
        assert result == []

    def test_scrub_message(self, filter_instance):
        """Test scrubbing message to keep only essential fields."""
        msg = {
            "id": "msg1",
            "parentId": "msg0",
            "role": "user",
            "content": "test",
            "timestamp": 1700000000,
            "extra_field": "should be removed",
            "another_extra": 123
        }
        result = filter_instance._scrub_message(msg)
        assert "id" in result
        assert "parentId" in result
        assert "role" in result
        assert "content" in result
        assert "timestamp" in result
        assert "extra_field" not in result
        assert "another_extra" not in result

    def test_build_text_only_message(self, filter_instance):
        """Test building text-only message."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {}}
            ],
            "timestamp": 1700000000
        }
        result = filter_instance._build_text_only_message(msg)
        assert result is not None
        assert result["content"] == "Hello"
        assert result["role"] == "user"
        assert result["timestamp"] == 1700000000

    def test_build_text_only_message_empty(self, filter_instance):
        """Test building text-only message with no text content."""
        msg = {
            "role": "user",
            "content": {"type": "image_url", "image_url": {}}
        }
        result = filter_instance._build_text_only_message(msg)
        assert result is None

    def test_format_token_count_small(self, filter_instance):
        """Test formatting small token count."""
        result = filter_instance._format_token_count(500)
        assert result == "500"

    def test_format_token_count_large(self, filter_instance):
        """Test formatting large token count."""
        result = filter_instance._format_token_count(5000)
        assert result == "5.0k"

    def test_get_chat_id_from_metadata(self, filter_instance):
        """Test extracting chat_id from metadata."""
        body = {}
        metadata = {"chat_id": "chat123"}
        result = filter_instance._get_chat_id(body, metadata)
        assert result == "chat123"

    def test_get_chat_id_from_body(self, filter_instance):
        """Test extracting chat_id from body."""
        body = {"chat_id": "chat456"}
        metadata = None
        result = filter_instance._get_chat_id(body, metadata)
        assert result == "chat456"

    def test_get_chat_id_from_meta_nested(self, filter_instance):
        """Test extracting chat_id from nested meta."""
        body = {"meta": {"chat_id": "chat789"}}
        metadata = None
        result = filter_instance._get_chat_id(body, metadata)
        assert result == "chat789"

    def test_get_chat_id_none(self, filter_instance):
        """Test when no chat_id is found."""
        body = {}
        metadata = None
        result = filter_instance._get_chat_id(body, metadata)
        assert result is None
