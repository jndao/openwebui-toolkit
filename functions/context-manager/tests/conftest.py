"""
Shared fixtures and mocks for context_manager tests.
"""

import sys
from pathlib import Path
import pytest
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock OpenWebUI modules before importing context_manager
mock_users = MagicMock()
mock_chats = MagicMock()
mock_db = MagicMock()

sys.modules['open_webui'] = MagicMock()
sys.modules['open_webui.models'] = MagicMock()
sys.modules['open_webui.models.users'] = mock_users
sys.modules['open_webui.models.chats'] = mock_chats
sys.modules['open_webui.utils'] = MagicMock()
sys.modules['open_webui.utils.chat'] = MagicMock()
sys.modules['open_webui.internal'] = MagicMock()
sys.modules['open_webui.internal.db'] = mock_db
sys.modules['open_webui.config'] = MagicMock()

# Import directly from the module file
import importlib.util
spec = importlib.util.spec_from_file_location(
    "context_manager",
    Path(__file__).parent.parent / "context_manager.py"
)
context_manager = importlib.util.module_from_spec(spec)
sys.modules["context_manager"] = context_manager
spec.loader.exec_module(context_manager)


@pytest.fixture
def sample_messages() -> List[Dict[str, Any]]:
    """Standard message list for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "Tell me about Python."},
        {"role": "assistant", "content": "Python is a versatile programming language..."},
    ]


@pytest.fixture
def media_messages() -> List[Dict[str, Any]]:
    """Messages with image/file content."""
    return [
        {"role": "user", "content": "What's in this image?"},
        {
            "role": "user",
            "content": {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"}
            }
        },
        {"role": "assistant", "content": "I see a landscape photo."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this file:"},
                {"type": "file", "file": {"url": "data:application/pdf;base64,abc123"}}
            ]
        },
    ]


@pytest.fixture
def tool_call_messages() -> List[Dict[str, Any]]:
    """Messages with tool calls."""
    return [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Sydney"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": '{"temperature": 22, "condition": "sunny"}'
        },
        {"role": "assistant", "content": "It's 22°C and sunny in Sydney."},
    ]


@pytest.fixture
def long_tool_id_messages() -> List[Dict[str, Any]]:
    """Messages with tool call IDs exceeding 64 characters."""
    long_id = "call_" + "x" * 100  # 105 characters total
    return [
        {"role": "user", "content": "Execute something"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": long_id,
                    "type": "function",
                    "function": {"name": "long_operation", "arguments": "{}"}
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": long_id,
            "content": "Result of long operation"
        },
    ]


@pytest.fixture
def messages_with_timestamps() -> List[Dict[str, Any]]:
    """Messages with various timestamp formats."""
    return [
        {"role": "user", "content": "First", "timestamp": 1700000000},
        {"role": "assistant", "content": "Second", "timestamp": 1700000100},
        {"role": "user", "content": "Third", "timestamp": "1700000200"},
        {"role": "assistant", "content": "Fourth", "timestamp": "2023-11-15T00:03:20Z"},
        {"role": "user", "content": "Fifth", "created_at": 1700000300},
    ]


@pytest.fixture
def messages_with_children() -> List[Dict[str, Any]]:
    """Messages with children structure (OpenWebUI format)."""
    return [
        {
            "role": "user",
            "content": "Parent message",
            "children": [{"content": "Child content", "role": "user"}]
        },
        {
            "role": "assistant",
            "content": "Another parent",
            "children": []
        },
    ]


@pytest.fixture
def detail_block_messages() -> List[Dict[str, Any]]:
    """Messages with <details type="tool_calls"> blocks."""
    return [
        {
            "role": "assistant",
            "content": '''Here's the result:
<details type="tool_calls" result="This is a very long tool result that should be trimmed when it exceeds the token threshold. It contains detailed output from a tool execution.">
<summary>Tool Output</summary>
</details>'''
        },
    ]


@pytest.fixture
def mock_db_session():
    """Mock SQLAlchemy session."""
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.commit = MagicMock()
    session.add = MagicMock()
    session.query = MagicMock()
    return session


@pytest.fixture
def mock_chat_record():
    """Mock chat record from Chats.get_chat_by_id."""
    record = MagicMock()
    record.chat = {
        "messages": [
            {"role": "user", "content": "DB message 1"},
            {"role": "assistant", "content": "DB message 2"},
        ]
    }
    return record


@pytest.fixture
def mock_user():
    """Mock user object."""
    user = MagicMock()
    user.id = "test-user-123"
    user.email = "test@example.com"
    return user


@pytest.fixture
def filter_instance():
    """Filter instance with default valves."""
    from context_manager import Filter
    instance = Filter()
    instance.valves.debug_logging = False
    return instance


@pytest.fixture
def summary_state():
    """Sample SummaryState."""
    from context_manager import SummaryState
    return SummaryState(
        content="## Current State\n- User prefers Python (95%)\n- Working on testing (90%)",
        until_ts=1700000100,
        raw={"content": "summary", "until_timestamp": 1700000100}
    )


@pytest.fixture
def mock_event_emitter():
    """Mock async event emitter."""
    return AsyncMock()


@pytest.fixture
def mock_request():
    """Mock FastAPI Request."""
    request = MagicMock()
    request.scope = {"type": "http"}
    return request
