"""Tests for SearchService restore and retrieval edge cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from claude_kb.models import ErrorResult, GetResult
from claude_kb.search import SearchService


@dataclass
class FakePoint:
    """Minimal Qdrant point shape for tests."""

    id: str
    payload: dict | None


class FakeClient:
    """Fake Qdrant client that supports paginated scroll()."""

    def __init__(self, points: list[FakePoint]):
        self.points = points
        self.scroll_calls: list[dict] = []

    def scroll(
        self,
        collection_name: str,
        scroll_filter,
        limit: int,
        offset,
        with_payload: bool,
    ):
        del collection_name, scroll_filter
        self.scroll_calls.append({"limit": limit, "offset": offset, "with_payload": with_payload})

        start = 0 if offset is None else offset
        end = min(start + limit, len(self.points))
        page = self.points[start:end]
        if not with_payload:
            page = [FakePoint(id=p.id, payload=None) for p in page]

        next_offset = end if end < len(self.points) else None
        return page, next_offset


class FakeDB:
    """Fake DB adapter for SearchService tests."""

    def __init__(self, points: list[FakePoint]):
        self.client = FakeClient(points)

    def get_thread_context(self, collection: str, message_id: str, depth: int):
        del collection, message_id, depth
        return []

    def get_by_id(self, collection: str, point_id: str):
        del collection, point_id
        return None


def make_point(
    point_id: str,
    *,
    role: str,
    timestamp: str,
    content: str | list | dict,
    conversation_id: str = "conv-1",
) -> FakePoint:
    """Build a fake point payload."""
    return FakePoint(
        id=point_id,
        payload={
            "message_id": point_id,
            "role": role,
            "timestamp": timestamp,
            "content": content,
            "project_path": "/tmp/project",
            "conversation_id": conversation_id,
        },
    )


def make_service(points: list[FakePoint]) -> SearchService:
    """Create service with fake DB and fixed collection."""
    service = SearchService(db=FakeDB(points))
    service._collection = "conversations"  # Bypass auto-detection during tests.
    return service


def test_get_conversation_restores_clean_text_and_metadata():
    """Restore mode should keep readable text and strip tool/thinking/system blocks."""
    points = [
        make_point(
            "m3",
            role="user",
            timestamp="2026-01-01T10:00:03+00:00",
            content=[{"type": "text", "text": "late message"}],
        ),
        make_point(
            "m1",
            role="user",
            timestamp="2026-01-01T10:00:00",
            content="<system-reminder>remove this</system-reminder>Hello there",
        ),
        make_point(
            "m2",
            role="assistant",
            timestamp="2026-01-01T10:00:01Z",
            content=json.dumps(
                [
                    {"type": "thinking", "thinking": "hidden"},
                    {"type": "text", "text": "Answer one"},
                    {"type": "tool_use", "name": "lookup"},
                    {"type": "tool_result", "content": "verbose data"},
                    {"type": "text", "text": "Final bit <system-reminder>x</system-reminder>"},
                ]
            ),
        ),
        make_point(
            "meta-1",
            role="system",
            timestamp="2026-01-01T10:00:02+00:00",
            content="meta",
        ),
    ]
    service = make_service(points)

    result = service.get_conversation("conv-1", up_to="m2", max_messages=10)

    assert isinstance(result, GetResult)
    assert result.thread is not None
    assert [msg.id for msg in result.thread] == ["m1", "m2"]
    assert [msg.role for msg in result.thread] == ["user", "assistant"]

    all_content = "\n".join(msg.content for msg in result.thread)
    assert "tool_use" not in all_content
    assert "tool_result" not in all_content
    assert "thinking" not in all_content
    assert "<system-reminder>" not in all_content
    assert "[user] Hello there" in result.thread[0].content
    assert "Answer one" in result.thread[1].content
    assert "Final bit" in result.thread[1].content
    assert result.thread[0].content.endswith("\n\n---")

    assert result.message.conversation_id == "conv-1"
    assert "message_count=2" in result.message.content
    assert "up_to=m2" in result.message.content


def test_get_conversation_applies_max_messages_after_up_to():
    """Restore mode should truncate to most recent N messages after up_to selection."""
    points = [
        make_point(
            f"m{i}",
            role="user" if i % 2 else "assistant",
            timestamp=f"2026-01-01T10:00:0{i}" + ("Z" if i % 2 else ""),
            content=[{"type": "text", "text": f"message {i}"}],
        )
        for i in range(1, 6)
    ]
    service = make_service(points)

    result = service.get_conversation("conv-1", up_to="m5", max_messages=2)

    assert isinstance(result, GetResult)
    assert result.thread is not None
    assert [msg.id for msg in result.thread] == ["m4", "m5"]
    assert "truncated_to_most_recent=2" in result.message.content


def test_get_validates_ambiguous_and_unsupported_restore_flags():
    """Service.get should reject mixed modes and restore-only incompatible flags."""
    service = make_service([])

    both_ids = service.get(message_id="m1", conversation_id="conv-1")
    assert isinstance(both_ids, ErrorResult)
    assert "mutually exclusive" in both_ids.error

    restore_with_include = service.get(conversation_id="conv-1", include_tool_results=True)
    assert isinstance(restore_with_include, ErrorResult)
    assert "not supported in conversation restore mode" in restore_with_include.error


def test_strip_system_reminders_and_parse_timestamp_fallback():
    """System reminder cleanup and timestamp parsing should be deterministic."""
    service = make_service([])

    cleaned = service._strip_system_reminders("a<system-reminder>drop</system-reminder>b\n\n\n\nc")
    assert cleaned == "ab\n\nc"

    cleaned_open = service._strip_system_reminders("prefix<system-reminder>drop rest")
    assert cleaned_open == "prefix"

    fallback = datetime(2000, 1, 1, tzinfo=UTC)
    assert service._parse_timestamp("not-a-time", fallback=fallback) == fallback
    assert service._parse_timestamp("2026-01-01T10:00:00").tzinfo is not None


def test_scroll_helpers_paginate_and_share_logic():
    """Count and restore scrolling should reuse the same pagination helper."""
    points = [
        make_point(
            f"m{i}",
            role="user",
            timestamp=f"2026-01-01T10:00:{i % 60:02d}Z",
            content="x",
        )
        for i in range(205)
    ]
    service = make_service(points)
    client = service.db.client

    count = service._get_conversation_message_count("conv-1")
    assert count == 205

    with_payload_values = [call["with_payload"] for call in client.scroll_calls]
    assert with_payload_values.count(False) >= 3  # 205 items, 100/page => 3 calls

    loaded = service._scroll_conversation_points("conv-1", with_payload=True)
    assert len(loaded) == 205
    assert loaded[0].payload is not None
