"""SSE delivery regressions: overflow, multiple readers, and reconnects."""
import json
import queue
import threading
import unittest
from unittest.mock import patch

from app import _EventChannel, _normalize_event, _queue_event, create_app


class EventChannelTests(unittest.TestCase):
    def test_long_live_stream_does_not_stop_after_replay_fills(self):
        channel = _EventChannel(maxsize=10)
        first, second = channel.subscribe(), channel.subscribe()
        for index in range(10000):
            event = _normalize_event("token", str(index))
            self.assertTrue(_queue_event(channel, event))
            self.assertEqual(first.get(timeout=0.1), event)
            self.assertEqual(second.get(timeout=0.1), event)
            self.assertLessEqual(len(channel._messages), 10)

    def test_disconnected_reader_cannot_block_new_progress_or_done(self):
        channel = _EventChannel(maxsize=5)
        slow = channel.subscribe()
        for index in range(1000):
            self.assertTrue(_queue_event(channel, _normalize_event("token", str(index))))
        for event_type in ("scene_written", "agent_active", "log", "status", "done"):
            self.assertTrue(_queue_event(channel, _normalize_event(event_type, {})))
        self.assertEqual([slow.get(timeout=0.1)["type"] for _ in range(5)],
                         ["scene_written", "agent_active", "log", "status", "done"])
        self.assertEqual(len(channel._messages), 5)

    def test_reconnect_starts_after_last_received_event(self):
        channel = _EventChannel(maxsize=5)
        first = channel.subscribe()
        channel.put_nowait(_normalize_event("token", "Hello "))
        self.assertEqual(first.get(timeout=0.1)["payload"], "Hello ")
        last_id = f"{channel._stream_id}:{first.cursor}"
        channel.put_nowait(_normalize_event("token", "world"))
        reconnect = channel.subscribe(last_id)
        self.assertEqual(reconnect.get(timeout=0.1)["payload"], "world")
        with self.assertRaises(queue.Empty):
            reconnect.get(timeout=0)

    def test_old_run_and_malformed_ids_do_not_skip_new_run(self):
        channel = _EventChannel(maxsize=5)
        channel.put_nowait(_normalize_event("status", "ready"))
        for last_id in ("other-run:999", "garbage", f"{channel._stream_id}:bad"):
            self.assertEqual(channel.subscribe(last_id).get(timeout=0.1)["payload"], "ready")

    def test_manual_channel_can_clear_and_reuse_existing_subscription(self):
        channel = _EventChannel(maxsize=2)
        first = channel.subscribe()
        channel.put_nowait(_normalize_event("status", "old"))
        channel.clear()
        channel.put_nowait(_normalize_event("status", "new"))
        self.assertEqual(first.get(timeout=0.1)["payload"], "new")

    def test_waiting_consumer_is_notified(self):
        channel = _EventChannel(maxsize=2)
        first = channel.subscribe()
        received = []
        worker = threading.Thread(target=lambda: received.append(first.get(timeout=1)))
        worker.start()
        channel.put_nowait(_normalize_event("status", "ready"))
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(received[0]["payload"], "ready")


class SSEEndpointTests(unittest.TestCase):
    def test_generation_and_combine_resume_with_sse_ids(self):
        app = create_app()
        for queue_name, route, terminal in (
            ("_event_queues", "generate", "done"),
            ("_combine_queues", "combine", "combine_done"),
        ):
            with self.subTest(route=route):
                channel = _EventChannel(maxsize=5)
                channel.put_nowait(_normalize_event("status", "already received"))
                channel.put_nowait(_normalize_event("status", "new update"))
                channel.put_nowait(_normalize_event(terminal, {}))
                with patch(f"app.{queue_name}", {"sse_test": channel}):
                    response = app.test_client().get(
                        f"/api/project/sse_test/{route}/stream",
                        headers={"Last-Event-ID": f"{channel._stream_id}:1"},
                    )
                    body = response.get_data(as_text=True)
                    response.close()
                self.assertNotIn("already received", body)
                self.assertIn(f"id: {channel._stream_id}:2", body)
                self.assertIn(f"id: {channel._stream_id}:3", body)
                messages = [json.loads(line[6:]) for line in body.splitlines()
                            if line.startswith("data: ")]
                self.assertEqual([m["type"] for m in messages], ["status", terminal])

    def test_progress_and_completion_survive_full_replay(self):
        channel = _EventChannel(maxsize=3)
        for index in range(1000):
            _queue_event(channel, _normalize_event("token", str(index)))
        for event_type in ("scene_complete", "chapter_complete", "done"):
            _queue_event(channel, _normalize_event(event_type, {}))
        with patch("app._event_queues", {"sse_test": channel}):
            response = create_app().test_client().get("/api/project/sse_test/generate/stream")
            body = response.get_data(as_text=True)
            response.close()
        self.assertIn('"type": "scene_complete"', body)
        self.assertIn('"type": "chapter_complete"', body)
        self.assertIn('"type": "done"', body)


if __name__ == "__main__":
    unittest.main()
