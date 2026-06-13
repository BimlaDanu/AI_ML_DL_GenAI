"""Unit tests for the notification bundler."""
import sys, io, unittest
from datetime import datetime
sys.path.insert(0, ".")
from notifications import process_events, format_message

def make_event(ts_str, user, friend_name, friend_id="X"):
    return {"timestamp": datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S"),
            "user_id": user, "friend_id": friend_id, "friend_name": friend_name}

class TestFormatMessage(unittest.TestCase):
    def test_single_tour(self):
        self.assertEqual(format_message([("t","Alice")]), "Alice went on a tour")

    def test_same_friend_multi(self):
        tours = [("t","Alice"),("t","Alice"),("t","Alice")]
        self.assertEqual(format_message(tours), "Alice went on 3 tours")

    def test_two_friends(self):
        tours = [("t","Alice"),("t","Bob")]
        self.assertEqual(format_message(tours), "Alice and 1 other went on a tour")

    def test_many_friends(self):
        tours = [("t","Alice"),("t","Bob"),("t","Carol")]
        self.assertEqual(format_message(tours), "Alice and 2 others went on a tour")


class TestBundling(unittest.TestCase):
    def test_immediate_single(self):
        events = [make_event("2018-01-01 10:00:00", "U1", "Alice")]
        out = process_events(events)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["tours"], "1")

    def test_rapid_events_bundled(self):
        events = [
            make_event("2018-01-01 10:00:00", "U1", "Alice"),
            make_event("2018-01-01 10:01:00", "U1", "Bob"),
            make_event("2018-01-01 10:02:00", "U1", "Carol"),
        ]
        out = process_events(events)
        # All 3 within a 5-min window → 1 bundle
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["tours"], "3")

    def test_daily_cap_4(self):
        # 8 events spread across day → should not exceed 4 notifications (soft cap)
        events = []
        for i in range(8):
            ts = f"2018-01-01 {10+i}:00:00"
            events.append(make_event(ts, "U2", f"Friend{i}"))
        out = process_events(events)
        notifs_for_u2 = [r for r in out if r["receiver_id"] == "U2"]
        # With cap logic, daily sends should be ≤ soft cap (some may spill)
        self.assertLessEqual(len(notifs_for_u2), 4 + 1)  # allow 1 overflow

    def test_different_users_independent(self):
        events = [
            make_event("2018-01-01 10:00:00", "U1", "Alice"),
            make_event("2018-01-01 10:00:00", "U2", "Alice"),
        ]
        out = process_events(events)
        users = {r["receiver_id"] for r in out}
        self.assertIn("U1", users)
        self.assertIn("U2", users)

    def test_widely_spaced_not_bundled(self):
        events = [
            make_event("2018-01-01 10:00:00", "U1", "Alice"),
            make_event("2018-01-01 12:00:00", "U1", "Bob"),  # 2h later
        ]
        out = process_events(events)
        self.assertEqual(len(out), 2)  # two separate notifications

if __name__ == "__main__":
    unittest.main(verbosity=2)
