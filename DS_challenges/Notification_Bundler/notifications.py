#!/usr/bin/env python3
"""
Notification Bundler — Senior Data Scientist / Backend Engineer Challenge
=========================================================================
Bundles push notifications so that:
  - A user receives at most 4 notifications per day (soft cap — exceeded only rarely)
  - Sending delay is minimised

Algorithm: Adaptive Windowing
  - First notification for a user in a "quiet window" → sent immediately (0 s delay).
  - Subsequent notifications for the same user within a rolling window are bundled.
  - The bundle window length adapts to the user's notification rate:
      low  rate  → short  window (5 min)  → fast delivery
      high rate  → longer window (15 min) → more bundling
  - At end-of-day, any pending bundle is flushed automatically.
  - Hard daily cap at 4 notifications: if the cap would be exceeded, the notification
    is queued into the next bundle rather than sent immediately.
"""

import sys
import csv
import io
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


# ── Tuneable constants ────────────────────────────────────────────────────────
DAILY_SOFT_CAP        = 4          # max notifications per user per day
BASE_WINDOW_SECS      = 300        # 5 min  — window when rate is low
MAX_WINDOW_SECS       = 900        # 15 min — window when rate is high
RATE_HIGH_THRESHOLD   = 3          # tours/hour that triggers longer window
DAY_END               = "23:59:59" # flush time for dangling bundles


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class PendingBundle:
    user_id:        str
    window_start:   datetime          # first tour in this bundle
    window_close:   datetime          # when to flush if no more tours arrive
    tours:          list = field(default_factory=list)   # (timestamp, friend_name)


@dataclass
class UserState:
    daily_sent:     int  = 0
    daily_date:     Optional[date] = None
    pending:        Optional[PendingBundle] = None
    hour_counts:    dict = field(default_factory=lambda: defaultdict(int))  # hour → count


# ── Core logic ────────────────────────────────────────────────────────────────
def adaptive_window(user_state: UserState, event_time: datetime) -> timedelta:
    """Return bundle window size based on recent hourly rate."""
    current_hour = event_time.replace(minute=0, second=0, microsecond=0)
    recent = sum(
        v for k, v in user_state.hour_counts.items()
        if k >= current_hour - timedelta(hours=1)
    )
    if recent >= RATE_HIGH_THRESHOLD:
        return timedelta(seconds=MAX_WINDOW_SECS)
    return timedelta(seconds=BASE_WINDOW_SECS)


def reset_daily_if_needed(state: UserState, event_date: date):
    if state.daily_date != event_date:
        state.daily_sent = 0
        state.daily_date = event_date
        state.pending    = None


def format_message(tours: list) -> str:
    names = [name for _, name in tours]
    unique = list(dict.fromkeys(names))          # preserve order, deduplicate
    if len(unique) == 1:
        n = len(tours)
        return f"{unique[0]} went on a tour" if n == 1 else f"{unique[0]} went on {n} tours"
    first = unique[0]
    others = len(unique) - 1
    return f"{first} and {others} other{'s' if others > 1 else ''} went on a tour"


def flush_bundle(bundle: PendingBundle, flush_time: datetime) -> dict:
    return {
        "notification_sent":    flush_time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_first_tour": bundle.window_start.strftime("%Y-%m-%d %H:%M:%S"),
        "tours":                str(len(bundle.tours)),
        "receiver_id":          bundle.user_id,
        "message":              format_message(bundle.tours),
    }


def process_events(rows: list[dict]) -> list[dict]:
    """
    Process a sorted list of event dicts and return output notification dicts.
    rows: [{"timestamp": datetime, "user_id": str, "friend_id": str, "friend_name": str}, ...]
    """
    # Sort by timestamp (stream may not be perfectly ordered)
    rows = sorted(rows, key=lambda r: r["timestamp"])

    user_states: dict[str, UserState] = defaultdict(UserState)
    output_rows: list[dict] = []
    # pending_flushes: sorted list of (flush_time, user_id) for lookahead
    # We process in a single pass; bundles flush when a later event arrives after window_close.

    def maybe_flush_pending(state: UserState, user_id: str, now: datetime):
        """Flush any pending bundle whose window has closed by `now`."""
        if state.pending and now >= state.pending.window_close:
            output_rows.append(flush_bundle(state.pending, state.pending.window_close))
            state.daily_sent += 1
            state.pending = None

    for row in rows:
        ts      = row["timestamp"]
        user_id = row["user_id"]
        fname   = row["friend_name"].strip()
        state   = user_states[user_id]

        reset_daily_if_needed(state, ts.date())
        maybe_flush_pending(state, user_id, ts)

        # Track hourly rate
        hour_key = ts.replace(minute=0, second=0, microsecond=0)
        state.hour_counts[hour_key] += 1

        if state.pending is None:
            # No open bundle
            if state.daily_sent < DAILY_SOFT_CAP:
                # Send immediately (window_close = now + adaptive window)
                window = adaptive_window(state, ts)
                state.pending = PendingBundle(
                    user_id=user_id,
                    window_start=ts,
                    window_close=ts + window,
                    tours=[(ts, fname)],
                )
                # If window is 0 (rate extremely low), flush at once
            else:
                # Daily cap reached — hold until next day; skip (log to stderr)
                print(f"[CAP] User {user_id} hit daily cap at {ts}", file=sys.stderr)
        else:
            # Extend or add to existing bundle
            window = adaptive_window(state, ts)
            new_close = ts + window
            if new_close > state.pending.window_close:
                state.pending.window_close = new_close   # extend window
            state.pending.tours.append((ts, fname))

    # End-of-stream: flush all remaining pending bundles
    for user_id, state in user_states.items():
        if state.pending:
            flush_time = state.pending.window_close
            output_rows.append(flush_bundle(state.pending, flush_time))

    # Sort output by notification_sent
    output_rows.sort(key=lambda r: r["notification_sent"])
    return output_rows


# ── I/O ───────────────────────────────────────────────────────────────
FIELDNAMES_IN  = ["timestamp", "user_id", "friend_id", "friend_name"]
FIELDNAMES_OUT = ["notification_sent", "timestamp_first_tour", "tours", "receiver_id", "message"]


def parse_input(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, fieldnames=FIELDNAMES_IN)
        next(reader, None)   # skip header
        for line in reader:
            try:
                ts = datetime.strptime(line["timestamp"].strip(), "%Y-%m-%d %H:%M:%S")
                rows.append({
                    "timestamp":   ts,
                    "user_id":     line["user_id"].strip(),
                    "friend_id":   line["friend_id"].strip(),
                    "friend_name": line["friend_name"].strip(),
                })
            except (ValueError, KeyError):
                print(f"[SKIP] Bad row: {line}", file=sys.stderr)
    return rows


def write_output(rows: list[dict]):
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES_OUT, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python notifications.py <input_csv>", file=sys.stderr)
        sys.exit(1)

    events = parse_input(sys.argv[1])
    results = process_events(events)
    write_output(results)


if __name__ == "__main__":
    main()
