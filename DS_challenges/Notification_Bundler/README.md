### Notification Bundler — Solution

#### Overview

A command-line tool that reads a CSV stream of tour events and outputs bundled push notifications, balancing:

- **≤ 4 notifications per user per day** (soft cap)
- **Minimal sending delay** — users hear about tours as fast as possible

---

### Quick Start

```bash
# Run on the provided sample data
python notifications.py sample_notifications.csv

# Run on  CSV file
python notifications.py /path/to/your/events.csv

# Save output to file
python notifications.py sample_notifications.csv > output.csv

# Run tests
python test_notifications.py
```

No dependencies outside Python 3.8+ standard library.

---

### Input Format

```
timestamp,user_id,friend_id,friend_name
2018-02-14 11:50:02,EB96305EADU2,84BE9DC3BFLL,Matthew
```

### Output Format

```
notification_sent,timestamp_first_tour,tours,receiver_id,message
2018-02-14 11:55:02,2018-02-14 11:50:02,1,EB96305EADU2,Matthew went on a tour
2018-02-14 12:07:10,2018-02-14 11:50:02,3,0B4E1F74A818,Matthew and 2 others went on a tour
```

---

### Algorithm - Adaptive Windowing

```
Event arrives for user U
  ├─ No open bundle?
  │     └─ Daily cap OK? → Open new bundle, start adaptive timer
  └─ Bundle open?
        └─ Extend window close; add tour to bundle

Window expires → Flush as one notification → increment daily counter
```

**Window length adapts to notification rate:**
| Rate | Window | Effect |
|---|---|---|
| < 3 tours/hour | 5 minutes | Near-instant delivery for casual users |
| ≥ 3 tours/hour | 15 minutes | More bundling for power users |

See `solution_walkthrough.ipynb` for a full visual explanation.

---

### Files

| File | Description |
|---|---|
| `notifications.py` | Main application (run this) |
| `test_notifications.py` | Unit tests (9 tests) |
| `sample_notifications.csv` | Sample input data |
| `solution_walkthrough.ipynb` | Algorithm explanation + result analysis |

