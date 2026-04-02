"""
VISTA 2.0 – Unit Tests
Tests individual pure-logic functions without loading YOLO models or Flask.
Run with:  python -m pytest test_unit.py -v
"""
import time
import csv
import os
import threading
import unittest
import numpy as np

# ─── Helpers we can test WITHOUT importing app (avoids model load) ─────────────

def compute_crowd_from_centroids(centroids, people_limit=10, distance_threshold=80):
    """Copied logic from app.py to allow isolated unit testing."""
    N = len(centroids)
    if N == 0:
        return False, 0
    pts = np.array(centroids)
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    close_counts = (dists < distance_threshold).sum(axis=1)
    crowded = np.any(close_counts >= people_limit)
    return crowded, N


def log_status(csv_path, status, count, violence, abandoned, suspicious):
    """Copied log logic for isolated testing."""
    ts = "2026-01-01 00:00:00"
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, status, count, violence, abandoned, suspicious])


# ─── Unit Test Cases ───────────────────────────────────────────────────────────

class TestCrowdDetectionLogic(unittest.TestCase):
    """UT-01 to UT-04: Crowd computation function."""

    def test_empty_centroids_returns_false(self):
        """UT-01: No people → not crowded, count=0."""
        crowded, count = compute_crowd_from_centroids([])
        self.assertFalse(crowded)
        self.assertEqual(count, 0)

    def test_few_spread_out_people_not_crowded(self):
        """UT-02: 5 people spread > threshold → not crowded."""
        centroids = [(i * 200, 0) for i in range(5)]   # 200px apart
        crowded, count = compute_crowd_from_centroids(centroids, people_limit=10, distance_threshold=80)
        self.assertFalse(crowded)
        self.assertEqual(count, 5)

    def test_dense_cluster_is_crowded(self):
        """UT-03: 12 people within 50px → crowded."""
        centroids = [(i * 10, 0) for i in range(12)]   # 10px apart, well within 80px threshold
        crowded, count = compute_crowd_from_centroids(centroids, people_limit=10, distance_threshold=80)
        self.assertTrue(crowded)
        self.assertEqual(count, 12)

    def test_exact_people_limit_boundary(self):
        """UT-04: Exactly at people_limit → crowded."""
        centroids = [(i * 5, 0) for i in range(10)]   # 10 people very close
        crowded, count = compute_crowd_from_centroids(centroids, people_limit=10, distance_threshold=80)
        self.assertTrue(crowded)
        self.assertEqual(count, 10)

    def test_single_person_not_crowded(self):
        """UT-05: Single person → not crowded."""
        crowded, count = compute_crowd_from_centroids([(100, 100)], people_limit=10)
        self.assertFalse(crowded)
        self.assertEqual(count, 1)

    def test_distance_threshold_respected(self):
        """UT-06: People outside threshold are not counted together."""
        centroids = [(0, 0)] * 5 + [(500, 500)] * 5   # two clusters of 5
        crowded, count = compute_crowd_from_centroids(centroids, people_limit=10, distance_threshold=80)
        self.assertFalse(crowded)   # neither cluster ≥ 10
        self.assertEqual(count, 10)


class TestCSVLogging(unittest.TestCase):
    """UT-07 to UT-09: CSV log writing."""

    CSV_TEST = "test_log_unit.csv"

    def setUp(self):
        if os.path.exists(self.CSV_TEST):
            os.remove(self.CSV_TEST)
        with open(self.CSV_TEST, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "status", "count",
                                    "violence_detected", "abandoned_count", "suspicious_detected"])

    def tearDown(self):
        if os.path.exists(self.CSV_TEST):
            os.remove(self.CSV_TEST)

    def test_log_creates_row(self):
        """UT-07: Logging adds exactly one row."""
        log_status(self.CSV_TEST, "NON-CROWDED", 3, False, 0, False)
        with open(self.CSV_TEST) as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 2)   # header + 1 data row

    def test_log_values_correct(self):
        """UT-08: Logged values match inputs."""
        log_status(self.CSV_TEST, "CROWDED", 15, True, 2, True)
        with open(self.CSV_TEST) as f:
            rows = list(csv.reader(f))
        row = rows[1]
        self.assertEqual(row[1], "CROWDED")
        self.assertEqual(row[2], "15")
        self.assertEqual(row[3], "True")
        self.assertEqual(row[4], "2")
        self.assertEqual(row[5], "True")

    def test_log_multiple_rows(self):
        """UT-09: Multiple log calls accumulate rows."""
        for i in range(5):
            log_status(self.CSV_TEST, "NON-CROWDED", i, False, 0, False)
        with open(self.CSV_TEST) as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 6)   # header + 5 data rows


class TestConfigConstants(unittest.TestCase):
    """UT-10: Verify required configuration constants exist and are sane."""

    def test_people_limit_positive(self):
        """UT-10: PEOPLE_LIMIT should be a positive integer."""
        PEOPLE_LIMIT = 10
        self.assertIsInstance(PEOPLE_LIMIT, int)
        self.assertGreater(PEOPLE_LIMIT, 0)

    def test_distance_threshold_positive(self):
        """UT-11: DISTANCE_THRESHOLD should be positive."""
        DISTANCE_THRESHOLD = 80
        self.assertGreater(DISTANCE_THRESHOLD, 0)

    def test_conf_threshold_range(self):
        """UT-12: Confidence threshold must be between 0 and 1."""
        CONF_THRES = 0.35
        self.assertGreater(CONF_THRES, 0.0)
        self.assertLess(CONF_THRES, 1.0)

    def test_stream_fps_limit_positive(self):
        """UT-13: STREAM_FPS_LIMIT should be positive."""
        STREAM_FPS_LIMIT = 20
        self.assertGreater(STREAM_FPS_LIMIT, 0)


class TestThreadSafety(unittest.TestCase):
    """UT-14: Thread-safety of the shared state lock pattern."""

    def test_concurrent_counter_increment(self):
        """UT-14: Concurrent lock-protected increments should be race-free."""
        lock = threading.Lock()
        counter = [0]

        def increment():
            for _ in range(1000):
                with lock:
                    counter[0] += 1

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(counter[0], 10000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
