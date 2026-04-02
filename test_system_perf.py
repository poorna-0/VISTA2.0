"""
VISTA 2.0 – System & Performance Tests
System tests validate end-to-end workflows.
Performance tests measure response time, throughput, and concurrent load.
Run with:  python -m pytest test_system_perf.py -v
"""
import io
import json
import sys
import os
import time
import csv
import threading
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# ─── Same mock setup as integration tests ─────────────────────────────────────
mock_yolo_instance = MagicMock()
mock_yolo_instance.side_effect = lambda frame, **kw: _make_fake_results(0)
mock_yolo_class = MagicMock(return_value=mock_yolo_instance)


def _make_fake_results(n=0):
    fake_box = MagicMock()
    if n == 0:
        fake_box.xyxy = MagicMock()
        fake_box.xyxy.cpu.return_value.numpy.return_value = np.empty((0, 4))
        fake_box.conf = MagicMock()
        fake_box.conf.cpu.return_value.numpy.return_value = np.empty((0,))
    else:
        coords = np.array([[10, 10, 50, 50]] * n, dtype=float)
        confs  = np.array([0.9] * n)
        fake_box.xyxy = MagicMock()
        fake_box.xyxy.cpu.return_value.numpy.return_value = coords
        fake_box.conf = MagicMock()
        fake_box.conf.cpu.return_value.numpy.return_value = confs
        fake_box.cls  = MagicMock()
        fake_box.cls.cpu.return_value.numpy.return_value = np.zeros(n)
    fake_result = MagicMock()
    fake_result.boxes = fake_box
    return [fake_result]


sys.modules['ultralytics'] = MagicMock(YOLO=mock_yolo_class)
sys.modules['winsound']    = MagicMock()

sys.path.insert(0, os.path.dirname(__file__))
with patch('cv2.VideoCapture') as _:
    import app as vista_app   # noqa: E402

vista_app.app.config['TESTING'] = True
vista_app.app.config['UPLOAD_FOLDER'] = 'uploads_test'
os.makedirs(vista_app.app.config['UPLOAD_FOLDER'], exist_ok=True)


# ═══════════════════════════════════════════════════════
#  SYSTEM TESTS
# ═══════════════════════════════════════════════════════

class TestSystemWorkflows(unittest.TestCase):
    """End-to-end workflow tests that verify multiple components working together."""

    def setUp(self):
        self.client = vista_app.app.test_client()

    # ── ST-01 ──────────────────────────────────────────
    def test_full_dashboard_load(self):
        """ST-01: Dashboard HTML loads successfully with status 200."""
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(len(resp.data), 100)

    # ── ST-02 ──────────────────────────────────────────
    def test_source_switch_then_status_poll(self):
        """ST-02: Switch source to RTSP, then poll status — should remain consistent."""
        self.client.post('/set_source',
                         json={'type': 'rtsp', 'value': 'rtsp://10.0.0.1/cam'})
        resp = self.client.get('/status')
        data = json.loads(resp.data)
        self.assertIn('crowd_status', data)

    # ── ST-03 ──────────────────────────────────────────
    def test_image_upload_end_to_end(self):
        """ST-03: Upload image → receive annotated JPEG response."""
        import cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)
        img_bytes = io.BytesIO(buf.tobytes())

        with patch.object(vista_app, 'process_frame',
                          return_value=(img, True, 12, True, 1, False)):
            resp = self.client.post('/upload',
                                    data={'file': (img_bytes, 'scene.jpg')},
                                    content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content_type, 'image/jpeg')

    # ── ST-04 ──────────────────────────────────────────
    def test_log_download_after_status_poll(self):
        """ST-04: After polling status, log file is downloadable."""
        self.client.get('/status')
        if not os.path.exists(vista_app.CSV_LOG):
            with open(vista_app.CSV_LOG, 'w', newline='') as f:
                csv.writer(f).writerow(['timestamp', 'status', 'count',
                                        'violence_detected', 'abandoned_count',
                                        'suspicious_detected'])
        resp = self.client.get('/download_log')
        self.assertEqual(resp.status_code, 200)

    # ── ST-05 ──────────────────────────────────────────
    def test_source_switch_webcam_to_file(self):
        """ST-05: Switch from webcam to file source returns ok."""
        r1 = self.client.post('/set_source', json={'type': 'webcam', 'value': '0'})
        self.assertTrue(json.loads(r1.data)['ok'])
        r2 = self.client.post('/set_source', json={'type': 'file', 'value': 'sample.mp4'})
        self.assertTrue(json.loads(r2.data)['ok'])

    # ── ST-06 ──────────────────────────────────────────
    def test_status_fields_always_present(self):
        """ST-06: All 5 status fields always present after state changes."""
        self.client.post('/set_source', json={'type': 'webcam', 'value': '0'})
        resp = self.client.get('/status')
        data = json.loads(resp.data)
        required_keys = {'crowd_status', 'count', 'violence_detected',
                         'abandoned_count', 'suspicious_detected'}
        self.assertTrue(required_keys.issubset(set(data.keys())))

    # ── ST-07 ──────────────────────────────────────────
    def test_multiple_uploads_sequential(self):
        """ST-07: Multiple sequential image uploads all succeed."""
        import cv2
        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            _, buf = cv2.imencode('.png', img)
            img_bytes = io.BytesIO(buf.tobytes())
            with patch.object(vista_app, 'process_frame',
                              return_value=(img, False, 0, False, 0, False)):
                resp = self.client.post('/upload',
                                        data={'file': (img_bytes, f'img_{i}.png')},
                                        content_type='multipart/form-data')
            self.assertEqual(resp.status_code, 200)

    # ── ST-08 ──────────────────────────────────────────
    def test_invalid_source_does_not_break_status(self):
        """ST-08: Invalid source switch doesn't break the /status endpoint."""
        self.client.post('/set_source', json={'type': 'bad', 'value': 'x'})
        resp = self.client.get('/status')
        self.assertEqual(resp.status_code, 200)


# ═══════════════════════════════════════════════════════
#  PERFORMANCE TESTS
# ═══════════════════════════════════════════════════════

class TestPerformanceMetrics(unittest.TestCase):
    """Measure response times, throughput, and concurrent load."""

    def setUp(self):
        self.client = vista_app.app.test_client()

    # ── PT-01 ──────────────────────────────────────────
    def test_status_response_time_under_100ms(self):
        """PT-01: /status responds in < 100 ms (no model inference)."""
        start = time.perf_counter()
        resp = self.client.get('/status')
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertEqual(resp.status_code, 200)
        print(f"\n  PT-01 /status latency: {elapsed_ms:.2f} ms")
        self.assertLess(elapsed_ms, 100,
                        f"/status too slow: {elapsed_ms:.1f} ms > 100 ms")

    # ── PT-02 ──────────────────────────────────────────
    def test_index_response_time_under_200ms(self):
        """PT-02: GET / responds in < 200 ms."""
        start = time.perf_counter()
        resp = self.client.get('/')
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"\n  PT-02 GET / latency: {elapsed_ms:.2f} ms")
        self.assertLess(elapsed_ms, 200)

    # ── PT-03 ──────────────────────────────────────────
    def test_status_throughput_50rps(self):
        """PT-03: /status endpoint handles ≥ 50 requests in ≤ 2 seconds."""
        N = 50
        start = time.perf_counter()
        for _ in range(N):
            r = self.client.get('/status')
            self.assertEqual(r.status_code, 200)
        elapsed = time.perf_counter() - start
        rps = N / elapsed
        print(f"\n  PT-03 throughput: {rps:.1f} req/s over {N} requests ({elapsed:.2f}s)")
        self.assertLessEqual(elapsed, 2.0,
                             f"50 requests took {elapsed:.2f}s, expected ≤ 2s")

    # ── PT-04 ──────────────────────────────────────────
    def test_concurrent_status_requests(self):
        """PT-04: 20 concurrent /status requests all succeed."""
        results = []
        errors  = []

        def hit_status():
            try:
                # Each thread needs its own client
                with vista_app.app.test_client() as c:
                    r = c.get('/status')
                    results.append(r.status_code)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=hit_status) for _ in range(20)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start
        print(f"\n  PT-04 concurrent (20 threads) completed in {elapsed:.2f}s")
        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertTrue(all(s == 200 for s in results))

    # ── PT-05 ──────────────────────────────────────────
    def test_set_source_response_time(self):
        """PT-05: POST /set_source responds in < 200 ms."""
        start = time.perf_counter()
        resp = self.client.post('/set_source', json={'type': 'webcam', 'value': '0'})
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"\n  PT-05 /set_source latency: {elapsed_ms:.2f} ms")
        self.assertLess(elapsed_ms, 200)

    # ── PT-06 ──────────────────────────────────────────
    def test_image_upload_processing_time(self):
        """PT-06: Image upload + mock process_frame completes in < 500 ms."""
        import cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)
        img_bytes = io.BytesIO(buf.tobytes())

        start = time.perf_counter()
        with patch.object(vista_app, 'process_frame',
                          return_value=(img, False, 0, False, 0, False)):
            resp = self.client.post('/upload',
                                    data={'file': (img_bytes, 'perf_test.jpg')},
                                    content_type='multipart/form-data')
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"\n  PT-06 image upload latency: {elapsed_ms:.2f} ms")
        self.assertEqual(resp.status_code, 200)
        self.assertLess(elapsed_ms, 500)

    # ── PT-07 ──────────────────────────────────────────
    def test_csv_log_write_speed(self):
        """PT-07: Writing 500 log rows takes < 1 second."""
        test_csv = "perf_log_test.csv"
        try:
            start = time.perf_counter()
            with open(test_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['timestamp', 'status', 'count',
                             'violence', 'abandoned', 'suspicious'])
                for i in range(500):
                    w.writerow(['2026-01-01 00:00:00', 'NON-CROWDED',
                                i % 20, False, 0, False])
            elapsed = time.perf_counter() - start
            print(f"\n  PT-07 500 CSV rows written in {elapsed:.4f}s")
            self.assertLess(elapsed, 1.0)
        finally:
            if os.path.exists(test_csv):
                os.remove(test_csv)

    # ── PT-08 ──────────────────────────────────────────
    def test_crowd_logic_performance_large_input(self):
        """PT-08: crowd logic runs in < 50 ms for 1000 centroids."""
        from test_unit import compute_crowd_from_centroids
        centroids = [(i % 640, (i // 640) * 10) for i in range(1000)]
        start = time.perf_counter()
        crowded, count = compute_crowd_from_centroids(centroids, people_limit=10, distance_threshold=80)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"\n  PT-08 crowd logic (1000 centroids): {elapsed_ms:.2f} ms")
        self.assertLess(elapsed_ms, 50,
                        f"Crowd logic too slow for 1000 centroids: {elapsed_ms:.1f}ms")


if __name__ == '__main__':
    unittest.main(verbosity=2)
