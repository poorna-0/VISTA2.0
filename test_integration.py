"""
VISTA 2.0 – Integration Tests
Tests Flask routes and their interactions using the test client.
Models are mocked so tests can run without GPU or model files.
Run with:  python -m pytest test_integration.py -v
"""
import io
import json
import sys
import os
import csv
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# ─── Mock heavy dependencies BEFORE importing app ─────────────────────────────
# Prevent actual model loading during import
mock_yolo_instance = MagicMock()

def _make_fake_results(n_detections=0):
    """Build a fake YOLO result list matching the interface app.py uses."""
    fake_box = MagicMock()
    if n_detections == 0:
        fake_box.xyxy = MagicMock()
        fake_box.xyxy.cpu.return_value.numpy.return_value = np.empty((0, 4))
        fake_box.conf = MagicMock()
        fake_box.conf.cpu.return_value.numpy.return_value = np.empty((0,))
    else:
        coords = np.array([[10, 10, 50, 50]] * n_detections, dtype=float)
        confs  = np.array([0.9] * n_detections)
        fake_box.xyxy = MagicMock()
        fake_box.xyxy.cpu.return_value.numpy.return_value = coords
        fake_box.conf = MagicMock()
        fake_box.conf.cpu.return_value.numpy.return_value = confs
        fake_box.cls  = MagicMock()
        fake_box.cls.cpu.return_value.numpy.return_value = np.zeros(n_detections)
    fake_result = MagicMock()
    fake_result.boxes = fake_box
    return [fake_result]

mock_yolo_instance.side_effect = lambda frame, **kw: _make_fake_results(0)

mock_yolo_class = MagicMock(return_value=mock_yolo_instance)

sys.modules['ultralytics'] = MagicMock(YOLO=mock_yolo_class)
sys.modules['winsound']    = MagicMock()

# ─── Now safe to import the app ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# Patch cv2.VideoCapture to prevent real camera access
with patch('cv2.VideoCapture') as _:
    import app as vista_app   # noqa: E402

vista_app.app.config['TESTING'] = True
vista_app.app.config['UPLOAD_FOLDER'] = 'uploads_test'
os.makedirs(vista_app.app.config['UPLOAD_FOLDER'], exist_ok=True)


class TestFlaskRoutes(unittest.TestCase):
    """Integration tests for all Flask HTTP routes."""

    def setUp(self):
        self.client = vista_app.app.test_client()

    # ── IT-01 ──────────────────────────────────────────────────────────────────
    def test_index_route_returns_200(self):
        """IT-01: GET / returns HTTP 200 and HTML content."""
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'html', resp.data.lower())

    # ── IT-02 ──────────────────────────────────────────────────────────────────
    def test_status_route_returns_json(self):
        """IT-02: GET /status returns valid JSON with expected keys."""
        resp = self.client.get('/status')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        for key in ['crowd_status', 'count', 'violence_detected', 'abandoned_count', 'suspicious_detected']:
            self.assertIn(key, data)

    # ── IT-03 ──────────────────────────────────────────────────────────────────
    def test_status_crowd_status_is_string(self):
        """IT-03: crowd_status value is a string."""
        resp = self.client.get('/status')
        data = json.loads(resp.data)
        self.assertIsInstance(data['crowd_status'], str)

    # ── IT-04 ──────────────────────────────────────────────────────────────────
    def test_status_count_is_int(self):
        """IT-04: count value is an integer."""
        resp = self.client.get('/status')
        data = json.loads(resp.data)
        self.assertIsInstance(data['count'], int)

    # ── IT-05 ──────────────────────────────────────────────────────────────────
    def test_set_source_webcam(self):
        """IT-05: POST /set_source with webcam type succeeds."""
        resp = self.client.post('/set_source',
                                json={'type': 'webcam', 'value': '0'})
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data['ok'])
        self.assertEqual(data['type'], 'webcam')

    # ── IT-06 ──────────────────────────────────────────────────────────────────
    def test_set_source_rtsp(self):
        """IT-06: POST /set_source with rtsp type stores the URL."""
        resp = self.client.post('/set_source',
                                json={'type': 'rtsp', 'value': 'rtsp://192.168.1.1/stream'})
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data['ok'])
        self.assertEqual(data['type'], 'rtsp')

    # ── IT-07 ──────────────────────────────────────────────────────────────────
    def test_set_source_invalid_type(self):
        """IT-07: POST /set_source with invalid type returns 400."""
        resp = self.client.post('/set_source',
                                json={'type': 'bluetooth', 'value': '0'})
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertFalse(data['ok'])

    # ── IT-08 ──────────────────────────────────────────────────────────────────
    def test_set_source_missing_params(self):
        """IT-08: POST /set_source without required fields returns 400."""
        resp = self.client.post('/set_source', json={})
        self.assertEqual(resp.status_code, 400)

    # ── IT-09 ──────────────────────────────────────────────────────────────────
    def test_upload_no_file_redirects(self):
        """IT-09: POST /upload with no file redirects to index."""
        resp = self.client.post('/upload', data={})
        self.assertIn(resp.status_code, [302, 200])

    # ── IT-10 ──────────────────────────────────────────────────────────────────
    def test_upload_image_returns_jpeg(self):
        """IT-10: POST /upload with a valid PNG image returns annotated JPEG."""
        # create a tiny in-memory PNG (red square)
        import cv2
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (0, 0, 200)
        ok, buf = cv2.imencode('.png', img)
        self.assertTrue(ok)
        img_bytes = io.BytesIO(buf.tobytes())

        # mock process_frame to avoid needing real models
        with patch.object(vista_app, 'process_frame',
                          return_value=(img, False, 0, False, 0, False)):
            resp = self.client.post('/upload',
                                    data={'file': (img_bytes, 'test_img.png')},
                                    content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content_type, 'image/jpeg')

    # ── IT-11 ──────────────────────────────────────────────────────────────────
    def test_download_log_returns_file(self):
        """IT-11: GET /download_log returns a CSV attachment."""
        # Ensure the CSV log file exists
        if not os.path.exists(vista_app.CSV_LOG):
            with open(vista_app.CSV_LOG, 'w', newline='') as f:
                csv.writer(f).writerow(['timestamp', 'status', 'count',
                                        'violence_detected', 'abandoned_count', 'suspicious_detected'])
        resp = self.client.get('/download_log')
        self.assertEqual(resp.status_code, 200)

    # ── IT-12 ──────────────────────────────────────────────────────────────────
    def test_video_feed_content_type(self):
        """IT-12: GET /video_feed returns multipart MIME type header."""
        resp = self.client.get('/video_feed', buffered=False)
        self.assertIn('multipart', resp.content_type)

    # ── IT-13 ──────────────────────────────────────────────────────────────────
    def test_set_source_webcam_invalid_value(self):
        """IT-13: Webcam source with non-integer value returns 400."""
        resp = self.client.post('/set_source',
                                json={'type': 'webcam', 'value': 'not_an_int'})
        self.assertEqual(resp.status_code, 400)

    # ── IT-14 ──────────────────────────────────────────────────────────────────
    def test_status_violence_is_bool(self):
        """IT-14: violence_detected value is boolean."""
        resp = self.client.get('/status')
        data = json.loads(resp.data)
        self.assertIsInstance(data['violence_detected'], bool)


if __name__ == '__main__':
    unittest.main(verbosity=2)
