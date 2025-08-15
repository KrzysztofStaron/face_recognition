import os
import sys
import types
import json
import tempfile
import importlib.util

import numpy as np
import pytest


def _ensure_stubbed_deps():
    """Provide minimal stubs for heavy optional deps so import never fails in CI/local."""
    # Stub insightface and insightface.app.FaceAnalysis if missing
    if 'insightface' not in sys.modules:
        insightface_mod = types.ModuleType('insightface')
        sys.modules['insightface'] = insightface_mod
    if 'insightface.app' not in sys.modules:
        insightface_app_mod = types.ModuleType('insightface.app')

        class FaceAnalysis:  # minimal stub
            def __init__(self, name=None):
                self.name = name

            def prepare(self, ctx_id=-1):
                return None

            def get(self, img):
                return []

        insightface_app_mod.FaceAnalysis = FaceAnalysis
        sys.modules['insightface.app'] = insightface_app_mod

    # Stub cv2 if missing
    if 'cv2' not in sys.modules:
        cv2_mod = types.ModuleType('cv2')

        def _imread(_):
            # Return a dummy image-like numpy array
            return np.zeros((2, 2, 3), dtype=np.uint8)

        cv2_mod.imread = _imread
        sys.modules['cv2'] = cv2_mod


def _import_main_module():
    """Import face/main.py as a module, after ensuring its directory is on sys.path."""
    tests_dir = os.path.dirname(__file__)
    face_dir = os.path.abspath(os.path.join(tests_dir, os.pardir))
    main_path = os.path.join(face_dir, 'main.py')
    assert os.path.exists(main_path), f"main.py not found at {main_path}"

    # Ensure imports like `import embedding_cache` resolve
    if face_dir not in sys.path:
        sys.path.insert(0, face_dir)

    spec = importlib.util.spec_from_file_location('face_main', main_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class FakeEmbeddingCache:
    """Lightweight in-memory fake for EmbeddingCache used by the API routes."""

    # Class-level test-controlled fixtures
    url_to_faces = {}
    stats_payload = {
        'total_basenames': 0,
        'total_files': 0,
        'cache_size_mb': 0,
        'total_faces': 0,
    }

    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir

    def _get_url_cache_file(self, url):
        # Deterministic dummy path string for responses
        safe = str(abs(hash(url)))
        return os.path.join(self.cache_dir, 'embeddings', f'temp_reference_{safe}_embeddings.pkl')

    # Cache admin
    def clear_cache(self):  # noqa: D401
        return None

    def get_cache_stats(self):
        return dict(FakeEmbeddingCache.stats_payload)

    def remove_invalid_cache_entries(self):
        return 0

    # URL-based faces/embeddings
    def get_or_compute_url_faces(self, url, image_path):
        return FakeEmbeddingCache.url_to_faces.get(url, [])


@pytest.fixture(scope='module')
def app_module():
    _ensure_stubbed_deps()
    mod = _import_main_module()
    return mod


@pytest.fixture()
def client(app_module, monkeypatch):
    # Replace heavy components with fakes
    monkeypatch.setattr(app_module, 'EmbeddingCache', FakeEmbeddingCache)

    def fake_download_image_from_url(url):
        fd, p = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        with open(p, 'wb') as f:
            f.write(b'\xff\xd8\xff\xd9')  # minimal junk bytes
        return p

    monkeypatch.setattr(app_module, 'download_image_from_url', fake_download_image_from_url)
    return app_module.app.test_client()


def test_health_check(client):
    res = client.get('/api/health')
    assert res.status_code == 200
    payload = res.get_json()
    assert payload['status'] == 'healthy'
    assert payload['service'] == 'face-finder-api'


def test_embed_endpoint_success(client):
    # Seed faces for URLs
    url_a = 'http://example.com/a.jpg'
    url_b = 'http://example.com/b.jpg'
    FakeEmbeddingCache.url_to_faces = {
        url_a: [
            {'embedding': np.array([1.0, 0.0, 0.0]), 'bbox': [0, 0, 1, 1], 'det_score': 0.9},
        ],
        url_b: [
            {'embedding': np.array([0.0, 1.0, 0.0]), 'bbox': [0, 0, 2, 2], 'det_score': 0.8},
            {'embedding': np.array([0.0, 0.0, 1.0]), 'bbox': [1, 1, 3, 3], 'det_score': 0.85},
        ],
    }

    res = client.post('/api/v0/embed', json={'urls': [url_a, url_b]})
    assert res.status_code == 200
    body = res.get_json()
    assert body['success'] is True
    assert body['total_urls'] == 2
    assert len(body['results']) == 2
    # Validate one entry
    entry_a = next(x for x in body['results'] if x['url'] == url_a)
    assert entry_a['success'] is True
    assert entry_a['cached'] is True
    assert entry_a['num_faces'] == 1


def test_find_in_v0_success(client):
    target = 'http://example.com/target.jpg'
    scope1 = 'http://example.com/scope1.jpg'
    scope2 = 'http://example.com/scope2.jpg'

    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([0.0, 1.0, 0.0])
    vec_c = np.array([0.0, 0.0, 1.0])

    FakeEmbeddingCache.url_to_faces = {
        target: [
            {'embedding': vec_a, 'bbox': [10, 10, 20, 20], 'det_score': 0.95},
            {'embedding': vec_b, 'bbox': [30, 30, 40, 40], 'det_score': 0.90},
        ],
        scope1: [
            {'embedding': vec_a, 'bbox': [5, 5, 15, 15], 'det_score': 0.92},
        ],
        scope2: [
            {'embedding': vec_c, 'bbox': [0, 0, 5, 5], 'det_score': 0.80},
        ],
    }

    payload = {
        'scope': [scope1, scope2],
        'target': target,
        'threshold': 0.6,
        'include_details': True,
    }
    res = client.post('/api/v0/findIn', json=payload)
    assert res.status_code == 200
    body = res.get_json()
    assert body['success'] is True
    assert body['total_scope_images'] == 2
    assert body['total_matches'] == 1
    assert body['urls'] == [scope1]
    assert body['matches'][0]['url'] == scope1
    assert body['matches'][0]['target_faces_found'] == 1


def test_find_in_v0_invalid_payload(client):
    res = client.post('/api/v0/findIn', json={'target': 'http://example.com/target.jpg'})
    assert res.status_code == 400
    body = res.get_json()
    assert body['success'] is False


def test_inspect_endpoint_success(client):
    img_url = 'http://example.com/inspect.jpg'
    FakeEmbeddingCache.url_to_faces = {
        img_url: [
            {'embedding': np.array([1.0, 0.0, 0.0]), 'bbox': [1, 2, 3, 4], 'det_score': 0.88},
            {'embedding': np.array([0.0, 1.0, 0.0]), 'bbox': [2, 3, 4, 5], 'det_score': 0.77},
        ]
    }
    res = client.post('/api/v0/inspect', json={'url': img_url})
    assert res.status_code == 200
    body = res.get_json()
    assert body['success'] is True
    assert body['url'] == img_url
    assert body['faces_count'] == 2
    assert len(body['faces']) == 2
    assert 'bbox' in body['faces'][0]
    assert 'score' in body['faces'][0]


def test_cache_endpoints(client):
    FakeEmbeddingCache.stats_payload = {
        'total_basenames': 3,
        'total_files': 7,
        'cache_size_mb': 1.23,
        'total_faces': 15,
    }

    # stats
    res_stats = client.get('/api/cache/stats')
    assert res_stats.status_code == 200
    body_stats = res_stats.get_json()
    assert body_stats['success'] is True
    assert body_stats['cache_stats']['total_files'] == 7

    # clear
    res_clear = client.post('/api/cache/clear', json={})
    assert res_clear.status_code == 200
    body_clear = res_clear.get_json()
    assert body_clear['success'] is True

    # cleanup
    res_cleanup = client.post('/api/cache/cleanup', json={})
    assert res_cleanup.status_code == 200
    body_cleanup = res_cleanup.get_json()
    assert body_cleanup['success'] is True
    assert 'Cleaned up' in body_cleanup['message']



def test_embed_endpoint_invalid_payload(client):
    res = client.post('/api/v0/embed', json={})
    assert res.status_code == 400
    body = res.get_json()
    assert body['success'] is False


def test_inspect_endpoint_invalid_payload(client):
    res = client.post('/api/v0/inspect', json={})
    assert res.status_code == 400
    body = res.get_json()
    assert body['success'] is False


