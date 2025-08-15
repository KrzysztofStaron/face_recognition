from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import cv2
import numpy as np
import os
import tempfile
import uuid
from findAll import find_matching_photos, load_reference_face
from insightface.app import FaceAnalysis
from embedding_cache import EmbeddingCache

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

def download_image_from_url(url):
    """Download image from URL and save to temporary file"""
    try:
        # Download the image
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        filename = f"temp_reference_{uuid.uuid4().hex}.jpg"
        temp_path = os.path.join(temp_dir, filename)
        
        # Save image to temporary file
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return temp_path
    except Exception as e:
        raise Exception(f"Failed to download image: {str(e)}")


def _compute_area(bbox):
    """Compute area from bbox [x1, y1, x2, y2]"""
    try:
        if not bbox or len(bbox) < 4:
            return 0.0
        x1, y1, x2, y2 = bbox[:4]
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        return w * h
    except Exception:
        return 0.0


def _select_target_face_indices(faces, selector):
    """Select which target faces to use.
    selector can be:
      - None or 'all' (default): use all faces
      - 'largest': face with largest bbox area
      - 'best': face with highest det_score
      - int: use specific index
      - list[int]: use specific indices (validated in range)
    Returns list of indices.
    """
    total = len(faces)
    if total == 0:
        return []

    # Default: all
    if selector is None or selector == 'all':
        return list(range(total))

    # Single index
    if isinstance(selector, int):
        idx = selector
        if -total <= idx < total:
            if idx < 0:
                idx += total
            return [idx]
        return []

    # List of indices
    if isinstance(selector, list):
        indices = []
        for idx in selector:
            if isinstance(idx, int) and -total <= idx < total:
                if idx < 0:
                    idx += total
                indices.append(idx)
        # unique and sorted for stability
        return sorted(set(indices))

    # Named strategies
    if selector == 'largest':
        areas = [(_compute_area(face.get('bbox')), i) for i, face in enumerate(faces)]
        best_idx = max(areas, key=lambda t: t[0])[1]
        return [best_idx]

    if selector == 'best':
        scores = [((face.get('det_score') or -1.0), i) for i, face in enumerate(faces)]
        best_idx = max(scores, key=lambda t: t[0])[1]
        return [best_idx]

    # Fallback: all
    return list(range(total))


def _cosine_similarity(a, b):
    import numpy as _np
    a = _np.asarray(a)
    b = _np.asarray(b)
    denom = _np.linalg.norm(a) * _np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(_np.dot(a, b) / denom)

@app.route('/api/findIn', methods=['POST'])
def find_all():
    """API endpoint to find all photos containing a person from a reference URL"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({
                'error': 'Missing required field: url',
                'success': False
            }), 400
        
        image_url = data['url']
        threshold = data.get('threshold', 0.6)  # Default threshold
        data_directory = data.get('data_directory', 'data')  # Default to 'data' folder
        
        # Validate threshold
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            return jsonify({
                'error': 'Threshold must be a number between 0 and 1',
                'success': False
            }), 400
        
        # Download reference image from URL
        temp_image_path = None
        try:
            temp_image_path = download_image_from_url(image_url)
            
            # Find matching photos
            matches = find_matching_photos(temp_image_path, data_directory, threshold)
            
            # Format response
            response_data = {
                'success': True,
                'reference_url': image_url,
                'threshold': threshold,
                'data_directory': data_directory,
                'total_matches': len(matches),
                'matches': []
            }
            
            # Sort matches by similarity (highest first)
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            for match in matches:
                response_data['matches'].append({
                    'filename': match['filename'],
                    'path': match['path'],
                    'similarity': round(float(match['similarity']), 4),
                    'matching_faces': match['matching_faces'],
                    'all_similarities': [round(float(s), 4) for s in match['all_similarities']]
                })
            
            return jsonify(response_data), 200
            
        finally:
            # Clean up temporary file
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except:
                    pass  # Ignore cleanup errors
                    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'face-finder-api'
    }), 200

@app.route('/api/cache/stats', methods=['GET'])
def cache_stats():
    """Get cache statistics"""
    try:
        cache = EmbeddingCache()
        stats = cache.get_cache_stats()
        return jsonify({
            'success': True,
            'cache_stats': stats
        }), 200
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all cached embeddings"""
    try:
        cache = EmbeddingCache()
        cache.clear_cache()
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully'
        }), 200
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/cache/cleanup', methods=['POST'])
def cleanup_cache():
    """Remove invalid cache entries"""
    try:
        cache = EmbeddingCache()
        removed = cache.remove_invalid_cache_entries()
        return jsonify({
            'success': True,
            'message': f'Cleaned up {removed} invalid cache entries',
            'removed_entries': removed
        }), 200
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/v0/embed', methods=['POST'])
def embed_images():
    """Pre-warm cache by downloading and creating embeddings for multiple images"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'urls' not in data:
            return jsonify({
                'error': 'Missing required field: urls',
                'success': False
            }), 400
        
        urls = data['urls']
        if not isinstance(urls, list):
            return jsonify({
                'error': 'urls must be an array',
                'success': False
            }), 400
        
        # Initialize cache
        cache = EmbeddingCache()
        results = []
        
        for url in urls:
            temp_image_path = None
            try:
                # Download image
                temp_image_path = download_image_from_url(url)
                
                # Get or compute faces (will cache faces and embeddings)
                faces = cache.get_or_compute_url_faces(url, temp_image_path)
                embeddings = [f.get('embedding') for f in faces] if faces is not None else []

                results.append({
                    'url': url,
                    'success': True,
                    'cached': True,
                    'num_faces': len(faces) if faces else 0,
                    'cache_file': cache._get_url_cache_file(url)
                })
                
            except Exception as e:
                results.append({
                    'url': url,
                    'success': False,
                    'error': str(e)
                })
            finally:
                # Clean up temporary file
                if temp_image_path and os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except:
                        pass
        
        return jsonify({
            'success': True,
            'total_urls': len(urls),
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/v0/findIn', methods=['POST'])
def find_in_scope():
    """Find target person in a scope of images with robust multi-face handling"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Missing request body', 'success': False}), 400

        if 'scope' not in data or 'target' not in data:
            return jsonify({'error': 'Missing required fields: scope and target', 'success': False}), 400

        scope_urls = data['scope']
        target_url = data['target']
        threshold = data.get('threshold', 0.6)
        target_face_selector = data.get('target_face', 'all')  # 'all' | 'largest' | 'best' | int | [int]
        include_details = bool(data.get('include_details', False))
        max_results = data.get('max_results')

        if not isinstance(scope_urls, list):
            return jsonify({'error': 'scope must be an array of URLs', 'success': False}), 400

        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            return jsonify({'error': 'Threshold must be a number between 0 and 1', 'success': False}), 400

        cache = EmbeddingCache()

        # Download and get faces for target
        target_temp_path = None
        try:
            target_temp_path = download_image_from_url(target_url)
            target_faces = cache.get_or_compute_url_faces(target_url, target_temp_path)
            if not target_faces:
                return jsonify({'error': 'No face detected in target image', 'success': False}), 400

            selected_target_indices = _select_target_face_indices(target_faces, target_face_selector)
            if not selected_target_indices:
                return jsonify({'error': 'Target face selector did not select any faces', 'success': False}), 400

            # Prepare selected target embeddings and optional metadata
            selected_targets = [(i, target_faces[i]) for i in selected_target_indices]

            # Process each scope image
            matches = []
            for scope_url in scope_urls:
                scope_temp_path = None
                try:
                    scope_temp_path = download_image_from_url(scope_url)
                    scope_faces = cache.get_or_compute_url_faces(scope_url, scope_temp_path)
                    if not scope_faces:
                        continue

                    # Build candidate pairs above threshold
                    candidate_pairs = []  # (similarity, target_idx, scope_idx)
                    best_similarity = -1.0
                    for t_local_idx, (t_idx, t_face) in enumerate(selected_targets):
                        t_emb = t_face.get('embedding')
                        for s_idx, s_face in enumerate(scope_faces):
                            sim = _cosine_similarity(t_emb, s_face.get('embedding'))
                            if sim > best_similarity:
                                best_similarity = sim
                            if sim >= threshold:
                                candidate_pairs.append((sim, t_idx, s_idx))

                    if not candidate_pairs:
                        continue

                    # Greedy one-to-one matching by highest similarity
                    candidate_pairs.sort(key=lambda x: x[0], reverse=True)
                    used_t = set()
                    used_s = set()
                    accepted = []
                    for sim, t_idx, s_idx in candidate_pairs:
                        if t_idx in used_t or s_idx in used_s:
                            continue
                        used_t.add(t_idx)
                        used_s.add(s_idx)
                        accepted.append((sim, t_idx, s_idx))

                    if not accepted:
                        continue

                    # Build result entry
                    accepted_sorted = sorted(accepted, key=lambda x: x[0], reverse=True)
                    detailed_matches = []
                    for sim, t_idx, s_idx in accepted_sorted:
                        match_entry = {
                            'target_face': t_idx,
                            'scope_face': s_idx,
                            'similarity': round(float(sim), 4)
                        }
                        if include_details:
                            # add bboxes/scores if available
                            try:
                                match_entry['target_bbox'] = target_faces[t_idx].get('bbox')
                                match_entry['target_score'] = target_faces[t_idx].get('det_score')
                                match_entry['scope_bbox'] = scope_faces[s_idx].get('bbox')
                                match_entry['scope_score'] = scope_faces[s_idx].get('det_score')
                            except Exception:
                                pass
                        detailed_matches.append(match_entry)

                    result_entry = {
                        'url': scope_url,
                        'similarity': round(float(accepted_sorted[0][0] if accepted_sorted else best_similarity), 4),
                        'matching_faces': len(accepted_sorted),
                        'target_faces_found': len({t for _, t, _ in accepted_sorted}),
                        'target_face_indices': sorted({t for _, t, _ in accepted_sorted}),
                        'all_similarities': [round(float(sim), 4) for sim, _, _ in accepted_sorted],
                        'detailed_matches': detailed_matches
                    }

                    if include_details:
                        result_entry['scope_faces_count'] = len(scope_faces)

                    matches.append(result_entry)
                finally:
                    if scope_temp_path and os.path.exists(scope_temp_path):
                        try:
                            os.remove(scope_temp_path)
                        except:
                            pass

            # Sort and trim results
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            if isinstance(max_results, int) and max_results > 0:
                matches = matches[:max_results]

            response = {
                'success': True,
                'target_url': target_url,
                'target_faces_count': len(target_faces),
                'threshold': threshold,
                'total_scope_images': len(scope_urls),
                'total_matches': len(matches),
                'urls': [m['url'] for m in matches],
                'matches': matches
            }

            if include_details:
                response['selected_target_indices'] = selected_target_indices
                response['target_summary'] = [
                    {
                        'index': i,
                        'bbox': f.get('bbox'),
                        'score': f.get('det_score')
                    } for i, f in enumerate(target_faces)
                ]

            return jsonify(response), 200
        finally:
            if target_temp_path and os.path.exists(target_temp_path):
                try:
                    os.remove(target_temp_path)
                except:
                    pass
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/v0/inspect', methods=['POST'])
def inspect_image_faces():
    """Return face metadata (bbox, score) for a given image URL so client can choose target face."""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'Missing required field: url', 'success': False}), 400
        url = data['url']
        cache = EmbeddingCache()
        temp_path = None
        try:
            temp_path = download_image_from_url(url)
            faces = cache.get_or_compute_url_faces(url, temp_path)
            if faces is None:
                faces = []
            response_faces = []
            for i, f in enumerate(faces):
                response_faces.append({
                    'index': i,
                    'bbox': f.get('bbox'),
                    'score': f.get('det_score')
                })
            return jsonify({
                'success': True,
                'url': url,
                'faces_count': len(response_faces),
                'faces': response_faces
            }), 200
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

# Initialize data directory check and logging
def init_app():
    """Initialize application settings"""
    # Ensure data directory exists
    if not os.path.exists('data'):
        print("Warning: 'data' directory not found. Make sure it exists with images to search.")
    
    # Log available endpoints
    print("🚀 Face Finder API initialized")
    print("Available endpoints:")
    print("🔄 POST /api/v0/embed - Pre-warm cache with image URLs")
    print("🔍 POST /api/v0/findIn - Find target in scope of images") 
    print("🔍 POST /api/findIn - Find matches in data directory (legacy)")
    print("💚 GET /api/health - Health check")
    print("📊 GET /api/cache/stats - Get cache statistics")
    print("🧹 POST /api/cache/clear - Clear all cached embeddings")
    print("🧹 POST /api/cache/cleanup - Remove invalid cache entries")

# Initialize app when imported
init_app()

if __name__ == '__main__':
    # Development server - not used in production
    app.run(host='0.0.0.0', port=5003, debug=True)
