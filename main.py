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
                
                # Get or compute embeddings (will cache automatically)
                embeddings = cache.get_or_compute_url_embeddings(url, temp_image_path)
                
                results.append({
                    'url': url,
                    'success': True,
                    'cached': True,
                    'num_faces': len(embeddings) if embeddings else 0,
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
    """Find target person in a scope of images"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Missing request body',
                'success': False
            }), 400
        
        if 'scope' not in data or 'target' not in data:
            return jsonify({
                'error': 'Missing required fields: scope and target',
                'success': False
            }), 400
        
        scope_urls = data['scope']
        target_url = data['target']
        threshold = data.get('threshold', 0.6)
        
        if not isinstance(scope_urls, list):
            return jsonify({
                'error': 'scope must be an array of URLs',
                'success': False
            }), 400
        
        # Validate threshold
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            return jsonify({
                'error': 'Threshold must be a number between 0 and 1',
                'success': False
            }), 400
        
        # Initialize cache
        cache = EmbeddingCache()
        
        # Download and get embeddings for target
        target_temp_path = None
        try:
            target_temp_path = download_image_from_url(target_url)
            target_embeddings = cache.get_or_compute_url_embeddings(target_url, target_temp_path)
            
            if not target_embeddings:
                return jsonify({
                    'error': 'No face detected in target image',
                    'success': False
                }), 400
            
            # Use first face from target
            target_embedding = target_embeddings[0]
            
            # Process each scope image
            matches = []
            
            for scope_url in scope_urls:
                scope_temp_path = None
                try:
                    # Download scope image
                    scope_temp_path = download_image_from_url(scope_url)
                    
                    # Get embeddings
                    scope_embeddings = cache.get_or_compute_url_embeddings(scope_url, scope_temp_path)
                    
                    if not scope_embeddings:
                        continue
                    
                    # Check each face in scope against target
                    best_similarity = -1
                    matching_faces = []
                    
                    for face_embedding in scope_embeddings:
                        from findAll import cosine_similarity
                        similarity = cosine_similarity(target_embedding, face_embedding)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                        
                        if similarity >= threshold:
                            matching_faces.append(similarity)
                    
                    if matching_faces:
                        matches.append({
                            'url': scope_url,
                            'similarity': round(float(best_similarity), 4),
                            'matching_faces': len(matching_faces),
                            'all_similarities': [round(float(s), 4) for s in matching_faces]
                        })
                        
                finally:
                    # Clean up temporary file
                    if scope_temp_path and os.path.exists(scope_temp_path):
                        try:
                            os.remove(scope_temp_path)
                        except:
                            pass
            
            # Sort matches by similarity
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            return jsonify({
                'success': True,
                'target_url': target_url,
                'threshold': threshold,
                'total_scope_images': len(scope_urls),
                'total_matches': len(matches),
                'matches': matches
            }), 200
            
        finally:
            # Clean up target temp file
            if target_temp_path and os.path.exists(target_temp_path):
                try:
                    os.remove(target_temp_path)
                except:
                    pass
                    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

if __name__ == '__main__':
    # Ensure data directory exists
    if not os.path.exists('data'):
        print("Warning: 'data' directory not found. Make sure it exists with images to search.")
    
    print("🚀 Starting Face Finder API...")

    print("🔄 POST /api/v0/embed - Pre-warm cache with image URLs")
    print("🔍 POST /api/v0/findIn - Find target in scope of images")
    print("💚 GET /api/health - Health check")
    print("📊 GET /api/cache/stats - Get cache statistics")
    print("🧹 POST /api/cache/clear - Clear all cached embeddings")
    print("🧹 POST /api/cache/cleanup - Remove invalid cache entries")
    app.run(host='0.0.0.0', port=5003, debug=True)
