from flask import Flask, request, jsonify
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

@app.route('/api/findAll', methods=['POST'])
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

if __name__ == '__main__':
    # Ensure data directory exists
    if not os.path.exists('data'):
        print("Warning: 'data' directory not found. Make sure it exists with images to search.")
    
    print("🚀 Starting Face Finder API...")
    print("📍 POST /api/findAll - Find matching faces")
    print("💚 GET /api/health - Health check")
    print("📊 GET /api/cache/stats - Get cache statistics")
    print("🧹 POST /api/cache/clear - Clear all cached embeddings")
    print("🧹 POST /api/cache/cleanup - Remove invalid cache entries")
    app.run(host='0.0.0.0', port=5000, debug=True)
