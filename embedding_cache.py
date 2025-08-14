import os
import pickle
import hashlib
import json
import numpy as np
from datetime import datetime
import cv2
from insightface.app import FaceAnalysis

class EmbeddingCache:
    """Cache system for face embeddings to avoid recalculating them each time"""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        self.embeddings_dir = os.path.join(cache_dir, "embeddings")
        self.metadata_file = os.path.join(cache_dir, "metadata.json")
        self.app = None
        
        # Create cache directories
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Load or create metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata or create empty if doesn't exist"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to disk"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_file_hash(self, file_path):
        """Calculate MD5 hash of file to detect changes"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None
    
    def _get_cache_key(self, file_path):
        """Generate cache key from file path"""
        return hashlib.md5(file_path.encode()).hexdigest()
    
    def _init_face_analysis(self):
        """Initialize face analysis app if not already done"""
        if self.app is None:
            print("Initializing face analysis for cache...")
            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=-1)  # Use CPU
    
    def is_cached(self, file_path):
        """Check if embeddings for this file are cached and valid"""
        cache_key = self._get_cache_key(file_path)
        
        if cache_key not in self.metadata:
            return False
        
        # Check if file still exists
        if not os.path.exists(file_path):
            return False
        
        # Check if file has been modified
        current_hash = self._get_file_hash(file_path)
        if current_hash != self.metadata[cache_key].get('file_hash'):
            return False
        
        # Check if cache file exists
        cache_file = os.path.join(self.embeddings_dir, f"{cache_key}.pkl")
        if not os.path.exists(cache_file):
            return False
        
        return True
    
    def get_embeddings(self, file_path):
        """Get embeddings from cache"""
        if not self.is_cached(file_path):
            return None
        
        cache_key = self._get_cache_key(file_path)
        cache_file = os.path.join(self.embeddings_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    
    def cache_embeddings(self, file_path, embeddings):
        """Cache embeddings for a file"""
        cache_key = self._get_cache_key(file_path)
        cache_file = os.path.join(self.embeddings_dir, f"{cache_key}.pkl")
        
        # Save embeddings
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
        except Exception as e:
            print(f"Warning: Failed to cache embeddings for {file_path}: {e}")
            return False
        
        # Update metadata
        file_hash = self._get_file_hash(file_path)
        self.metadata[cache_key] = {
            'file_path': file_path,
            'file_hash': file_hash,
            'cached_at': datetime.now().isoformat(),
            'num_faces': len(embeddings) if embeddings else 0
        }
        self._save_metadata()
        return True
    
    def get_or_compute_embeddings(self, file_path):
        """Get embeddings from cache or compute them if not cached"""
        # Try to get from cache first
        embeddings = self.get_embeddings(file_path)
        if embeddings is not None:
            return embeddings
        
        # Not in cache, compute embeddings
        self._init_face_analysis()
        
        try:
            img = cv2.imread(file_path)
            if img is None:
                print(f"Warning: Could not load image {file_path}")
                return []
            
            faces = self.app.get(img)
            embeddings = [face.normed_embedding for face in faces]
            
            # Cache the results
            self.cache_embeddings(file_path, embeddings)
            
            return embeddings
            
        except Exception as e:
            print(f"Error computing embeddings for {file_path}: {e}")
            return []
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            os.makedirs(self.embeddings_dir, exist_ok=True)
            self.metadata = {}
            self._save_metadata()
            print("Cache cleared successfully")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    
    def get_cache_stats(self):
        """Get statistics about the cache"""
        stats = {
            'total_files': len(self.metadata),
            'cache_size_mb': 0,
            'total_faces': 0
        }
        
        # Calculate cache size
        try:
            if os.path.exists(self.embeddings_dir):
                for filename in os.listdir(self.embeddings_dir):
                    file_path = os.path.join(self.embeddings_dir, filename)
                    if os.path.isfile(file_path):
                        stats['cache_size_mb'] += os.path.getsize(file_path)
                stats['cache_size_mb'] = round(stats['cache_size_mb'] / (1024 * 1024), 2)
        except:
            pass
        
        # Count total faces
        for metadata in self.metadata.values():
            stats['total_faces'] += metadata.get('num_faces', 0)
        
        return stats
    
    def remove_invalid_cache_entries(self):
        """Remove cache entries for files that no longer exist or have been modified"""
        invalid_keys = []
        
        for cache_key, metadata in self.metadata.items():
            file_path = metadata['file_path']
            
            # Check if file exists and hasn't been modified
            if not os.path.exists(file_path) or self._get_file_hash(file_path) != metadata['file_hash']:
                invalid_keys.append(cache_key)
                
                # Remove cache file
                cache_file = os.path.join(self.embeddings_dir, f"{cache_key}.pkl")
                if os.path.exists(cache_file):
                    try:
                        os.remove(cache_file)
                    except:
                        pass
        
        # Remove from metadata
        for key in invalid_keys:
            del self.metadata[key]
        
        if invalid_keys:
            self._save_metadata()
            print(f"Removed {len(invalid_keys)} invalid cache entries")
        
        return len(invalid_keys)
