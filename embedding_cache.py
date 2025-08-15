import os
import pickle
import hashlib
import json
import numpy as np
from datetime import datetime
import cv2
from insightface.app import FaceAnalysis
import re

class EmbeddingCache:
    """Cache system for face embeddings organized by basename to avoid recalculating them each time"""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        self.embeddings_dir = os.path.join(cache_dir, "embeddings")
        self.metadata_file = os.path.join(cache_dir, "metadata.json")
        self.app = None
        
        # Create cache directories
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Load or create metadata
        self.metadata = self._load_metadata()
        
        # Auto-migrate from old cache format if needed
        self.migrate_old_cache_format()
    
    def _extract_basename(self, file_path):
        """Extract basename from file path (e.g., demowki063.jpg -> demowki)"""
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        # Remove numbers to get the base name (e.g., demowki063 -> demowki)
        base_name = re.sub(r'\d+', '', base_name)
        return base_name
    
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
    
    def _get_basename_cache_file(self, basename):
        """Get cache file path for a basename"""
        return os.path.join(self.embeddings_dir, f"{basename}_embeddings.pkl")
    
    def _init_face_analysis(self):
        """Initialize face analysis app if not already done"""
        if self.app is None:
            print("Initializing face analysis for cache...")
            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=-1)  # Use CPU
    
    def is_cached(self, file_path):
        """Check if embeddings for this file are cached and valid"""
        basename = self._extract_basename(file_path)
        
        # Check if basename exists in metadata
        if basename not in self.metadata:
            return False
        
        basename_data = self.metadata[basename]
        filename = os.path.basename(file_path)
        
        # Check if this specific file is in the basename cache
        if filename not in basename_data.get('files', {}):
            return False
        
        file_data = basename_data['files'][filename]
        
        # Check if file still exists
        if not os.path.exists(file_path):
            return False
        
        # Check if file has been modified
        current_hash = self._get_file_hash(file_path)
        if current_hash != file_data.get('file_hash'):
            return False
        
        # Check if cache file exists
        cache_file = self._get_basename_cache_file(basename)
        if not os.path.exists(cache_file):
            return False
        
        return True
    
    def get_embeddings(self, file_path):
        """Get embeddings from cache"""
        if not self.is_cached(file_path):
            return None
        
        basename = self._extract_basename(file_path)
        filename = os.path.basename(file_path)
        cache_file = self._get_basename_cache_file(basename)
        
        try:
            with open(cache_file, 'rb') as f:
                basename_cache = pickle.load(f)
                return basename_cache.get(filename)
        except:
            return None
    
    def cache_embeddings(self, file_path, embeddings):
        """Cache embeddings for a file organized by basename"""
        basename = self._extract_basename(file_path)
        filename = os.path.basename(file_path)
        cache_file = self._get_basename_cache_file(basename)
        
        # Load existing basename cache or create new one
        basename_cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    basename_cache = pickle.load(f)
            except:
                basename_cache = {}
        
        # Add/update embeddings for this file
        basename_cache[filename] = embeddings
        
        # Save updated basename cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(basename_cache, f)
        except Exception as e:
            print(f"Warning: Failed to cache embeddings for {file_path}: {e}")
            return False
        
        # Update metadata
        file_hash = self._get_file_hash(file_path)
        
        if basename not in self.metadata:
            self.metadata[basename] = {
                'basename': basename,
                'cache_file': f"{basename}_embeddings.pkl",
                'files': {},
                'last_updated': datetime.now().isoformat()
            }
        
        self.metadata[basename]['files'][filename] = {
            'file_path': file_path,
            'file_hash': file_hash,
            'cached_at': datetime.now().isoformat(),
            'num_faces': len(embeddings) if embeddings else 0
        }
        self.metadata[basename]['last_updated'] = datetime.now().isoformat()
        
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
            'total_basenames': len(self.metadata),
            'total_files': 0,
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
        
        # Count total files and faces
        for basename_data in self.metadata.values():
            if 'files' in basename_data:
                stats['total_files'] += len(basename_data['files'])
                for file_data in basename_data['files'].values():
                    stats['total_faces'] += file_data.get('num_faces', 0)
        
        return stats
    
    def remove_invalid_cache_entries(self):
        """Remove cache entries for files that no longer exist or have been modified"""
        basenames_to_remove = []
        total_removed = 0
        
        for basename, basename_data in self.metadata.items():
            if 'files' not in basename_data:
                basenames_to_remove.append(basename)
                continue
                
            files_to_remove = []
            
            for filename, file_data in basename_data['files'].items():
                file_path = file_data['file_path']
                
                # Check if file exists and hasn't been modified
                if not os.path.exists(file_path) or self._get_file_hash(file_path) != file_data['file_hash']:
                    files_to_remove.append(filename)
                    total_removed += 1
            
            # Remove invalid files from this basename
            for filename in files_to_remove:
                del basename_data['files'][filename]
            
            # If no files remain for this basename, remove the basename entirely
            if not basename_data['files']:
                basenames_to_remove.append(basename)
                
                # Remove cache file
                cache_file = self._get_basename_cache_file(basename)
                if os.path.exists(cache_file):
                    try:
                        os.remove(cache_file)
                    except:
                        pass
            else:
                # Update basename cache file to remove invalid entries
                cache_file = self._get_basename_cache_file(basename)
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'rb') as f:
                            basename_cache = pickle.load(f)
                        
                        # Remove files that were marked as invalid
                        for filename in files_to_remove:
                            basename_cache.pop(filename, None)
                        
                        # Save updated cache
                        with open(cache_file, 'wb') as f:
                            pickle.dump(basename_cache, f)
                    except:
                        pass
        
        # Remove empty basenames from metadata
        for basename in basenames_to_remove:
            del self.metadata[basename]
        
        if total_removed > 0 or basenames_to_remove:
            self._save_metadata()
            print(f"Removed {total_removed} invalid cache entries across {len(basenames_to_remove)} basenames")
        
        return total_removed
    
    def _get_url_hash(self, url):
        """Calculate SHA256 hash of URL for unique identification"""
        return hashlib.sha256(url.encode()).hexdigest()[:16]
    
    def _get_url_cache_file(self, url):
        """Get cache file path for a URL"""
        url_hash = self._get_url_hash(url)
        return os.path.join(self.embeddings_dir, f"temp_reference_{url_hash}_embeddings.pkl")
    
    def is_url_cached(self, url):
        """Check if embeddings for this URL are cached"""
        cache_file = self._get_url_cache_file(url)
        return os.path.exists(cache_file)
    
    def get_url_embeddings(self, url):
        """Get embeddings from cache for a URL"""
        cache_file = self._get_url_cache_file(url)
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    
    def cache_url_embeddings(self, url, embeddings):
        """Cache embeddings for a URL"""
        cache_file = self._get_url_cache_file(url)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            return True
        except Exception as e:
            print(f"Warning: Failed to cache embeddings for URL {url}: {e}")
            return False
    
    def _get_url_faces_cache_file(self, url):
        """Get cache file path for URL face metadata (embeddings + bbox, scores)"""
        url_hash = self._get_url_hash(url)
        return os.path.join(self.embeddings_dir, f"temp_reference_{url_hash}_faces.pkl")
    
    def get_url_faces(self, url):
        """Get face metadata list from cache for a URL. Fallback to embeddings-only cache if needed."""
        faces_cache_file = self._get_url_faces_cache_file(url)
        if os.path.exists(faces_cache_file):
            try:
                with open(faces_cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        # Fallback: load embeddings-only cache and wrap as face dicts (no bbox available)
        embeddings = self.get_url_embeddings(url)
        if embeddings is None:
            return None
        
        faces = []
        try:
            for emb in embeddings:
                faces.append({
                    'embedding': emb,
                    'bbox': None,
                    'kps': None,
                    'det_score': None
                })
            return faces
        except Exception as e:
            print(f"Warning: Failed to adapt embeddings to faces for URL {url}: {e}")
            return None
    
    def cache_url_faces(self, url, faces):
        """Cache face metadata list for a URL"""
        faces_cache_file = self._get_url_faces_cache_file(url)
        try:
            with open(faces_cache_file, 'wb') as f:
                pickle.dump(faces, f)
            return True
        except Exception as e:
            print(f"Warning: Failed to cache face metadata for URL {url}: {e}")
            return False
    
    def get_or_compute_url_embeddings(self, url, image_path):
        """Get embeddings from cache or compute them for a URL"""
        # Try to get from cache first
        embeddings = self.get_url_embeddings(url)
        if embeddings is not None:
            return embeddings
        
        # Not in cache, compute embeddings
        self._init_face_analysis()
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not load image from {image_path}")
                return []
            
            faces = self.app.get(img)
            embeddings = [face.normed_embedding for face in faces]
            
            # Cache the results
            self.cache_url_embeddings(url, embeddings)
            
            return embeddings
            
        except Exception as e:
            print(f"Error computing embeddings for URL {url}: {e}")
            return []
    
    def get_or_compute_url_faces(self, url, image_path):
        """Get face metadata list (embedding + bbox + score) from cache or compute for a URL.
        Also ensures embeddings-only cache is created for compatibility.
        """
        # Try to get faces from cache first
        faces = self.get_url_faces(url)
        if faces is not None:
            return faces
        
        # Not in cache, compute
        self._init_face_analysis()
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not load image from {image_path}")
                return []
            
            detected_faces = self.app.get(img)
            faces = []
            embeddings = []
            for face in detected_faces:
                try:
                    bbox = face.bbox.tolist() if hasattr(face, 'bbox') else None
                except Exception:
                    bbox = None
                try:
                    kps = face.kps.tolist() if hasattr(face, 'kps') else None
                except Exception:
                    kps = None
                try:
                    det_score = float(face.det_score) if hasattr(face, 'det_score') else None
                except Exception:
                    det_score = None
                emb = face.normed_embedding
                faces.append({
                    'embedding': emb,
                    'bbox': bbox,
                    'kps': kps,
                    'det_score': det_score
                })
                embeddings.append(emb)
            
            # Cache both faces metadata and plain embeddings for compatibility
            self.cache_url_faces(url, faces)
            # Only write embeddings cache if not already present
            if not self.is_url_cached(url):
                self.cache_url_embeddings(url, embeddings)
            
            return faces
        except Exception as e:
            print(f"Error computing face metadata for URL {url}: {e}")
            return []
    
    def migrate_old_cache_format(self):
        """Migrate from old hash-based cache format to new basename format"""
        print("Checking for old cache format to migrate...")
        
        # Check if we have old-style metadata (hash keys instead of basenames)
        old_entries = []
        for key, value in self.metadata.items():
            # Old format has hash keys and 'file_path' directly in the value
            if isinstance(value, dict) and 'file_path' in value and 'files' not in value:
                old_entries.append((key, value))
        
        if not old_entries:
            print("No old cache format detected.")
            return 0
        
        print(f"Found {len(old_entries)} old cache entries to migrate...")
        migrated_count = 0
        
        # Group old entries by basename
        basename_groups = {}
        for hash_key, old_data in old_entries:
            file_path = old_data['file_path']
            if not os.path.exists(file_path):
                continue
                
            basename = self._extract_basename(file_path)
            filename = os.path.basename(file_path)
            
            if basename not in basename_groups:
                basename_groups[basename] = {}
            
            # Load embeddings from old cache file
            old_cache_file = os.path.join(self.embeddings_dir, f"{hash_key}.pkl")
            if os.path.exists(old_cache_file):
                try:
                    with open(old_cache_file, 'rb') as f:
                        embeddings = pickle.load(f)
                    basename_groups[basename][filename] = {
                        'embeddings': embeddings,
                        'metadata': old_data
                    }
                    migrated_count += 1
                except:
                    continue
        
        # Create new basename-based cache files
        for basename, files_data in basename_groups.items():
            # Create new cache file for this basename
            cache_file = self._get_basename_cache_file(basename)
            basename_cache = {}
            
            # Create new metadata structure
            self.metadata[basename] = {
                'basename': basename,
                'cache_file': f"{basename}_embeddings.pkl",
                'files': {},
                'last_updated': datetime.now().isoformat()
            }
            
            for filename, data in files_data.items():
                embeddings = data['embeddings']
                old_metadata = data['metadata']
                
                # Store embeddings in basename cache
                basename_cache[filename] = embeddings
                
                # Store file metadata
                self.metadata[basename]['files'][filename] = {
                    'file_path': old_metadata['file_path'],
                    'file_hash': old_metadata['file_hash'],
                    'cached_at': old_metadata.get('cached_at', datetime.now().isoformat()),
                    'num_faces': old_metadata.get('num_faces', len(embeddings) if embeddings else 0)
                }
            
            # Save new basename cache file
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(basename_cache, f)
            except Exception as e:
                print(f"Warning: Failed to create new cache file for {basename}: {e}")
                continue
        
        # Remove old entries from metadata and delete old cache files
        for hash_key, old_data in old_entries:
            # Remove old metadata entry
            if hash_key in self.metadata:
                del self.metadata[hash_key]
            
            # Remove old cache file
            old_cache_file = os.path.join(self.embeddings_dir, f"{hash_key}.pkl")
            if os.path.exists(old_cache_file):
                try:
                    os.remove(old_cache_file)
                except:
                    pass
        
        # Save updated metadata
        self._save_metadata()
        
        print(f"Migration completed! Migrated {migrated_count} cache entries to basename format.")
        return migrated_count
