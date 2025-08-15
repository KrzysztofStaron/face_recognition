import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
import os
import glob
import re
from embedding_cache import EmbeddingCache

# Extract the base name of the image from the url
def parse_image_url(url):
    #https://klient.fotoklaser.pl/download.php?mode=api_preview&access=oGywJNAeoELTy4k_2_KE&file=demowki083.jpg

    file_name = url.split("file=")[1]
    base_name = os.path.splitext(file_name)[0]
    base_name = re.sub(r'\d+', '', base_name)
    return base_name

def cosine_similarity(a, b):
    """Calculate cosine similarity between two face embeddings"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_reference_face(cache, reference_path):
    """Load and encode reference face from the given image path using cache"""
    if not os.path.exists(reference_path):
        print(f"❌ Reference image not found: {reference_path}")
        return None
    
    # Try to get embeddings from cache first
    embeddings = cache.get_or_compute_embeddings(reference_path)
    
    if not embeddings:
        print(f"❌ No face detected in reference image: {reference_path}")
        return None
    
    print(f"✓ Loaded reference face from {reference_path}")
    return embeddings[0]  # Return the first face embedding

def find_matching_photos(reference_path, data_directory, threshold=0.6):
    """Find all photos in data directory where the reference person is present"""
    
    # Initialize cache system
    print("Initializing embedding cache...")
    cache = EmbeddingCache()
    
    # Load reference face
    print(f"Loading reference face from {reference_path}...")
    reference_embedding = load_reference_face(cache, reference_path)
    
    if reference_embedding is None:
        return []
    
    # Get all jpg files in data directory
    data_pattern = os.path.join(data_directory, "*.jpg")
    image_files = glob.glob(data_pattern)
    
    if not image_files:
        print(f"❌ No .jpg files found in {data_directory}")
        return []
    
    print(f"📂 Found {len(image_files)} images to analyze in {data_directory}")
    
    matches = []
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"{i}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            # Get embeddings from cache or compute them
            face_embeddings = cache.get_or_compute_embeddings(image_path)
            
            if not face_embeddings:
                print(f"   ⚠ No faces detected in {os.path.basename(image_path)}")
                continue
            
            # Check each detected face against the reference
            best_similarity = -1
            matching_faces = []
            
            # print(f"   📊 Found {len(face_embeddings)} face(s) in image")
            
            for j, face_embedding in enumerate(face_embeddings):
                similarity = cosine_similarity(reference_embedding, face_embedding)
                print(f"      Face {j+1}: similarity = {similarity:.4f}")
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    
                if similarity >= threshold:
                    matching_faces.append(similarity)
            
            if matching_faces:
                matches.append({
                    'filename': os.path.basename(image_path),
                    'path': f"https://klient.fotoklaser.pl/download.php?mode=api_preview&access=oGywJNAeoELTy4k_2_KE&file={os.path.basename(image_path)}",
                    'similarity': best_similarity,
                    'matching_faces': len(matching_faces),
                    'all_similarities': matching_faces
                })
                if len(matching_faces) == 1:
                    print(f"   ✅ MATCH FOUND! 1 matching face with similarity: {best_similarity:.4f}")
                else:
                    print(f"   ✅ MATCH FOUND! {len(matching_faces)} matching faces (best: {best_similarity:.4f})")
            else:
                # print(f"   ❌ No match (best similarity: {best_similarity:.4f})")
                pass
                
        except Exception as e:
            print(f"   ❌ Error processing {image_path}: {str(e)}")
            continue
    
    return matches

def main():
    reference_path = "eval/person_b.jpg"
    data_directory = "data"
    threshold = 0.6  # Similarity threshold for matching
    
    print("=" * 60)
    print("🔍 FACE MATCHING ANALYSIS")
    print("=" * 60)
    print(f"Reference image: {reference_path}")
    print(f"Search directory: {data_directory}")
    print(f"Similarity threshold: {threshold}")
    print("=" * 60)
    
    # Find matching photos
    matches = find_matching_photos(reference_path, data_directory, threshold)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    
    if matches:
        print(f"✅ Found {len(matches)} matching photos:")
        print()
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        for i, match in enumerate(matches, 1):
            if match['matching_faces'] == 1:
                print(f"{i:2d}. {match['filename']:20s} (similarity: {match['similarity']:.4f})")
            else:
                print(f"{i:2d}. {match['filename']:20s} (best: {match['similarity']:.4f}, {match['matching_faces']} matching faces)")
        
        print()
        print("📝 Detailed results:")
        for match in matches:
            if match['matching_faces'] == 1:
                print(f"   {match['path']} - Single match: {match['similarity']:.4f}")
            else:
                similarities_str = ", ".join([f"{s:.4f}" for s in match['all_similarities']])
                print(f"   {match['path']} - Group photo: {match['matching_faces']} matches [{similarities_str}]")
            
    else:
        print("❌ No matching photos found.")
    
    print("\n" + "=" * 60)

def clear_cache():
    """Clear all cached embeddings"""
    cache = EmbeddingCache()
    cache.clear_cache()

def cache_stats():
    """Show cache statistics"""
    cache = EmbeddingCache()
    stats = cache.get_cache_stats()
    
    print("=" * 60)
    print("📊 EMBEDDING CACHE STATISTICS")
    print("=" * 60)
    print(f"Total basenames: {stats.get('total_basenames', 'N/A')}")
    print(f"Total cached files: {stats['total_files']}")
    print(f"Total faces cached: {stats['total_faces']}")
    print(f"Cache size: {stats['cache_size_mb']} MB")
    print("=" * 60)

def cleanup_cache():
    """Remove invalid cache entries"""
    cache = EmbeddingCache()
    removed = cache.remove_invalid_cache_entries()
    print(f"Cleaned up {removed} invalid cache entries")

def migrate_cache():
    """Migrate cache to new basename format"""
    cache = EmbeddingCache()
    migrated = cache.migrate_old_cache_format()
    print(f"Migration completed: {migrated} entries migrated")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "clear-cache":
            clear_cache()
        elif command == "cache-stats":
            cache_stats()
        elif command == "cleanup-cache":
            cleanup_cache()
        elif command == "migrate-cache":
            migrate_cache()
        else:
            print("Unknown command. Available commands:")
            print("  python findAll.py clear-cache    - Clear all cached embeddings")
            print("  python findAll.py cache-stats    - Show cache statistics")
            print("  python findAll.py cleanup-cache  - Remove invalid cache entries")
            print("  python findAll.py migrate-cache  - Migrate to new basename cache format")
    else:
        main()