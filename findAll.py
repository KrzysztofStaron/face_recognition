import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
import os
import glob

def cosine_similarity(a, b):
    """Calculate cosine similarity between two face embeddings"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_reference_face(app, reference_path):
    """Load and encode reference face from the given image path"""
    if not os.path.exists(reference_path):
        print(f"❌ Reference image not found: {reference_path}")
        return None
    
    img = cv2.imread(reference_path)
    if img is None:
        print(f"❌ Could not load image: {reference_path}")
        return None
    
    faces = app.get(img)
    if not faces:
        print(f"❌ No face detected in reference image: {reference_path}")
        return None
    
    print(f"✓ Loaded reference face from {reference_path}")
    return faces[0].normed_embedding

def find_matching_photos(reference_path, data_directory, threshold=0.6):
    """Find all photos in data directory where the reference person is present"""
    
    # Initialize face analysis
    print("Initializing face analysis...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1)  # Use CPU (-1), change to 0 for GPU
    
    # Load reference face
    print(f"Loading reference face from {reference_path}...")
    reference_embedding = load_reference_face(app, reference_path)
    
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
            # Load and analyze the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"   ⚠ Could not load image: {image_path}")
                continue
                
            faces = app.get(img)
            
            if not faces:
                print(f"   ⚠ No faces detected in {os.path.basename(image_path)}")
                continue
            
            # Check each detected face against the reference
            best_similarity = -1
            matching_faces = []
            
            # print(f"   📊 Found {len(faces)} face(s) in image")
            
            for j, face in enumerate(faces):
                similarity = cosine_similarity(reference_embedding, face.normed_embedding)
                print(f"      Face {j+1}: similarity = {similarity:.4f}")
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    
                if similarity >= threshold:
                    matching_faces.append(similarity)
            
            if matching_faces:
                matches.append({
                    'filename': os.path.basename(image_path),
                    'path': image_path,
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

if __name__ == "__main__":
    main()