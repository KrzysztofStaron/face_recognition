import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
import os
import time

def cosine_similarity(a, b):
    """Calculate cosine similarity between two face embeddings"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_reference_faces(app):
    """Load and encode reference faces from person3.jpg and person4.png"""
    reference_faces = {}
    
    # Load person3.jpg
    if os.path.exists("person3.jpg"):
        img3 = cv2.imread("person3.jpg")
        faces3 = app.get(img3)
        if faces3:
            reference_faces["person3"] = faces3[0].normed_embedding
            print("✓ Loaded person3.jpg")
        else:
            print("⚠ No face detected in person3.jpg")
    else:
        print("⚠ person3.jpg not found")
    
    # Load person4.png
    if os.path.exists("person4.png"):
        img4 = cv2.imread("person4.png")
        faces4 = app.get(img4)
        if faces4:
            reference_faces["person4"] = faces4[0].normed_embedding
            print("✓ Loaded person4.png")
        else:
            print("⚠ No face detected in person4.png")
    else:
        print("⚠ person4.png not found")
    
    return reference_faces

def capture_and_compare():
    """Capture photo from webcam and compare with reference faces"""
    # Initialize face analysis
    print("Initializing face analysis...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1)  # Use CPU (-1), change to 0 for GPU
    
    # Load reference faces
    print("Loading reference faces...")
    reference_faces = load_reference_faces(app)
    
    if not reference_faces:
        print("❌ No reference faces loaded. Please ensure person3.jpg and/or person4.png exist.")
        return
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("📷 Webcam ready! Press SPACE to capture photo, ESC to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        # Display the frame
        cv2.imshow('Webcam - Press SPACE to capture, ESC to exit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC key
            print("Exiting...")
            break
        elif key == 32:  # SPACE key
            print("📸 Capturing photo...")
            
            # Analyze the captured frame
            faces = app.get(frame)
            
            if not faces:
                print("❌ No face detected in captured image")
                continue
            
            # Get the first detected face embedding
            captured_embedding = faces[0].normed_embedding
            
            print(f"\n🔍 Face detected! Comparing with reference faces...")
            print("-" * 50)
            
            best_match = None
            best_similarity = -1
            
            # Compare with each reference face
            for person_name, ref_embedding in reference_faces.items():
                similarity = cosine_similarity(captured_embedding, ref_embedding)
                print(f"{person_name}: {similarity:.4f}")
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_name
            
            print("-" * 50)
            
            # Determine if it's a match (threshold of 0.6 is commonly used)
            threshold = 0.6
            if best_similarity > threshold:
                print(f"✅ MATCH FOUND: {best_match} (similarity: {best_similarity:.4f})")
            else:
                print(f"❌ NO MATCH: Best similarity was {best_match} with {best_similarity:.4f} (threshold: {threshold})")
            
            print(f"\nPress SPACE to capture again, ESC to exit")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_compare()
