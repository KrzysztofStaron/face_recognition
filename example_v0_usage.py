#!/usr/bin/env python3
"""
Example usage of the Face Finder API v0 endpoints
"""

import requests
import json
import time

BASE_URL = "http://localhost:5003"

def example_pre_warm_cache():
    """Example: Pre-warm cache with multiple images"""
    print("Example 1: Pre-warming cache with multiple images")
    print("-" * 50)
    
    # List of images to cache
    images_to_cache = [
        "https://klient.fotoklaser.pl/download.php?mode=api_preview&access=oGywJNAeoELTy4k_2_KE&file=demowki001.jpg",
        "https://klient.fotoklaser.pl/download.php?mode=api_preview&access=oGywJNAeoELTy4k_2_KE&file=demowki010.jpg",
        "https://klient.fotoklaser.pl/download.php?mode=api_preview&access=oGywJNAeoELTy4k_2_KE&file=demowki020.jpg",
        "https://klient.fotoklaser.pl/download.php?mode=api_preview&access=oGywJNAeoELTy4k_2_KE&file=demowki030.jpg",
        "https://klient.fotoklaser.pl/download.php?mode=api_preview&access=oGywJNAeoELTy4k_2_KE&file=demowki040.jpg"
    ]
    
    # Call the embed endpoint
    response = requests.post(
        f"{BASE_URL}/api/v0/embed",
        json={"urls": images_to_cache}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Successfully cached {result['total_urls']} images")
        for item in result['results']:
            status = "✓" if item['success'] else "✗"
            faces = item.get('num_faces', 0)
            print(f"  {status} {item['url'].split('file=')[-1]} - {faces} face(s)")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())
    
    print()

def example_find_person():
    """Example: Find a specific person in multiple images"""
    print("Example 2: Finding a person in multiple images")
    print("-" * 50)
    
    # The person we want to find
    target_person = "https://klient.fotoklaser.pl/download.php?mode=api_preview&access=oGywJNAeoELTy4k_2_KE&file=demowki089.jpg"
    
    # Images to search in
    search_scope = [
        f"https://klient.fotoklaser.pl/download.php?mode=api_preview&access=oGywJNAeoELTy4k_2_KE&file=demowki{i:03d}.jpg"
        for i in range(1, 141)
    ]
    
    # Call the findIn endpoint
    response = requests.post(
        f"{BASE_URL}/api/v0/findIn",
        json={
            "target": target_person,
            "scope": search_scope,
            "threshold": 0.4
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Search completed!")
        print(f"   Target: {target_person.split('file=')[-1]}")
        print(f"   Searched {result['total_scope_images']} images")
        print(f"   Found {result['total_matches']} matches with threshold {result['threshold']}")
        
        if result['matches']:
            print("\n   Matches:")
            for match in result['matches']:
                filename = match['url'].split('file=')[-1]
                print(f"   • {filename} - similarity: {match['similarity']}")
                if result.get('target_faces_count', 1) > 1:
                    print(f"     Found {match.get('target_faces_found', 1)} of {result['target_faces_count']} target faces")
                if match['matching_faces'] > 1:
                    print(f"     {match['matching_faces']} total face matches in this image")
                
                # Show detailed matches if available
                if 'detailed_matches' in match and len(match['detailed_matches']) > 0:
                    print(f"     Details:")
                    for detail in match['detailed_matches']:
                        print(f"       Target face {detail['target_face']} → Scope face {detail['scope_face']} (similarity: {detail['similarity']})")
        else:
            print("\n   No matches found")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())
    
    print()

def example_batch_search():
    """Example: Search for multiple people in a batch"""
    print("Example 3: Batch search for multiple people")
    print("-" * 50)
    
    # Multiple people to search for
    targets = [
        "https://klient.fotoklaser.pl/download.php?mode=api_preview&access=oGywJNAeoELTy4k_2_KE&file=demowki001.jpg",
        "https://klient.fotoklaser.pl/download.php?mode=api_preview&access=oGywJNAeoELTy4k_2_KE&file=demowki005.jpg"
    ]
    
    # Common search scope - all images from 001 to 140
    search_scope = [
        f"https://klient.fotoklaser.pl/download.php?mode=api_preview&access=oGywJNAeoELTy4k_2_KE&file=demowki{i:03d}.jpg"
        for i in range(1, 141)  # Search in images 001-140
    ]
    
    # First, pre-warm the cache for better performance
    print(f"Pre-warming cache for batch search with {len(search_scope)} images...")
    print("This may take a while for 140 images...")
    
    # Split into batches for better handling
    batch_size = 20
    for i in range(0, len(search_scope), batch_size):
        batch = search_scope[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(search_scope) + batch_size - 1)//batch_size} ({len(batch)} images)...")
        response = requests.post(f"{BASE_URL}/api/v0/embed", json={"urls": batch})
        if response.status_code != 200:
            print(f"Warning: Batch {i//batch_size + 1} failed")
    
    print("Cache pre-warming completed!")
    
    # Search for each target
    for target in targets:
        print(f"\nSearching for {target.split('file=')[-1]}...")
        
        response = requests.post(
            f"{BASE_URL}/api/v0/findIn",
            json={
                "target": target,
                "scope": search_scope,
                "threshold": 0.65
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  Found in {result['total_matches']} images:")
            for match in result['matches'][:5]:  # Show top 5 matches
                filename = match['url'].split('file=')[-1]
                print(f"  • {filename} (similarity: {match['similarity']})")
            if result['total_matches'] > 5:
                print(f"  ... and {result['total_matches'] - 5} more")
    
    print()

if __name__ == "__main__":
    print("Face Finder API v0 - Usage Examples")
    print("=" * 50)
    print()
    
    # Run examples
    example_pre_warm_cache()
    time.sleep(1)  # Small delay between examples
    
    example_find_person()
    time.sleep(1)
    
    example_batch_search()
    
    print("\nDone! Check the cache directory to see the saved embeddings.")
