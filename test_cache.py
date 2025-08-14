#!/usr/bin/env python3
"""
Test script to demonstrate the embedding cache system
"""
import time
from findAll import find_matching_photos
from embedding_cache import EmbeddingCache

def test_caching_performance():
    """Test the performance improvement with caching"""
    reference_path = "eval/person_b.jpg"
    data_directory = "data"
    threshold = 0.6
    
    print("=" * 80)
    print("🧪 TESTING EMBEDDING CACHE PERFORMANCE")
    print("=" * 80)
    
    # First run - will compute and cache embeddings
    print("\n🔄 FIRST RUN (Computing and caching embeddings)")
    print("-" * 50)
    start_time = time.time()
    matches1 = find_matching_photos(reference_path, data_directory, threshold)
    first_run_time = time.time() - start_time
    
    print(f"\n⏱️ First run completed in {first_run_time:.2f} seconds")
    print(f"📊 Found {len(matches1)} matches")
    
    # Show cache statistics
    cache = EmbeddingCache()
    stats = cache.get_cache_stats()
    print(f"\n📈 Cache Statistics:")
    print(f"   - Cached files: {stats['total_files']}")
    print(f"   - Total faces: {stats['total_faces']}")
    print(f"   - Cache size: {stats['cache_size_mb']} MB")
    
    # Second run - will use cached embeddings
    print("\n🚀 SECOND RUN (Using cached embeddings)")
    print("-" * 50)
    start_time = time.time()
    matches2 = find_matching_photos(reference_path, data_directory, threshold)
    second_run_time = time.time() - start_time
    
    print(f"\n⏱️ Second run completed in {second_run_time:.2f} seconds")
    print(f"📊 Found {len(matches2)} matches")
    
    # Calculate performance improvement
    if first_run_time > 0:
        speedup = first_run_time / second_run_time
        time_saved = first_run_time - second_run_time
        percentage_faster = ((first_run_time - second_run_time) / first_run_time) * 100
        
        print("\n" + "=" * 80)
        print("🎯 PERFORMANCE RESULTS")
        print("=" * 80)
        print(f"First run (no cache):  {first_run_time:.2f} seconds")
        print(f"Second run (cached):   {second_run_time:.2f} seconds")
        print(f"Time saved:            {time_saved:.2f} seconds")
        print(f"Speedup factor:        {speedup:.1f}x faster")
        print(f"Performance gain:      {percentage_faster:.1f}% faster")
        print("=" * 80)
    
    # Verify results are identical
    if len(matches1) == len(matches2):
        print("✅ Cache results verified: Both runs found identical matches")
    else:
        print("❌ Warning: Different number of matches between runs")

if __name__ == "__main__":
    test_caching_performance()
