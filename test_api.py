#!/usr/bin/env python3
"""
Test script for the Face Finder API
"""

import requests
import json

def test_find_all_api():
    """Test the /api/findAll endpoint"""
    
    # API endpoint
    url = "http://localhost:5000/api/findAll"
    
    # Test data - replace with actual image URL
    test_data = {
        "url": "https://example.com/child_photo.jpg",  # Replace with actual URL
        "threshold": 0.6,  # Optional: similarity threshold (0.0 to 1.0)
        "data_directory": "data"  # Optional: directory to search in
    }
    
    try:
        print("🔍 Testing Face Finder API...")
        print(f"📡 Sending POST request to: {url}")
        print(f"📝 Payload: {json.dumps(test_data, indent=2)}")
        print("-" * 50)
        
        # Send POST request
        response = requests.post(url, json=test_data, timeout=60)
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📄 Response Body:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"\n✅ Success! Found {result['total_matches']} matching photos")
                for i, match in enumerate(result['matches'], 1):
                    print(f"  {i}. {match['filename']} (similarity: {match['similarity']})")
            else:
                print(f"\n❌ API returned error: {result.get('error', 'Unknown error')}")
        else:
            print(f"\n❌ HTTP Error: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        print("\n💡 Make sure the API server is running:")
        print("   python main.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        print(f"💚 Health Check: {response.json()}")
        return response.status_code == 200
    except:
        print("❌ Health check failed - API server not running")
        return False

if __name__ == "__main__":
    print("🧪 Face Finder API Test Script")
    print("=" * 50)
    
    # First check if API is running
    if test_health_check():
        print("\n" + "=" * 50)
        test_find_all_api()
    else:
        print("\n💡 Start the API server first:")
        print("   python main.py")
        print("\n📝 Then update the test_data['url'] in this script with a real image URL")
