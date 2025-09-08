#!/usr/bin/env python3
"""
Quick API test to verify the production system is working
"""

import requests
import json

def test_api():
    """Test the running API with a simple classification request."""
    print("🧪 Testing Contract Correspondence Classification API")
    print("=" * 60)
    
    base_url = "http://127.0.0.1:8000"
    
    # Test classification endpoint
    print("📝 Testing classification...")
    
    test_text = "Request for extension of time due to weather delays in highway construction project. The contractor requests 45 days additional time for completion."
    
    payload = {
        "text": test_text,
        "approach": "hybrid_rag",
        "confidence_threshold": 0.5,
        "max_results": 5
    }
    
    try:
        response = requests.post(f"{base_url}/classify", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Classification successful!")
            print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"   Approach used: {result.get('approach_used')}")
            print(f"   Confidence score: {result.get('confidence_score', 0):.3f}")
            
            issues = result.get('identified_issues', [])
            if issues:
                print("   🔍 Top Issues:")
                for issue in issues[:3]:
                    print(f"     • {issue['issue_type']} (confidence: {issue['confidence']:.3f})")
            
            categories = result.get('categories', [])
            if categories:
                print("   📂 Categories:")
                for cat in categories[:3]:
                    print(f"     • {cat['category']} (confidence: {cat['confidence']:.3f})")
            
            print("\n🎉 API is working perfectly!")
            return True
            
        else:
            print(f"❌ Classification failed with status {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    if success:
        print("\n" + "=" * 60)
        print("🚀 PRODUCTION API IS RUNNING SUCCESSFULLY!")
        print("=" * 60)
        print("\n🌐 Access your API at:")
        print("   • Main API: http://127.0.0.1:8000")
        print("   • Documentation: http://127.0.0.1:8000/docs")
        print("   • Test endpoint: http://127.0.0.1:8000/classify")
        print("\n📋 Ready for production use!")
    else:
        print("\n❌ API test failed. Check the server logs.")