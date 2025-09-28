#!/usr/bin/env python3
"""
Test script to verify that LLM provider tracking is working in the classification results
"""

import requests
import json

def test_provider_tracking():
    """Test that the API returns provider information in results."""
    
    print("🧪 Testing LLM Provider Tracking in Classification Results")
    print("=" * 60)
    
    # Test text
    test_text = """
    Dear Contractor,
    There has been a delay in the project timeline due to weather conditions.
    The completion date needs to be revised from December 2024 to January 2025.
    Please provide an updated schedule showing the revised milestones.
    """
    
    # Test with both approaches
    approaches = ["pure_llm", "hybrid_rag"]
    
    for approach in approaches:
        print(f"\n🔍 Testing {approach.upper()} approach...")
        
        payload = {
            "text": test_text,
            "approach": approach,
            "confidence_threshold": 0.3
        }
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/classify", 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {approach} classification successful")
                print(f"   Status: {result.get('status', 'unknown')}")
                print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
                
                # Check for provider information
                llm_provider = result.get('llm_provider_used')
                if llm_provider:
                    print(f"   🤖 LLM Provider used: {llm_provider.upper()}")
                else:
                    print(f"   ⚠️  No LLM provider information found")
                
                # Check categories
                categories = result.get('categories', [])
                print(f"   📊 Categories found: {len(categories)}")
                if categories:
                    for cat in categories[:2]:
                        print(f"      - {cat.get('category', 'Unknown')}: {cat.get('confidence', 0):.3f}")
                        
            else:
                print(f"❌ {approach} failed with status {response.status_code}")
                print(f"   Error: {response.text[:200]}")
                
        except Exception as e:
            print(f"❌ {approach} request failed: {e}")
    
    print(f"\n🎉 Provider tracking test completed!")

if __name__ == "__main__":
    test_provider_tracking()