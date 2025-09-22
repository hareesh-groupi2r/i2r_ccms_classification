#!/usr/bin/env python3
"""
Simple test for Gemini API after enabling
"""

import os
import google.generativeai as genai

def test_gemini():
    """Test Gemini API with simple call"""
    
    print("🧪 Testing Gemini API")
    print("=" * 30)
    
    # Configure with API key only
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY not found")
        return False
    
    genai.configure(api_key=api_key)
    print(f"✅ Configured with API key: {api_key[:15]}...")
    
    try:
        # Test model creation
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("✅ Model created successfully")
        
        # Test simple API call
        response = model.generate_content("Hello! Respond with just 'Hi there!'")
        print(f"✅ API call successful!")
        print(f"📝 Response: {response.text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if "SERVICE_DISABLED" in str(e):
            print("💡 The API still needs to be enabled - please wait a few more minutes")
        return False

if __name__ == "__main__":
    print("🚀 Simple Gemini API Test")
    print("Run this after enabling the Generative Language API")
    print()
    
    if test_gemini():
        print("\n🎉 Gemini API is working perfectly!")
        print("Ready to restart the classification API server")
    else:
        print("\n⏳ If you just enabled the API, wait 2-3 minutes and try again")