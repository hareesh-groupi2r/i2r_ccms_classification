#!/usr/bin/env python3
"""
Script to fix Gemini API configuration and test access
"""

import os
import google.generativeai as genai

def enable_gemini_api():
    """
    Enable Gemini API and test connectivity.
    """
    
    print("🔧 Fixing Gemini API Configuration")
    print("=" * 50)
    
    # Check current API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment")
        return False
    
    print(f"✅ GOOGLE_API_KEY found: {api_key[:10]}...")
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    print("✅ Gemini API configured")
    
    # Test if we can list models (this should work even if generative calls don't)
    try:
        models = list(genai.list_models())
        print(f"✅ Can access {len(models)} Gemini models")
        return True
    except Exception as e:
        print(f"❌ Cannot access Gemini models: {e}")
        
        # The error indicates we need to enable the API
        if "SERVICE_DISABLED" in str(e):
            print("\n🛠️  SOLUTION:")
            print("   1. The Generative Language API is disabled for your project")
            print("   2. Go to: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
            print("   3. Make sure you're in the correct project")
            print("   4. Click 'Enable' button")
            print("   5. Wait a few minutes for activation")
            
            # Extract project ID from error
            import re
            project_match = re.search(r'project (\d+)', str(e))
            if project_match:
                project_id = project_match.group(1)
                print(f"   6. Your project ID: {project_id}")
                print(f"   7. Direct link: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project={project_id}")
        
        return False

def test_direct_api_call():
    """Test direct API call to see specific error"""
    print("\n🧪 Testing Direct API Call")
    print("-" * 30)
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello")
        print(f"✅ SUCCESS: {response.text[:50]}...")
        return True
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Gemini API diagnosis...")
    
    # Test basic configuration
    can_access_models = enable_gemini_api()
    
    if can_access_models:
        # Test actual API call
        api_call_works = test_direct_api_call()
        
        if api_call_works:
            print("\n🎉 Gemini API is fully functional!")
        else:
            print("\n⚠️  Gemini API is configured but calls are failing")
    else:
        print("\n❌ Gemini API needs to be enabled first")
    
    print("\nNext steps:")
    print("1. Enable the Generative Language API in Google Cloud Console")
    print("2. Wait 2-3 minutes for activation")
    print("3. Test again with this script")
    print("4. Restart the classification API server")