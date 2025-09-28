#!/usr/bin/env python3
"""
Diagnose and fix Gemini API issues by using only GOOGLE_API_KEY
"""

import os
import google.generativeai as genai

def diagnose_gemini_setup():
    """
    Diagnose current Gemini setup and test different configurations
    """
    print("🔧 Diagnosing Gemini API Setup")
    print("=" * 50)
    
    # Check current environment variables
    google_api_key = os.getenv('GOOGLE_API_KEY')
    google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    print(f"📋 Current Environment:")
    print(f"   GOOGLE_API_KEY: {'✅ Found' if google_api_key else '❌ Not found'}")
    if google_api_key:
        print(f"   Key preview: {google_api_key[:15]}...")
    print(f"   GOOGLE_APPLICATION_CREDENTIALS: {'✅ Set' if google_creds else '❌ Not set'}")
    if google_creds:
        print(f"   Credentials file: {google_creds}")
    
    if not google_api_key:
        print("\n❌ GOOGLE_API_KEY is required for Gemini API")
        return False
    
    # Test 1: Basic configuration
    print(f"\n🧪 Test 1: Basic Gemini Configuration")
    try:
        genai.configure(api_key=google_api_key)
        print("✅ Gemini configured successfully")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False
    
    # Test 2: List available models
    print(f"\n🧪 Test 2: List Available Models")
    try:
        models = list(genai.list_models())
        print(f"✅ Found {len(models)} available models:")
        for model in models[:3]:  # Show first 3
            print(f"   - {model.name}")
        if len(models) > 3:
            print(f"   ... and {len(models) - 3} more")
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        if "SERVICE_DISABLED" in str(e):
            print(f"\n🛠️  SOLUTION: Enable Generative Language API")
            print(f"   1. Go to: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
            print(f"   2. Make sure correct project is selected")  
            print(f"   3. Click 'ENABLE'")
            return False
        return False
    
    # Test 3: Simple API call
    print(f"\n🧪 Test 3: Test API Call")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say hello in one word")
        print(f"✅ API call successful!")
        print(f"   Response: {response.text}")
        return True
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

def remove_conflicting_credentials():
    """
    Temporarily unset Google Cloud credentials to avoid conflicts
    """
    print(f"\n🔄 Removing Conflicting Credentials")
    print("-" * 30)
    
    # Remove from current environment
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        print(f"✅ Unsetting GOOGLE_APPLICATION_CREDENTIALS from environment")
        del os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    
    # Test again without credentials
    return diagnose_gemini_setup()

def main():
    print("🚀 Gemini API Diagnostic Tool")
    print("Checking if GOOGLE_API_KEY alone can work with Gemini")
    print()
    
    # First test with current setup
    if diagnose_gemini_setup():
        print(f"\n🎉 Gemini API is working perfectly!")
        print(f"💡 No changes needed - ready to use")
        return True
    
    print(f"\n🔄 Trying without Google Cloud credentials...")
    if remove_conflicting_credentials():
        print(f"\n🎉 Gemini API works without GOOGLE_APPLICATION_CREDENTIALS!")
        print(f"💡 Recommendation: Remove or comment out GOOGLE_APPLICATION_CREDENTIALS")
        print(f"   in your .env file for Gemini API usage")
        return True
    
    print(f"\n❌ Gemini API still not working")
    print(f"🛠️  Next steps:")
    print(f"   1. Enable the Generative Language API in Google Cloud Console")
    print(f"   2. Make sure you're in the correct Google Cloud project")
    print(f"   3. Wait 2-3 minutes after enabling")
    print(f"   4. Run this script again")
    
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)