#!/usr/bin/env python3
"""
Find the correct Google Cloud project for your API key
"""

import os
import requests

def find_correct_project():
    """
    Try to determine which project your API key actually belongs to
    """
    
    print("🔍 Finding Your Correct Google Cloud Project")
    print("=" * 50)
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY not found")
        return
    
    print(f"✅ Using API Key: {api_key[:15]}...")
    
    # Try to make a simple API call to see what project is used
    print(f"\n🧪 Testing API key to find project...")
    
    try:
        # Make a request to any Google API to see error details
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            print("✅ API call successful! Generative Language API is working")
            return True
        elif response.status_code == 403:
            error_data = response.json()
            print(f"❌ 403 Error (as expected)")
            
            # Look for project information in the error
            error_message = error_data.get('error', {}).get('message', '')
            print(f"📋 Error message: {error_message[:200]}...")
            
            # Try to extract project info
            if 'project' in error_message:
                import re
                project_match = re.search(r'project (\d+)', error_message)
                if project_match:
                    project_id = project_match.group(1)
                    print(f"\n🎯 Your API key belongs to project: {project_id}")
                    print(f"🔗 Enable API for YOUR project:")
                    print(f"   https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project={project_id}")
                    return project_id
        else:
            print(f"❌ Unexpected response: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")
    
    return None

def check_api_key_ownership():
    """
    Help user figure out which project their API key belongs to
    """
    
    print(f"\n💡 How to Find Your Correct Project:")
    print(f"1. Go to: https://console.cloud.google.com/apis/credentials")
    print(f"2. Make sure you're logged in with the correct Google account")
    print(f"3. Look for your API key: AIzaSyAM5OgW2nU...")
    print(f"4. The project shown at the top is your ACTUAL project")
    print(f"5. Enable Generative Language API in THAT project")
    
    print(f"\n🔄 Alternative: Create New API Key")
    print(f"1. If you want to use project 1007551429936:")
    print(f"2. Make sure you have access to that project")
    print(f"3. Create a new API key in that project")
    print(f"4. Update your GOOGLE_API_KEY environment variable")

if __name__ == "__main__":
    print("🚀 Google Cloud Project Detective")
    print("Finding which project your API key actually belongs to...")
    print()
    
    project_id = find_correct_project()
    
    if not project_id:
        check_api_key_ownership()
    
    print(f"\n🎯 Next Steps:")
    if project_id:
        print(f"✅ Enable Generative Language API in YOUR project: {project_id}")
        print(f"🔗 Direct link: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project={project_id}")
    else:
        print(f"🔍 Find your correct project in Google Cloud Console")
        print(f"💡 Or get access to project 1007551429936 if that's intended")