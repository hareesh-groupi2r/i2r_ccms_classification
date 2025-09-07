# How to Check Google Cloud API Key and Service Status

## 🔍 Check API Key in Google Cloud Console

### **Step 1: Go to Credentials Page**
**Direct Link**: https://console.cloud.google.com/apis/credentials?project=1007551429936

### **Step 2: Find Your API Key**
- Look for "API Keys" section
- You should see your key: `AIzaSyAM5OgW2nU...`
- Click on it to see details

### **Step 3: Verify API Key Settings**
Check that your API key has:
- ✅ **Status**: Enabled
- ✅ **API restrictions**: Either "None" or includes "Generative Language API"
- ✅ **Application restrictions**: Appropriate for your setup

## 🔧 Check API Service Status

### **Method 1: APIs & Services Dashboard**
**Link**: https://console.cloud.google.com/apis/dashboard?project=1007551429936
- Look for "Generative Language API" in the list
- Status should show "Enabled"

### **Method 2: Direct API Page**
**Link**: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com?project=1007551429936
- Should show "API Enabled" at the top
- If it shows "Enable" button, click it again

### **Method 3: API Library**
**Link**: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project=1007551429936
- Should show "API Enabled" or "Manage" button
- If shows "Enable", the activation didn't work

## 🛠️ Common Issues & Solutions

### **Issue 1: API Key Restrictions**
If your API key is restricted:
1. Go to: https://console.cloud.google.com/apis/credentials?project=1007551429936
2. Click on your API key
3. Under "API restrictions":
   - Choose "Don't restrict key" (for testing)
   - OR add "Generative Language API" to allowed APIs

### **Issue 2: API Not Actually Enabled**
1. Go to: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project=1007551429936
2. If you see "ENABLE" button, click it again
3. Wait for confirmation message

### **Issue 3: Wrong Project Selected**
- Make sure the project dropdown shows: `1007551429936`
- If different project is selected, switch to the correct one

## 📋 Quick Checklist

Visit these URLs and verify:

1. **✅ Project**: https://console.cloud.google.com/home/dashboard?project=1007551429936
   - Confirm you're in project `1007551429936`

2. **✅ API Key**: https://console.cloud.google.com/apis/credentials?project=1007551429936
   - Your key `AIzaSyAM5O...` should be listed and enabled

3. **✅ API Status**: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com?project=1007551429936
   - Should show "API Enabled" at the top

4. **✅ API Dashboard**: https://console.cloud.google.com/apis/dashboard?project=1007551429936
   - "Generative Language API" should appear in enabled APIs list

## 🧪 Test After Verification

Once everything looks correct in the console:
```bash
python test_gemini_simple.py
```