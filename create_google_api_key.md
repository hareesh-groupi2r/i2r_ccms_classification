# How to Create Google API Key - Step by Step

## 🎯 Creating API Key in Your New Project

### **Step 1: Make Sure You're in Your New Project**
1. Go to: https://console.cloud.google.com/
2. At the top left, click the **project dropdown** (next to "Google Cloud")
3. Select **your new project** (not project 1007551429936)

### **Step 2: Navigate to APIs & Services**
1. Click the **hamburger menu** (☰) in the top left
2. Go to **"APIs & Services"**
3. Click **"Credentials"**

OR use direct link (replace `YOUR-PROJECT-ID` with your actual project):
https://console.cloud.google.com/apis/credentials?project=YOUR-PROJECT-ID

### **Step 3: Create API Key**
1. Click the **"+ CREATE CREDENTIALS"** button at the top
2. Select **"API key"** from the dropdown
3. A popup will show your new API key
4. **Copy the API key** - it looks like: `AIzaSy...`
5. Click **"RESTRICT KEY"** (recommended) or "CLOSE"

### **Step 4: (Optional) Restrict the API Key**
1. Give it a name like "Gemini API Key"
2. Under **"API restrictions"**:
   - Select **"Restrict key"**
   - Check **"Generative Language API"**
3. Click **"SAVE"**

### **Step 5: Enable Generative Language API**
1. Go to: https://console.cloud.google.com/apis/library
2. Search for **"Generative Language API"**
3. Click on it
4. Click the **"ENABLE"** button
5. Wait for activation (usually instant in your own project)

## 🔧 Alternative Method: Direct Navigation

### **Method 1: Through Menu**
```
Google Cloud Console → ☰ Menu → APIs & Services → Credentials → + CREATE CREDENTIALS → API key
```

### **Method 2: Search Bar**
1. In Google Cloud Console, use the **search bar** at the top
2. Type: **"credentials"**
3. Click **"Credentials - APIs & Services"**
4. Follow Step 3 above

## 📋 What You Should See

### **In Credentials Page:**
- **"+ CREATE CREDENTIALS"** button at the top
- Dropdown with options: API key, OAuth 2.0 Client ID, Service account key

### **After Creating API Key:**
- A popup showing your new key: `AIzaSy...`
- Options to "RESTRICT KEY" or "CLOSE"

## 🚨 If You Don't See "CREATE CREDENTIALS" Button

### **Possible Issues:**
1. **Wrong Project**: Make sure you're in YOUR project (not 1007551429936)
2. **Permissions**: You might not have the right permissions
3. **Browser Issues**: Try refreshing or different browser

### **Quick Fixes:**
1. **Check Project**: Look at project name in top dropdown
2. **Try Direct URL**: https://console.cloud.google.com/apis/credentials
3. **Refresh Page**: Sometimes takes a moment to load

## ✅ Once You Have Your New API Key

1. **Copy the key**: Something like `AIzaSyXXXXX...`
2. **Update your .env file**:
   ```
   GOOGLE_API_KEY=AIzaSyXXXXX_YOUR_NEW_KEY_HERE
   ```
3. **Test it**: `python test_gemini_simple.py`

## 🎉 Expected Result

After creating your own API key in your own project:
- ✅ No permission issues
- ✅ Generative Language API enabled instantly
- ✅ Gemini API calls work immediately