# Enable Gemini API Instructions

## Current Status
- ❌ **Gemini API**: `SERVICE_DISABLED` - needs to be enabled
- ✅ **Anthropic Claude**: Working with `claude-sonnet-4-20250514`
- ❌ **OpenAI**: Quota exceeded (expected)

## To Enable Gemini API

### Method 1: Google Cloud Console (Recommended)
1. Go to: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project=1007551429936
2. Click the **"ENABLE"** button
3. Wait 2-3 minutes for activation
4. Test with: `python fix_gemini_api.py`

### Method 2: Direct Link
1. Visit: https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview?project=1007551429936
2. Click **"Enable API"**
3. Wait for activation

## Test Commands

### Test Gemini API
```bash
source venv/bin/activate && python fix_gemini_api.py
```

### Test Complete Hierarchical System  
```bash
source venv/bin/activate && python test_complete_hierarchical_system.py
```

## Expected Behavior After Enabling

The hierarchical system should work as:

1. **🟢 Gemini (First)**: Should work after enabling API
2. **🔄 OpenAI (Second)**: Will fail due to quota (expected)
3. **🟢 Anthropic (Third)**: Now working with Claude Sonnet 4

## Project Details
- **Project ID**: `1007551429936`
- **API**: Generative Language API (generativelanguage.googleapis.com)
- **Status**: Currently disabled, needs manual activation

Once enabled, the system will have full redundancy across all three LLM providers!