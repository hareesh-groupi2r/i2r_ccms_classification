#!/usr/bin/env python3
"""
Test improved configuration for broader issue type detection
"""

print("🔧 HYBRID RAG CONFIGURATION ANALYSIS")
print("=" * 60)

current_config = {
    "top_k": 12,
    "window_size": 3, 
    "overlap": 1,
    "similarity_threshold": 0.20,
    "max_issues": 10,
    "min_llm_confidence": 0.1,
    "enable_confidence_filtering": True,
    "confidence_divergence_threshold": 0.15,
    "top_percentage_filter": 0.20
}

print("📊 CURRENT CONFIG:")
for key, value in current_config.items():
    print(f"   {key}: {value}")

print()
print("🚨 POTENTIAL ISSUES:")
print("   • top_k=12 may not be enough for diverse issue discovery")
print("   • confidence_divergence_threshold=0.15 might filter out valid issues")  
print("   • top_percentage_filter=0.20 only keeps top 20% - too restrictive")
print("   • Need to ensure semantic search finds diverse issue types")

print()
print("💡 SUGGESTED IMPROVEMENTS:")
suggested_config = {
    "top_k": 20,  # More candidates for better coverage
    "window_size": 3,  # Keep same  
    "overlap": 1,  # Keep same
    "similarity_threshold": 0.15,  # Lower threshold for more recall
    "max_issues": 15,  # Allow more issues to be considered
    "min_llm_confidence": 0.05,  # Much lower threshold  
    "enable_confidence_filtering": False,  # Disable aggressive filtering
    "confidence_divergence_threshold": 0.10,  # Less restrictive
    "top_percentage_filter": 0.40  # Keep more candidates (40%)
}

print("📈 IMPROVED CONFIG:")
for key, value in suggested_config.items():
    change = "🔄" if current_config.get(key) != value else "✓"
    print(f"   {change} {key}: {value}")

print()
print("🎯 EXPECTED IMPACT:")
print("   ✅ More semantic search candidates (top_k: 12→20)")
print("   ✅ Lower similarity threshold for broader matching (0.20→0.15)")  
print("   ✅ More issues considered (max_issues: 10→15)")
print("   ✅ Much lower LLM confidence threshold (0.1→0.05)")
print("   ✅ Disabled confidence filtering to avoid missing valid issues")
print("   ✅ Less restrictive filtering (20%→40% kept)")

print()
print("📋 NEXT STEPS:")
print("   1. Update config.yaml with improved parameters")
print("   2. Restart integrated backend")  
print("   3. Test with same 2 files to see if we find more issue types")
print("   4. Should find Authority Engineer, Contractor obligations, etc.")