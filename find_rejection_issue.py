#!/usr/bin/env python3
import pandas as pd

# Search for the rejection issue type
df = pd.read_excel("issues_to_category_mapping_normalized.xlsx")

print("Searching for 'rejection' or 'reject' issues...")
rejection_issues = df[df['Issue_type'].str.contains('reject', case=False, na=False)]

print(f"Found {len(rejection_issues)} matching issues:")
for _, row in rejection_issues.iterrows():
    print(f"  • {row['Issue_type']} → {row['Category']}")
    
# Also search the original CSV
print("\nSearching in original CSV...")
try:
    with open("integrated_backend/Issue_to_category_mapping.csv", 'r') as f:
        content = f.read()
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'reject' in line.lower():
                print(f"  Line {i+1}: {line}")
except Exception as e:
    print(f"Could not read CSV: {e}")