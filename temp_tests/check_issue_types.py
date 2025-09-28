#!/usr/bin/env python3
"""
Check for issue type variations that need normalization
"""

import pandas as pd
import difflib

# Load the training data
df = pd.read_excel('data/raw/Consolidated_labeled_data.xlsx')

# Get all unique issue types
unique_issues = sorted(df['issue_type'].dropna().unique())
print(f'Total unique issue types: {len(unique_issues)}')

print('\n=== CHECKING FOR ISSUE TYPE VARIATIONS ===')

# Look for obvious variations
variations_found = []

for i, issue1 in enumerate(unique_issues):
    for j, issue2 in enumerate(unique_issues[i+1:], i+1):
        # Check for exact match after case normalization
        if issue1.lower().strip() == issue2.lower().strip() and issue1 != issue2:
            count1 = df[df['issue_type'] == issue1].shape[0]
            count2 = df[df['issue_type'] == issue2].shape[0]
            variations_found.append(('Case difference', issue1, count1, issue2, count2))
        
        # Check for punctuation differences  
        elif (issue1.replace('.', '').replace(',', '').replace(' ', ' ').strip().lower() == 
              issue2.replace('.', '').replace(',', '').replace(' ', ' ').strip().lower() and issue1 != issue2):
            count1 = df[df['issue_type'] == issue1].shape[0]
            count2 = df[df['issue_type'] == issue2].shape[0]
            variations_found.append(('Punctuation/spacing difference', issue1, count1, issue2, count2))
        
        # Check for very high similarity (potential typos)
        elif difflib.SequenceMatcher(None, issue1.lower(), issue2.lower()).ratio() > 0.92:
            count1 = df[df['issue_type'] == issue1].shape[0]
            count2 = df[df['issue_type'] == issue2].shape[0]
            variations_found.append(('High similarity (>92%)', issue1, count1, issue2, count2))

if variations_found:
    print(f'\nFound {len(variations_found)} potential variations:')
    print('=' * 80)
    
    for i, (type_diff, issue1, count1, issue2, count2) in enumerate(variations_found, 1):
        print(f'\n{i}. {type_diff}:')
        print(f'   "{issue1}" ({count1} samples)')
        print(f'   "{issue2}" ({count2} samples)')
        
        if count1 >= count2:
            print(f'   → Suggest keeping: "{issue1}"')
        else:
            print(f'   → Suggest keeping: "{issue2}"')
    
    print(f'\nPotential for normalization: {len(unique_issues)} → {len(unique_issues) - len(variations_found)}')
else:
    print('\n✅ No obvious variations found in issue types!')
    print('The 125 issue types appear to be properly normalized.')

# Also check for common patterns that might indicate variations
print('\n=== CHECKING FOR COMMON PATTERNS ===')

# Group by common words/patterns
word_groups = {}
for issue in unique_issues:
    words = set(issue.lower().split())
    for word in words:
        if len(word) > 3:  # Skip short words
            if word not in word_groups:
                word_groups[word] = []
            word_groups[word].append(issue)

# Show groups with multiple similar issues
print('\nIssue types sharing common significant words:')
for word, issues in sorted(word_groups.items()):
    if len(issues) > 3:  # Show groups with more than 3 issues
        print(f'\n"{word}" appears in {len(issues)} issue types:')
        for issue in sorted(issues)[:5]:  # Show first 5
            count = df[df['issue_type'] == issue].shape[0]
            print(f'   - "{issue}" ({count} samples)')
        if len(issues) > 5:
            print(f'   ... and {len(issues) - 5} more')

print('\n' + '=' * 80)