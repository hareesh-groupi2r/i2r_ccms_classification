#!/usr/bin/env python3
"""
Create sample training data for testing
"""

import pandas as pd
import numpy as np

# Sample issue types and categories
issue_types = [
    "Payment Delay", "Change of Scope", "Material Delivery Delay",
    "Quality Issue", "Safety Concern", "Extension Request",
    "Cost Overrun", "Resource Allocation", "Design Change",
    "Contract Amendment"
]

categories = [
    "Financial Management", "Contract Modification", "Schedule Impact",
    "Quality Control", "Safety Management", "Time Extension",
    "Cost Management", "Resource Management", "Design Management",
    "Legal Compliance", "Risk Management"
]

# Create sample data
data = []
np.random.seed(42)

for i in range(100):
    # Pick random issue type
    issue = np.random.choice(issue_types)
    
    # Pick 1-3 categories
    n_categories = np.random.randint(1, 4)
    selected_categories = np.random.choice(categories, n_categories, replace=False)
    category_str = ", ".join(selected_categories)
    
    # Generate sample text
    subject = f"Regarding {issue} - Project Update {i+1}"
    body = f"""
    Dear Project Manager,
    
    We are writing to address the {issue} issue that has recently occurred in our project.
    This matter requires immediate attention as it impacts our operations.
    
    The situation has implications for {category_str.lower()}.
    We request your review and approval of our proposed solution.
    
    Please respond at your earliest convenience.
    
    Best regards,
    Contractor Team
    """
    
    reference = f"This matter requires immediate attention as it impacts our operations."
    
    data.append({
        'issue_type': issue,
        'category': category_str,
        'reference_sentence': reference,
        'source_file': f'document_{i+1}.pdf',
        'subject': subject,
        'body': body.strip()
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
output_path = 'data/raw/Consolidated_labeled_data.xlsx'
df.to_excel(output_path, index=False)

print(f"✅ Created sample training data with {len(df)} records")
print(f"📁 Saved to: {output_path}")
print(f"📊 Issue types: {len(issue_types)}")
print(f"📊 Categories: {len(categories)}")
print(f"\nYou can now run: python test_classifiers.py")