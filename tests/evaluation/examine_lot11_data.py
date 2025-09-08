#!/usr/bin/env python3
"""
Examine Lot-11 data structure and ground truth labels
"""

import pandas as pd
import os
from pathlib import Path

def examine_lot11_data():
    """
    Examine the structure of Lot-11 data and ground truth labels
    """
    
    print("🔍 Examining Lot-11 Data Structure")
    print("=" * 50)
    
    # Paths
    lot11_dir = Path("data/Lot-11")
    excel_path = lot11_dir / "EDMS-Lot 11.xlsx"
    
    # Check if files exist
    if not lot11_dir.exists():
        print(f"❌ Directory not found: {lot11_dir}")
        return
    
    if not excel_path.exists():
        print(f"❌ Excel file not found: {excel_path}")
        return
    
    # Count PDF files
    pdf_files = list(lot11_dir.glob("*.pdf"))
    print(f"📄 Found {len(pdf_files)} PDF files in Lot-11")
    
    # Read Excel file
    print(f"\n📊 Reading ground truth from: {excel_path}")
    try:
        df = pd.read_excel(excel_path)
        print(f"✅ Excel loaded with {len(df)} rows and {len(df.columns)} columns")
        
        # Show column names
        print(f"\n📋 Columns: {list(df.columns)}")
        
        # Show first few rows
        print(f"\n📋 First 10 rows:")
        print(df.head(10).to_string())
        
        # Check column C (filename) and column F (categories)
        if len(df.columns) >= 6:  # At least 6 columns (A, B, C, D, E, F)
            col_c = df.iloc[:, 2]  # Column C (0-indexed)
            col_f = df.iloc[:, 5]  # Column F (0-indexed)
            
            print(f"\n🔍 Column C (Filenames) sample:")
            print(col_c.head(10).tolist())
            
            print(f"\n🔍 Column F (Categories) sample:")
            print(col_f.head(10).tolist())
            
            # Find unique filenames
            unique_files = col_c.dropna().unique()
            print(f"\n📊 Found {len(unique_files)} unique filenames in ground truth")
            
            # Check for files spanning multiple rows
            file_counts = col_c.value_counts()
            multi_row_files = file_counts[file_counts > 1]
            print(f"📊 Files with multiple rows: {len(multi_row_files)}")
            if len(multi_row_files) > 0:
                print(f"Examples: {multi_row_files.head().to_dict()}")
            
            # Check unique categories
            unique_categories = col_f.dropna().unique()
            print(f"\n🏷️  Found {len(unique_categories)} unique categories:")
            for i, cat in enumerate(unique_categories[:20]):  # Show first 20
                print(f"  {i+1}. {cat}")
            if len(unique_categories) > 20:
                print(f"  ... and {len(unique_categories) - 20} more")
                
        else:
            print("❌ Excel file doesn't have expected columns (need at least F columns)")
            
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return
    
    # Match PDF files with ground truth
    print(f"\n🔗 Matching PDF files with ground truth...")
    
    if len(df.columns) >= 3:
        filenames_in_gt = set(df.iloc[:, 2].dropna().astype(str))  # Column C
        pdf_basenames = set([f.stem for f in pdf_files])  # Without .pdf extension
        
        # Add .pdf extension to ground truth names for comparison
        filenames_in_gt_with_pdf = set([f + '.pdf' if not f.endswith('.pdf') else f for f in filenames_in_gt])
        pdf_filenames = set([f.name for f in pdf_files])
        
        print(f"📊 Ground truth files: {len(filenames_in_gt)}")
        print(f"📊 PDF files found: {len(pdf_files)}")
        
        # Find matches
        matches = pdf_filenames.intersection(filenames_in_gt_with_pdf)
        print(f"✅ Matching files: {len(matches)}")
        
        # Show mismatches
        missing_in_gt = pdf_filenames - filenames_in_gt_with_pdf
        missing_pdfs = filenames_in_gt_with_pdf - pdf_filenames
        
        if missing_in_gt:
            print(f"\n❌ PDF files not in ground truth ({len(missing_in_gt)}):")
            for f in sorted(missing_in_gt)[:5]:
                print(f"  - {f}")
            if len(missing_in_gt) > 5:
                print(f"  ... and {len(missing_in_gt) - 5} more")
                
        if missing_pdfs:
            print(f"\n❌ Ground truth files not found as PDFs ({len(missing_pdfs)}):")
            for f in sorted(missing_pdfs)[:5]:
                print(f"  - {f}")
            if len(missing_pdfs) > 5:
                print(f"  ... and {len(missing_pdfs) - 5} more")
    
    return df

if __name__ == "__main__":
    examine_lot11_data()