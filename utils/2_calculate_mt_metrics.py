"""
calculate_mt_metrics.py
Calculate standard MT evaluation metrics: BLEU, chrF, and COMET
Processes all CSVs in batches for maximum efficiency
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import MT evaluation libraries
import sacrebleu

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_COMBINED_PATH = "/home/owusus/Documents/GitHub/nsanku/output_combined"
BATCH_SIZE = 32  # Batch size for processing
DEBUG_MODE = True
# ==============================================================================

print("="*70)
print("MT METRICS CALCULATOR (BLEU, chrF)")
print("="*70)

def calculate_bleu(hypothesis: str, reference: str) -> float:
    """Calculate BLEU score for a single sentence pair"""
    try:
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
        return bleu.score
    except:
        return 0.0

def calculate_chrf(hypothesis: str, reference: str) -> float:
    """Calculate chrF score for a single sentence pair"""
    try:
        chrf = sacrebleu.sentence_chrf(hypothesis, [reference])
        return chrf.score
    except:
        return 0.0

def collect_all_missing_pairs(csv_files: List[str], debug: bool = False) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Phase 1: Collect all missing metric pairs across all CSVs
    """
    print("\n" + "-"*70)
    print("PHASE 1: Collecting all missing metric calculations...")
    print("-"*70)
    
    all_pairs = []
    file_stats = defaultdict(int)
    
    for file_path in tqdm(csv_files, desc="Scanning CSVs"):
        try:
            df = pd.read_csv(file_path)
            
            # Check required columns
            required_cols = ['translated', 'ref']
            if not all(col in df.columns for col in required_cols):
                if debug:
                    print(f"  Skipping (missing columns): {os.path.basename(file_path)}")
                continue
            
            # Add metric columns if missing
            metric_cols = ['bleu_score', 'chrf_score']
            needs_save = False
            for col in metric_cols:
                if col not in df.columns:
                    df[col] = np.nan
                    needs_save = True
            
            if needs_save:
                df.to_csv(file_path, index=False)
            
            # Find rows missing ANY metric
            missing_mask = (
                df['bleu_score'].isna() | 
                df['chrf_score'].isna()
            )
            missing_indices = df[missing_mask].index.tolist()
            
            if not missing_indices:
                file_stats['complete'] += 1
                continue
            
            # Collect pairs that need calculation
            for idx in missing_indices:
                translated = str(df.loc[idx, 'translated']).strip()
                ref = str(df.loc[idx, 'ref']).strip()
                
                if translated and ref and translated != 'nan' and ref != 'nan':
                    all_pairs.append({
                        'file_path': file_path,
                        'row_index': idx,
                        'translated': translated,
                        'ref': ref,
                        'needs_bleu': pd.isna(df.loc[idx, 'bleu_score']),
                        'needs_chrf': pd.isna(df.loc[idx, 'chrf_score'])
                    })
                else:
                    # Mark invalid pairs
                    all_pairs.append({
                        'file_path': file_path,
                        'row_index': idx,
                        'translated': None,
                        'ref': None,
                        'needs_bleu': True,
                        'needs_chrf': True
                    })
            
            file_stats['pending'] += len(missing_indices)
            
        except Exception as e:
            print(f"  ✗ Error reading {file_path}: {e}")
    
    total_pairs = len(all_pairs)
    print(f"\n✓ Found {total_pairs} pairs needing calculation across {len(csv_files)} files")
    print(f"  Complete files: {file_stats['complete']}")
    print(f"  Pending calculations: {file_stats['pending']}")
    
    return all_pairs, dict(file_stats)

def process_all_pairs(pairs: List[Dict], debug: bool = False) -> List[Dict]:
    """
    Phase 2: Calculate all metrics for collected pairs
    """
    if not pairs:
        return []
    
    print("\n" + "-"*70)
    print("PHASE 2: Calculating MT metrics...")
    print("-"*70)
    
    # Separate valid and invalid pairs
    valid_pairs = [p for p in pairs if p['translated'] is not None]
    invalid_pairs = [p for p in pairs if p['translated'] is None]
    
    print(f"Valid pairs to process: {len(valid_pairs)}")
    print(f"Invalid pairs (set to 0): {len(invalid_pairs)}")
    
    if not valid_pairs:
        for p in invalid_pairs:
            p['bleu_score'] = 0.0
            p['chrf_score'] = 0.0
        return invalid_pairs
    
    results = []
    
    # Calculate BLEU and chrF
    print("\nCalculating BLEU and chrF scores...")
    for pair in tqdm(valid_pairs, desc="BLEU & chrF"):
        if pair['needs_bleu']:
            pair['bleu_score'] = calculate_bleu(pair['translated'], pair['ref'])
        
        if pair['needs_chrf']:
            pair['chrf_score'] = calculate_chrf(pair['translated'], pair['ref'])
    
    # Add all valid pairs to results
    results.extend(valid_pairs)
    
    # Add invalid pairs with 0 scores
    for pair in invalid_pairs:
        pair['bleu_score'] = 0.0
        pair['chrf_score'] = 0.0
        results.append(pair)
    
    print(f"\n✓ Processed {len(results)} pairs")
    
    return results

def update_csvs_with_results(results: List[Dict], debug: bool = False):
    """
    Phase 3: Update all CSVs with calculated metrics
    """
    print("\n" + "-"*70)
    print("PHASE 3: Updating CSV files with metrics...")
    print("-"*70)
    
    # Group results by file
    file_groups = defaultdict(list)
    for result in results:
        file_groups[result['file_path']].append(result)
    
    print(f"Updating {len(file_groups)} files...")
    
    for file_path, file_results in tqdm(file_groups.items(), desc="Updating CSVs"):
        try:
            df = pd.read_csv(file_path)
            
            # Ensure columns exist
            for col in ['bleu_score', 'chrf_score']:
                if col not in df.columns:
                    df[col] = np.nan
            
            # Update rows
            update_count = 0
            for result in file_results:
                idx = result['row_index']
                if idx in df.index:
                    if 'bleu_score' in result and result.get('needs_bleu'):
                        df.loc[idx, 'bleu_score'] = result['bleu_score']
                    if 'chrf_score' in result and result.get('needs_chrf'):
                        df.loc[idx, 'chrf_score'] = result['chrf_score']
                    update_count += 1
            
            # Calculate average score
            if 'avg_score' not in df.columns:
                df['avg_score'] = np.nan
            
            # Calculate average of BLEU and chrF
            metric_cols = ['bleu_score', 'chrf_score']
            
            df['avg_score'] = df[metric_cols].mean(axis=1)
            
            # Save once per file
            df.to_csv(file_path, index=False)
            
            if debug:
                print(f"  Updated {update_count} rows in {os.path.basename(file_path)}")
                
        except Exception as e:
            print(f"  ✗ Error updating {file_path}: {e}")

def find_csv_files(folder_path: str) -> List[str]:
    """Find all CSV files recursively"""
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return sorted(csv_files)

def main():
    print("\n" + "="*70)
    print("MT METRICS CALCULATOR")
    print("="*70)
    print(f"Output folder: {OUTPUT_COMBINED_PATH}")
    print(f"Metrics: BLEU, chrF")
    print(f"Debug mode: {DEBUG_MODE}")
    print("="*70)
    
    if not os.path.exists(OUTPUT_COMBINED_PATH):
        print(f"\n✗ Error: Path does not exist: {OUTPUT_COMBINED_PATH}")
        return
    
    # Find all CSV files
    print("\nScanning for CSV files...")
    csv_files = find_csv_files(OUTPUT_COMBINED_PATH)
    
    if not csv_files:
        print("⚠ No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Phase 1: Collect all missing pairs
    all_pairs, stats = collect_all_missing_pairs(csv_files, DEBUG_MODE)
    
    if not all_pairs:
        print("\n✓ All metrics are already calculated!")
        return
    
    # Phase 2: Process all pairs
    results = process_all_pairs(all_pairs, DEBUG_MODE)
    
    # Phase 3: Update all CSVs
    update_csvs_with_results(results, DEBUG_MODE)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total pairs processed: {len(results)}")
    print(f"Files updated: {len(set(r['file_path'] for r in results))}")
    print(f"Metrics calculated: BLEU, chrF")
    print("✓ Processing complete!")
    print("="*70)

if __name__ == "__main__":
    main()
