"""
reporting_combined.py
OPTIMIZED version with efficient MPNet similarity calculation
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import re
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import gc

# ------------------------------------------------------------------
# OPTIMIZED: Configure GPU and model settings
# ------------------------------------------------------------------
pio.kaleido.scope.default_width = 1200
pio.kaleido.scope.default_height = 800
pio.templates.default = "plotly_white"

# GPU optimization settings
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize similarity model with optimizations
similarity_model_name = "sentence-transformers/all-mpnet-base-v2"
similarity_model = SentenceTransformer(
    similarity_model_name,
    device=device,
    cache_folder="./model_cache"  # Cache model to avoid re-downloading
)

# OPTIMIZED: Configure model for inference
similarity_model.eval()
if device == "cuda":
    similarity_model = similarity_model.half()  # Use half precision for T4 GPU
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner

# ------------------------------------------------------------------
# OPTIMIZED: Batch processing with GPU memory management
# ------------------------------------------------------------------

def calculate_similarity_batch_optimized(df, batch_size=64, max_length=384):
    """
    OPTIMIZED: Calculate similarity scores with proper batching and GPU memory management
    """
    print("Calculating similarity scores with optimized batch processing...")
    
    # Check if required columns exist
    if 'translated' not in df.columns or 'ref' not in df.columns:
        print("Error: Missing required columns 'translated' or 'ref'")
        return df
    
    # Check if similarity already calculated
    if "similarity_score" in df.columns and not df["similarity_score"].isna().all():
        print("Similarity scores already exist, skipping calculation")
        return df
    
    # Prepare data
    translated_texts = df['translated'].fillna('').astype(str).tolist()
    ref_texts = df['ref'].fillna('').astype(str).tolist()
    
    # OPTIMIZED: Filter out empty pairs and track indices
    valid_indices = []
    valid_translated = []
    valid_ref = []
    
    for i, (t, r) in enumerate(zip(translated_texts, ref_texts)):
        if t.strip() and r.strip():  # Only process non-empty texts
            valid_indices.append(i)
            valid_translated.append(t)
            valid_ref.append(r)
    
    print(f"Processing {len(valid_indices)} valid text pairs out of {len(df)} total")
    
    if not valid_indices:
        print("No valid text pairs found for similarity calculation")
        df['similarity_score'] = 0.0
        return df
    
    # OPTIMIZED: Calculate similarity in batches with memory management
    similarities = np.zeros(len(df), dtype=np.float32)
    
    total_batches = (len(valid_translated) + batch_size - 1) // batch_size
    
    with torch.no_grad():  # Disable gradient calculation for inference
        for i in tqdm(range(0, len(valid_translated), batch_size), 
                     desc="Calculating similarities", total=total_batches):
            batch_end = min(i + batch_size, len(valid_translated))
            
            batch_translated = valid_translated[i:batch_end]
            batch_ref = valid_ref[i:batch_end]
            
            try:
                # OPTIMIZED: Encode both batches simultaneously
                embeddings_translated = similarity_model.encode(
                    batch_translated,
                    convert_to_tensor=True,
                    device=device,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=min(batch_size, 16)  # Smaller batch for encoding
                )
                
                embeddings_ref = similarity_model.encode(
                    batch_ref,
                    convert_to_tensor=True,
                    device=device,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=min(batch_size, 16)
                )
                
                # OPTIMIZED: Calculate cosine similarities efficiently
                batch_similarities = util.pytorch_cos_sim(embeddings_translated, embeddings_ref)
                batch_diagonal = batch_similarities.diag().cpu().numpy().astype(np.float32)
                
                # Store results for valid indices
                batch_indices = valid_indices[i:batch_end]
                similarities[batch_indices] = batch_diagonal
                
                # OPTIMIZED: Clear GPU memory
                del embeddings_translated, embeddings_ref, batch_similarities
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU out of memory at batch {i//batch_size + 1}, reducing batch size...")
                    # Reduce batch size and retry
                    reduced_batch_size = max(batch_size // 2, 8)
                    return calculate_similarity_batch_optimized(df, reduced_batch_size)
                else:
                    raise e
    
    df['similarity_score'] = similarities
    print("Similarity calculation completed!")
    return df

def check_similarity_calculation_needed(df, file_path):
    """
    OPTIMIZED: Check if similarity calculation is needed for this file
    """
    # Check if similarity_score column exists and has valid data
    if "similarity_score" in df.columns:
        # Check if there are any non-null, non-zero values
        valid_scores = df["similarity_score"].dropna()
        if len(valid_scores) > 0 and not (valid_scores == 0).all():
            print(f"Similarity scores already exist in {os.path.basename(file_path)}, skipping...")
            return False
    
    # Check if file was modified after a certain date (optional)
    # You could add logic here to check modification dates if needed
    return True

def combine_all_datasets_optimized(input_dir="output", force_recalculate=False):
    """
    OPTIMIZED: Combine datasets with intelligent similarity calculation
    """
    all_data = []
    recipes = get_available_recipes()
    
    print("Combining all datasets with optimized similarity calculation...")
    
    # First pass: collect all files and check what needs processing
    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            
            folder = os.path.basename(root)
            if "-" not in folder:
                continue
            
            file_path = os.path.join(root, file)
            files_to_process.append((root, file, file_path))
    
    print(f"Found {len(files_to_process)} CSV files to process")
    
    # Process files with progress indication
    for root, file, file_path in tqdm(files_to_process, desc="Processing files"):
        try:
            src, tgt = os.path.basename(root).split("-", 1)
            recipe = extract_recipe_name_from_filename(file, recipes)
            
            df = pd.read_csv(file_path)
            
            # OPTIMIZED: Check if similarity calculation is needed
            needs_calculation = force_recalculate or check_similarity_calculation_needed(df, file_path)
            
            if needs_calculation:
                print(f"Calculating similarity for {file}...")
                df = calculate_similarity_batch_optimized(df)
                
                # OPTIMIZED: Save the file with similarity scores immediately
                df.to_csv(file_path, index=False)
                print(f"Updated {file} with similarity scores")
            else:
                print(f"Skipping similarity calculation for {file}")
            
            # Add metadata columns
            df['language_pair'] = f"{src}-{tgt}"
            df['source_lang'] = src
            df['target_lang'] = tgt
            df['model'] = recipe
            df['file_path'] = file_path
            
            # Convert similarity score to percentage
            df['similarity_score_pct'] = df['similarity_score'] * 100
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    if not all_data:
        print("No data files found!")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Language pairs: {combined_df['language_pair'].nunique()}")
    print(f"Models: {combined_df['model'].nunique()}")
    
    # OPTIMIZED: Clear memory
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return combined_df

# Replace the original functions with optimized versions
calculate_similarity_batch = calculate_similarity_batch_optimized
combine_all_datasets = combine_all_datasets_optimized

# ------------------------------------------------------------------
# The rest of your functions remain the same, but update the main function
# ------------------------------------------------------------------

def generate_report_optimized(input_dir="output", output_dir="reports_combined", force_recalculate=False):
    """OPTIMIZED: Main function with optimized similarity calculation"""
    print("Generating performance reports with OPTIMIZED similarity calculation...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Combine all datasets with optimized similarity calculation
    combined_df = combine_all_datasets_optimized(input_dir, force_recalculate)
    
    # Step 2: Generate quadrant reports if we have data
    if not combined_df.empty:
        generate_quadrant_reports(combined_df, output_dir)
    else:
        print("No combined data found for quadrant analysis.")
    
    # Step 3: Collect results for traditional reporting
    results, source_breakdown = collect_results(input_dir)
    
    if not results:
        print("No processed results found. Please run translations first.")
        return
    
    # Step 4: Generate traditional reports
    generate_language_specific_reports(results, source_breakdown, output_dir)
    overall_summary = generate_overall_summary(results, source_breakdown, output_dir)
    
    print(f"Reports generated successfully in {output_dir}/")
    print("Both traditional reports and quadrant analysis have been created!")
    print("Note: HTML and PNG charts are now generated")
    
    if overall_summary:
        print(f"Overall best model: {overall_summary['best_overall_model']} ({overall_summary['best_overall_score']:.2f}%)")
        print(f"Best performing language: {overall_summary['best_language']} ({overall_summary['best_language_score']:.2f}%)")
    
    return results, overall_summary

if __name__ == "__main__":
    # Add command line argument support for force recalculate
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-recalculate", action="store_true", 
                       help="Force recalculation of similarity scores even if they exist")
    args = parser.parse_args()
    
    generate_report_optimized(force_recalculate=args.force_recalculate)
