"""
calculate_similarity.py
Standalone script to calculate and save MPNet similarity scores for all translation files
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import gc
import re
import argparse

# ------------------------------------------------------------------
# GPU optimization settings
# ------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize similarity model with optimizations
similarity_model_name = "sentence-transformers/all-mpnet-base-v2"
similarity_model = SentenceTransformer(
    similarity_model_name,
    device=device,
    cache_folder="./model_cache"
)

# Configure model for inference
similarity_model.eval()
if device == "cuda":
    similarity_model = similarity_model.half()  # Use half precision for T4 GPU
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner

def get_available_recipes(recipes_dir="recipes"):
    """Get available recipe names"""
    recipes = []
    for file in os.listdir(recipes_dir):
        if file.endswith(".py") and file != "__init__.py":
            recipes.append(file[:-3])
    return recipes

def extract_recipe_name_from_filename(filename, available_recipes):
    """Extract recipe name from filename"""
    name_without_ext = os.path.splitext(filename)[0]
    for recipe in available_recipes:
        if f"_{recipe}" in name_without_ext:
            return recipe
    match = re.search(r'_([^_]+)$', name_without_ext)
    if match:
        return match.group(1)
    return "unknown_recipe"

def check_similarity_calculation_needed(df, file_path):
    """
    Check if similarity calculation is needed for this file
    """
    if "similarity_score" in df.columns:
        valid_scores = df["similarity_score"].dropna()
        if len(valid_scores) > 0 and not (valid_scores == 0).all():
            print(f"✓ Similarity scores already exist in {os.path.basename(file_path)}")
            return False
    return True

def calculate_similarity_batch_optimized(df, batch_size=64):
    """
    Calculate similarity scores with proper batching and GPU memory management
    """
    # Check if required columns exist
    if 'translated' not in df.columns or 'ref' not in df.columns:
        print(f"✗ Missing required columns 'translated' or 'ref'")
        return df
    
    # Prepare data
    translated_texts = df['translated'].fillna('').astype(str).tolist()
    ref_texts = df['ref'].fillna('').astype(str).tolist()
    
    # Filter out empty pairs and track indices
    valid_indices = []
    valid_translated = []
    valid_ref = []
    
    for i, (t, r) in enumerate(zip(translated_texts, ref_texts)):
        if t.strip() and r.strip():  # Only process non-empty texts
            valid_indices.append(i)
            valid_translated.append(t)
            valid_ref.append(r)
    
    print(f"  Processing {len(valid_indices)} valid text pairs out of {len(df)} total")
    
    if not valid_indices:
        print("  No valid text pairs found for similarity calculation")
        df['similarity_score'] = 0.0
        return df
    
    # Calculate similarity in batches with memory management
    similarities = np.zeros(len(df), dtype=np.float32)
    
    total_batches = (len(valid_translated) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_translated), batch_size), 
                     desc="  Calculating similarities", total=total_batches, leave=False):
            batch_end = min(i + batch_size, len(valid_translated))
            
            batch_translated = valid_translated[i:batch_end]
            batch_ref = valid_ref[i:batch_end]
            
            try:
                # Encode both batches simultaneously
                embeddings_translated = similarity_model.encode(
                    batch_translated,
                    convert_to_tensor=True,
                    device=device,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=min(batch_size, 16)
                )
                
                embeddings_ref = similarity_model.encode(
                    batch_ref,
                    convert_to_tensor=True,
                    device=device,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=min(batch_size, 16)
                )
                
                # Calculate cosine similarities efficiently
                batch_similarities = util.pytorch_cos_sim(embeddings_translated, embeddings_ref)
                batch_diagonal = batch_similarities.diag().cpu().numpy().astype(np.float32)
                
                # Store results for valid indices
                batch_indices = valid_indices[i:batch_end]
                similarities[batch_indices] = batch_diagonal
                
                # Clear GPU memory
                del embeddings_translated, embeddings_ref, batch_similarities
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  GPU out of memory, reducing batch size from {batch_size} to {max(batch_size//2, 8)}")
                    return calculate_similarity_batch_optimized(df, max(batch_size//2, 8))
                else:
                    raise e
    
    df['similarity_score'] = similarities
    return df

def process_single_file(file_path, force_recalculate=False):
    """
    Process a single CSV file and calculate similarity scores if needed
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check if similarity calculation is needed
        if not force_recalculate and not check_similarity_calculation_needed(df, file_path):
            return True, "skipped"
        
        # Calculate similarity scores
        df_updated = calculate_similarity_batch_optimized(df)
        
        # Save the updated file
        df_updated.to_csv(file_path, index=False)
        return True, "calculated"
        
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False, str(e)

def calculate_similarity_for_all_files(input_dir="output", force_recalculate=False):
    """
    Main function to calculate similarity scores for all CSV files
    """
    print("=" * 60)
    print("MPNet Similarity Calculation Tool")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Force recalculate: {force_recalculate}")
    print(f"Using device: {device}")
    print()
    
    recipes = get_available_recipes()
    files_processed = 0
    files_skipped = 0
    files_failed = 0
    
    # Collect all CSV files
    all_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    
    print(f"Found {len(all_files)} CSV files to process")
    print()
    
    # Process files
    for file_path in tqdm(all_files, desc="Overall progress"):
        success, status = process_single_file(file_path, force_recalculate)
        
        if success:
            if status == "calculated":
                files_processed += 1
            else:
                files_skipped += 1
        else:
            files_failed += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("SIMILARITY CALCULATION SUMMARY")
    print("=" * 60)
    print(f"✓ Files processed: {files_processed}")
    print(f"○ Files skipped: {files_skipped}")
    print(f"✗ Files failed: {files_failed}")
    print(f"Total files: {len(all_files)}")
    
    # Clear memory
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return files_processed, files_skipped, files_failed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MPNet similarity scores for translation files")
    parser.add_argument("--input-dir", default="output", help="Input directory containing CSV files")
    parser.add_argument("--force-recalculate", action="store_true", 
                       help="Force recalculation of similarity scores even if they exist")
    
    args = parser.parse_args()
    
    calculate_similarity_for_all_files(args.input_dir, args.force_recalculate)
