"""
optimized_similarity_calculator.py
FIXED VERSION - Proper GPU utilization with diagnostics
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import gc
import time

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_COMBINED_PATH = "./output_combined"
BATCH_SIZE = 32
DEBUG_MODE = True
# ==============================================================================

# Enhanced GPU detection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {gpu_props.total_memory / 1024**3:.2f} GB")
    print(f"GPU Compute: {gpu_props.major}.{gpu_props.minor}")
    
    # Test GPU functionality
    try:
        test_tensor = torch.tensor([1.0]).cuda()
        print("✓ GPU basic functionality test passed")
        del test_tensor
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        device = "cpu"

# Initialize model with better GPU handling
print("\nLoading MPNet model...")
model_load_start = time.time()

try:
    similarity_model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        device=device,
        cache_folder="./model_cache"
    )
    
    # Force model to specified device
    similarity_model = similarity_model.to(device)
    
    if device == "cuda":
        # Use mixed precision for better performance
        from torch.cuda.amp import autocast
        similarity_model = similarity_model.half()  # FP16
        torch.backends.cudnn.benchmark = True
        print("✓ Model configured for FP16 on GPU")
    
    similarity_model.eval()
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Falling back to CPU")
    device = "cpu"
    similarity_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")

model_load_time = time.time() - model_load_start
print(f"✓ Model loaded in {model_load_time:.2f} seconds on {device}")

def calculate_similarity_for_indices(df, indices, batch_size=32, debug=False):
    """
    Calculate similarity scores with proper GPU utilization
    """
    if not indices:
        return df
    
    if debug:
        print(f"\n  [DEBUG] Starting calculation for {len(indices)} rows")
        print(f"  [DEBUG] Using device: {device}")
        if device == "cuda":
            print(f"  [DEBUG] GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    calc_start_time = time.time()
    
    # Extract texts
    texts_to_process = []
    ref_texts_to_process = []
    valid_indices = []
    
    for idx in indices:
        translated = str(df.loc[idx, 'translated']).strip()
        ref = str(df.loc[idx, 'ref']).strip()
        
        if translated and ref and translated != 'nan' and ref != 'nan':
            texts_to_process.append(translated)
            ref_texts_to_process.append(ref)
            valid_indices.append(idx)
        else:
            df.loc[idx, 'similarity_score'] = 0.0
            if debug:
                print(f"  [DEBUG] Row {idx}: Invalid text pair, set to 0.0")
    
    if not texts_to_process:
        print("  No valid text pairs to process")
        return df
    
    if debug:
        print(f"  [DEBUG] Valid pairs: {len(texts_to_process)}")
        print(f"  [DEBUG] First text sample: '{texts_to_process[0][:50]}...'")
    
    # Calculate similarities
    similarities = []
    total_batches = (len(texts_to_process) + batch_size - 1) // batch_size
    
    if debug:
        print(f"  [DEBUG] Processing {total_batches} batches")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts_to_process), batch_size), 
                     desc="  Calculating", total=total_batches, leave=False):
            batch_end = min(i + batch_size, len(texts_to_process))
            batch_translated = texts_to_process[i:batch_end]
            batch_ref = ref_texts_to_process[i:batch_end]
            
            try:
                # Encode with GPU optimization
                if device == "cuda":
                    # Use autocast for mixed precision
                    with torch.cuda.amp.autocast():
                        embeddings_translated = similarity_model.encode(
                            batch_translated,
                            convert_to_tensor=True,
                            device=device,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            batch_size=min(16, len(batch_translated))
                        )
                        
                        embeddings_ref = similarity_model.encode(
                            batch_ref,
                            convert_to_tensor=True,
                            device=device,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            batch_size=min(16, len(batch_ref))
                        )
                else:
                    # CPU encoding
                    embeddings_translated = similarity_model.encode(
                        batch_translated,
                        convert_to_tensor=True,
                        device=device,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=min(16, len(batch_translated))
                    )
                    
                    embeddings_ref = similarity_model.encode(
                        batch_ref,
                        convert_to_tensor=True,
                        device=device,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=min(16, len(batch_ref))
                    )
                
                # Calculate similarity
                batch_similarities = util.pytorch_cos_sim(embeddings_translated, embeddings_ref)
                batch_diagonal = batch_similarities.diag().cpu().numpy().astype(np.float32)
                similarities.extend(batch_diagonal)
                
                # Memory cleanup
                del embeddings_translated, embeddings_ref, batch_similarities
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  ⚠ GPU OOM, reducing batch size")
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    return calculate_similarity_for_indices(df, indices, max(batch_size//2, 4), debug)
                else:
                    raise e
    
    calc_time = time.time() - calc_start_time
    
    # Assign results
    for idx, similarity in zip(valid_indices, similarities):
        df.loc[idx, 'similarity_score'] = float(similarity)
    
    if debug:
        print(f"  [DEBUG] Calculation time: {calc_time:.2f}s")
        print(f"  [DEBUG] Speed: {calc_time/len(valid_indices):.3f}s per row")
        print(f"  [DEBUG] Score range: {min(similarities):.4f} to {max(similarities):.4f}")
        if device == "cuda":
            print(f"  [DEBUG] Final GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    return df

# Rest of the functions remain the same as your working version
def process_csv_file(file_path, batch_size=32, debug=False):
    """Process a single CSV file"""
    try:
        if debug:
            print(f"\n  [DEBUG] Reading CSV: {file_path}")
        
        df = pd.read_csv(file_path)
        
        if debug:
            print(f"  [DEBUG] CSV shape: {df.shape}")
            print(f"  [DEBUG] Columns: {df.columns.tolist()}")
        
        if 'translated' not in df.columns or 'ref' not in df.columns:
            print(f"  ⚠ Skipping: missing 'translated' or 'ref' columns")
            return 0, 0
        
        if 'similarity_score' not in df.columns:
            print(f"  + Adding similarity_score column")
            df['similarity_score'] = np.nan
        
        missing_mask = df['similarity_score'].isna() | (df['similarity_score'] == 0.0)
        missing_indices = df[missing_mask].index.tolist()
        
        if not missing_indices:
            print(f"  ✓ All rows have similarity scores")
            return 0, len(df)
        
        print(f"  → {len(missing_indices)}/{len(df)} rows need calculation")
        
        df = calculate_similarity_for_indices(df, missing_indices, batch_size, debug)
        
        df.to_csv(file_path, index=False)
        print(f"  ✓ Saved updated file")
        
        return len(missing_indices), len(df) - len(missing_indices)
        
    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return 0, 0

def process_output_combined_folder(output_combined_path, batch_size=32, debug=False):
    """Process all CSV files"""
    print("\n" + "="*70)
    print("PROCESSING CSV FILES FOR SIMILARITY SCORES")
    print("="*70)
    print(f"Folder: {output_combined_path}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Debug mode: {'ON' if debug else 'OFF'}")
    print("="*70 + "\n")
    
    csv_files = []
    for root, dirs, files in os.walk(output_combined_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print("⚠ No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files to process\n")
    
    total_calculated = 0
    total_existing = 0
    files_processed = 0
    files_skipped = 0
    
    overall_start = time.time()
    
    for i, file_path in enumerate(csv_files, 1):
        relative_path = os.path.relpath(file_path, output_combined_path)
        print(f"\n[{i}/{len(csv_files)}] Processing: {relative_path}")
        
        file_start = time.time()
        calculated, existing = process_csv_file(file_path, batch_size, debug)
        file_time = time.time() - file_start
        
        if calculated > 0:
            files_processed += 1
            total_calculated += calculated
            total_existing += existing
            print(f"  Time: {file_time:.2f}s ({file_time/max(calculated,1):.3f}s per row)")
        else:
            if existing > 0:
                files_skipped += 1
                total_existing += existing
        
        if device == "cuda" and i % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    overall_time = time.time() - overall_start
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Files processed: {files_processed}")
    print(f"Files skipped (complete): {files_skipped}")
    print(f"Total rows calculated: {total_calculated}")
    print(f"Total rows with existing scores: {total_existing}")
    print(f"Total time: {overall_time:.2f}s")
    if total_calculated > 0:
        print(f"Average time per row: {overall_time/total_calculated:.3f}s")
    print("="*70)

def main():
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"Output folder: {OUTPUT_COMBINED_PATH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Debug mode: {DEBUG_MODE}")
    print("="*70)
    
    if not os.path.exists(OUTPUT_COMBINED_PATH):
        print(f"\n✗ Error: Path does not exist: {OUTPUT_COMBINED_PATH}")
        return
    
    process_output_combined_folder(OUTPUT_COMBINED_PATH, BATCH_SIZE, DEBUG_MODE)
    print("\n✓ Processing complete!")

if __name__ == "__main__":
    main()
