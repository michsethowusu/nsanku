import os
import pandas as pd
import importlib.util
from pathlib import Path
import re
import sys
import json
import random
import time
from datetime import timedelta

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def setup_api_key():
    """Check for .env file and create it if missing with user-provided API key"""
    env_file = '.env'
    if not os.path.exists(env_file):
        print("No .env file found.")
        print("Please enter your NVIDIA API key obtained from https://build.nvidia.com/")
        api_key = input("API Key: ").strip()
        
        with open(env_file, 'w') as f:
            f.write(f'NVIDIA_BUILD_API_KEY={api_key}\n')
        
        print(".env file created with your API key.")
        return api_key
    else:
        # Load existing API key from .env file
        nvidia_key = None
        
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('NVIDIA_BUILD_API_KEY='):
                    nvidia_key = line.strip().split('=', 1)[1]
        
        # If we have the NVIDIA key, return it
        if nvidia_key:
            return nvidia_key
        
        # If we get here, the .env file exists but doesn't contain the key
        print("Existing .env file found but no NVIDIA API key detected.")
        api_key = input("Please enter your NVIDIA API key: ").strip()
        
        with open(env_file, 'a') as f:
            f.write(f'\nNVIDIA_BUILD_API_KEY={api_key}\n')
        
        print("API key added to .env file.")
        return api_key

def load_recipes(recipes_dir="recipes"):
    recipes = {}
    for file in os.listdir(recipes_dir):
        if file.endswith(".py") and file != "__init__.py":
            module_name = file[:-3]
            spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(recipes_dir, file)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            recipes[module_name] = module
    return recipes

def extract_language_pair_from_filename(filename):
    """Extract language pair from filename in format source-target.csv"""
    pattern = r'^([a-zA-Z]+)-([a-zA-Z]+)\.csv$'
    match = re.match(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def load_processing_state(state_file="processing_state.json"):
    """Load the processing state from a JSON file"""
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except:
            # If file is corrupted, create a new one
            print("State file corrupted, creating a new one")
            with open(state_file, 'w') as f:
                json.dump({}, f)
            return {}
    else:
        # Create an empty state file if it doesn't exist
        print("Creating new state file")
        with open(state_file, 'w') as f:
            json.dump({}, f)
        return {}

def save_processing_state(state, state_file="processing_state.json"):
    """Save the processing state to a JSON file"""
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"State saved with {len(state)} entries")
    except Exception as e:
        print(f"Error saving state: {str(e)}")

def get_common_sentence_ids(input_dir, sample_size=200, random_seed=42):
    """Get common sentence IDs that exist in all language pairs"""
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    
    if not csv_files:
        raise ValueError("No CSV files found in input directory")
    
    # Read the first file to get available sentence IDs
    first_file = os.path.join(input_dir, csv_files[0])
    first_df = pd.read_csv(first_file)
    
    # Get all available sentence IDs from first file
    all_sentence_ids = list(range(len(first_df)))
    
    # Sample once for all languages
    random.seed(random_seed)
    selected_ids = random.sample(all_sentence_ids, min(sample_size, len(all_sentence_ids)))
    selected_ids.sort()  # Sort for consistent ordering
    
    print(f"Selected sentence IDs for all languages: {selected_ids}")
    return selected_ids

def load_sampled_data(input_dir, sentence_ids):
    """Load the same sentence IDs from all CSV files"""
    samples = {}
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    
    print(f"Loading same {len(sentence_ids)} sentences for all {len(csv_files)} language pairs...")
    
    for file in csv_files:
        input_path = os.path.join(input_dir, file)
        df = pd.read_csv(input_path)
        
        # Select the same sentence IDs for all files
        if len(df) > max(sentence_ids):
            sampled_df = df.iloc[sentence_ids].reset_index(drop=True)
            samples[file] = sampled_df
            print(f"  - {file}: loaded sentences {sentence_ids}")
        else:
            print(f"  - WARNING: {file} has only {len(df)} rows, cannot load all requested sentences")
            # Use available sentences
            available_ids = [sid for sid in sentence_ids if sid < len(df)]
            sampled_df = df.iloc[available_ids].reset_index(drop=True)
            samples[file] = sampled_df
    
    return samples

def process_csv(sampled_df, recipe_module, source_lang, target_lang, mode="translation_only"):
    # Process with the specified language codes using pre-sampled data
    if mode == "translation_only" and hasattr(recipe_module, 'translation_only'):
        processed_df = recipe_module.translation_only(sampled_df, source_lang=source_lang, target_lang=target_lang)
    elif mode == "similarity_only" and hasattr(recipe_module, 'similarity_only'):
        processed_df = recipe_module.similarity_only(sampled_df)
    else:
        processed_df = recipe_module.translation_only(sampled_df, source_lang=source_lang, target_lang=target_lang)
    
    return processed_df

def get_output_filename(input_filename, recipe_name):
    """Generate output filename with recipe prefix"""
    name, ext = os.path.splitext(input_filename)
    return f"{name}_{recipe_name}{ext}"

def run_translation_only(input_dir, output_dir, recipes, state):
    """Run only the translation part with consistent sampling and ETA"""
    print("Running translation only...")
    print(f"Initial state: {len(state)} entries")
    
    # Get common sentence IDs that will be used for ALL languages
    sentence_ids = get_common_sentence_ids(input_dir, sample_size=200, random_seed=42)
    
    # Load the same sentences for all language pairs
    samples = load_sampled_data(input_dir, sentence_ids)
    
    # Calculate total tasks for ETA
    total_tasks = 0
    completed_tasks = 0
    
    # First pass: count total tasks
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    for file in csv_files:
        source_lang, target_lang = extract_language_pair_from_filename(file)
        if not source_lang or not target_lang:
            continue
            
        for recipe_name, recipe_module in recipes.items():
            if hasattr(recipe_module, 'translation_only'):
                state_key = f"{source_lang}-{target_lang}/{file}/{recipe_name}"
                if not state.get(state_key, {}).get('translation_completed', False):
                    total_tasks += 1
    
    print(f"Total tasks to process: {total_tasks}")
    start_time = time.time()
    
    # Process each CSV file in the input directory
    for file in csv_files:
        # Extract language pair from filename
        source_lang, target_lang = extract_language_pair_from_filename(file)
        if not source_lang or not target_lang:
            print(f"Skipping {file}: filename should be in format 'source-target.csv'")
            continue
            
        input_path = os.path.join(input_dir, file)
        
        # Create output directory for this language pair
        lang_pair_dir = os.path.join(output_dir, f"{source_lang}-{target_lang}")
        os.makedirs(lang_pair_dir, exist_ok=True)
        
        # Get the pre-sampled data for this language pair (same sentences as all others)
        sampled_df = samples[file]
        print(f"\nUsing same {len(sampled_df)} sentences for {file} ({source_lang}-{target_lang})")
        
        for recipe_name, recipe_module in recipes.items():
            # Generate recipe-specific output filename
            output_filename = get_output_filename(file, recipe_name)
            output_path = os.path.join(lang_pair_dir, output_filename)
            
            # Check if this recipe has already completed translation for this file
            state_key = f"{source_lang}-{target_lang}/{file}/{recipe_name}"
            if state.get(state_key, {}).get('translation_completed', False):
                print(f"Skipping translation for {recipe_name} on {file} - already completed")
                continue
            
            print(f"Processing {file} with recipe {recipe_name} for {source_lang}-{target_lang}")
            
            try:
                # Check if recipe supports translation only mode
                if hasattr(recipe_module, 'translation_only'):
                    result_df = process_csv(sampled_df, recipe_module, 
                                          source_lang, target_lang, "translation_only")
                    result_df.to_csv(output_path, index=False)
                    
                    # Update state
                    if state_key not in state:
                        state[state_key] = {}
                    state[state_key]['translation_completed'] = True
                    state[state_key]['rows_processed'] = len(result_df)
                    state[state_key]['timestamp'] = pd.Timestamp.now().isoformat()
                    save_processing_state(state)
                    
                    completed_tasks += 1
                    
                    # Calculate ETA
                    elapsed_time = time.time() - start_time
                    if completed_tasks > 0:
                        time_per_task = elapsed_time / completed_tasks
                        remaining_tasks = total_tasks - completed_tasks
                        eta_seconds = time_per_task * remaining_tasks
                        eta_str = str(timedelta(seconds=int(eta_seconds)))
                        
                        progress = (completed_tasks / total_tasks) * 100
                        print(f"Progress: {completed_tasks}/{total_tasks} ({progress:.1f}%) - ETA: {eta_str}")
                    
                    print(f"Completed translation with {recipe_name} on {file}")
                else:
                    print(f"Recipe {recipe_name} doesn't support translation-only mode")
            except Exception as e:
                print(f"Error applying {recipe_name} to {file} for {source_lang}-{target_lang}: {str(e)}")
    
    total_time = time.time() - start_time
    print(f"\nTranslation process completed in {timedelta(seconds=int(total_time))}! Final state: {len(state)} entries")

def main():
    # Setup API key first
    api_key = setup_api_key()
    os.environ['NVIDIA_BUILD_API_KEY'] = api_key
    
    # Define input and output directories
    input_dir = "input"
    output_dir = "output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load recipes
    recipes = load_recipes()
    
    # Load processing state
    state = load_processing_state()
    print(f"Loaded state with {len(state)} entries")
    
    # Run translation automatically without menu
    run_translation_only(input_dir, output_dir, recipes, state)
    
    print("Translation completed! Now run reporting_combined.py to calculate similarity and generate reports.")

if __name__ == "__main__":
    main()
