import pandas as pd
import time
import os
import re
import concurrent.futures
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize Gemini client using the new SDK
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables.")
    
client = genai.Client(api_key=api_key)

# Mock/Import utils - adjust this if your path differs
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from reporting import get_language_name
except ImportError:
    # Fallback if utility isn't found
    def get_language_name(code):
        return code

def translate_text_with_gemini(text, source_lang, target_lang, max_retries=5):
    """
    Translate text using the Gemma model with Thinking enabled.
    """
    source_lang_name = get_language_name(source_lang)
    target_lang_name = get_language_name(target_lang)

    text = str(text).strip()
    if not text:
        return ""

    prompt = f"""Translate the following {source_lang_name} text into {target_lang_name}. 
Return ONLY the translation, no explanations or additional text.

Text to translate: {text}

Translation:"""

    model_name = 'gemma-4-31b-it' 
    
    # Configure generation parameters, safety settings, and thinking config
    config = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=1024,
        top_p=0.9,
        top_k=40,
        thinking_config=types.ThinkingConfig(
            thinking_level="HIGH",
        ),
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
        ]
    )

    for attempt in range(max_retries):
        try:
            wait_time = (2 ** attempt) + 1
            
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
            
            if not response or not response.candidates:
                logger.warning(f"No candidates in response on attempt {attempt+1}")
                time.sleep(wait_time)
                continue

            # Check if content was blocked by safety filters
            finish_reason = response.candidates[0].finish_reason
            if finish_reason and getattr(finish_reason, 'name', '') == 'SAFETY':
                logger.warning(f"Content blocked by safety filters for: {text[:30]}")
                return "[Blocked by Safety]"
            
            response_text = response.text
            
            if response_text:
                # Clean the response
                response_text = response_text.strip()
                # Remove common markdown artifacting if LLM ignores "ONLY translation" instruction
                response_text = re.sub(r'^[\[\]"\'‘’“”]+|[\[\]"\'‘’“”]+$', '', response_text)
                return response_text
                
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed: {str(e)}")
            if "429" in str(e): # Rate limit
                time.sleep(wait_time * 2)
            else:
                time.sleep(wait_time)
                
    return ""

def process_single_row(index, text, source_lang, target_lang, total):
    """Worker function to process a single translation task."""
    print(f"Task {index+1}/{total} started: {text[:40]}...")
    translation = translate_text_with_gemini(text, source_lang, target_lang)
    
    if translation:
        print(f"  → Task {index+1} success: {translation[:40]}...")
    else:
        print(f"  → Task {index+1} [Failed]")
        
    return index, translation

def translation_only(df, source_lang, target_lang):
    """Perform translation using parallel processing"""
    result_df = df.copy()
    result_df['translated'] = ""

    total_texts = len(result_df)
    failed_translations = 0
    
    # Process up to 5 sentences simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all rows to the thread pool
        future_to_index = {
            executor.submit(process_single_row, i, row['text'], source_lang, target_lang, total_texts): i
            for i, row in result_df.iterrows()
        }
        
        # As each task completes, grab the result and update the dataframe
        for future in concurrent.futures.as_completed(future_to_index):
            index, translation = future.result()
            
            if translation:
                result_df.at[index, 'translated'] = translation
            else:
                failed_translations += 1

    print(f"\nBatch complete. Failed translations: {failed_translations}")
    return result_df

def process_dataframe(df, source_lang, target_lang):
    """Process the dataframe for translation only"""
    df = translation_only(df, source_lang, target_lang)
    return df
