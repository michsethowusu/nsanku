import pandas as pd
import time
import os
import re
from openai import OpenAI
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

# Add utils to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from reporting import get_language_name
except ImportError:
    def get_language_name(code):
        return code

# Initialize similarity model (kept for downstream compatibility)
similarity_model_name = "sentence-transformers/all-mpnet-base-v2"
similarity_model = SentenceTransformer(similarity_model_name)

# --- Client Initializers ---
def get_openai_compatible_client(provider):
    if provider == "nvidia":
        return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_BUILD_API_KEY"))
    elif provider == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "mistral":
        return OpenAI(base_url="https://api.mistral.ai/v1", api_key=os.getenv("MISTRAL_API_KEY"))
    elif provider == "perplexity":
        return OpenAI(base_url="https://api.perplexity.ai", api_key=os.getenv("PERPLEXITY_API_KEY"))
    elif provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
    return None

def extract_bracketed_text(text):
    """Extract text from brackets if present (for LLMs instructed to return [text])"""
    match = re.search(r'\[(.*?)\]', text, flags=re.S)
    if match:
        return match.group(1).strip()
    return text.strip()

def translate_llm(client, text, source_lang, target_lang, model_id, provider, max_retries=5):
    """Generic LLM translation handler"""
    source_lang_name = get_language_name(source_lang)
    target_lang_name = get_language_name(target_lang)

    prompt = f"Translate the following {source_lang_name} text into {target_lang_name} and return ONLY the translation inside square brackets:\n\n{text}"

    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                response = client.messages.create(
                    model=model_id,
                    max_tokens=2024,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text
            elif provider == "gemini":
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model = genai.GenerativeModel(model_id)
                response = model.generate_content(prompt, generation_config={"temperature": 0.3})
                response_text = response.text
            else:
                # Standard OpenAI compatible format (NVIDIA, OpenAI, Mistral, Perplexity)
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2024,
                    top_p=0.95,
                    stream=False
                )
                response_text = completion.choices[0].message.content
            
            return extract_bracketed_text(response_text)
                
        except Exception as e:
            print(f"Attempt {attempt+1} failed for text '{text[:20]}...': {str(e)}")
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + 1)
            else:
                return ""

def translation_only(df, source_lang, target_lang, model_id, provider):
    """Perform translation dynamically based on provider"""
    print(f"Translation: {provider.upper()}")
    print(f"Model: {model_id}")

    result_df = df.copy()
    result_df['translated'] = ""
    total_texts = len(result_df)
    
    # BCP-47 mapping shared across NLLB variants
    NLLB_LANG_MAP = {
        'en': 'eng_Latn', 'fr': 'fra_Latn', 'es': 'spa_Latn',
        'de': 'deu_Latn', 'it': 'ita_Latn', 'zh': 'zho_Hans',
        'ar': 'arb_Arab', 'pt': 'por_Latn', 'ru': 'rus_Cyrl',
        'ja': 'jpn_Jpan', 'ko': 'kor_Hang', 'nl': 'nld_Latn',
        'pl': 'pol_Latn', 'tr': 'tur_Latn', 'vi': 'vie_Latn',
    }

    # 0. Handle CTranslate2 quantized NLLB (3.3B ct2-int8)
    if provider == "nllb-ct2":
        import ctranslate2
        import torch
        from transformers import NllbTokenizer

        src_bcp47 = NLLB_LANG_MAP.get(source_lang, f"{source_lang}_Latn")
        tgt_bcp47 = NLLB_LANG_MAP.get(target_lang, f"{target_lang}_Latn")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "int8_float16" if device == "cuda" else "int8"
        print(f"CTranslate2 device={device}, compute_type={compute_type}")

        # model_id should be the HF repo, e.g. "OpenNMT/nllb-200-3.3B-ct2-int8"
        translator_ct2 = ctranslate2.Translator(
            model_id,
            device=device,
            compute_type=compute_type,
        )
        tokenizer = NllbTokenizer.from_pretrained(model_id, src_lang=src_bcp47)

        for i, row in result_df.iterrows():
            text = row['text']
            print(f"Translating {i+1}/{total_texts}: {text[:50]}...")
            try:
                tokens = tokenizer.convert_ids_to_tokens(
                    tokenizer.encode(text, add_special_tokens=True)
                )
                results = translator_ct2.translate_batch(
                    [tokens],
                    target_prefix=[[tgt_bcp47]],
                    beam_size=4,
                    max_decoding_length=256,
                )
                output_tokens = results[0].hypotheses[0][1:]  # strip the lang token prefix
                translation = tokenizer.convert_tokens_to_string(output_tokens)
                result_df.at[i, 'translated'] = translation
                print(f"  → {translation[:50]}...")
            except Exception as e:
                print(f"  → [Failed]: {e}")
                result_df.at[i, 'translated'] = ""

        return result_df

    # 1. Handle Local/Transformers Models (Load ONCE per dataframe to save time)
    if provider in ["nllb", "opus-mt"]:
        from transformers import pipeline
        
        if provider == "nllb":
            # Basic NLLB BCP-47 Mapping (Expand as needed)
            nllb_map = NLLB_LANG_MAP
            src = nllb_map.get(source_lang, f"{source_lang}_Latn")
            tgt = nllb_map.get(target_lang, f"{target_lang}_Latn")
            # Use fp16 for smaller models on GPU for speed
            import torch
            use_fp16 = torch.cuda.is_available()
            translator = pipeline(
                'translation', model=model_id,
                src_lang=src, tgt_lang=tgt,
                torch_dtype=torch.float16 if use_fp16 else torch.float32,
                device=0 if use_fp16 else -1,
            )
            
        elif provider == "opus-mt":
            # Opus-MT requires specific directional models
            hf_model_id = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            try:
                translator = pipeline('translation', model=hf_model_id)
            except Exception as e:
                print(f"Could not load Opus-MT model for {source_lang}-{target_lang}: {e}")
                return result_df

        for i, row in result_df.iterrows():
            text = row['text']
            print(f"Translating {i+1}/{total_texts}: {text[:50]}...")
            try:
                translation = translator(text)[0]['translation_text']
                result_df.at[i, 'translated'] = translation
                print(f"  → {translation[:50]}...")
            except Exception as e:
                print(f"  → [Failed]: {e}")
                
        return result_df

    # 2. Handle Google Translate (Free API via py-googletrans)
    elif provider == "googletrans":
        from googletrans import Translator
        translator_client = Translator()
        
        for i, row in result_df.iterrows():
            text = row['text']
            print(f"Translating {i+1}/{total_texts}: {text[:50]}...")
            try:
                res = translator_client.translate(text, src=source_lang, dest=target_lang)
                result_df.at[i, 'translated'] = res.text
                print(f"  → {res.text[:50]}...")
            except Exception as e:
                print(f"  → [Failed]: {e}")
            time.sleep(0.5) # Gentle rate limiting
            
        return result_df

    # 3. Handle APIs (LLMs)
    else:
        client = get_openai_compatible_client(provider) if provider != "gemini" else None
        delay = 2.0
        
        for i, row in result_df.iterrows():
            text = row['text']
            print(f"Translating {i+1}/{total_texts}: {text[:50]}...")
            
            translation = translate_llm(client, text, source_lang, target_lang, model_id, provider)
            result_df.at[i, 'translated'] = translation
            
            if translation:
                print(f"  → {translation[:50]}...")
            else:
                print("  → [Translation failed]")
            
            if i < total_texts - 1:
                time.sleep(delay)

        return result_df

# Retained downstream compatibility functions
def similarity_only(df, batch_size=32):
    # Same as your existing similarity calculation logic to not break downstream tasks
    print("Calculating similarity scores...")
    result_df = df.copy()
    if 'translated' not in result_df.columns or 'ref' not in result_df.columns:
        return result_df
    
    translated_texts = result_df['translated'].fillna('').tolist()
    ref_texts = result_df['ref'].fillna('').tolist()
    similarities = []
    
    for i in range(0, len(translated_texts), batch_size):
        batch_trans = translated_texts[i:i+batch_size]
        batch_ref = ref_texts[i:i+batch_size]
        emb_trans = similarity_model.encode(batch_trans, convert_to_tensor=True)
        emb_ref = similarity_model.encode(batch_ref, convert_to_tensor=True)
        batch_sims = util.pytorch_cos_sim(emb_trans, emb_ref)
        similarities.extend(batch_sims.diag().cpu().numpy())
    
    result_df['similarity_score'] = similarities
    return result_df

def process_dataframe(df, source_lang, target_lang, model_id, provider):
    df = translation_only(df, source_lang, target_lang, model_id, provider)
    df = similarity_only(df)
    return df
