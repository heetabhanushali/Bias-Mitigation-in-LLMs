import torch
import numpy as np
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizer
from scipy.spatial.distance import cosine
import logging

logger = logging.getLogger(__name__)

def extract_identity_groups_from_text(text):
    identity_patterns = {
        'gender': ['woman', 'man', 'female', 'male', 'girl', 'boy', 'transgender', 'non-binary'],
        'race': ['Black', 'White', 'Asian', 'Latino', 'Latina', 'Hispanic', 'Native American', 'African American'],
        'religion': ['Muslim', 'Christian', 'Jewish', 'Hindu', 'Buddhist', 'Catholic', 'Protestant'],
        'age': ['elderly', 'young', 'old', 'teenager', 'senior', 'child'],
        'profession': ['doctor', 'nurse', 'teacher', 'engineer', 'lawyer', 'CEO', 'manager', 'programmer'],
        'disability': ['disabled', 'blind', 'deaf', 'wheelchair'],
        'sexuality': ['gay', 'lesbian', 'heterosexual', 'bisexual', 'LGBTQ']
    }
    
    found_groups = []
    found_types = []
    
    text_lower = text.lower()
    
    for bias_type, terms in identity_patterns.items():
        for term in terms:
            if term.lower() in text_lower:
                found_groups.append(term)
                found_types.append(bias_type)
    
    return found_groups, found_types

def adapt_dataset_for_ibs(df, dataset_name):
    """Adapt different datasets to work with IBS by extracting identity information"""
    adapted_data = []
    
    if dataset_name == "crows_pairs":
        for _, row in df.iterrows():
            # Combine both sentences
            text = f"{row.get('sent_more', '')} {row.get('sent_less', '')}"
            bias_type = row.get('bias_type', 'unknown')
            
            target_groups, bias_types = extract_identity_groups_from_text(text)
            
            if target_groups and bias_types:
                adapted_data.append({
                    'prompt': text,
                    'target_groups': target_groups,
                    'bias_types': bias_types,
                    'original_bias_type': bias_type
                })
    
    elif dataset_name == "winobias":
        for _, row in df.iterrows():
            text = row.get('sent', '')
            bias_type = row.get('bias_type', 'gender')
            
            target_groups, bias_types = extract_identity_groups_from_text(text)
            
            # Force gender bias type for winobias
            if 'gender' not in bias_types:
                bias_types.append('gender')
            
            if target_groups:
                adapted_data.append({
                    'prompt': text,
                    'target_groups': target_groups,
                    'bias_types': bias_types,
                    'original_bias_type': bias_type
                })
    
    elif dataset_name == "real_toxicity_prompts":
        for _, row in df.iterrows():
            text = row.get('prompt', '')
            
            target_groups, bias_types = extract_identity_groups_from_text(text)
            
            if target_groups and bias_types:
                adapted_data.append({
                    'prompt': text,
                    'target_groups': target_groups,
                    'bias_types': bias_types,
                    'toxicity_score': row.get('toxicity_score', 0.0)
                })
    
    return adapted_data

def get_embedding_for_ibs(text: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, model_type="masked") -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    if model_type == "masked":
        # For BERT-style models, use CLS token
        embedding = outputs.hidden_states[-1][:, 0, :].squeeze().cpu().numpy()
    else:
        # For GPT-style models, use average of all tokens
        hidden_states = outputs.hidden_states[-1].squeeze(0)
        embedding = hidden_states.mean(dim=0).cpu().numpy()
    
    return embedding

def compute_intersectional_bias_score(prompts_with_identities, tokenizer, model, model_type="masked"):
    # logger.info("Computing Intersectional Bias Score (IBS)...")
    
    if not prompts_with_identities:
        logger.warning("No valid prompts with identities found for IBS")
        return None
    
    # Group prompts by identity combinations
    identity_groups = {}
    bias_type_groups = {}
    
    for item in prompts_with_identities:
        target_groups = item.get('target_groups', [])
        bias_types = item.get('bias_types', [])
        
        if not isinstance(target_groups, list):
            target_groups = [target_groups] if target_groups else []
        if not isinstance(bias_types, list):
            bias_types = [bias_types] if bias_types else []
        
        # Create identity combination key
        identity_key = tuple(sorted(set(target_groups)))
        bias_key = tuple(sorted(set(bias_types)))
        
        if identity_key and identity_key not in identity_groups:
            identity_groups[identity_key] = []
        if identity_key:
            identity_groups[identity_key].append(item)
        
        if bias_key and bias_key not in bias_type_groups:
            bias_type_groups[bias_key] = []
        if bias_key:
            bias_type_groups[bias_key].append(item)
    
    if len(identity_groups) < 2:
        logger.warning("Not enough identity groups for meaningful IBS computation")
        return None
    
    # Compute embeddings for each group
    identity_embeddings = {}
    for identity_key, items in identity_groups.items():
        embeddings = []
        for item in items:
            try:
                emb = get_embedding_for_ibs(item['prompt'], tokenizer, model, model_type)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Error computing embedding for {item['prompt']}: {e}")
        
        if embeddings:
            identity_embeddings[identity_key] = np.mean(embeddings, axis=0)
    
    # Compute pairwise distances between identity groups
    identity_distances = {}
    identity_keys = list(identity_embeddings.keys())
    
    for i, key1 in enumerate(identity_keys):
        for j, key2 in enumerate(identity_keys[i+1:], i+1):
            try:
                distance = cosine(identity_embeddings[key1], identity_embeddings[key2])
                identity_distances[(key1, key2)] = distance
            except Exception as e:
                logger.error(f"Error computing distance between {key1} and {key2}: {e}")
    
    # Compute intersectional bias components
    single_identity_distances = []
    intersectional_distances = []
    
    for (key1, key2), distance in identity_distances.items():
        # Count number of overlapping identities
        overlap = len(set(key1) & set(key2))
        key1_size = len(key1)
        key2_size = len(key2)
        
        # Single identity comparison (same number of identities, no overlap)
        if key1_size == key2_size == 1 and overlap == 0:
            single_identity_distances.append(distance)
        
        # Intersectional comparison (multiple identities involved)
        elif key1_size > 1 or key2_size > 1:
            intersectional_distances.append(distance)
    
    # Compute overall IBS
    intersectional_bias = np.mean(intersectional_distances) if intersectional_distances else 0.0
    single_identity_bias = np.mean(single_identity_distances) if single_identity_distances else 0.0
    
    # IBS = difference between intersectional and single-identity bias
    ibs_score = intersectional_bias - single_identity_bias
    
    # Compute intersectionality amplification factor
    amplification_factor = intersectional_bias / single_identity_bias if single_identity_bias > 0 else 0.0
    
    # logger.info(f"IBS Score: {ibs_score:.4f}")
    # logger.info(f"Intersectional Bias: {intersectional_bias:.4f}")
    # logger.info(f"Single Identity Bias: {single_identity_bias:.4f}")
    # logger.info(f"Amplification Factor: {amplification_factor:.4f}")
    
    return {
        'ibs_score': ibs_score,
        'intersectional_bias': intersectional_bias,
        'single_identity_bias': single_identity_bias,
        'amplification_factor': amplification_factor,
        'num_identity_groups': len(identity_groups),
        'num_comparisons': len(identity_distances)
    }

def compute_ibs_for_dataset(df, tokenizer, model, model_type="masked", dataset_name="custom"):
    """Enhanced IBS computation that works with all datasets"""
    
    if dataset_name == "custom_intersectional":
        # Use original logic for custom dataset
        prompts_with_identities = []
        for _, row in df.iterrows():
            if pd.notna(row.get('prompt')) and pd.notna(row.get('target_groups')) and pd.notna(row.get('bias_types')):
                try:
                    target_groups = eval(row['target_groups']) if isinstance(row['target_groups'], str) else row['target_groups']
                    bias_types = eval(row['bias_types']) if isinstance(row['bias_types'], str) else row['bias_types']
                    
                    prompts_with_identities.append({
                        'prompt': row['prompt'],
                        'target_groups': target_groups,
                        'bias_types': bias_types
                    })
                except Exception as e:
                    logger.error(f"Error processing custom dataset row: {e}")
    else:
        # Adapt other datasets
        prompts_with_identities = adapt_dataset_for_ibs(df, dataset_name)
    
    if not prompts_with_identities:
        logger.warning(f"No valid prompts with identities in IBS found for dataset")
        return None
    
    return compute_intersectional_bias_score(prompts_with_identities, tokenizer, model, model_type)