import os
import pandas as pd
import logging
import torch
from typing import Tuple, List


logger = logging.getLogger(__name__)

def load_saved_datasets(dir_path="processed_datasets") -> dict:
    datasets = {}
    if not os.path.exists(dir_path):
        logger.warning(f"Directory {dir_path} does not exist")
        return datasets
        
    for file in os.listdir(dir_path):
        if file.endswith(".csv"):
            name = file.replace(".csv", "")
            path = os.path.join(dir_path, file)
            try:
                df = pd.read_csv(path)
                datasets[name] = df
                logger.info(f"Loaded {name}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
    return datasets


def prepare_word_sets(dataset_name : str, df : pd.DataFrame) -> Tuple[list[str] , List[str], List[str], List[str]] :
    X, Y, A, B = [], [], [], []

    # ____________________ CUSTOM - DATASET ______________________
    if dataset_name == "custom_intersectional":
        all_targets = set()
        all_bias = set()
        
        for _, row in df.iterrows():
            if pd.notna(row.get('target_groups')):
                try:
                    targets = eval(row['target_groups']) if isinstance(row['target_groups'], str) else row['target_groups']
                    if isinstance(targets, list):
                        all_targets.update(targets)
                except:
                    continue
            
            if pd.notna(row.get('bias_types')):
                try:
                    biases = eval(row['bias_types']) if isinstance(row['bias_types'], str) else row['bias_types']
                    if isinstance(biases, list):
                        all_bias.update(biases)
                except:
                    continue
        
        all_targets = list(all_targets)
        all_bias = list(all_bias)
        
        half = len(all_targets) // 2
        X = all_targets[:half] if half > 0 else ["woman", "female"]
        Y = all_targets[half:] if half > 0 else ["man", "male"]
        
        half_b = len(all_bias) // 2
        A = all_bias[:half_b] if half_b > 0 else ["positive"]
        B = all_bias[half_b:] if half_b > 0 else ["negative"]


    # ____________________ CROWS - PAIRS ______________________
    elif dataset_name == "crows_pairs":
        if "bias_type" not in df.columns:
            return [], [], [], []
        
        positive_attrs = ["good", "positive", "pleasant", "successful", "competent"]
        negative_attrs = ["bad", "negative", "unpleasant", "unsuccessful", "incompetent"]

        target_groups = set()
        for _, row in df.iterrows():
            sent_more = str(row.get("sent_more", ""))
            sent_less = str(row.get("sent_less", ""))
        
        words = (sent_more + " " + sent_less).lower().split()

        demographic_terms = ["woman", "man", "girl", "boy", "black", "white", "asian", 
                               "muslim", "christian", "jewish", "elderly", "young"]
            
        for word in words:
            if word in demographic_terms:
                target_groups.add(word)
        
        target_groups = list(target_groups)

        if len(target_groups) >= 2:
            half = len(target_groups) // 2
            X = target_groups[:half]
            Y = target_groups[half:]
        else:
            X = ["stereotypical"]
            Y = ["anti-stereotypical"]
        
        A = positive_attrs
        B = negative_attrs
            

# ____________________ WINOBIAS ______________________
    elif dataset_name == "winobias":
        if "bias_type" not in df.columns or "sent" not in df.columns:
            print(f"Missing columns in winobias: {df.columns}")
            return [], [], [], []
        A = ["anti_stereotypical"]
        B = ["pro_stereotypical"]
        gendered_terms_male = ['he', 'him', 'his', 'man', 'male', 'boy']
        gendered_terms_female = ['she', 'her', 'hers', 'woman', 'female', 'girl']
        X = []
        Y = []
        for _, row in df.iterrows():
            sentence = str(row["sent"]).lower()
            bias = str(row["bias_type"]).strip().lower()
            tokens = sentence.split()
            male_words = [w for w in tokens if w in gendered_terms_male]
            female_words = [w for w in tokens if w in gendered_terms_female]
            if bias == A[0]:
                if female_words:
                    X.extend(female_words)
                else:
                    X.extend([w for w in tokens if w not in gendered_terms_male + gendered_terms_female][:4])
            elif bias == B[0]:
                if male_words:
                    Y.extend(male_words)
                else:
                    Y.extend([w for w in tokens if w not in gendered_terms_male + gendered_terms_female][:4])
            if len(X) >= 10 and len(Y) >= 10:
                break
        X = list(dict.fromkeys(X))[:10]
        Y = list(dict.fromkeys(Y))[:10]


# ____________________ TOXICITY ______________________
    elif dataset_name == "real_toxicity_prompts":
        prompts = df['prompt'].dropna().tolist()

        high_toxicity = []
        low_toxicity = []
        
        for _, row in df.iterrows():
            prompt = row.get('prompt', '')
            toxicity = row.get('toxicity_score', 0.0)
            
            if toxicity > 0.5:
                high_toxicity.append(prompt)
            else:
                low_toxicity.append(prompt)
        
        X = high_toxicity[:10] if high_toxicity else prompts[:10]
        Y = low_toxicity[:10] if low_toxicity else prompts[10:20]
        A = ["toxic", "harmful", "offensive"]
        B = ["safe", "neutral", "appropriate"]


    # Ensure all elements are strings and limit length
    X = list(map(str, X))[:10]
    Y = list(map(str, Y))[:10]
    A = list(map(str, A))
    B = list(map(str, B))

    return X, Y, A, B



def generate_text_samples(model_name : str, tokenizer, model, prompts : list[str], max_length : int=300, num_samples : int = 10) -> List[str]:
    generated_texts = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try: 
        model.to(device)

        valid_prompts = [p for p in prompts if isinstance(p, str)][:num_samples]
        if not valid_prompts:
            logger.warning(f"No valid prompts provided for {model_name}. Returning empty list.")
            return ["Sample generated text"]
        
        for prompt in valid_prompts:
            try:
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                inputs = tokenizer(prompt, return_tensors="pt" , truncation = True , max_length = 128).to(device)

                if model_name in ["gpt2", "distilgpt2"]:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length = min(max_length, 1000),
                            do_sample=True,
                            top_k=50,
                            top_p=0.95,
                            num_return_sequences=1,
                            pad_token_id=tokenizer.eos_token_id,
                            temperature=0.7
                        )
                    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_texts.append(text)
                else:
                    generated_texts.append(prompt)
            except Exception as e:
                logger.error(f"Error generating text for prompt '{prompt}': {e}")
                generated_texts.append(prompt)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except Exception as e:
        logger.error(f"Error in text generation setup: {e}")
        return prompts[:num_samples] 
    return generated_texts


