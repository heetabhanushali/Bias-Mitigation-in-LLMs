import os
import pandas as pd
import logging
import torch
import datetime
from src.models import load_all_models
from src.utils import load_saved_datasets, prepare_word_sets ,generate_text_samples
from src.metrics.weat import weat_score
from src.metrics.toxicity import score_toxicity_texts
from src.metrics.cat import get_contextual_embedding, cat_score
from src.metrics.ibs import compute_ibs_for_dataset
from src.mitigation.context_injection import apply_context_injection
from src.mitigation.post_filtering import post_filtering_pipeline
from src.mitigation.prompt_cleaning import batch_clean_prompts
from src.mitigation.prompt_engineering import batch_engineer_prompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


STRATEGIES = [
    # "context_injection",
    # "post_filtering",
    # "prompt_cleaning",
    "prompt_engineering"
]

BATCH_SIZE = 100

def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]
        print(f"Processed batch {i // batch_size + 1} of {len(lst) // batch_size + 1}")


def run_all_mitigation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = load_saved_datasets()
    models = load_all_models()
    results = []
    batch_size = 100 
    

    for strategy in STRATEGIES:
        logger.info(f"\n{'#'*60}\nApplying Prompt Strategy: {strategy}\n{'#'*60}")

        for ds_name, df in datasets.items():
            try:
                X, Y, A, B = prepare_word_sets(ds_name, df)
                print(f"Prepared word sets for {ds_name}: X={X}, Y={Y}, A={A}, B={B}")
                if not all([X, Y, A, B]):
                    logger.warning(f"Skipping {ds_name} for {strategy} due to invalid word sets.")
                    continue
            except Exception as e:
                logger.error(f"Error preparing word sets for {ds_name}: {e}")
                continue

            # Select text column based on dataset
            if ds_name == "real_toxicity_prompts":
                prompts = df['prompt'].dropna().tolist()
            elif ds_name == "custom_intersectional":
                prompts = df['prompt'].dropna().tolist()
            elif ds_name == "crows_pairs":
                prompts = df['sent_more'].dropna().tolist()
            elif ds_name == "winobias":
                prompts = df['sent'].dropna().tolist()
            else:
                prompts = df.iloc[:, 0].dropna().astype(str).tolist()

            # logger.info(f"Cleaning prompts for {ds_name} in batches of {batch_size}")
            # cleaned_prompts = batch_clean_prompts(prompts , batch_size= BATCH_SIZE)

            engineered_prompts = batch_engineer_prompts(prompts, batch_size=BATCH_SIZE)
            

            for model_name, (tokenizer, model) in models.items():
                logger.info(f"Running {strategy} on {ds_name} using {model_name}")
                model_type = "masked" if model_name in ["distilbert", "tinybert"] else "causal"

                gen_texts = []

                if strategy == "prompt_cleaning":
                    for batch in batchify(cleaned_prompts, batch_size):
                        gen_texts.extend(generate_text_samples(model_name, tokenizer, model, batch))
                
                if strategy == "prompt_engineering":
                    # logger.info(f"engineering prompts for {ds_name}")
                    for batch in batchify(engineered_prompts, batch_size):
                        gen_texts.extend(generate_text_samples(model_name, tokenizer, model, batch))


                elif strategy == "post_filtering":
                    if model_type == "causal":
                      
                        try:
                            gen_texts = post_filtering_pipeline(tokenizer,model, prompts,model_name, num_samples=3, generation_batch_size = int(BATCH_SIZE/4))
                        except Exception as e:
                            logger.error(f"Post-filtering error: {e}")
                            gen_texts = prompts 
                    else:
                        logger.info(f"Skipping post-filtering for masked model {model_name}")
                        continue  

                else:
                    mitigated_prompts = apply_context_injection(prompts, strategy=strategy)
                    for batch in batchify(mitigated_prompts, batch_size):
                        gen_texts.extend(generate_text_samples(model_name, tokenizer, model, batch))

                # Now calculate metrics on gen_texts
                try:
                    m_tox_result = score_toxicity_texts(gen_texts)
                except Exception as e:
                    logger.error(f"Toxicity error: {e}")
                    m_tox_result = None

                try:
                    m_weat_result = weat_score(X, Y, A, B, tokenizer, model, model_type=model_type)
                except Exception as e:
                    logger.error(f"WEAT error: {e}")
                    m_weat_result = None

                try:
                    X_embs = [get_contextual_embedding(tokenizer, model, x, device) for x in X]
                    Y_embs = [get_contextual_embedding(tokenizer, model, y, device) for y in Y]
                    A_embs = [get_contextual_embedding(tokenizer, model, a, device) for a in A]
                    B_embs = [get_contextual_embedding(tokenizer, model, b, device) for b in B]
                    m_cat_result = cat_score(X_embs, Y_embs, A_embs, B_embs)
                except Exception as e:
                    logger.error(f"CAT error: {e}")
                    m_cat_result = None

                try:
                    m_ibs_result = compute_ibs_for_dataset(df, tokenizer, model, model_type, ds_name).get("ibs_score", None)
                except Exception as e:
                    logger.error(f"IBS error: {e}")
                    m_ibs_result = None

                results.append({
                    "Dataset": ds_name,
                    "Model": model_name,
                    "Mitigation": strategy,
                    "WEAT": m_weat_result,
                    "Toxicity": m_tox_result,
                    "CAT": m_cat_result,
                    "Custom IBS": m_ibs_result,
                    "Model_Type": model_type
                })
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return pd.DataFrame(results)



def print_mitigation_results(df_results):
    print("\n" + "=" * 80)
    print("MITIGATION RESULTS (Grouped by Strategy)")
    print("=" * 80)

    metrics = ["WEAT", "Toxicity", "CAT", "Custom IBS"]

    for strategy in df_results["Mitigation"].unique():
        print(f"\n{'-'*30} STRATEGY: {strategy} {'-'*30}")
        subset = df_results[df_results["Mitigation"] == strategy]

        for metric in metrics:
            print(f"\n--- {metric} ---")
            pivot = subset.pivot_table(index="Dataset", columns="Model", values=metric, aggfunc="mean")
            print(pivot.round(4).fillna("N/A"))


def save_results(df_results, output_dir="results/report"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"mitigation_results_{timestamp}.csv")
    df_results.to_csv(path, index=False)
    logger.info(f"Saved mitigation results to {path}")


if __name__ == "__main__":
    df_results = run_all_mitigation()
    if not df_results.empty:
        print_mitigation_results(df_results)
        save_results(df_results)
    else:
        logger.error("No mitigated results generated.")
