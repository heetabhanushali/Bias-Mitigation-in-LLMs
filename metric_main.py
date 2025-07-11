import os
import pandas as pd
from src.models import load_all_models
from src.metrics.weat import weat_score
from src.metrics.toxicity import score_toxicity_texts
from src.metrics.cat import get_contextual_embedding, cat_score
from src.metrics.ibs import compute_ibs_for_dataset
import logging
import torch
import datetime
from src.utils import load_saved_datasets, prepare_word_sets, generate_text_samples


logger = logging.getLogger(__name__)


def run_all_metrics_and_collect():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    datasets = load_saved_datasets()
    models = load_all_models()

    if not datasets:
        logger.error("No datasets loaded!")
        return pd.DataFrame()
    
    if not models:
        logger.error("No models loaded!")
        return pd.DataFrame()
    
    results = []
    total_combinations = len(datasets) * len(models)
    current_combination = 0

    for ds_name, df in datasets.items():
        print(f"\n{'='*50}")
        print(f"Processing dataset: {ds_name} ({len(df)} rows)")
        print(f"{'='*50}")

        try:
            X, Y, A, B = prepare_word_sets(ds_name, df)
            if not all([X, Y, A, B]):
                print(f"Skipping {ds_name} due to missing or invalid word sets.")
                continue
            logger.info(f"Prepared word sets for {ds_name}: X = {len(X)}, Y = {len(Y)}, A = {len(A)}, B = {len(B)}")
        except Exception as e:
            logger.error(f"Error preparing word sets for {ds_name}: {e}")
            continue

        try:
            if ds_name == "real_toxicity_prompts":
                texts = df['prompt'].dropna().tolist()
            elif ds_name == "custom_intersectional":
                texts = df['prompt'].dropna().tolist() if 'prompt' in df.columns else []
            elif ds_name == "crows_pairs":
                texts = df['sent_more'].dropna().tolist() if 'sent_more' in df.columns else []
            elif ds_name == "winobias":
                texts = df['sent'].dropna().tolist() if 'sent' in df.columns else []
            else:
                texts = df.iloc[:, 0].dropna().astype(str).tolist()
            
            if not texts:
                texts = ["Sample text for evaluation"]
        except Exception as e:
            logger.error(f"Error preparing texts for {ds_name}: {e}")
            texts = ["Sample text for evaluation"]
        
        for model_name, (tokenizer, model) in models.items():
            current_combination += 1
            logger.info(f"\n[{current_combination}/{total_combinations}] Model: {model_name} on dataset: {ds_name}")
            model_type = "masked" if model_name in ["distilbert", "tinybert"] else "causal"

            weat_result = None
            tox_result = None
            cat_result = None
            ibs_result = None

            # WEAT score
            try:
                weat_result = weat_score(X, Y, A, B, tokenizer, model, model_type=model_type)
                # logger.info(f"WEAT Score: {weat_result:.4f}" if weat_result else "WEAT: Failed")
            except Exception as e:
                logger.error(f"WEAT error: {e}")

            # Toxicity score
            try:
                generated_texts = generate_text_samples(model_name, tokenizer, model, texts)
                tox_result = score_toxicity_texts(generated_texts)
                # logger.info(f"Toxicity Score: {tox_result:.4f}" if tox_result else "Toxicity: Failed")
            except Exception as e:
                logger.error(f"Toxicity scoring error: {e}")

            # CAT score
            try:
                X_embs = [get_contextual_embedding(tokenizer, model, x, device) for x in X]
                Y_embs = [get_contextual_embedding(tokenizer, model, y, device) for y in Y]
                A_embs = [get_contextual_embedding(tokenizer, model, a, device) for a in A]
                B_embs = [get_contextual_embedding(tokenizer, model, b, device) for b in B]
                cat_result = cat_score(X_embs, Y_embs, A_embs, B_embs)
                # logger.info(f"CAT Score: {cat_result:.4f}" if cat_result else "CAT: Failed")
            except Exception as e:
                logger.error(f"CAT error: {e}")

            # IBS score
            try:
                ibs_result_dict = compute_ibs_for_dataset(df, tokenizer, model, model_type, ds_name)
                ibs_result = ibs_result_dict["ibs_score"] if ibs_result_dict else None
                # logger.info(f"IBS Score: {ibs_result:.4f}" if ibs_result else "IBS: Failed")
            except Exception as e:
                logger.error(f"IBS error: {e}")


            results.append({
                "Dataset": ds_name,
                "Model": model_name,
                "WEAT": weat_result,
                "Toxicity": tox_result,
                "CAT": cat_result,
                "Custom IBS": ibs_result,
                "Model_Type": model_type
            })

    return pd.DataFrame(results)


def print_results_matrix(df_results):
    print(f"\n{'='*80}")
    print("BIAS EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for metric in ["WEAT", "Toxicity", "CAT", "Custom IBS"]:
        print(f"\n=== {metric} Scores Matrix ===")
        pivot = df_results.pivot(index="Dataset", columns="Model", values=metric)
        print(pivot.fillna("N/A"))
        
        # Print basic statistics
        metric_values = df_results[metric].dropna()
        if not metric_values.empty:
            print(f"\n{metric} Statistics:")
            print(f"  Mean: {metric_values.mean():.4f}")
            print(f"  Std:  {metric_values.std():.4f}")
            print(f"  Min:  {metric_values.min():.4f}")
            print(f"  Max:  {metric_values.max():.4f}")


def save_results(df_results, output_dir="results/report"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"bias_evaluation_results_{timestamp}.csv")
    df_results.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")



if __name__ == "__main__":
    try:
        logger.info("Starting bias evaluation framework...")
        df_results = run_all_metrics_and_collect()
        
        if not df_results.empty:
            print_results_matrix(df_results)
            save_results(df_results)
        else:
            logger.error("No results generated!")
            
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        raise