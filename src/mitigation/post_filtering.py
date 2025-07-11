import torch
from detoxify import Detoxify
import logging
from typing import List
from transformers import PreTrainedTokenizer, PreTrainedModel

# Set logging level to DEBUG for detailed output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Detoxify model once globally
detox_model = Detoxify('original')

def generate_multiple_samples_batch(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prompts: List[str],
    model_name: str,
    num_samples: int = 3,
    max_length: int = 75,
    batch_size: int = 50
) -> List[List[str]]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_outputs = []

    logger.info(f"Starting text generation for {len(prompts)} prompts...")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        logger.info(f"Generating batch {i // batch_size + 1} of {len(prompts) // (batch_size + 1)}")

        # Process each prompt individually to avoid repeat_interleave issues
        batch_generated = []
        
        for prompt in batch_prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)

            try:
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    num_return_sequences=num_samples,
                    pad_token_id=tokenizer.eos_token_id
                )

                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                batch_generated.append(decoded)

                logger.debug(f"Generated {len(decoded)} outputs for prompt: {prompt[:50]}...")
                # logger.debug(f"\ninput: {prompt} \n output: {decoded}")  # Log first two outputs for brevity

            except Exception as e:
                logger.error(f"Error generating for prompt '{prompt[:50]}...': {e}")
                batch_generated.append([""]*num_samples)  # fallback

        all_outputs.extend(batch_generated)
        # logger.debug(f"Completed batch {i // batch_size + 1}")

    # logger.debug("Completed text generation for all prompts.")
    return all_outputs


def select_least_toxic_outputs_for_batches(
    all_texts: List[List[str]],
    scoring_batch_size: int = 50
) -> List[str]:
    least_toxic_outputs = []
    logger.info("Scoring toxicity for each set of outputs...")

    for i, texts in enumerate(all_texts):
        try:
            toxicity_scores = []
            for j in range(0, len(texts), scoring_batch_size):
                batch = texts[j:j + scoring_batch_size]
                scores = detox_model.predict(batch)['toxicity']
                toxicity_scores.extend(scores)

            min_index = toxicity_scores.index(min(toxicity_scores))
            least_toxic = texts[min_index]
            least_toxic_outputs.append(least_toxic)

            # logger.debug(f"Prompt {i}: Least toxic = {toxicity_scores[min_index]:.4f} | Output = {least_toxic[:60]}")

        except Exception as e:
            logger.error(f"Toxicity scoring failed at prompt {i}: {e}")
            least_toxic_outputs.append(texts[0] if texts else "")  # fallback

    logger.info("Completed toxicity filtering for all prompts.")
    return least_toxic_outputs


def post_filtering_pipeline(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prompts: List[str],
    model_name: str,
    num_samples: int = 3,
    generation_batch_size: int = 50
) -> List[str]:
    try:
        generated_samples = generate_multiple_samples_batch(
            tokenizer, model, prompts, model_name,
            num_samples=num_samples,
            batch_size=generation_batch_size
        )

        filtered_outputs = select_least_toxic_outputs_for_batches(generated_samples)
        logger.info("Post-filtering completed successfully.")
        return filtered_outputs

    except Exception as e:
        logger.error(f"Post-filtering pipeline failed: {e}")
        return prompts  # fallback if failure