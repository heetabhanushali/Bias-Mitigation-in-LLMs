from detoxify import Detoxify
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Load detoxify model once
detox_model = Detoxify('original')

def score_toxicity_texts(texts):
    if not texts:
        logger.warning("Empty text list for toxicity scoring.")
        return None
    
    results = []
    for text in texts:
        try:
            scores = detox_model.predict(text)
            toxicity_score = scores.get('toxicity', 0.0)
            results.append(toxicity_score)
        except Exception as e:
            logger.error(f"Error scoring text: {e}")
    
    if results:
        avg_score = np.mean(results)
        # logger.info(f"Average detoxify toxicity score: {avg_score:.4f}")
        return avg_score
    else:
        logger.warning("No toxicity scores computed.")
        return None