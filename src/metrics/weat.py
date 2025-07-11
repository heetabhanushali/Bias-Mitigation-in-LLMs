import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


def get_embedding_gpt(text: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> np.ndarray:
    """Get average hidden state embedding from causal LM (GPT-style)."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1].squeeze(0)  # (seq_len, hidden_size)
    avg_embedding = last_hidden.mean(dim=0)
    return avg_embedding.cpu().numpy()


def get_embedding_bert(text: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> np.ndarray:
    """Get [CLS] token embedding from masked LM (BERT-style)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    cls_embedding = outputs.hidden_states[-1][:, 0, :]  # shape: (1, hidden_size)
    return cls_embedding.squeeze(0).cpu().numpy()


def weat_score(X, Y, A, B, tokenizer, model, model_type="masked") -> float:
    # logger.info(f"Computing WEAT score for model type '{model_type}'...")

    if model_type == "masked":
        get_embedding = lambda w: get_embedding_bert(w, tokenizer, model)
    else:
        get_embedding = lambda w: get_embedding_gpt(w, tokenizer, model)

    X_vecs = [get_embedding(w) for w in X]
    Y_vecs = [get_embedding(w) for w in Y]
    A_vecs = [get_embedding(w) for w in A]
    B_vecs = [get_embedding(w) for w in B]

    def cosine_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def s(w, A_vecs, B_vecs):
        return np.mean([cosine_sim(w, a) for a in A_vecs]) - np.mean([cosine_sim(w, b) for b in B_vecs])

    s_X = np.mean([s(x, A_vecs, B_vecs) for x in X_vecs])
    s_Y = np.mean([s(y, A_vecs, B_vecs) for y in Y_vecs])
    score = s_X - s_Y

    # logger.info(f"WEAT score: {score:.4f}")
    return score