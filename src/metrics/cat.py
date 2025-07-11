import torch
import numpy as np
from scipy.spatial.distance import cosine
import logging

logger = logging.getLogger(__name__)

def get_contextual_embedding(tokenizer, model, text, device='cpu'):

    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states = True)
    
    # For masked LM models (like BERT), use CLS token embedding
    if hasattr(outputs, 'last_hidden_state'):
        embedding = outputs.last_hidden_state[:,0,:].squeeze().cpu().numpy()
    # For causal LM models (like GPT2), use last token hidden state
    elif hasattr(outputs, 'hidden_states'):
        # take last layer hidden state of last token
        hidden_states = outputs.hidden_states[-1].squeeze(0)
        embedding = hidden_states[-1].cpu().numpy()
    else:
        raise ValueError("Model output does not have expected hidden states")
    
    return embedding

def association(w, A_embs, B_embs):
    sim_a = np.mean([1 - cosine(w, a) for a in A_embs])
    sim_b = np.mean([1 - cosine(w, b) for b in B_embs])
    return sim_a - sim_b

def cat_score(X_embs, Y_embs, A_embs, B_embs):
    assoc_X = [association(x, A_embs, B_embs) for x in X_embs]
    assoc_Y = [association(y, A_embs, B_embs) for y in Y_embs]
    return np.mean(assoc_X) - np.mean(assoc_Y)