from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel
)
from typing import Tuple, Dict
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")

#  CASUAL LANGUAGE MODELS

def load_gpt2_small() -> Tuple:
    logger.info("Loading GPT-2 small...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

def load_distilgpt2() -> Tuple:
    logger.info("Loading DistilGPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizer, model


#  MASKED LANGUAGE MODELS

def load_distilbert() -> Tuple:
    logger.info("Loading DistilBERT...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    return tokenizer, model

def load_tinybert() -> Tuple:
    logger.info("Loading TinyBERT...")
    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    return tokenizer, model

def load_all_models() -> Dict[str, Tuple]:
    logger.info("Loading all models...")
    models = {
        "gpt2": load_gpt2_small(),
        "distilgpt2": load_distilgpt2(),
        "distilbert": load_distilbert(),
        "tinybert": load_tinybert()
    }
    logger.info("All models loaded successfully.")
    return models