import re
import logging
from detoxify import Detoxify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model once
detox_model = Detoxify('original')


def clean_toxic_prompt(prompt, threshold=0.65):
    try:
        original_score = detox_model.predict([prompt])['toxicity'][0]
        if original_score < threshold and not is_softly_biased(prompt):
            return prompt

        cleaned_prompt = prompt
        cleaned_prompt = remove_toxic_words(cleaned_prompt)
        cleaned_prompt = neutralize_descriptors(cleaned_prompt)
        cleaned_prompt = remove_inflammatory_language(cleaned_prompt)
        cleaned_prompt = replace_biased_terms(cleaned_prompt)

        new_score = detox_model.predict([cleaned_prompt])['toxicity'][0]
        if new_score < threshold and cleaned_prompt != prompt:
            return cleaned_prompt
        else:
            return generate_neutral_alternative(prompt)

    except Exception as e:
        logger.error(f"Cleaning error: {e}")
        return prompt


def remove_toxic_words(text):
    toxic_words = {
        'damn', 'shit', 'fuck', 'bitch', 'bastard', 'ass', 'crap',
        'idiot', 'moron', 'stupid', 'dumb', 'retard', 'loser',
        'hate', 'disgust', 'despise', 'pathetic', 'worthless',
        'nasty', 'scum', 'garbage', 'trash', 'sissy', 'whore'
    }
    pattern = r'\b(' + '|'.join(re.escape(word) for word in toxic_words) + r')\b'
    return re.sub(pattern, '', text, flags=re.IGNORECASE)


def neutralize_descriptors(text):
    replacements = {
        r'\b(awful|terrible|horrible|disgusting|repulsive|vile)\b': 'different',
        r'\b(evil|wicked|vicious|cruel|malicious)\b': 'challenging',
        r'\b(inferior|worthless|useless|backward|dumb)\b': 'alternative',
        r'\b(aggressive|violent|dangerous)\b': 'assertive',
        r'\b(lazy|incompetent|incapable)\b': 'developing',
        r'\b(ugly|hideous|repugnant)\b': 'unique',
        r'\b(crazy|insane|nuts|psychotic)\b': 'unconventional',
        r'\b(failed|failure|hopeless)\b': 'learning',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def remove_inflammatory_language(text):
    patterns = [
        r'\b(obviously|clearly|definitely)\s+(wrong|bad|inferior)',
        r'\b(all|every|most)\s+(women|men|people)\s+(are|can\'t|don\'t)',
        r'\b(never|always|constantly)\s+(fails|sucks|wrong|complains)',
        r'\b(typical|classic)\s+(woman|man|person)',
        r'\b(just|only)\s+(a|an)\s+(woman|man|girl|boy)',
        r'\b(they|she|he)\s+(probably|must have)\s+(gotten|received|been given).*?(job|position|award)',
        r'\b(they|she|he)\s+(don’t|doesn’t|can’t|won’t)\s+(deserve|belong|qualify)',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text


def replace_biased_terms(text):
    replacements = {
        r'\b(chairman|chairwoman)\b': 'chairperson',
        r'\b(policeman|policewoman)\b': 'police officer',
        r'\b(fireman|firewoman)\b': 'firefighter',
        r'\b(mankind)\b': 'humanity',
        r'\b(manpower)\b': 'workforce',
        r'\b(old people|elderly)\b': 'older adults',
        r'\b(young people|kids|child)\b': 'younger individuals',
        r'\b(disabled|handicapped|crippled)\b': 'people with disabilities',
        r'\b(normal people)\b': 'typical individuals',
        r'\b(poor people|the poor)\b': 'people with lower incomes',
        r'\b(rich people|the rich)\b': 'people with higher incomes',
        r'\b(the blacks|the gays|the muslims|the jews)\b': 'members of that community'
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def is_softly_biased(text):
    # Detect phrases like "emotional woman", "strong man", etc.
    soft_bias_phrases = [
        r'\bemotional\s+woman\b',
        r'\bstrong\s+man\b',
        r'\bweak\s+girl\b',
        r'\bassertive\s+man\b',
        r'\btimid\s+woman\b',
        r'\bsurprisingly\s+competent\b',
        r'\bmust have gotten the job through.*\b',
        r'\bprobably didn’t deserve\b',
    ]
    for pattern in soft_bias_phrases:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def generate_neutral_alternative(original_prompt):
    if any(word in original_prompt.lower() for word in ['doctor', 'medical', 'hospital']):
        return "The healthcare professional provided care to patients."
    elif any(word in original_prompt.lower() for word in ['teacher', 'school', 'student']):
        return "The educator worked with students in the classroom."
    elif any(word in original_prompt.lower() for word in ['work', 'job', 'career']):
        return "The person pursued their professional goals."
    elif any(word in original_prompt.lower() for word in ['family', 'parent', 'child']):
        return "The family members spent time together."
    else:
        return "The individual engaged in daily activities."


def batch_clean_prompts(prompts, batch_size=10, threshold=0.65):
    cleaned_prompts = []
    cleaning_stats = {"cleaned": 0, "unchanged": 0, "failed": 0}

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        try:
            batch_cleaned = [clean_toxic_prompt(p, threshold) for p in batch]
            for original, cleaned in zip(batch, batch_cleaned):
                cleaned_prompts.append(cleaned)
                if cleaned != original:
                    cleaning_stats["cleaned"] += 1
                else:
                    cleaning_stats["unchanged"] += 1
        except Exception as e:
            logger.error(f"Batch error: {e}")
            cleaned_prompts.extend(batch)
            cleaning_stats["failed"] += len(batch)

    logger.info(f"Cleaning stats: {cleaning_stats}")
    return cleaned_prompts
