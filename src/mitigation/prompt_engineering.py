import re
import logging

logger = logging.getLogger(__name__)

class PromptEngineering:
    def __init__(self):
        pass
        
    def apply(self, prompt: str) -> str:
        """Apply gender-balanced prompt engineering."""
        return self._gender_balanced_prompt(prompt)

    def _gender_balanced_prompt(self, prompt):
        """Replace gendered terms with gender-balanced alternatives and handle multiple bias types."""
        substitutions = {
            # Professions (Gender bias)
            r"\bdoctor\b": "doctor (who could be any gender)",
            r"\bnurse\b": "nurse (who could be any gender)",
            r"\bengineer\b": "engineer (who could be any gender)",
            r"\bteacher\b": "teacher (who could be any gender)",
            r"\bCEO\b": "CEO (who could be any gender)",
            r"\blawyer\b": "lawyer (who could be any gender)",
            r"\bscientist\b": "scientist (who could be any gender)",
            r"\bpilot\b": "pilot (who could be any gender)",
            r"\bmechanic\b": "mechanic (who could be any gender)",
            r"\breceptionist\b": "receptionist (who could be any gender)",
            r"\bcashier\b": "cashier (who could be any gender)",
            r"\bmaid\b": "cleaner (who could be any gender)",
            
            # Pronouns and possessives (Gender bias)
            r"\bHe\b": "He or she",
            r"\bShe\b": "He or she",
            r"\bHis\b": "His or her",
            r"\bHer\b": "His or her",
            r"\bHim\b": "Him or her",
            r"\bhis\b": "his or her",
            r"\bher\b": "his or her",
            r"\bhim\b": "him or her",
            r"\bhimself\b": "himself or herself",
            r"\bherself\b": "himself or herself",
            
            # Generic terms (Gender bias)
            r"\bman\b": "person",
            r"\bwoman\b": "person",
            r"\bguys\b": "people",
            r"\bgirls\b": "people",
            r"\bboys\b": "people",
            r"\bmale\b": "person",
            r"\bfemale\b": "person",
            
            # Race/Ethnicity bias (inclusive framing)
            r"\bblack man\b": "person",
            r"\bblack woman\b": "person",
            r"\bwhite man\b": "person",
            r"\bwhite woman\b": "person",
            r"\bAsian man\b": "person",
            r"\bAsian woman\b": "person",
            r"\bblack\b": "person of African descent",
            r"\bwhite\b": "person of European descent",
            r"\bAsian\b": "person of Asian descent",
            
            # Age bias (neutral framing)
            r"\bold\b": "elderly",
            r"\bold\b": "elderly",
            r"\bteenager\b": "young",
            r"\bsenior citizen\b": "elderly",
            
            # Socioeconomic bias (neutral framing)
            r"\bpoor\b": "person with limited resources",
            r"\brich\b": "person with financial resources",
            r"\bhomeless\b": "person experiencing homelessness",
            r"\bwelfare recipient\b": "person receiving assistance",
            
            # Religious bias (neutral framing)
            r"\bChristian\b": "The person of faith",
            r"\bMuslim\b": "The person of faith",
            r"\bJewish\b": "The person of faith",
            r"\batheist\b": "The person",
            r"\bHindu\b": "The person of faith",
            r"\bBuddhist\b": "The person of faith",
            r"\bSikh\b": "The person of faith",
            
            # Ability bias (inclusive framing)
            r"\bdisabled person\b": "person with disabilities",
            r"\bblind person\b": "person with visual impairment",
            r"\bdeaf woman\b": "person with hearing impairment",

            
            # Sexual orientation bias (inclusive framing)
            r"\bgay\b": "person",
            r"\blesbian\b": "person",
            r"\bstraight\b": "person",
            r"\btransgender\b": "person",
            r"\bqueer\b": "person",

            
            # Appearance bias (neutral framing)
            r"\bfat person\b": "person",
            r"\bthin person\b": "person",
            r"\battractive person\b": "person",
            r"\bugly person\b": "person",
            
            # Mental health bias (respectful framing)
            r"\bcrazy person\b": "person with mental health challenges",
            r"\bmentally ill\b": "person with mental health conditions",
            r"\bdepressed person\b": "person experiencing depression",
        }
        
        modified_prompt = prompt
        for pattern, replacement in substitutions.items():
            modified_prompt = re.sub(pattern, replacement, modified_prompt, flags=re.IGNORECASE)
        
        # logger.debug(f"Gender-balanced (multi-bias) prompt: {modified_prompt}")
        return modified_prompt


def batch_engineer_prompts(prompts, batch_size=100):
    pe = PromptEngineering()
    engineered = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        engineered.extend([pe.apply(p) for p in batch])
        logger.info(f"engineered batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}")

    return engineered