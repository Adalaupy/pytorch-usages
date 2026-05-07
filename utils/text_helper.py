
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import re
from use_case.Stock_Price_Prediction.constant.financial_terms import financial_terms
from use_case.Stock_Price_Prediction.constant.negative_words import NEGATION_PREFIX_MAP


from nltk.stem import WordNetLemmatizer as grammer_handler


# ================================================================================================
# Main function to get Abbreviation and phrase mapping
# ================================================================================================

def get_phrase_map():
    
    ref = financial_terms

    # --------------------------------------------------------------------------------
    # Extract abbreviation map from financial_terms entries that match
    # "Full Name (ABBREV)" pattern.
    # Returns: {"abbrev": "full_name_normalized", ...}
    # Example: "Earnings Per Share (EPS)" -> {"eps": "earnings_per_share"}
    # --------------------------------------------------------------------------------

    def build_abbreviation_map(terms: list) -> dict:
        pattern = re.compile(r"^(.+?)\s+\(([^)]+)\)\s*$")
        result = {}
        for term in terms:
            match = pattern.match(term.strip())
            if not match:
                continue
            full_name, abbrev = match.group(1).strip(), match.group(2).strip()
            if len(abbrev) <= 1:
                continue
            key = abbrev.lower()
            value = re.sub(r"[^a-zA-Z0-9]+", "_", full_name.lower()).strip("_")
            result[key] = value
        
        return result


    # --------------------------------------------------------------------------------
    # Extract phrase map from financial_terms entries that are multi-word terms
    # without parenthetical abbreviations.
    # Returns: {"full phrase": "normalized_token", ...}
    # Example: "Bear Market" -> {"bear market": "bear_market"}
    # --------------------------------------------------------------------------------

    def build_phrase_map(terms: list) -> dict:
        abbrev_pattern = re.compile(r"\([^)]+\)")
        result = {}
        for term in terms:
            term = term.strip()
            if len(term) <= 1 or term.lower().startswith("see complete"):
                continue
            if term[0].isdigit() or term[0] == "#":
                continue
            cleaned = abbrev_pattern.sub("", term).strip()
            words = cleaned.split()
            if len(words) < 2:
                continue
            key = cleaned.lower().strip()
            value = re.sub(r"[^a-zA-Z0-9]+", "_", key).strip("_")
            result[key] = value
        
        return result


    abbrev_map = build_abbreviation_map(ref)
    phrase_map = build_phrase_map(ref)



    return abbrev_map,phrase_map



# ================================================================================================
# Main function for NLP data cleaning
# ================================================================================================

abbrev_map , phrase_map = get_phrase_map()

# --------------------------------------------------------------------------------
# Remove unnecessary from a single text and normalize spaces.
# --------------------------------------------------------------------------------

def remove_unused_from_text(text: str) -> str:

    PATTERNS = [
        r"http\S+|www\S+", # URL
        r"\$[a-z]*(\s+)?(-\s+|:\s+)?", # Stock Code
        r"\#[a-z]+"
    ]
    cleaned = text

    for pattern in PATTERNS:
        
        sub_pattern = re.compile(pattern, flags=re.IGNORECASE)

        cleaned = sub_pattern.sub("", str(cleaned))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    return cleaned


# --------------------------------------------------------------------------------
# Map phrase and concat them with "_"
# --------------------------------------------------------------------------------

def apply_phrase_map(text: str, phrase_map: dict) -> str:

    for phrase, replacement in sorted(phrase_map.items(), key=lambda x: -len(x[0])):
        text = text.replace(phrase, replacement)
    
    return text

# --------------------------------------------------------------------------------
# Map abbreviation and revert them to words
# --------------------------------------------------------------------------------
def apply_abbreviation_map(text: str, abbrev_map: dict) -> str:
    tokens = text.split()
    return " ".join(abbrev_map.get(tok, tok) for tok in tokens)

# --------------------------------------------------------------------------------
# Convert negation + next token into one token.
# --------------------------------------------------------------------------------
def apply_negation_concat(sentence: str, neg_map: dict) -> str:
    """
    
    Examples:
    - "not good" -> "not_good"
    - "don't like" -> "do_not_like"
    - "doesn't grow" -> "does_not_grow"
    """
    sentence = str(sentence).lower().replace("’", "'")
    tokens = sentence.split()

    result = []
    i = 0
    while i < len(tokens):
        prefix = neg_map.get(tokens[i])

        if prefix and i + 1 < len(tokens):
            result.append(f"{prefix}_{tokens[i + 1]}")
            i += 2
        else:
            result.append(tokens[i])
            i += 1

    return " ".join(result)



# --------------------------------------------------------------------------------
# lemmatization / Stemming
# --------------------------------------------------------------------------------
grammer = grammer_handler()

def normalize_grammar(sentence: str) -> str:

    tokens = str(sentence).split()
    result = []

    for tok in tokens:
        if "_" in tok:
            result.append(tok)
        else:
            # Try verb lemmatization first so "celebrated" -> "celebrate"
            verb_form = grammer.lemmatize(tok, "v")
            if verb_form != tok:
                result.append(verb_form)
            else:
                result.append(grammer.lemmatize(tok))

    return " ".join(result)


# --------------------------------------------------------------------------------
# Main data cleaning
# --------------------------------------------------------------------------------

def NLP_data_cleaning(sentense: str):
    
    # Step 1: Lower Case + replace "-" to "_"
    cleaned_sentense = sentense.lower()
    cleaned_sentense = cleaned_sentense.replace('-','_')

    # Step 2: Remove unnecessary text
    cleaned_sentense = remove_unused_from_text( cleaned_sentense )
    
    # Step 2.1: Remove punctuation marks ".", ",", "!"
    cleaned_sentense = re.sub(r"[.,!]", " ", cleaned_sentense)
    cleaned_sentense = re.sub(r"\s+", " ", cleaned_sentense).strip()

    # Step 2.2:  Replace from "'s" to '_s'
    cleaned_sentense = re.sub(r"(\s+)?'s", "_s", cleaned_sentense).strip()

    # Step 3: Concat Negative word
    cleaned_sentense = apply_negation_concat( cleaned_sentense, NEGATION_PREFIX_MAP)

    # Step 4: Concat Phrase
    cleaned_sentense = apply_phrase_map(cleaned_sentense , phrase_map)

    # Step 5: Convert abbreviation to words
    cleaned_sentense = apply_abbreviation_map(cleaned_sentense , abbrev_map)

    # Step 6: Stemming/Lemmatization
    cleaned_sentense = normalize_grammar(cleaned_sentense)


    return cleaned_sentense


# ================================================================================================
# Split text into list from a sentense
# ================================================================================================


def is_number(text) -> bool:
    
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False
    

def tokenize(text: str):
    
    stop_words = { "a", "an", "the" ,"in", "at", "on" }
    text = re.sub(r"[^a-zA-Z0-9 ]+", " ", str(text).lower())

    text_list = []
    
    for item in text.split():        

        if item not in stop_words and len(item)> 1 and not is_number(item):
            
            text_list.append(item)

    return text_list

# ================================================================================================
# Map text to your vocab dictionary's Id
# ================================================================================================

def encode_text(text: str, vocab: dict, unk_idx: int, pad_idx: int, max_len: int):
    
    ids = [vocab.get(tok, unk_idx) for tok in tokenize(text)]
    ids = ids[:max_len]
    ids += [pad_idx] * (max_len - len(ids))
    
    return ids


