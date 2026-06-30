import re
import json
from typing import Dict

PUNCT = set(".,;:!?")

def extract_prompt_features(prompt: str) -> Dict[str, float]:
    """Lightweight, admission-time features for workload prediction."""
    text = prompt or ""
    n_chars = len(text)
    n_lines = text.count("\n") + 1 if text else 0
    n_words = len(text.split())
    n_qmarks = text.count("?")
    n_emarks = text.count("!")
    n_backticks = text.count("`")
    n_fences = len(re.findall(r"```", text))
    has_url = int(bool(re.search(r"https?://", text)))
    has_table = int(("|" in text and "---" in text) or ("table" in text.lower()))
    has_list = int(bool(re.search(r"^\s*[-*]\s", text, flags=re.M)))

    # UPDATED: detect LaTeX math OR simple arithmetic expressions with + - * /
    math_pattern = r"(\$[^$]+\$)|\\begin\{equation\}|\d+\s*[\+\-\*/]\s*\d+"
    has_math = int(bool(re.search(math_pattern, text)))

    caps_ratio = sum(1 for c in text if c.isupper()) / max(1, n_chars)
    punct_density = sum(1 for c in text if c in PUNCT) / max(1, n_chars)
    avg_line_len = n_chars / max(1, n_lines)
    avg_word_len = sum(len(w) for w in text.split()) / max(1, n_words)

    return {
        "char_len": n_chars,
        "line_count": n_lines,
        "word_count": n_words,
        "question_marks": n_qmarks,
        "exclamation_marks": n_emarks,
        "backticks": n_backticks,
        "code_fences": n_fences,
        "has_url": has_url,
        "has_table": has_table,
        "has_list": has_list,
        "has_math": has_math,
        "caps_ratio": round(caps_ratio, 5),
        "punct_density": round(punct_density, 5),
        "avg_line_len": round(avg_line_len, 3),
        "avg_word_len": round(avg_word_len, 3),
    }

if __name__ == "__main__":
    demo = "Write Python code to add two numbers.\n```python\nprint(1+1)\n```"
    print(json.dumps(extract_prompt_features(demo), indent=2))
