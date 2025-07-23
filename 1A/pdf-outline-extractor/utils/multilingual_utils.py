# multilingual_utils.py - Enhanced multilingual text processing utilities

import re
import unicodedata

def detect_language(text_sample):
    """Detect the language of text to determine appropriate processing"""
    if not text_sample:
        return "en"
    
    # Simple language detection based on character patterns
    # Japanese (Hiragana, Katakana, Kanji)
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text_sample):
        return "ja"
    
    # Chinese (Simplified/Traditional)
    if re.search(r'[\u4E00-\u9FAF]', text_sample):
        return "zh"
    
    # Korean (Hangul)
    if re.search(r'[\uAC00-\uD7AF]', text_sample):
        return "ko"
    
    # Arabic
    if re.search(r'[\u0600-\u06FF]', text_sample):
        return "ar"
    
    # Hindi/Devanagari
    if re.search(r'[\u0900-\u097F]', text_sample):
        return "hi"
    
    # Russian/Cyrillic
    if re.search(r'[\u0400-\u04FF]', text_sample):
        return "ru"
    
    # French (common accented characters)
    if re.search(r'[àâäéèêëïîôùûüÿç]', text_sample):
        return "fr"
    
    # German (common characters)
    if re.search(r'[äöüßÄÖÜ]', text_sample):
        return "de"
    
    # Spanish (common accented characters)
    if re.search(r'[ñáéíóúü¿¡]', text_sample):
        return "es"
    
    # Default to English
    return "en"

def normalize_multilingual(text):
    """Enhanced text normalization with multilingual support"""
    # Handle various Unicode characters and normalize whitespace
    text = text.replace("\u2013", "-").replace("–", "-").replace("\u2019", "'").replace("'", "'")
    text = text.replace("\u201C", '"').replace("\u201D", '"').replace("\u2018", "'").replace("\u2019", "'")
    text = " ".join(text.strip().split())
    
    # Fix common OCR/extraction issues for Latin scripts
    if re.search(r'[a-zA-Z]', text):  # Only apply to text with Latin characters
        text = re.sub(r'\b([A-Z]) ([a-z]{1,3})\b', r'\1\2', text)
        text = re.sub(r'\b([A-Z]) ([A-Z]{2,})\b', r'\1\2', text)
        text = re.sub(r'\s+([!?.])', r'\1', text)
        
        if text.endswith('!') and not text.endswith('! '):
            text += ' '
    
    # Handle CJK (Chinese, Japanese, Korean) text spacing
    if re.search(r'[\u4E00-\u9FAF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]', text):
        # Remove unnecessary spaces around CJK characters
        text = re.sub(r'\s+(?=[\u4E00-\u9FAF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF])', '', text)
        text = re.sub(r'(?<=[\u4E00-\u9FAF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF])\s+', '', text)
    
    return text

def is_likely_false_positive_multilingual(text, document_context=None):
    """Enhanced false positive detection with multilingual support"""
    text_clean = text.strip()
    
    if len(text_clean) <= 2:
        return True
    
    # Multilingual character analysis
    total_chars = len(text_clean)
    alpha_chars = sum(c.isalpha() for c in text_clean)
    # Count CJK characters as alphabetic
    cjk_chars = len(re.findall(r'[\u4E00-\u9FAF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]', text_clean))
    # Count Arabic characters
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text_clean))
    # Count other script characters (Devanagari, Cyrillic, etc.)
    other_script_chars = len(re.findall(r'[\u0900-\u097F\u0400-\u04FF]', text_clean))
    
    meaningful_ratio = (alpha_chars + cjk_chars + arabic_chars + other_script_chars) / total_chars
    if meaningful_ratio < 0.3:
        return True
    
    # Common patterns across languages
    if re.search(r'(.{1,3})\1{4,}', text_clean):
        return True
    
    if re.match(r'^[\s\-_\.=]+$', text_clean):
        return True
    
    # Language-agnostic URL and contact detection
    if document_context:
        # Check if text has characteristics of non-heading content
        dot_ratio = text_clean.count('.') / len(text_clean)
        if dot_ratio > 0.15:  # High dot density suggests URLs/addresses
            return True
        
        # Check for patterns that suggest contact information (universal)
        if any(char in text_clean.lower() for char in ['@', 'www', 'http']):
            return True
        
        # Check for patterns with consecutive uppercase and dots (like URLs)
        if re.search(r'[A-Z]{3,}.*\..*[A-Z]{3,}', text_clean):
            return True
    
    return False

def get_heading_patterns_by_language(language):
    """Get language-specific heading patterns"""
    patterns = {
        "en": [
            r'^\d+\.?\s+[A-Z]',  # 1. Introduction
            r'^Chapter\s+\d+',    # Chapter 1
            r'^Section\s+\d+',    # Section 1
        ],
        "ja": [
            r'^第\d+章',          # 第1章 (Chapter 1)
            r'^第\d+節',          # 第1節 (Section 1)
            r'^\d+\.?\s*',        # 1. or 1
        ],
        "zh": [
            r'^第\d+章',          # 第1章 (Chapter 1)
            r'^第\d+节',          # 第1节 (Section 1)
            r'^\d+\.?\s*',        # 1. or 1
        ],
        "ko": [
            r'^제\d+장',          # 제1장 (Chapter 1)
            r'^제\d+절',          # 제1절 (Section 1)
            r'^\d+\.?\s*',        # 1. or 1
        ],
        "ar": [
            r'^الفصل\s+\d+',      # الفصل 1 (Chapter 1)
            r'^\d+\.?\s*',        # 1. or 1
        ],
        "de": [
            r'^Kapitel\s+\d+',    # Kapitel 1
            r'^Abschnitt\s+\d+',  # Abschnitt 1
            r'^\d+\.?\s+[A-Z]',   # 1. Introduction
        ],
        "fr": [
            r'^Chapitre\s+\d+',   # Chapitre 1
            r'^Section\s+\d+',    # Section 1
            r'^\d+\.?\s+[A-Z]',   # 1. Introduction
        ],
        "es": [
            r'^Capítulo\s+\d+',   # Capítulo 1
            r'^Sección\s+\d+',    # Sección 1
            r'^\d+\.?\s+[A-Z]',   # 1. Introduction
        ],
        "ru": [
            r'^Глава\s+\d+',      # Глава 1 (Chapter 1)
            r'^Раздел\s+\d+',     # Раздел 1 (Section 1)
            r'^\d+\.?\s+[А-Я]',   # 1. Введение
        ],
        "hi": [
            r'^अध्याय\s+\d+',     # अध्याय 1 (Chapter 1)
            r'^\d+\.?\s*',        # 1. or 1
        ]
    }
    
    return patterns.get(language, patterns["en"])

def is_title_case_multilingual(text, language):
    """Check if text follows title case conventions for the given language"""
    if language in ["zh", "ja", "ko"]:
        # CJK languages don't have title case
        return True
    elif language == "ar":
        # Arabic doesn't have upper/lower case distinction
        return True
    else:
        # For Latin-based languages, check title case
        return text.istitle() or text.isupper()
