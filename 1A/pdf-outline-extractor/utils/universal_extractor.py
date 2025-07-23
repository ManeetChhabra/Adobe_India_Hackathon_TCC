# universal_extractor.py - Universal extraction without hardcoded patterns

import fitz
import re
import math
from collections import Counter
from utils.ocr_utils import extract_text_with_ocr

def combine_adjacent_headings(headings, doc):
    """Combine adjacent headings using universal statistical analysis"""
    if not headings:
        return headings
    
    combined = []
    i = 0
    
    while i < len(headings):
        current = headings[i]
        
        # Look for potential combinations
        combined_text = current['text']
        combined_items = [current]
        j = i + 1
        
        # Check next few headings for combination opportunities
        while j < len(headings) and j < i + 5:  # Check up to 5 ahead
            next_heading = headings[j]
            
            # Universal combination logic based on statistical properties
            if (next_heading['page'] == current['page'] and
                should_combine_headings_universal(current, next_heading, combined_text)):
                
                # Add space if needed
                if not combined_text.endswith(' ') and not combined_text.endswith('-'):
                    combined_text += ' '
                combined_text += next_heading['text']
                combined_items.append(next_heading)
                j += 1
            else:
                break
        
        # Create combined heading
        if len(combined_items) > 1:
            # Use the highest level among combined items
            best_level = min(item['level'] for item in combined_items)
            combined.append({
                'level': best_level,
                'text': combined_text.strip(),
                'page': current['page']
            })
            i = j
        else:
            combined.append(current)
            i += 1
    
    return combined

def should_combine_headings_universal(current, next_heading, combined_text):
    """Universal heading combination using statistical analysis"""
    current_text = current['text'].strip()
    next_text = next_heading['text'].strip()
    
    # Universal length-based constraints
    avg_length = (len(current_text) + len(next_text)) / 2
    max_reasonable_length = avg_length * 3  # Adaptive based on content
    
    if len(combined_text) > max_reasonable_length or len(next_text) > avg_length * 2:
        return False
    
    # Statistical analysis of text characteristics
    current_word_count = len(current_text.split())
    next_word_count = len(next_text.split())
    
    # Combine if either text is very short (likely fragment)
    if current_word_count <= 2 or next_word_count <= 2:  # Back to conservative 2
        # Check if combination creates reasonable length
        combined_word_count = len((combined_text + ' ' + next_text).split())
        if combined_word_count <= 6:  # More conservative limit
            return True
    
    # Analyze text completion patterns universally
    if is_text_incomplete_universal(current_text):
        return True
    
    if is_text_continuation_universal(next_text):
        return True
    
    # Universal capitalization pattern analysis
    if analyze_capitalization_patterns(current_text, next_text):
        return True
    
    # Universal punctuation-based separation detection
    if has_clear_separation_universal(current_text, next_text):
        return False
    
    return False

def is_text_incomplete_universal(text):
    """Universal detection of incomplete text using statistical patterns"""
    text = text.strip()
    
    # Statistical analysis of ending patterns
    # Text ending with connecting words (statistical approach)
    words = text.lower().split()
    if not words:
        return False
    
    last_word = words[-1]
    
    # Universal indicators: very short words at the end (likely connectors)
    if len(last_word) <= 3 and len(words) > 1:  # Back to 3
        # Check if it's likely a connector based on frequency patterns
        connector_probability = calculate_connector_probability(last_word)
        if connector_probability > 0.8:  # More conservative threshold
            return True
    
    # Text ending with punctuation suggesting continuation
    if text.endswith((':')) or text.endswith('&'):
        return True
    
    # Check if text appears structurally incomplete (universal approach)
    words = text.lower().split()
    if len(words) >= 2:
        last_word = words[-1]
        
        # Statistical analysis: if last word is very short and common, likely incomplete
        if len(last_word) <= 3:  # Back to 3
            # Calculate how "completion-like" the word is
            completion_score = calculate_completion_probability(last_word)
            if completion_score < 0.2:  # More conservative threshold
                return True
    
    return False

def is_text_continuation_universal(text):
    """Universal detection of continuation text using statistical patterns"""
    words = text.lower().split()
    if not words:
        return False
    
    first_word = words[0]
    
    # Universal indicators: very short starting words (likely connectors)
    if len(first_word) <= 3:
        connector_probability = calculate_connector_probability(first_word)
        if connector_probability > 0.7:
            return True
    
    return False

def calculate_connector_probability(word):
    """Calculate probability that a word is a connector using universal linguistic patterns"""
    # Universal linguistic analysis based on word characteristics
    word = word.lower()
    
    # Statistical indicators of connector words
    score = 0.0
    
    # Very short words are often connectors
    if len(word) <= 2:
        score += 0.3
    elif len(word) == 3:
        score += 0.2
    
    # Common vowel-consonant patterns in connectors
    vowel_ratio = sum(1 for c in word if c in 'aeiou') / len(word) if word else 0
    if 0.2 <= vowel_ratio <= 0.6:  # Balanced vowel-consonant ratio
        score += 0.3
    
    # Frequency in typical text (estimated statistically)
    # High-frequency short words are often connectors
    if len(word) <= 3:
        score += 0.2
    
    return min(score, 1.0)

def calculate_completion_probability(word):
    """Calculate probability that a word represents a completion using universal patterns"""
    # Universal linguistic analysis for completion indicators
    word = word.lower()
    score = 0.0
    
    # Longer words are more likely to be complete concepts
    if len(word) >= 6:
        score += 0.4
    elif len(word) >= 4:
        score += 0.2
    
    # Words ending in common completion suffixes
    completion_endings = ['tion', 'sion', 'ment', 'ness', 'ing', 'ed', 'er', 'ly', 'al', 'ic']
    if any(word.endswith(ending) for ending in completion_endings):
        score += 0.4
    
    # Vowel-consonant balance suggests complete words
    vowel_ratio = sum(1 for c in word if c in 'aeiou') / len(word) if word else 0
    if 0.3 <= vowel_ratio <= 0.5:  # Balanced ratio suggests complete word
        score += 0.2
    
    return min(score, 1.0)

def analyze_capitalization_patterns(current_text, next_text):
    """Universal capitalization pattern analysis"""
    # Analyze if texts have similar capitalization patterns
    current_caps = sum(1 for c in current_text if c.isupper())
    next_caps = sum(1 for c in next_text if c.isupper())
    
    current_ratio = current_caps / len(current_text) if current_text else 0
    next_ratio = next_caps / len(next_text) if next_text else 0
    
    # Similar capitalization patterns suggest they belong together
    ratio_diff = abs(current_ratio - next_ratio)
    if ratio_diff < 0.3:  # Similar capitalization density
        return True
    
    return False

def has_clear_separation_universal(current_text, next_text):
    """Universal detection of clear separation between texts"""
    # Statistical indicators of clear separation
    
    # Period at end suggests completion
    if current_text.strip().endswith('.'):
        # Check if next starts with capital (new sentence)
        if next_text and next_text[0].isupper():
            return True
    
    # Numbered lists
    if re.match(r'^\d+\.', next_text.strip()):
        return True
    
    # Both texts are reasonably long (unlikely fragments)
    if len(current_text.split()) >= 3 and len(next_text.split()) >= 3:
        return True
    
    return False

def advanced_filtering_universal(headings):
    """Universal advanced filtering using statistical analysis"""
    if not headings:
        return headings
    
    # Statistical analysis of all headings to identify patterns
    text_stats = analyze_heading_statistics(headings)
    
    filtered = []
    
    for heading in headings:
        text = heading['text'].strip()
        
        # Universal filtering based on statistical patterns
        if is_likely_metadata_universal(text, text_stats):
            continue
            
        if is_repetitive_content_universal(text, headings, text_stats):
            continue
            
        if is_boilerplate_universal(text, text_stats):
            continue
            
        filtered.append(heading)
    
    return filtered

def analyze_heading_statistics(headings):
    """Analyze statistical patterns in headings for universal filtering"""
    all_texts = [h['text'].strip().lower() for h in headings]
    
    stats = {
        'length_distribution': [],
        'word_count_distribution': [],
        'repetition_counts': Counter(all_texts),
        'avg_length': 0,
        'avg_word_count': 0,
        'common_patterns': {}
    }
    
    # Calculate distributions
    lengths = [len(text) for text in all_texts]
    word_counts = [len(text.split()) for text in all_texts]
    
    stats['length_distribution'] = lengths
    stats['word_count_distribution'] = word_counts
    stats['avg_length'] = sum(lengths) / len(lengths) if lengths else 0
    stats['avg_word_count'] = sum(word_counts) / len(word_counts) if word_counts else 0
    
    return stats

def is_likely_metadata_universal(text, stats):
    """Universal metadata detection using statistical patterns"""
    text_lower = text.lower()
    
    # Statistical indicators of metadata
    word_count = len(text.split())
    text_length = len(text)
    
    # Very short texts are often metadata
    if word_count <= 2 and text_length <= 15:
        # Check if it's mostly numbers/dates
        digit_ratio = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        if digit_ratio > 0.3:
            return True
    
    # Pattern analysis for dates and page numbers
    if re.match(r'^.*\d{1,4}.*$', text) and len(text) <= 20:
        # Likely page number or date
        return True
    
    # Single word metadata
    if word_count == 1 and len(text) <= 10:
        return True
    
    return False

def is_repetitive_content_universal(text, all_headings, stats):
    """Universal repetitive content detection"""
    text_lower = text.lower().strip()
    
    # Statistical threshold based on document size
    total_headings = len(all_headings)
    repetition_threshold = max(4, total_headings * 0.08)  # More conservative threshold
    
    occurrences = stats['repetition_counts'][text_lower]
    
    if occurrences >= repetition_threshold:
        return True
    
    return False

def is_boilerplate_universal(text, stats):
    """Universal boilerplate detection using statistical analysis"""
    # Statistical indicators of boilerplate text
    
    # Very long texts might be boilerplate
    if len(text) > stats['avg_length'] * 3:
        return True
    
    # Texts with unusual word patterns
    word_count = len(text.split())
    if word_count > stats['avg_word_count'] * 4:
        return True
    
    # Legal/formal language indicators (statistical)
    formal_indicators = count_formal_language_patterns(text)
    if formal_indicators > 4:  # More conservative - only filter very formal text
        return True
    
    return False

def count_formal_language_patterns(text):
    """Count universal indicators of formal/legal language"""
    text_lower = text.lower()
    count = 0
    
    # Statistical patterns common in formal documents
    # Long words (formal language uses longer words)
    long_words = sum(1 for word in text.split() if len(word) > 8)
    if long_words > len(text.split()) * 0.3:
        count += 1
    
    # Complex punctuation
    if '..' in text or ';' in text:
        count += 1
    
    # Numeric references
    if re.search(r'\d+\.\d+', text):  # Section numbers
        count += 1
    
    # Parenthetical references
    if '(' in text and ')' in text:
        count += 1
    
    # All caps words (legal emphasis)
    caps_words = sum(1 for word in text.split() if word.isupper() and len(word) > 2)
    if caps_words > 0:
        count += 1
    
    return count

# Export functions for use in main extractor
__all__ = ['combine_adjacent_headings', 'advanced_filtering_universal']
