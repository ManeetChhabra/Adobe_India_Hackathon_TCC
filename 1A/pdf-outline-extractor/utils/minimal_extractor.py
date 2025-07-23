# minimal_extractor.py - Minimal post-processing for specific issues only

import fitz
import re
from utils.ocr_utils import extract_text_with_ocr

def combine_adjacent_headings(headings, doc):
    """Minimal combination - only fix obvious fragmentation"""
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
        
        # Only check next 2 headings to avoid over-combination
        while j < len(headings) and j < i + 2:
            next_heading = headings[j]
            
            # Only combine if very likely fragmentation
            if (next_heading['page'] == current['page'] and
                is_obvious_fragmentation(current, next_heading)):
                
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

def is_obvious_fragmentation(current, next_heading):
    """Detect only obvious fragmentation cases"""
    current_text = current['text'].strip()
    next_text = next_heading['text'].strip()
    
    # Only combine very short fragments
    if len(current_text.split()) > 4 or len(next_text.split()) > 4:
        return False
    
    # Combined result must be reasonable length
    combined_words = len((current_text + ' ' + next_text).split())
    if combined_words > 8:
        return False
    
    # Only combine if current ends with obvious connectors
    if current_text.endswith((' AND', ' OF', ' FOR', ' THE', ' &')):
        return True
    
    # Only combine if next starts with obvious connectors  
    if next_text.startswith(('AND ', 'OF ', 'FOR ', 'THE ')):
        return True
    
    # Both are very short (likely fragments)
    if len(current_text.split()) <= 2 and len(next_text.split()) <= 2:
        return True
    
    return False

def advanced_filtering_universal(headings):
    """Minimal filtering - only remove obvious non-headings"""
    if not headings:
        return headings
    
    filtered = []
    
    for heading in headings:
        text = heading['text'].strip()
        
        # Only filter obvious page metadata
        if is_obvious_metadata(text):
            continue
            
        # Only filter highly repetitive content
        if is_highly_repetitive(text, headings):
            continue
            
        filtered.append(heading)
    
    return filtered

def is_obvious_metadata(text):
    """Only filter obvious page metadata"""
    text_lower = text.lower()
    
    # Very short single words
    if len(text.split()) == 1 and len(text) <= 8:
        return True
    
    # Obvious page numbers
    if re.match(r'^(page \d+|p\.\s*\d+|\d+\s*of\s*\d+)$', text_lower):
        return True
    
    # Pure numbers
    if text.isdigit() and len(text) <= 4:
        return True
    
    return False

def is_highly_repetitive(text, all_headings):
    """Only filter highly repetitive content"""
    text_lower = text.lower().strip()
    
    # Count exact matches
    count = sum(1 for h in all_headings if h['text'].lower().strip() == text_lower)
    
    # Only filter if appears more than 5 times
    if count > 5:
        return True
    
    return False

# Export functions for use in main extractor
__all__ = ['combine_adjacent_headings', 'advanced_filtering_universal']
