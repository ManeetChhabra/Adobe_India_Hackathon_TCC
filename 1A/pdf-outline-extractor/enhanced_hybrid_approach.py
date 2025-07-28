#!/usr/bin/env python3
"""
Enhanced Hybrid Approach with Improved Combination Logic
Addresses form field detection and quality-based selection
"""

import os
import sys
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use the working ML extraction function directly
def ml_extract_outline(pdf_path):
    try:
        # Import the working ML function
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
        from extractor import extract_outline
        return extract_outline(pdf_path)
    except Exception as e:
        print(f"ML extraction error: {e}")
        return "Document", []

from enhanced_visual_extractor import EnhancedVisualExtractor
from multilingual_support import enhance_with_multilingual_support

# Form field patterns for better filtering
FORM_FIELD_PATTERNS = [
    r'^(Case Number|Category|Comments|Date|Time|Name|Phone|Email|Address):\s*',
    r'^(Current Age|Age Last Seen|Hair Color|Eye Color|Height|Weight):\s*',
    r'^(NCIC Number|Location Last Seen|Last Known Address):\s*',
    r'^(Alert Information|Person Information|Picture\(s\)|Status Update)$',
    r'^(AGENDA ITEM|To:|From:|Subject:)',
    r'^M\s+\d{8,}',  # Case numbers like "M 103973526"
    r'^(Application:|Form:|Signature:|Date:|Phone:|Email:)',  # Only with colons to be more specific
]

def is_form_field(text):
    """Detect if text is a form field, not a heading"""
    for pattern in FORM_FIELD_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False

def evaluate_extraction_quality_enhanced(headings, pdf_path=""):
    """Enhanced quality evaluation with form field penalties"""
    if not headings:
        return 0.0
    
    heading_count = len(headings)
    if heading_count == 0:
        return 0.0
    
    score = 0.0
    total_checks = 0
    
    # Check 1: Reasonable heading count (with form field penalty)
    form_field_count = sum(1 for h in headings if is_form_field(h.get('text', '')))
    real_heading_count = heading_count - form_field_count
    
    if 1 <= real_heading_count <= 50:
        score += 1.0
    elif real_heading_count > 0:
        score += 0.7
    total_checks += 1
    
    # Penalty for too many form fields
    if form_field_count > 0:
        form_field_ratio = form_field_count / heading_count
        score *= (1.0 - form_field_ratio * 0.8)  # Heavy penalty for form fields
    
    # Check 2: Text diversity and quality
    texts = [h.get('text', '').strip().lower() for h in headings if not is_form_field(h.get('text', ''))]
    unique_texts = set(texts)
    
    if len(unique_texts) >= len(texts) * 0.8:
        score += 0.3
    total_checks += 1
    
    # Check 3: Enhanced text quality (excluding form fields)
    good_headings = 0
    for heading in headings:
        text = heading.get('text', '').strip()
        if (not is_form_field(text) and
            len(text) >= 3 and 
            len(text.split()) <= 15 and 
            not text.lower().startswith(('page ', 'figure ', 'table '))):
            good_headings += 1
    
    if heading_count > 0:
        text_quality = good_headings / heading_count
        score += text_quality
    total_checks += 1
    
    # Check 4: No obvious duplicates
    if len(unique_texts) == len(texts):
        score += 1.0
    elif len(unique_texts) >= len(texts) * 0.8:
        score += 0.5
    total_checks += 1
    
    return score / total_checks if total_checks > 0 else 0.0

def text_similarity(text1, text2):
    """Calculate text similarity using sequence matching"""
    return SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()

def assess_heading_quality(heading, source_confidence):
    """Enhanced heading quality assessment with form field detection"""
    text = heading.get('text', '').strip()
    
    # Start with base quality
    quality_score = 0.5
    
    # PENALTY: Form fields get heavy penalty
    if is_form_field(text):
        quality_score *= 0.1  # Very heavy penalty
        return quality_score * source_confidence
    
    # PENALTY: Other non-heading patterns
    if any(bad in text.lower() for bad in [
        'case number', 'phone', 'email', 'address', 'ssn',
        'date of birth', 'social security', 'driver license'
    ]):
        quality_score *= 0.2
    
    # BONUS: Good heading indicators
    if len(text) >= 5: quality_score += 0.1
    if len(text.split()) >= 2: quality_score += 0.1
    if text.endswith(':') and not is_form_field(text): quality_score += 0.2
    if text[0].isupper(): quality_score += 0.1
    if re.match(r'^\d+\.\s+[A-Z]', text): quality_score += 0.3  # Numbered sections
    if text.isupper() and len(text.split()) <= 4: quality_score += 0.15  # All caps short headings
    
    # PENALTY: Bad indicators
    if not any(char in text for char in ['(', ')', '[', ']']): quality_score += 0.05
    if len(text) > 100: quality_score *= 0.7  # Too long
    
    # Weight by source confidence
    final_quality = (quality_score * 0.8) + (source_confidence * 0.2)
    return min(1.0, final_quality)

def select_best_title(ml_title, visual_title, ml_conf, visual_conf):
    """Always use enhanced visual extractor title - it has superior title detection"""
    # Clean the visual title
    visual_title_clean = visual_title.strip() if visual_title else ""
    
    # Filter out obvious non-titles from visual extractor
    if visual_title_clean and is_form_field(visual_title_clean):
        visual_title_clean = ""
    
    # Always use visual title if available, otherwise empty string
    if visual_title_clean:
        return visual_title_clean, "Visual"
    else:
        return "", "Default"

def combine_results_enhanced(ml_result, visual_result, pdf_path=""):
    """
    ENHANCED: Intelligent combination with quality-based selection and form field filtering
    """
    ml_title, ml_headings = ml_result
    visual_title, visual_headings = visual_result
    
    # Evaluate overall confidence with enhanced metrics
    ml_confidence = evaluate_extraction_quality_enhanced(ml_headings, pdf_path)
    visual_confidence = evaluate_extraction_quality_enhanced(visual_headings, pdf_path)
    
    print(f"   📊 Enhanced ML Confidence: {ml_confidence:.3f} ({len(ml_headings)} headings)")
    print(f"   📊 Enhanced Visual Confidence: {visual_confidence:.3f} ({len(visual_headings)} headings)")
    
    # Enhanced title selection
    best_title, title_source = select_best_title(ml_title, visual_title, ml_confidence, visual_confidence)
    print(f"   🎯 Title from {title_source}: '{best_title[:40]}...'")
    
    # Add quality scores to headings
    for heading in ml_headings:
        heading['quality'] = assess_heading_quality(heading, ml_confidence)
        heading['source'] = 'ML'
    
    for heading in visual_headings:
        heading['quality'] = assess_heading_quality(heading, visual_confidence)
        heading['source'] = 'Visual'
    
    # Intelligent combination with fuzzy deduplication
    combined_headings = []
    all_headings = ml_headings + visual_headings
    
    # Ensure all headings have quality scores
    for heading in all_headings:
        if 'quality' not in heading:
            source_conf = ml_confidence if heading.get('source') == 'ML' else visual_confidence
            heading['quality'] = assess_heading_quality(heading, source_conf)
    
    # Sort by quality (best first) - use existing enhanced_confidence if available
    all_headings.sort(key=lambda x: x.get('quality', x.get('enhanced_confidence', 0.5)), reverse=True)
    
    used_headings = []
    
    for heading in all_headings:
        text = heading.get('text', '').strip()
        
        # Get quality score (use enhanced if available, otherwise our calculated quality)
        heading_quality = heading.get('quality', heading.get('enhanced_confidence', 0.5))
        
        # Skip very low quality headings (especially form fields)
        if heading_quality < 0.3:
            continue
        
        # Check for fuzzy duplicates
        is_duplicate = False
        for used_heading in used_headings:
            used_text = used_heading.get('text', '').strip()
            used_quality = used_heading.get('quality', used_heading.get('enhanced_confidence', 0.5))
            
            # Multiple similarity checks
            similarity = text_similarity(text, used_text)
            if (similarity > 0.8 or  # High text similarity
                text.lower() in used_text.lower() or       # Substring match
                used_text.lower() in text.lower()):        # Reverse substring
                
                # Replace if this one is significantly better
                if heading_quality > used_quality + 0.15:
                    used_headings.remove(used_heading)
                    combined_headings.remove(used_heading)
                    break
                else:
                    is_duplicate = True
                    break
        
        if not is_duplicate and text:
            combined_headings.append(heading)
            used_headings.append(heading)
    
    # Sort by page and position
    combined_headings.sort(key=lambda x: (x.get('page', 0), x.get('y_position', 0)))
    
    # Apply multilingual processing
    print("   🌐 Applying multilingual enhancement...")
    combined_headings = enhance_with_multilingual_support(combined_headings, pdf_path)
    multilingual_filtered = len(combined_headings) - len([h for h in combined_headings if h.get('multilingual_quality', 1.0) >= 0.4])
    
    # Clean output: only keep essential fields per schema
    clean_headings = []
    for heading in combined_headings:
        clean_heading = {
            'level': heading.get('level', 'H1'),
            'text': heading.get('text', '').strip(),
            'page': heading.get('page', 0)  # Already 0-indexed
        }
        clean_headings.append(clean_heading)
    
    form_fields_filtered = len([h for h in all_headings if is_form_field(h.get('text', ''))])
    low_quality_filtered = len([h for h in all_headings if h.get('quality', h.get('enhanced_confidence', 0.5)) < 0.3])
    
    print(f"   🔄 Enhanced combination: ML({len(ml_headings)}) + Visual({len(visual_headings)}) → {len(clean_headings)}")
    print(f"   🚫 Filtered out: {form_fields_filtered} form fields, {low_quality_filtered} low-quality headings")
    if multilingual_filtered > 0:
        print(f"   🌐 Multilingual processing: {multilingual_filtered} low-quality multilingual headings filtered")
    
    return best_title, clean_headings, "Hybrid-Enhanced"

def test_enhanced_combination():
    """Test the enhanced combination on file01"""
    
    print("🧪 TESTING ENHANCED COMBINATION ON FILE01")
    print("="*50)
    
    pdf_path = "input/file01.pdf"
    
    try:
        # Get ML result
        print("🤖 Getting ML result...")
        ml_title, ml_headings = ml_extract_outline(pdf_path)
        
        # Get Visual result  
        print("👁️ Getting Visual result...")
        extractor = EnhancedVisualExtractor()
        visual_result = extractor.extract_headings_enhanced(pdf_path)
        visual_title = visual_result['title']
        visual_headings = visual_result['headings']
        
        # Apply enhanced combination
        print("\n⚡ Applying enhanced combination...")
        enhanced_title, enhanced_headings, method = combine_results_enhanced(
            (ml_title, ml_headings), 
            (visual_title, visual_headings), 
            pdf_path
        )
        
        print(f"\n📋 ENHANCED RESULTS:")
        print(f"   Title: '{enhanced_title}'")
        print(f"   Method: {method}")
        print(f"   Headings: {len(enhanced_headings)}")
        print(f"\n📝 Enhanced Headings:")
        for i, heading in enumerate(enhanced_headings, 1):
            text = heading.get('text', 'No text')
            level = heading.get('level', 'H1')
            is_form = "🚫" if is_form_field(text) else "✅"
            print(f"   {i}. {is_form} {level}: '{text}'")
        
        print(f"\n📊 COMPARISON WITH CURRENT:")
        with open("output_hybrid/file13.json", "r", encoding="utf-8") as f:
            current_result = json.load(f)
        
        current_headings = current_result['outline']
        print(f"   Current: {len(current_headings)} headings")
        print(f"   Enhanced: {len(enhanced_headings)} headings")
        
        current_form_fields = sum(1 for h in current_headings if is_form_field(h['text']))
        enhanced_form_fields = sum(1 for h in enhanced_headings if is_form_field(h.get('text', '')))
        
        print(f"   Current form fields: {current_form_fields}")
        print(f"   Enhanced form fields: {enhanced_form_fields}")
        print(f"   Improvement: {current_form_fields - enhanced_form_fields} fewer form fields")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_enhanced_combination()
