# enhanced_extractor.py - Combined multilingual and OCR extractor

import json
import re
import numpy as np
import fitz
from .extractor import (
    extract_with_ml_primary,
    is_likely_false_positive,
    normalize,
    extract_outline
)
from .multilingual_utils import (
    detect_language,
    normalize_multilingual, 
    is_likely_false_positive_multilingual,
    get_heading_patterns_by_language,
    is_title_case_multilingual
)
from .ocr_utils import (
    detect_scanned_page,
    extract_from_scanned_pdf,
    identify_headings_from_ocr,
    detect_text_language_ocr
)

def detect_document_language(pdf_path):
    """Detect the primary language of the document"""
    try:
        doc = fitz.open(pdf_path)
        sample_text = ""
        
        # Sample text from first few pages
        for page_num in range(min(3, doc.page_count)):
            page = doc[page_num]
            page_text = page.get_text()
            sample_text += page_text[:1000]  # First 1000 chars per page
            
            if len(sample_text) > 2000:  # Enough sample
                break
        
        doc.close()
        
        if sample_text.strip():
            return detect_language(sample_text)
        else:
            # No embedded text, try OCR on first page
            doc = fitz.open(pdf_path)
            page = doc[0]
            if detect_scanned_page(page):
                from .ocr_utils import extract_text_with_layout
                import numpy as np
                from PIL import Image
                import io
                
                # Convert first page to image for language detection
                pix = page.get_pixmap(dpi=150)  # Lower DPI for quick detection
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                image_array = np.array(image)
                
                # Extract some text for language detection
                text_elements = extract_text_with_layout(image_array, lang='eng')
                sample_text = " ".join([elem['text'] for elem in text_elements[:10]])
                
                if sample_text.strip():
                    return detect_language(sample_text)
            
            doc.close()
            return "en"  # Default to English
            
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "en"

def has_scanned_pages(pdf_path):
    """Check if PDF contains any scanned pages"""
    try:
        doc = fitz.open(pdf_path)
        scanned_count = 0
        
        for page_num in range(min(5, doc.page_count)):  # Check first 5 pages
            page = doc[page_num]
            if detect_scanned_page(page):
                scanned_count += 1
        
        doc.close()
        return scanned_count > 0
        
    except Exception:
        return False

def extract_with_enhanced_multilingual_ocr(pdf_path, use_ml=True):
    """Enhanced extraction with multilingual and OCR support"""
    try:
        # Detect document language
        doc_language = detect_document_language(pdf_path)
        print(f"Detected document language: {doc_language}")
        
        # Check if document has scanned pages
        has_scanned = has_scanned_pages(pdf_path)
        print(f"Document contains scanned pages: {has_scanned}")
        
        headings = []
        
        if has_scanned:
            # Use OCR extraction for scanned documents
            print("Using OCR extraction for scanned document")
            ocr_lang = detect_text_language_ocr(" ")  # Will use language detection
            if doc_language == "ja":
                ocr_lang = "jpn"
            elif doc_language == "zh":
                ocr_lang = "chi_sim"
            elif doc_language == "ko":
                ocr_lang = "kor"
            elif doc_language == "ar":
                ocr_lang = "ara"
            elif doc_language == "hi":
                ocr_lang = "hin"
            elif doc_language == "ru":
                ocr_lang = "rus"
            elif doc_language == "fr":
                ocr_lang = "fra"
            elif doc_language == "de":
                ocr_lang = "deu"
            elif doc_language == "es":
                ocr_lang = "spa"
            else:
                ocr_lang = "eng"
            
            text_elements = extract_from_scanned_pdf(pdf_path, ocr_lang)
            ocr_headings = identify_headings_from_ocr(text_elements)
            
            # Convert OCR headings to standard format
            for heading in ocr_headings:
                headings.append({
                    "title": normalize_multilingual(heading['text']),
                    "page": heading['page'],
                    "level": heading.get('level', 1)
                })
        
        else:
            # Use regular extraction method
            print("Using standard extraction with multilingual enhancement")
            
            # Get the standard outline first
            title, standard_headings = extract_outline(pdf_path)
            
            # Apply multilingual normalization to standard results
            for heading in standard_headings:
                normalized_text = normalize_multilingual(heading['text'])
                
                # Apply multilingual false positive detection
                if not is_likely_false_positive_multilingual(normalized_text):
                    headings.append({
                        "title": normalized_text,
                        "page": heading.get('page', 1),
                        "level": heading.get('level', 1)
                    })
        
        # Apply level detection based on content patterns
        if headings:
            headings = assign_heading_levels_multilingual(headings, doc_language)
        
        return headings
        
    except Exception as e:
        print(f"Enhanced extraction failed: {e}")
        # Fallback to standard extraction
        try:
            title, standard_headings = extract_outline(pdf_path)
            return [{"title": h['text'], "page": h.get('page', 1), "level": h.get('level', 1)} for h in standard_headings]
        except:
            return []

def assign_heading_levels_multilingual(headings, language):
    """Assign heading levels with multilingual awareness"""
    if not headings:
        return headings
    
    # Language-specific level patterns
    level_patterns = {
        "en": [
            (r'^\d+\.?\s+', 1),  # 1. or 1 
            (r'^\d+\.\d+\.?\s+', 2),  # 1.1. or 1.1
            (r'^\d+\.\d+\.\d+\.?\s+', 3),  # 1.1.1. or 1.1.1
        ],
        "ja": [
            (r'^第\d+章', 1),  # 第1章
            (r'^第\d+節', 2),  # 第1節
            (r'^\d+\.?\s*', 2),  # 1. or 1
        ],
        "zh": [
            (r'^第\d+章', 1),  # 第1章
            (r'^第\d+节', 2),  # 第1节
            (r'^\d+\.?\s*', 2),  # 1. or 1
        ]
    }
    
    patterns = level_patterns.get(language, level_patterns["en"])
    
    for heading in headings:
        text = heading['title']
        assigned = False
        
        for pattern, level in patterns:
            if re.match(pattern, text):
                heading['level'] = level
                assigned = True
                break
        
        if not assigned:
            # Default level assignment based on text characteristics
            if language in ["ja", "zh", "ko"]:
                # For CJK languages, use length as indicator
                if len(text) <= 10:
                    heading['level'] = 1
                else:
                    heading['level'] = 2
            else:
                # For other languages, use case and length
                if text.isupper() and len(text.split()) <= 5:
                    heading['level'] = 1
                else:
                    heading['level'] = 2
    
    return headings

def format_outline_multilingual(headings, language):
    """Format outline with multilingual considerations"""
    if not headings:
        return []
    
    formatted = []
    
    for heading in headings:
        title = heading['title']
        
        # Language-specific formatting
        if language in ["ja", "zh", "ko"]:
            # For CJK languages, ensure proper spacing
            title = re.sub(r'\s+', '', title)  # Remove excessive spaces
            title = re.sub(r'([。！？])(\S)', r'\1 \2', title)  # Add space after punctuation
        
        formatted.append({
            "title": title,
            "page": heading['page'],
            "level": heading.get('level', 1)
        })
    
    return formatted
