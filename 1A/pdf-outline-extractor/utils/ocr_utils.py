# ocr_utils.py - Enhanced OCR utilities for scanned documents

import pytesseract
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import io
import tempfile
import os
import re

# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Users\%s\AppData\Local\Programs\Tesseract-OCR\tesseract.exe' % os.getenv('USERNAME'),
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

def enhance_image_for_ocr(image_array):
    """Enhance image quality for better OCR results"""
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations to clean up
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def extract_text_with_layout(image_array, lang='eng', preserve_layout=True):
    """Extract text with layout information using OCR"""
    try:
        # Enhance image for better OCR
        enhanced_image = enhance_image_for_ocr(image_array)
        
        # Convert back to PIL Image
        pil_image = Image.fromarray(enhanced_image)
        
        # OCR configuration for better layout preservation
        config = '--oem 3 --psm 6'  # PSM 6: uniform block of text
        if preserve_layout:
            config += ' -c preserve_interword_spaces=1'
        
        # Extract text with bounding boxes
        data = pytesseract.image_to_data(
            pil_image, 
            lang=lang, 
            config=config,
            output_type=pytesseract.Output.DICT
        )
        
        # Group text by lines and blocks
        lines = {}
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:  # Only high-confidence text
                block_num = data['block_num'][i]
                par_num = data['par_num'][i]
                line_num = data['line_num'][i]
                
                line_key = (block_num, par_num, line_num)
                
                if line_key not in lines:
                    lines[line_key] = {
                        'text': [],
                        'bbox': [data['left'][i], data['top'][i], 
                                data['left'][i] + data['width'][i], 
                                data['top'][i] + data['height'][i]]
                    }
                
                # Extend bounding box
                bbox = lines[line_key]['bbox']
                bbox[0] = min(bbox[0], data['left'][i])
                bbox[1] = min(bbox[1], data['top'][i])
                bbox[2] = max(bbox[2], data['left'][i] + data['width'][i])
                bbox[3] = max(bbox[3], data['top'][i] + data['height'][i])
                
                lines[line_key]['text'].append(data['text'][i])
        
        # Convert to text elements with positions
        text_elements = []
        for line_key, line_data in lines.items():
            text = ' '.join(line_data['text']).strip()
            if text:
                text_elements.append({
                    'text': text,
                    'bbox': line_data['bbox'],
                    'block': line_key[0],
                    'paragraph': line_key[1],
                    'line': line_key[2]
                })
        
        return text_elements
        
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return []

def detect_scanned_page(pdf_page):
    """Detect if a PDF page is scanned (image-based) or has embedded text"""
    try:
        # Check if page has extractable text
        text = pdf_page.get_text()
        if text.strip() and len(text.strip()) > 50:
            return False  # Has embedded text
        
        # Check if page has images
        image_list = pdf_page.get_images()
        if len(image_list) > 0:
            return True  # Likely scanned
        
        return False
        
    except Exception:
        return True  # Assume scanned if we can't determine

def extract_from_scanned_pdf(pdf_path, language='eng'):
    """Extract outline from scanned PDF using OCR"""
    try:
        doc = fitz.open(pdf_path)
        all_text_elements = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Check if page needs OCR
            if detect_scanned_page(page):
                print(f"Processing scanned page {page_num + 1} with OCR...")
                
                # Convert PDF page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Load image with PIL
                image = Image.open(io.BytesIO(img_data))
                image_array = np.array(image)
                
                # Extract text with layout
                text_elements = extract_text_with_layout(image_array, lang=language)
                
                # Add page number to elements
                for element in text_elements:
                    element['page'] = page_num + 1
                    # Scale coordinates back to original size
                    element['bbox'] = [coord / 2.0 for coord in element['bbox']]
                
                all_text_elements.extend(text_elements)
            else:
                # Use regular text extraction for non-scanned pages
                text = page.get_text()
                if text.strip():
                    all_text_elements.append({
                        'text': text.strip(),
                        'page': page_num + 1,
                        'bbox': [0, 0, page.rect.width, page.rect.height],
                        'block': 0,
                        'paragraph': 0,
                        'line': 0
                    })
        
        doc.close()
        return all_text_elements
        
    except Exception as e:
        print(f"OCR processing failed: {e}")
        return []

def identify_headings_from_ocr(text_elements):
    """Identify potential headings from OCR text elements"""
    headings = []
    
    if not text_elements:
        return headings
    
    # Sort by page, then by vertical position
    sorted_elements = sorted(text_elements, key=lambda x: (x['page'], x['bbox'][1]))
    
    for element in sorted_elements:
        text = element['text'].strip()
        
        if not text or len(text) < 3:
            continue
        
        # Calculate text metrics
        bbox = element['bbox']
        text_height = bbox[3] - bbox[1]
        text_width = bbox[2] - bbox[0]
        
        # Skip if too small or too large
        if text_height < 10 or text_height > 100:
            continue
        
        # Check for heading patterns
        is_heading = False
        confidence = 0.0
        
        # Pattern-based detection
        if re.match(r'^\d+\.?\s*[A-Z]', text):
            is_heading = True
            confidence += 0.4
        
        if re.match(r'^[A-Z][A-Z\s]{2,}$', text):  # ALL CAPS
            is_heading = True
            confidence += 0.3
        
        # Check if text is short (typical for headings)
        if len(text.split()) <= 8:
            confidence += 0.2
        
        # Check positioning (isolated lines often headings)
        if text_height > 12:  # Larger text
            confidence += 0.2
        
        # Simple check for title case
        if text.istitle():
            confidence += 0.1
        
        if is_heading and confidence > 0.3:
            headings.append({
                'text': text,
                'page': element['page'],
                'level': 1,  # Default level, can be refined
                'bbox': bbox,
                'confidence': confidence
            })
    
    return headings

def get_supported_languages():
    """Get list of supported languages for OCR"""
    try:
        langs = pytesseract.get_languages()
        return langs
    except Exception:
        return ['eng']  # Default to English if can't detect

def detect_text_language_ocr(text_sample):
    """Detect language from OCR text for appropriate processing"""
    # Simple language detection for OCR results
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text_sample):
        return 'jpn'  # Japanese
    elif re.search(r'[\u4E00-\u9FAF]', text_sample):
        return 'chi_sim'  # Chinese Simplified
    elif re.search(r'[\uAC00-\uD7AF]', text_sample):
        return 'kor'  # Korean
    elif re.search(r'[\u0600-\u06FF]', text_sample):
        return 'ara'  # Arabic
    elif re.search(r'[\u0900-\u097F]', text_sample):
        return 'hin'  # Hindi
    elif re.search(r'[\u0400-\u04FF]', text_sample):
        return 'rus'  # Russian
    elif re.search(r'[àâäéèêëïîôùûüÿç]', text_sample):
        return 'fra'  # French
    elif re.search(r'[äöüßÄÖÜ]', text_sample):
        return 'deu'  # German
    elif re.search(r'[ñáéíóúü¿¡]', text_sample):
        return 'spa'  # Spanish
    else:
        return 'eng'  # English default

# Legacy function for backward compatibility
def extract_text_with_ocr(page):
    """Legacy OCR function for backward compatibility"""
    # Convert PyMuPDF page to pixmap → image
    pix = page.get_pixmap(dpi=300)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # OCR the image
    text = pytesseract.image_to_string(image, lang="eng")
    return text
