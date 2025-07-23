# extractor.py - SIMPLIFIED ML-First Approach

import fitz
import re
from utils.ocr_utils import extract_text_with_ocr
from feature_engineering import extract_features

# --- Load ML model with graceful fallback ---
try:
    import joblib
    import pandas as pd
    try:
        MODEL = joblib.load("heading_classifier_optimized.joblib")
    except FileNotFoundError:
        MODEL = joblib.load("heading_classifier.joblib")
    ML_AVAILABLE = True
except (FileNotFoundError, ImportError):
    MODEL = None
    ML_AVAILABLE = False

def normalize(text):
    text = text.replace("\u2013", "-").replace("–", "-").replace("\u2019", "'").replace("'", "'")
    text = " ".join(text.strip().split())
    
    # Fix common OCR/extraction issues
    text = re.sub(r'\b([A-Z]) ([a-z]{1,3})\b', r'\1\2', text)
    text = re.sub(r'\b([A-Z]) ([A-Z]{2,})\b', r'\1\2', text)
    text = re.sub(r'\s+([!?.])', r'\1', text)
    
    if text.endswith('!') and not text.endswith('! '):
        text += ' '
    
    return text

def combine_fragmented_headings(headings):
    """Combine headings that are split across lines but should be one heading"""
    if not headings:
        return headings
    
    combined_headings = []
    i = 0
    
    while i < len(headings):
        current = headings[i]
        
        # Look ahead to see if next heading should be combined
        if i + 1 < len(headings):
            next_heading = headings[i + 1]
            
            # More conservative combination - only combine if very likely to be fragments
            should_combine = (
                current['level'] == next_heading['level'] and
                current['page'] == next_heading['page'] and  # Must be same page
                (
                    # Current text is very short and doesn't end with punctuation
                    (len(current['text'].split()) <= 3 and 
                     not current['text'].rstrip().endswith(('.', '!', '?', ':', ';'))) or
                    # Next text clearly starts with continuation (lowercase)
                    (next_heading['text'] and next_heading['text'][0].islower() and
                     len(current['text'].split()) <= 6)
                )
            )
            
            if should_combine:
                # Combine the texts
                combined_text = current['text'].rstrip() + " " + next_heading['text'].lstrip()
                combined_heading = {
                    'level': current['level'],
                    'text': combined_text,
                    'page': current['page']
                }
                combined_headings.append(combined_heading)
                i += 2  # Skip both headings
            else:
                combined_headings.append(current)
                i += 1
        else:
            combined_headings.append(current)
            i += 1
    
    return combined_headings

def is_likely_false_positive(text, document_context=None):
    """Dynamic false positive detection based on document characteristics"""
    text_clean = text.strip()
    
    if len(text_clean) <= 2:
        return True
    
    alpha_ratio = sum(c.isalpha() for c in text_clean) / len(text_clean)
    if alpha_ratio < 0.3:
        return True
    
    if re.search(r'(.{1,3})\1{4,}', text_clean):
        return True
    
    if re.match(r'^[\s\-_\.=]+$', text_clean):
        return True
    
    # Dynamic pattern detection based on document context
    if document_context:
        # Check if text has characteristics of non-heading content
        dot_ratio = text_clean.count('.') / len(text_clean)
        if dot_ratio > 0.15:  # High dot density suggests URLs/addresses
            return True
        
        # Check for patterns that suggest contact information
        if any(char in text_clean.lower() for char in ['@', 'www', 'http']):
            return True
        
        # Check for patterns with consecutive uppercase and dots (like URLs)
        if re.search(r'[A-Z]{3,}.*\..*[A-Z]{3,}', text_clean):
            return True
    
    return False

def discover_document_structure_simple(spans_with_pos):
    """Simplified mathematical approach - only for forms"""
    if not spans_with_pos:
        return []
    
    # Group by lines and page
    lines = {}
    for span_data in spans_with_pos:
        if len(span_data) >= 6:  # New format with page number
            text, size, flags, x, y, page_num = span_data
        else:  # Old format without page number
            text, size, flags, x, y = span_data
            page_num = 0
            
        line_key = (page_num, round(y, 1))  # Group by page and y-coordinate
        if line_key not in lines:
            lines[line_key] = []
        lines[line_key].append((text, size, flags, x))
    
    # Reconstruct full lines
    line_features = []
    for (page_num, y_pos), line_spans in lines.items():
        line_spans.sort(key=lambda x: x[3])
        full_text = " ".join([span[0] for span in line_spans]).strip()
        
        if len(full_text) > 2:
            avg_size = sum([span[1] for span in line_spans]) / len(line_spans)
            is_bold = any(span[2] & 2**4 for span in line_spans)
            
            line_features.append({
                'text': full_text,
                'size': avg_size,
                'is_bold': is_bold,
                'length': len(full_text),
                'word_count': len(full_text.split()),
                'page': page_num
            })
    
    if not line_features:
        return []
    
    # Enhanced size-based detection with better hierarchy
    sizes = [f['size'] for f in line_features]
    max_size = max(sizes)
    avg_size = sum(sizes) / len(sizes)
    
    heading_candidates = []
    for feature in line_features:
        size_ratio = feature['size'] / max_size
        
        # Improved heading level detection with confidence-based hierarchy
        heading_level = None
        
        # H1: Largest or very bold and large
        if size_ratio >= 0.9 or (size_ratio >= 0.8 and feature['is_bold']):
            heading_level = "H1"
        # H2: Medium-large or bold medium
        elif size_ratio >= 0.75 or (size_ratio >= 0.65 and feature['is_bold']):
            heading_level = "H2"  
        # H3: Smaller but still above average and formatted
        elif size_ratio >= 0.6 and (feature['is_bold'] or feature['size'] > avg_size):
            heading_level = "H3"
        
        # Apply word count limits based on level
        word_limit = 12 if heading_level == "H3" else 10 if heading_level == "H2" else 8
        
        if heading_level and feature['word_count'] <= word_limit:
            heading_candidates.append({
                'text': feature['text'],
                'level': heading_level,
                'page': feature['page']
            })
    
    return heading_candidates

def extract_with_ml_primary(doc):
    """Enhanced ML extraction using proper feature format"""
    if not ML_AVAILABLE:
        return []
    
    all_features_list = []
    all_sizes = []
    
    # Collect font sizes for normalization (matching training format)
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["size"] > 0:
                        all_sizes.append(span["size"])
    
    avg_doc_size = sum(all_sizes) / len(all_sizes) if all_sizes else 12
    max_doc_size = max(all_sizes) if all_sizes else 12

    for page_num, page in enumerate(doc):
        page_width = page.rect.width
        page_height = page.rect.height
        blocks = page.get_text("dict")["blocks"]
        
        for i, block in enumerate(blocks):
            for j, line in enumerate(block.get("lines", [])):
                line_text = " ".join([span["text"] for span in line["spans"]]).strip()
                if not line_text or len(line_text) < 3:
                    continue
                
                sizes = [span["size"] for span in line["spans"]]
                avg_size = sum(sizes) / len(sizes) if sizes else 0
                is_bold = any(span["flags"] & 2**4 for span in line["spans"])
                is_italic = any(span["flags"] & 2**6 for span in line["spans"])
                is_all_caps = line_text.isupper() and len(line_text) > 1
                
                # Enhanced text features (matching training format)
                has_colon = ':' in line_text
                ends_with_colon = line_text.rstrip().endswith(':')
                is_numeric_start = line_text[0].isdigit() if line_text else False
                
                line_bbox = line['bbox']
                line_center = (line_bbox[0] + line_bbox[2]) / 2
                page_center = page_width / 2
                is_centered = abs(line_center - page_center) < (page_width * 0.1)
                
                # Enhanced positioning
                relative_x = line_bbox[0] / page_width if page_width > 0 else 0
                relative_y = line_bbox[1] / page_height if page_height > 0 else 0
                line_width = line_bbox[2] - line_bbox[0]
                relative_width = line_width / page_width if page_width > 0 else 0

                space_below = 0
                space_above = 0
                if j + 1 < len(block.get("lines", [])):
                    next_line_bbox = block.get("lines", [])[j+1]['bbox']
                    space_below = next_line_bbox[1] - line_bbox[3]
                elif i + 1 < len(blocks):
                    next_block_bbox = blocks[i+1]['bbox']
                    space_below = next_block_bbox[1] - line_bbox[3]
                
                if j > 0:
                    prev_line_bbox = block.get("lines", [])[j-1]['bbox']
                    space_above = line_bbox[1] - prev_line_bbox[3]
                elif i > 0:
                    prev_block_bbox = blocks[i-1]['bbox']
                    space_above = line_bbox[1] - prev_block_bbox[3]
                
                # Text analysis
                word_count = len(line_text.split())
                char_count = len(line_text)
                avg_word_length = char_count / word_count if word_count > 0 else 0
                has_punctuation = any(c in line_text for c in '.,!?;')
                punctuation_ratio = sum(1 for c in line_text if c in '.,!?;:') / len(line_text) if line_text else 0
                upper_ratio = sum(1 for c in line_text if c.isupper()) / len(line_text) if line_text else 0
                title_case = line_text.istitle()

                all_features_list.append({
                    "text": line_text, "page_num": page_num,
                    "avg_size": avg_size, 
                    "is_bold": is_bold,
                    "is_italic": is_italic,
                    "word_count": word_count,
                    "char_count": char_count,
                    "avg_word_length": avg_word_length,
                    "relative_size": avg_size / avg_doc_size if avg_doc_size > 0 else 1,
                    "size_ratio_to_max": avg_size / max_doc_size if max_doc_size > 0 else 1,
                    "x_pos": line_bbox[0],
                    "relative_x": relative_x,
                    "relative_y": relative_y,
                    "relative_width": relative_width,
                    "is_all_caps": is_all_caps,
                    "has_colon": has_colon,
                    "ends_with_colon": ends_with_colon,
                    "is_numeric_start": is_numeric_start,
                    "is_centered": is_centered,
                    "space_below": space_below,
                    "space_above": space_above,
                    "has_punctuation": has_punctuation,
                    "punctuation_ratio": punctuation_ratio,
                    "upper_ratio": upper_ratio,
                    "title_case": title_case,
                })

    if not all_features_list:
        return []
    
    predict_df = pd.DataFrame(all_features_list)
    features_for_model = predict_df.drop(columns=["text", "page_num"])
    
    # Get prediction probabilities for confidence filtering
    try:
        prediction_probs = MODEL.predict_proba(features_for_model)
        predictions = MODEL.predict(features_for_model)
    except:
        predictions = MODEL.predict(features_for_model)
        prediction_probs = None
    
    headings = []
    heading_candidates = []
    
    # Collect all heading candidates with their confidences
    for i, label in enumerate(predictions):
        if label in ["H1", "H2", "H3"]:
            confidence = 1.0
            if prediction_probs is not None:
                try:
                    label_idx = list(MODEL.classes_).index(label)
                    confidence = prediction_probs[i][label_idx]
                except (ValueError, IndexError):
                    confidence = 1.0
            
            text = predict_df.loc[i, "text"]
            heading_candidates.append({
                "level": label,
                "text": normalize(text),
                "page": int(predict_df.loc[i, "page_num"]),
                "confidence": confidence
            })
    
    # Adaptive confidence threshold based on document characteristics
    if heading_candidates:
        confidences = [h["confidence"] for h in heading_candidates]
        max_conf = max(confidences)
        avg_conf = sum(confidences) / len(confidences)
        
        # Improved dynamic threshold logic with better fallback
        if len(heading_candidates) > 25:
            # Too many candidates - likely forms/tables, be stricter
            threshold = max(0.6, max_conf * 0.8)
        elif max_conf < 0.3:
            # Very low confidence document - be more lenient
            threshold = 0.15  # Lower threshold for weak confidence docs
        elif max_conf < 0.5:
            # Low confidence document - moderate threshold
            threshold = 0.25  # More lenient than before
        else:
            # Normal document - balanced threshold
            threshold = max(0.25, min(0.55, avg_conf))  # More lenient range
        
        # Apply the adaptive threshold
        for candidate in heading_candidates:
            if candidate["confidence"] > threshold:
                headings.append({
                    "level": candidate["level"],
                    "text": candidate["text"],
                    "page": candidate["page"]
                })
    
    # Deduplicate headings based on text content
    if headings:
        unique_headings = []
        seen_texts = set()
        for heading in headings:
            text_key = heading["text"].lower().strip()
            if text_key not in seen_texts:
                unique_headings.append(heading)
                seen_texts.add(text_key)
        headings = unique_headings
    
    return headings

def extract_outline(pdf_path):
    doc = fitz.open(pdf_path)
    
    # Enhanced title extraction from document content only (no metadata)
    title = ""
    
    # Extract from document content - analyze first page structure
    first_page = doc[0]
    text_blocks = first_page.get_text("dict")["blocks"]
    title_candidates = []
    
    # Collect potential title candidates from first page
    for block_idx, block in enumerate(text_blocks[:5]):  # Only check first 5 blocks
        for line_idx, line in enumerate(block.get("lines", [])[:3]):  # Only first 3 lines per block
            line_text_parts = []
            line_font_size = 0
            line_is_bold = False
            
            # Reconstruct full line and get properties
            for span in line.get("spans", []):
                text = span["text"].strip()
                if text:
                    line_text_parts.append(text)
                    line_font_size = max(line_font_size, span["size"])
                    if span["flags"] & 2**4:  # Bold flag
                        line_is_bold = True
            
            full_line = " ".join(line_text_parts).strip()
            
            # Only consider lines that could be titles
            if (full_line and 
                5 <= len(full_line) <= 150 and  # Reasonable length
                not re.match(r'^(page \d+|^\d+$|name:|date:|signature:|form|agenda item)', full_line.lower()) and
                not re.search(r'(.{1,2})\1{3,}', full_line) and  # Not repetitive characters
                not full_line.startswith('•') and  # Not bullet points
                not full_line.startswith('-') and  # Not dashes
                not re.search(r'^[A-Z\s]+:$', full_line) and  # Not section labels ending with colon
                len(full_line.split()) <= 15):  # Not too many words for a title
                
                # Score based on position, size, formatting
                score = line_font_size
                if line_is_bold:
                    score += 2
                if block_idx == 0:  # First block bonus
                    score += 3
                if line_idx == 0:  # First line bonus
                    score += 2
                
                title_candidates.append((full_line, score, line_font_size))
    
    if title_candidates:
        # Sort by score (size + position + formatting)
        title_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take the highest scoring candidate
        best_candidate = title_candidates[0] 
        title = best_candidate[0]
        
        # If title is still too long, try to extract the most title-like part
        if len(title) > 100:
            # Split by common separators and take the first meaningful part
            parts = re.split(r'[:\n\r]', title)
            if parts and len(parts[0].strip()) >= 10:
                title = parts[0].strip()
    
    # Fallback to filename-based title if no good content title found
    if not title or len(title) < 5:
        import os
        filename = os.path.basename(pdf_path)
        if filename.lower().endswith('.pdf'):
            title = filename[:-4]
        else:
            title = filename
        # Clean up filename
        title = title.replace('_', ' ').replace('-', ' ')
        title = ' '.join(title.split())  # Normalize whitespace

    # **SIMPLIFIED APPROACH: ML FIRST**
    headings = []
    total_spans = 0
    
    # Try ML first
    if ML_AVAILABLE:
        headings = extract_with_ml_primary(doc)
        
        # Enhanced fallback logic - combine with mathematical when ML is weak
        should_use_fallback = False
        
        if len(headings) == 0:
            should_use_fallback = True
        elif len(headings) <= 2:
            # Check if ML confidence was generally low
            ml_headings_raw = extract_with_ml_primary(doc)  # Get raw candidates with confidence
            # If we only got very few headings, confidence might be too restrictive
            should_use_fallback = True
        
        if should_use_fallback:
            # Check if document has structural indicators
            spans_with_pos = []
            has_size_variation = False
            all_font_sizes = []
            
            for page_num, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            spans_with_pos.append((
                                span["text"], span["size"], span["flags"],
                                span["bbox"][0], span["bbox"][1], page_num
                            ))
                            all_font_sizes.append(span["size"])
                            total_spans += 1
            
            # Check if there's significant font size variation (indicating structure)
            if all_font_sizes:
                font_range = max(all_font_sizes) - min(all_font_sizes)
                avg_size = sum(all_font_sizes) / len(all_font_sizes)
                has_size_variation = font_range > (avg_size * 0.25)  # Lower threshold: 25% variation
            
            # More lenient fallback conditions
            if has_size_variation and total_spans > 30:  # Lower span requirement
                math_headings = discover_document_structure_simple(spans_with_pos)
                if math_headings:
                    # Merge mathematical results with ML results
                    existing_texts = {h["text"].lower() for h in headings}
                    for h in math_headings:
                        if h["text"].lower() not in existing_texts and len(h["text"].split()) <= 8:
                            headings.append({
                                "level": h["level"], 
                                "text": normalize(h["text"]), 
                                "page": h.get("page", 0)
                            })
    else:
        spans_with_pos = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        spans_with_pos.append((
                            normalize(span["text"]), span["size"], span["flags"],
                            span["bbox"][0], span["bbox"][1]
                        ))
                        total_spans += 1
        
        math_headings = discover_document_structure_simple(spans_with_pos)
        headings = [{"level": h["level"], "text": h["text"], "page": 0} for h in math_headings]
    
    # **MINIMAL POST-PROCESSING**
    if headings:
        # Create document context for dynamic false positive detection
        all_heading_texts = [h['text'] for h in headings]
        document_context = {
            'all_texts': all_heading_texts,
            'total_headings': len(headings)
        }
        
        # Only remove obvious false positives
        filtered_headings = []
        for heading in headings:
            if not is_likely_false_positive(heading['text'], document_context):
                filtered_headings.append(heading)
        headings = filtered_headings
        
        # Remove exact duplicates only
        seen = set()
        unique_headings = []
        for heading in headings:
            key = (heading['level'], heading['text'].lower().strip())
            if key not in seen:
                seen.add(key)
                unique_headings.append(heading)
        
        headings = unique_headings
        headings.sort(key=lambda x: x['page'])
    
    return title.strip(), headings