#!/usr/bin/env python3
"""
Multilingual Support Module for PDF Outline Extraction
Supports Hindi and other languages with language detection and processing
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional

class MultilingualProcessor:
    """
    Handles multilingual text processing for PDF outline extraction
    """
    
    def __init__(self):
        # Hindi language patterns and keywords
        self.hindi_heading_patterns = [
            # Common Hindi heading patterns
            r'^अध्याय\s+\d+',  # Chapter N
            r'^भाग\s+\d+',     # Part N  
            r'^खंड\s+\d+',     # Section N
            r'^प्रकरण\s+\d+',   # Chapter N (alternate)
            r'^अनुभाग\s+\d+',  # Subsection N
            r'^\d+\.\s*[\u0900-\u097F]',  # 1. Hindi text
            r'^[क-ह]\.\s*[\u0900-\u097F]',  # क. Hindi text (Devanagari bullets)
        ]
        
        # Hindi form field patterns (to filter out)
        self.hindi_form_patterns = [
            r'^नाम:\s*',       # Name:
            r'^पता:\s*',       # Address:
            r'^फोन:\s*',       # Phone:
            r'^ईमेल:\s*',      # Email:
            r'^दिनांक:\s*',     # Date:
            r'^हस्ताक्षर:\s*',   # Signature:
            r'^आवेदन:\s*',     # Application:
            r'^प्रपत्र:\s*',    # Form:
        ]
        
        # Common Hindi heading words (boost confidence)
        self.hindi_heading_indicators = [
            'परिचय', 'प्रस्तावना', 'अध्याय', 'भाग', 'खंड', 'प्रकरण',
            'अनुभाग', 'विषय', 'शीर्षक', 'मुख्य', 'महत्वपूर्ण',
            'निष्कर्ष', 'सारांश', 'परिणाम', 'सुझाव', 'सिफारिश'
        ]
        
        # Unicode ranges for different scripts
        self.script_ranges = {
            'devanagari': (0x0900, 0x097F),  # Hindi/Sanskrit
            'arabic': (0x0600, 0x06FF),      # Arabic
            'chinese': (0x4E00, 0x9FFF),     # CJK Unified Ideographs
            'latin': (0x0000, 0x024F),       # Latin scripts
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect the primary language/script of the text
        """
        if not text.strip():
            return 'unknown'
        
        script_counts = {script: 0 for script in self.script_ranges}
        
        for char in text:
            char_code = ord(char)
            for script, (start, end) in self.script_ranges.items():
                if start <= char_code <= end:
                    script_counts[script] += 1
                    break
        
        # Find the dominant script
        total_chars = sum(script_counts.values())
        if total_chars == 0:
            return 'unknown'
        
        dominant_script = max(script_counts, key=script_counts.get)
        if script_counts[dominant_script] / total_chars > 0.3:
            return dominant_script
        
        return 'mixed'
    
    def is_hindi_form_field(self, text: str) -> bool:
        """
        Check if text is a Hindi form field
        """
        for pattern in self.hindi_form_patterns:
            if re.match(pattern, text, re.UNICODE):
                return True
        return False
    
    def assess_hindi_heading_quality(self, text: str) -> float:
        """
        Assess the quality of a Hindi heading
        """
        if not text.strip():
            return 0.0
        
        quality_score = 0.5  # Base score
        
        # Check for Hindi heading patterns
        for pattern in self.hindi_heading_patterns:
            if re.match(pattern, text, re.UNICODE):
                quality_score += 0.3
                break
        
        # Check for Hindi heading indicators
        for indicator in self.hindi_heading_indicators:
            if indicator in text:
                quality_score += 0.2
                break
        
        # Length-based scoring (Hindi text)
        if 5 <= len(text) <= 100:
            quality_score += 0.1
        
        # Check for proper Devanagari script
        devanagari_chars = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        if devanagari_chars > 0:
            quality_score += 0.1
        
        # Penalty for very long text (likely paragraph, not heading)
        if len(text) > 150:
            quality_score *= 0.7
        
        return min(1.0, quality_score)
    
    def normalize_hindi_text(self, text: str) -> str:
        """
        Normalize Hindi text for better processing
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize Unicode (NFD -> NFC)
        text = unicodedata.normalize('NFC', text)
        
        # Remove common punctuation that might interfere
        text = re.sub(r'[।॥]+$', '', text)  # Remove trailing danda/double danda
        
        return text
    
    def enhance_multilingual_extraction(self, headings: List[Dict], pdf_path: str = "") -> List[Dict]:
        """
        Enhance heading extraction with multilingual support
        """
        enhanced_headings = []
        
        for heading in headings:
            text = heading.get('text', '').strip()
            if not text:
                continue
            
            # Detect language
            language = self.detect_language(text)
            
            # Process based on detected language
            if language == 'devanagari':  # Hindi
                # Skip form fields
                if self.is_hindi_form_field(text):
                    continue
                
                # Normalize text
                normalized_text = self.normalize_hindi_text(text)
                
                # Assess quality
                quality_score = self.assess_hindi_heading_quality(normalized_text)
                
                # Only include high-quality headings
                if quality_score >= 0.4:
                    enhanced_heading = heading.copy()
                    enhanced_heading['text'] = normalized_text
                    enhanced_heading['language'] = 'hindi'
                    enhanced_heading['multilingual_quality'] = quality_score
                    enhanced_headings.append(enhanced_heading)
            
            elif language == 'latin':  # English/Latin script
                # Use existing English processing
                enhanced_headings.append(heading)
            
            elif language == 'mixed':  # Mixed script (common in Indian documents)
                # Process as bilingual content
                normalized_text = self.normalize_hindi_text(text)
                enhanced_heading = heading.copy()
                enhanced_heading['text'] = normalized_text
                enhanced_heading['language'] = 'mixed'
                enhanced_heading['multilingual_quality'] = 0.6  # Medium confidence
                enhanced_headings.append(enhanced_heading)
            
            else:
                # Keep other languages as-is for now
                enhanced_headings.append(heading)
        
        return enhanced_headings
    
    def extract_multilingual_title(self, text_blocks: List[str], language_hints: List[str] = None) -> Tuple[str, str]:
        """
        Extract title with multilingual support
        """
        if not text_blocks:
            return "", "unknown"
        
        # Try to find the best title candidate
        best_title = ""
        best_language = "unknown"
        best_score = 0.0
        
        for text in text_blocks[:5]:  # Check first 5 text blocks
            if not text.strip():
                continue
            
            language = self.detect_language(text)
            
            # Score based on position (earlier = better) and language quality
            position_score = max(0.1, 1.0 - (text_blocks.index(text) * 0.2))
            
            if language == 'devanagari':
                # Hindi title processing
                if not self.is_hindi_form_field(text):
                    normalized = self.normalize_hindi_text(text)
                    quality = self.assess_hindi_heading_quality(normalized)
                    total_score = position_score * quality
                    
                    if total_score > best_score and len(normalized) <= 100:
                        best_title = normalized
                        best_language = 'hindi'
                        best_score = total_score
            
            elif language == 'latin':
                # English title processing
                if len(text) <= 100 and not any(field in text.lower() for field in ['name:', 'email:', 'phone:']):
                    total_score = position_score * 0.8
                    if total_score > best_score:
                        best_title = text.strip()
                        best_language = 'english'
                        best_score = total_score
        
        return best_title, best_language

# Global instance
multilingual_processor = MultilingualProcessor()


def enhance_with_multilingual_support(headings: List[Dict], pdf_path: str = "") -> List[Dict]:
    """
    Wrapper function to enhance headings with multilingual support
    """
    return multilingual_processor.enhance_multilingual_extraction(headings, pdf_path)


def extract_multilingual_title(text_blocks: List[str]) -> Tuple[str, str]:
    """
    Wrapper function to extract title with multilingual support
    """
    return multilingual_processor.extract_multilingual_title(text_blocks)
