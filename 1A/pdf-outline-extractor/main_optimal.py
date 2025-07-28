#!/usr/bin/env python3
"""
🎓 MAIN_OPTIMAL.PY - Optimal Balanced Approach
============================================
Refined approach combining the best insights:
1. Simplified but effective pattern learning
2. Balanced precision-recall optimization
3. Enhanced recall without sacrificing precision
"""

import os
import json
import time
import fitz  # PyMuPDF
import re
from collections import Counter, defaultdict
from enhanced_visual_extractor import EnhancedVisualExtractor

class OptimalPatternLearner:
    """Optimal pattern learning with balanced precision-recall focus"""
    
    def __init__(self):
        self.level_patterns = {}
        self.learned_patterns = False
        self.recall_boosters = {}
    
    def learn_optimal_patterns(self, ground_truth_dir="ground_truth"):
        """Learn patterns optimized for precision-recall balance"""
        
        print("🧠 Learning optimal patterns for balanced performance...")
        
        level_data = {'H1': [], 'H2': [], 'H3': [], 'H4': []}
        
        for filename in os.listdir(ground_truth_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(ground_truth_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    outline = data.get('outline', [])
                    for heading in outline:
                        level = heading.get('level', '').strip().upper()
                        text = heading.get('text', '').strip().lower()
                        page = heading.get('page', 0)
                        
                        if level in level_data:
                            features = self._extract_key_features(text, page)
                            level_data[level].append(features)
                            
                except Exception as e:
                    continue
        
        # Learn level-specific patterns
        for level, feature_list in level_data.items():
            if not feature_list:
                continue
                
            patterns = self._analyze_optimal_patterns(feature_list)
            self.level_patterns[level] = patterns
        
        # Learn recall boosters - patterns that improve content discovery
        self.recall_boosters = self._learn_recall_boosters(level_data)
        
        self.learned_patterns = True
        
        # Display learned patterns
        print("📊 Optimal Patterns Learned:")
        for level, patterns in self.level_patterns.items():
            print(f"   {level}: {patterns['sample_count']} samples")
            print(f"      Word range: {patterns['word_range']}")
            print(f"      Key pattern: {patterns['primary_pattern']}")
        
        print(f"\n🔄 Recall boosters learned: {len(self.recall_boosters)} patterns")
        
        return self.level_patterns
    
    def _extract_key_features(self, text, page):
        """Extract essential features for pattern learning"""
        words = text.split()
        
        return {
            'text': text,
            'word_count': len(words),
            'char_count': len(text),
            'page': page,
            'has_colon': text.endswith(':'),
            'has_numbers': bool(re.search(r'\d', text)),
            'has_section_pattern': bool(re.search(r'^\d+\.', text)),
            'has_subsection_pattern': bool(re.search(r'^\d+\.\d+', text)),
            'is_early_page': page <= 5,
            'word_length_avg': sum(len(w) for w in words) / len(words) if words else 0,
            'capital_words': sum(1 for w in words if w and w[0].isupper()),
        }
    
    def _analyze_optimal_patterns(self, feature_list):
        """Analyze patterns with focus on discriminative power"""
        if not feature_list:
            return {}
        
        # Basic statistics
        word_counts = [f['word_count'] for f in feature_list]
        
        patterns = {
            'sample_count': len(feature_list),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'word_range': (min(word_counts), max(word_counts)),
            'colon_probability': sum(1 for f in feature_list if f['has_colon']) / len(feature_list),
            'number_probability': sum(1 for f in feature_list if f['has_numbers']) / len(feature_list),
            'section_pattern_prob': sum(1 for f in feature_list if f['has_section_pattern']) / len(feature_list),
            'subsection_pattern_prob': sum(1 for f in feature_list if f['has_subsection_pattern']) / len(feature_list),
            'early_page_preference': sum(1 for f in feature_list if f['is_early_page']) / len(feature_list),
        }
        
        # Determine primary pattern for this level
        if patterns['colon_probability'] > 0.4:
            patterns['primary_pattern'] = 'colon_ended'
        elif patterns['section_pattern_prob'] > 0.3:
            patterns['primary_pattern'] = 'section_numbered'
        elif patterns['subsection_pattern_prob'] > 0.3:
            patterns['primary_pattern'] = 'subsection_numbered'
        elif patterns['early_page_preference'] > 0.6:
            patterns['primary_pattern'] = 'structural'
        else:
            patterns['primary_pattern'] = 'content'
        
        return patterns
    
    def _learn_recall_boosters(self, level_data):
        """Learn patterns that can boost recall without hurting precision"""
        boosters = {}
        
        all_texts = []
        for level, features in level_data.items():
            for f in features:
                all_texts.append(f['text'])
        
        # Find common meaningful patterns
        word_frequency = Counter()
        for text in all_texts:
            words = text.split()
            word_frequency.update(words)
        
        # Identify structural/important terms
        important_terms = []
        for word, freq in word_frequency.items():
            if (len(word) > 3 and 
                freq >= 2 and 
                freq <= len(all_texts) * 0.3 and  # Not too common
                not word.isdigit()):
                important_terms.append(word)
        
        boosters['structural_terms'] = important_terms[:20]
        
        # Pattern-based boosters
        boosters['numbered_sections'] = sum(1 for text in all_texts if re.search(r'^\d+\.', text))
        boosters['colon_sections'] = sum(1 for text in all_texts if text.endswith(':'))
        
        return boosters

def calculate_optimal_level_probability(features, learned_patterns):
    """Optimized level calculation balancing precision and recall"""
    
    if not learned_patterns:
        # Simple fallback rules
        if features['has_colon'] and features['word_count'] <= 6:
            return 'H3'
        elif features['has_subsection_pattern']:
            return 'H2'
        elif features['has_section_pattern']:
            return 'H1'
        else:
            return 'H1'
    
    level_scores = {}
    
    for level, patterns in learned_patterns.items():
        if patterns['sample_count'] == 0:
            level_scores[level] = 0
            continue
        
        score = 0
        
        # 1. Word count compatibility (major factor)
        word_diff = abs(features['word_count'] - patterns['avg_word_count'])
        if word_diff <= 2:
            score += 8  # Very close match
        elif word_diff <= 4:
            score += 5  # Reasonable match
        else:
            score += max(0, 3 - word_diff * 0.5)  # Penalty for big differences
        
        # 2. Primary pattern matching (major factor)
        primary_pattern = patterns['primary_pattern']
        
        if primary_pattern == 'colon_ended' and features['has_colon']:
            score += 10
        elif primary_pattern == 'section_numbered' and features['has_section_pattern']:
            score += 10
        elif primary_pattern == 'subsection_numbered' and features['has_subsection_pattern']:
            score += 10
        elif primary_pattern == 'structural' and features['is_early_page']:
            score += 6
        elif primary_pattern == 'content':
            score += 3  # Neutral boost for content headings
        
        # 3. Secondary pattern bonuses
        if features['has_colon']:
            score += patterns['colon_probability'] * 6
        
        if features['has_numbers']:
            score += patterns['number_probability'] * 6
        
        if features['is_early_page']:
            score += patterns['early_page_preference'] * 4
        
        # 4. Avoid extreme mismatches
        word_min, word_max = patterns['word_range']
        if not (word_min <= features['word_count'] <= word_max + 3):
            score *= 0.7  # Penalty for being outside expected range
        
        level_scores[level] = score
    
    # Return best scoring level
    if level_scores:
        return max(level_scores, key=level_scores.get)
    else:
        return 'H1'

def optimal_content_selection(headings, total_pages, learner):
    """Optimized content selection balancing precision and recall"""
    
    if not headings:
        return []
    
    scored_headings = []
    
    for heading in headings:
        text = heading.get('text', '').strip()
        page = heading.get('page', 0)
        features = learner._extract_key_features(text.lower(), page)
        
        # Base scoring
        score = 0
        
        # 1. Visual features (proven important)
        font_size = heading.get('font_size', 0)
        score += min(font_size / 2, 10)
        
        flags = heading.get('flags', 0)
        if flags & 2**4:  # Bold
            score += 4
        if flags & 2**6:  # Italic
            score += 1
        
        # 2. Structural pattern bonuses
        if features['has_section_pattern']:
            score += 6
        elif features['has_subsection_pattern']:
            score += 5
        
        if features['has_colon']:
            score += 4
        
        # 3. Recall boosters (without over-boosting)
        if learner.recall_boosters:
            structural_terms = learner.recall_boosters.get('structural_terms', [])
            text_words = text.lower().split()
            matches = sum(1 for word in text_words if word in structural_terms)
            if matches > 0:
                score += min(matches * 2, 4)  # Cap to prevent over-boosting
        
        # 4. Position-based scoring
        if features['is_early_page']:
            score += 2
        
        # 5. Quality penalties
        if features['word_count'] > 15:
            score -= 2
        elif features['word_count'] < 2:
            score -= 2
        
        # 6. Level compatibility boost
        if learner.learned_patterns:
            best_level = calculate_optimal_level_probability(features, learner.level_patterns)
            if best_level in learner.level_patterns:
                # Small boost for being a good fit for some level
                score += 1
        
        scored_headings.append({
            'heading': heading,
            'score': score,
            'features': features
        })
    
    # Sort by score
    scored_headings.sort(key=lambda x: x['score'], reverse=True)
    
    # Adaptive selection strategy
    if total_pages <= 1:
        max_count = min(5, len(scored_headings))
    elif total_pages <= 5:
        max_count = min(12, len(scored_headings))
    elif total_pages <= 15:
        max_count = min(20, len(scored_headings))
    else:
        max_count = min(30, len(scored_headings))
    
    # Quality-based selection with balanced thresholds
    if scored_headings:
        top_score = scored_headings[0]['score']
        
        # Two-tier selection for balance
        high_quality_threshold = top_score * 0.75
        medium_quality_threshold = top_score * 0.5
        
        selected = []
        
        # Tier 1: High quality (always include)
        for item in scored_headings:
            if item['score'] >= high_quality_threshold and len(selected) < max_count:
                selected.append(item)
        
        # Tier 2: Medium quality (include if needed for recall)
        remaining_slots = max_count - len(selected)
        if remaining_slots > 0 and len(selected) < max(3, total_pages // 2):
            for item in scored_headings:
                if (medium_quality_threshold <= item['score'] < high_quality_threshold 
                    and len(selected) < max_count):
                    selected.append(item)
                    remaining_slots -= 1
                    if remaining_slots == 0:
                        break
        
        return selected
    
    return []

def main():
    print("🎓 OPTIMAL BALANCED EXTRACTOR")
    print("=" * 60)
    print("🧠 Optimal ML approach balancing precision and recall...")
    print("=" * 60)
    
    # Initialize optimal learner
    learner = OptimalPatternLearner()
    
    # Learn optimal patterns
    learner.learn_optimal_patterns()
    
    input_folder = "input"
    output_folder = "output_final"
    
    os.makedirs(output_folder, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    print(f"📁 Processing {len(pdf_files)} files...")
    
    total_start_time = time.time()
    extractor = EnhancedVisualExtractor()
    
    for pdf_file in pdf_files:
        start_time = time.time()
        pdf_path = os.path.join(input_folder, pdf_file)
        
        print(f"📄 Processing: {pdf_file}")
        
        try:
            result = extractor.extract_headings_enhanced(pdf_path)
            headings = result.get('headings', [])
            
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            # Optimal selection
            selected_items = optimal_content_selection(headings, total_pages, learner)
            
            outline = []
            for item in selected_items:
                features = item['features']
                heading = item['heading']
                
                # Optimal level assignment
                level = calculate_optimal_level_probability(features, learner.level_patterns)
                
                outline.append({
                    "level": level,
                    "text": heading.get('text', '').strip(),
                    "page": heading.get('page', 0)
                })
            
            output = {
                "title": pdf_file.replace('.pdf', ''),
                "outline": outline
            }
            
            output_file = pdf_file.replace('.pdf', '.json')
            output_path = os.path.join(output_folder, output_file)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            processing_time = time.time() - start_time
            print(f"✅ Success: {len(outline)} headings in {processing_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {str(e)}")
    
    total_time = time.time() - total_start_time
    print(f"🏁 Completed in {total_time:.2f} seconds")
    print(f"📊 Average: {total_time/len(pdf_files):.2f}s per file")

if __name__ == "__main__":
    main()
