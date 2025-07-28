# train_model.py

import os
import json
import fitz  # PyMuPDF
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
# In train_model.py
from feature_engineering import extract_features


# --- Configuration ---
# Make sure your sample PDFs and ground-truth JSONs are in these folders
PDF_DIR = "input"
JSON_DIR = "ground_truth" # You may need to create this folder and place the provided JSONs here
MODEL_OUTPUT_PATH = "heading_classifier.joblib"

# --- 1. Feature Extraction ---
# In train_model.py

def extract_features_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    features = []
    
    # Analyze document-level stats for relative sizing
    all_sizes = [span["size"] for page in doc for block in page.get_text("dict")["blocks"] for line in block.get("lines", []) for span in line.get("spans", [])]
    avg_doc_size = sum(all_sizes) / len(all_sizes) if all_sizes else 12
    max_doc_size = max(all_sizes) if all_sizes else 12
    
    # Document-level analysis for better context
    all_texts = []
    all_positions = []
    
    for page_num, page in enumerate(doc):
        page_width = page.rect.width
        page_height = page.rect.height
        blocks = page.get_text("dict")["blocks"]
        
        for i, block in enumerate(blocks):
            for j, line in enumerate(block.get("lines", [])):
                line_text = " ".join([span["text"] for span in line["spans"]]).strip()
                if not line_text:
                    continue
                
                all_texts.append(line_text)
                all_positions.append(line['bbox'])

                # Basic features
                sizes = [span["size"] for span in line["spans"]]
                avg_size = sum(sizes) / len(sizes)
                is_bold = any(span["flags"] & 2**4 for span in line["spans"])
                is_italic = any(span["flags"] & 2**6 for span in line["spans"])
                
                # Enhanced features
                is_all_caps = line_text.isupper() and len(line_text) > 1
                has_colon = ':' in line_text
                ends_with_colon = line_text.rstrip().endswith(':')
                is_numeric_start = line_text[0].isdigit() if line_text else False
                
                # Position and layout features
                line_bbox = line['bbox']
                line_center = (line_bbox[0] + line_bbox[2]) / 2
                page_center = page_width / 2
                is_centered = abs(line_center - page_center) < (page_width * 0.1)
                
                # Relative positioning
                relative_x = line_bbox[0] / page_width if page_width > 0 else 0
                relative_y = line_bbox[1] / page_height if page_height > 0 else 0
                line_width = line_bbox[2] - line_bbox[0]
                relative_width = line_width / page_width if page_width > 0 else 0
                
                # Space calculations
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
                
                # Text analysis features
                word_count = len(line_text.split())
                char_count = len(line_text)
                avg_word_length = char_count / word_count if word_count > 0 else 0
                
                # Punctuation analysis
                has_punctuation = any(c in line_text for c in '.,!?;')
                punctuation_ratio = sum(1 for c in line_text if c in '.,!?;:') / len(line_text) if line_text else 0
                
                # Case analysis
                upper_ratio = sum(1 for c in line_text if c.isupper()) / len(line_text) if line_text else 0
                title_case = line_text.istitle()
                
                features.append({
                    "text": line_text, "page_num": page_num,
                    "avg_size": avg_size, 
                    "is_bold": is_bold,
                    "is_italic": is_italic,  # New
                    "word_count": word_count,
                    "char_count": char_count,  # New
                    "avg_word_length": avg_word_length,  # New
                    "relative_size": avg_size / avg_doc_size if avg_doc_size > 0 else 1,
                    "size_ratio_to_max": avg_size / max_doc_size if max_doc_size > 0 else 1,  # New
                    "x_pos": line_bbox[0],
                    "relative_x": relative_x,  # New
                    "relative_y": relative_y,  # New
                    "relative_width": relative_width,  # New
                    "is_all_caps": is_all_caps,
                    "has_colon": has_colon,  # New
                    "ends_with_colon": ends_with_colon,  # New
                    "is_numeric_start": is_numeric_start,  # New
                    "is_centered": is_centered,
                    "space_below": space_below,
                    "space_above": space_above,  # New
                    "has_punctuation": has_punctuation,  # New
                    "punctuation_ratio": punctuation_ratio,  # New
                    "upper_ratio": upper_ratio,  # New
                    "title_case": title_case,  # New
                })
    return features

# --- 2. Data Labeling ---
def create_labeled_dataset(pdf_files, pdf_dir, json_dir):
    all_features = []
    
    for filename in pdf_files:
        pdf_path = os.path.join(pdf_dir, filename)
        json_path = os.path.join(json_dir, filename.replace(".pdf", ".json"))
        
        if not os.path.exists(json_path):
            print(f"⚠️ Warning: No ground truth JSON found for {filename}")
            continue
            
        # Extract features from PDF
        doc = fitz.open(pdf_path)
        features = extract_features(doc)
        doc.close()
        
        # Load ground truth labels
        with open(json_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        # Create a quick lookup for headings
        true_headings = {}
        for item in ground_truth.get("outline", []):
            true_headings[item["text"].strip()] = item["level"]

        # Label the features
        for feature in features:
            label = true_headings.get(feature["text"], "Other")
            feature["label"] = label
        
        all_features.extend(features)
        
    return pd.DataFrame(all_features)

# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 Starting model training...")
    
    # Prepare file lists
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    
    # 1. Create the dataset
    print("📊 Creating labeled dataset...")
    dataset = create_labeled_dataset(pdf_files, PDF_DIR, JSON_DIR)
    
    if dataset.empty:
        print("❌ Dataset is empty. Check your PDF_DIR and JSON_DIR paths.")
    else:
        print(f"✅ Dataset created with {len(dataset)} lines.")
        print("Sample of dataset:")
        print(dataset.head())
        print("\nLabel distribution:")
        print(dataset['label'].value_counts())

        # 2. Prepare data for training
        features_df = dataset.drop(columns=["text", "page_num", "label"])
        labels = dataset["label"]
        
        # Handle class imbalance with SMOTE-like approach
        from collections import Counter
        label_counts = Counter(labels)
        print(f"Class distribution: {label_counts}")
        
        # 3. Enhanced model training with multiple algorithms
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        import numpy as np
        
        print("\n🧠 Training enhanced models...")
        
        # Model 1: Enhanced Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,  # More trees
            max_depth=15,      # Deeper trees
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42, 
            class_weight='balanced_subsample',  # Better for imbalanced data
            bootstrap=True,
            max_features='sqrt'
        )
        
        # Model 2: Gradient Boosting (often better for structured data)
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42,
            subsample=0.8
        )
        
        # Train and evaluate both models
        print("Training Random Forest...")
        rf_scores = cross_val_score(rf_model, features_df, labels, cv=3, scoring='f1_weighted')
        rf_model.fit(features_df, labels)
        
        print("Training Gradient Boosting...")
        gb_scores = cross_val_score(gb_model, features_df, labels, cv=3, scoring='f1_weighted')
        gb_model.fit(features_df, labels)
        
        print(f"Random Forest CV F1: {np.mean(rf_scores):.4f} (+/- {np.std(rf_scores) * 2:.4f})")
        print(f"Gradient Boosting CV F1: {np.mean(gb_scores):.4f} (+/- {np.std(gb_scores) * 2:.4f})")
        
        # Choose the better model
        if np.mean(gb_scores) > np.mean(rf_scores):
            print("✅ Gradient Boosting performs better, using GB model")
            best_model = gb_model
            model_type = "GradientBoosting"
        else:
            print("✅ Random Forest performs better, using RF model")
            best_model = rf_model
            model_type = "RandomForest"
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': features_df.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Top 10 Most Important Features ({model_type}):")
        print(feature_importance.head(10).to_string(index=False))
        
        # 4. Save the model
        joblib.dump(best_model, MODEL_OUTPUT_PATH)
        print(f"✅ {model_type} model saved to {MODEL_OUTPUT_PATH}")