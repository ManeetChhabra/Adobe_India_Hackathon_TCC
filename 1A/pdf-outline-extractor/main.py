print("🚀 Script started")

import os
import json
import sys

# Import the enhanced hybrid approach
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from enhanced_hybrid_approach import ml_extract_outline, combine_results_enhanced
from enhanced_visual_extractor import EnhancedVisualExtractor

INPUT_DIR = "input"
OUTPUT_DIR = "output"

def process_all_pdfs():
    print("🛠 Scanning input directory...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    input_files = os.listdir(INPUT_DIR)
    pdf_files = [f for f in input_files if f.endswith(".pdf")]
    
    print(f"📁 Found {len(pdf_files)} PDF files in {INPUT_DIR}")
    
    if not pdf_files:
        print("❌ No PDF files found in input directory")
        return

    extractor = EnhancedVisualExtractor()

    for i, filename in enumerate(pdf_files, 1):
        try:
            pdf_path = os.path.join(INPUT_DIR, filename)
            print(f"\n� [{i}/{len(pdf_files)}] Processing: {filename}")
            
            # Get ML extraction result
            print("   🤖 Running ML extraction...")
            ml_title, ml_headings = ml_extract_outline(pdf_path)
            print(f"   📊 ML: {len(ml_headings)} headings")
            
            # Get Enhanced Visual extraction result
            print("   👁️ Running Enhanced Visual extraction...")
            visual_result = extractor.extract_headings_enhanced(pdf_path)
            visual_title = visual_result['title']
            visual_headings = visual_result['headings']
            print(f"   📊 Visual: {len(visual_headings)} headings")
            
            # Apply enhanced combination
            print("   ⚡ Applying enhanced combination...")
            enhanced_title, enhanced_headings, method = combine_results_enhanced(
                (ml_title, ml_headings), 
                (visual_title, visual_headings), 
                pdf_path
            )
            
            # Format output according to schema
            wrapped_output = {
                "title": enhanced_title,
                "outline": enhanced_headings  # Already cleaned in combine_results_enhanced
            }

            output_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".json"))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(wrapped_output, f, indent=2, ensure_ascii=False)

            print(f"   ✅ Saved {len(enhanced_headings)} headings to {filename.replace('.pdf', '.json')}")

        except Exception as e:
            print(f"   ❌ Failed to process {filename}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n🎯 Processing complete! Check the '{OUTPUT_DIR}' directory for results.")

if __name__ == "__main__":
    process_all_pdfs()
