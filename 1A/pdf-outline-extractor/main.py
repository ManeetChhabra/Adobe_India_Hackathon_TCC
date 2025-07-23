print("🚀 Script started")

import os
import json
from utils.extractor import extract_outline

INPUT_DIR = "input"
OUTPUT_DIR = "output"

def process_all_pdfs():
    print("🛠 Scanning input directory...")
    input_files = os.listdir(INPUT_DIR)
    print(f"Files in {INPUT_DIR}: {input_files}")

    for filename in input_files:
        if filename.endswith(".pdf"):
            try:
                pdf_path = os.path.join(INPUT_DIR, filename)
                print(f"📄 Processing: {pdf_path}")

                title, outline = extract_outline(pdf_path)

                wrapped_output = {
                    "title": title,
                    "outline": outline
                }

                output_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".json"))
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(wrapped_output, f, indent=2, ensure_ascii=False)

                print(f"✅ Output written to {output_path}")

            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

if __name__ == "__main__":
    process_all_pdfs()
