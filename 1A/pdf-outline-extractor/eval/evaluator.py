import os
import json
import argparse
import sys
from difflib import SequenceMatcher
from time import time

GROUND_TRUTH_DIR = "ground_truth"
PREDICTIONS_DIR = "output_final"

def normalize_heading(h):
    # Handle both string ("H1", "H2") and integer (1, 2) level formats
    level = h["level"]
    if isinstance(level, int):
        level = f"H{level}"
    elif isinstance(level, str):
        level = level.strip().upper()
    
    return (
        level,
        h["text"].strip().lower(),
        h["page"]
    )

def evaluate_file(gt_path, pred_path):
    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        with open(pred_path, "r", encoding="utf-8") as f:
            pred_data = json.load(f)

        # Handle both 'outline' (ML) and 'headings' (clustering) keys
        gt_headings = gt_data.get("outline", [])
        pred_headings = pred_data.get("outline", pred_data.get("headings", []))

        gt_set = set(normalize_heading(h) for h in gt_headings)
        pred_set = set(normalize_heading(h) for h in pred_headings)

        # Debug output for first few files
        filename = os.path.basename(gt_path)
        if filename in ["file02.json", "file03.json"]:
            print(f"\n🔍 DEBUG for {filename}:")
            print(f"Ground Truth ({len(gt_set)}): {list(gt_set)[:3]}")
            print(f"Predictions ({len(pred_set)}): {list(pred_set)[:3]}")
            print(f"Matches: {gt_set & pred_set}")

        true_positives = len(gt_set & pred_set)
        precision = true_positives / len(pred_set) if pred_set else 0
        recall = true_positives / len(gt_set) if gt_set else 0
        return precision, recall

    except Exception as e:
        print(f"❌ Error processing {gt_path}: {e}")
        return 0, 0

def main():
    parser = argparse.ArgumentParser(description='Evaluate PDF outline extraction')
    parser.add_argument('--predicted_folder', default=PREDICTIONS_DIR, 
                       help='Folder containing predicted outline JSON files')
    parser.add_argument('--ground_truth_folder', default=GROUND_TRUTH_DIR,
                       help='Folder containing ground truth JSON files')
    parser.add_argument('--output_dir', default=None,
                       help='Specific output directory to evaluate (e.g., output_hybrid)')
    args = parser.parse_args()
    
    # Use output_dir if specified, otherwise use predicted_folder
    if args.output_dir:
        predictions_dir = args.output_dir
    else:
        predictions_dir = args.predicted_folder
    ground_truth_dir = args.ground_truth_folder
    
    print(f"📊 Evaluating predictions from: {predictions_dir}")
    print(f"📊 Against ground truth in: {ground_truth_dir}")
    print("=" * 50)
    
    start_time = time()
    precisions, recalls = [], []

    for filename in os.listdir(ground_truth_dir):
        if filename.endswith(".json"):
            gt_path = os.path.join(ground_truth_dir, filename)
            pred_path = os.path.join(predictions_dir, filename)

            if os.path.exists(pred_path):
                p, r = evaluate_file(gt_path, pred_path)
                precisions.append(p)
                recalls.append(r)
                print(f"✅ {filename}: Precision={p:.2f}, Recall={r:.2f}")
            else:
                print(f"⚠️ Prediction file missing: {filename}")

    if precisions and recalls:
        avg_p = sum(precisions) / len(precisions)
        avg_r = sum(recalls) / len(recalls)
        f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0
        print("\n📊 Final Evaluation:")
        print(f"Average Precision: {avg_p:.4f}")
        print(f"Average Recall:    {avg_r:.4f}")
        print(f"F1 Score:          {f1:.4f}")
    else:
        print("⚠️ No files evaluated!")

    print(f"\n⏱ Evaluation Time: {time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
