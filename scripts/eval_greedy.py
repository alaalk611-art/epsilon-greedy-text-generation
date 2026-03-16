import os
import json
import csv
import sys
sys.path.append(r"C:\Users\Dell\Desktop\Book Gen")
from compute_diversity import measure_diversity
from compute_mauve import measure_mauve
from compute_coherence import CoherenceEvaluator


# Paths to datasets
DATASETS = {
    "book": {
        "diversity_mauve": "C:/Users/Dell/Desktop/Book Gen/book/book_greedy_gpt2-xl_256_diversity_mauve_gen_length_result.json",
        "coherence": "C:/Users/Dell/Desktop/Book Gen/book/book_greedy_gpt2-xl_256_opt-2.7b_coherence_result.json"
    },
    "wikitext": {
        "diversity_mauve": "C:/Users/Dell/Desktop/Book Gen/wikitext/wikitext_greedy_gpt2-xl_256_diversity_mauve_gen_length_result.json",
        "coherence": "C:/Users/Dell/Desktop/Book Gen/wikitext/wikitext_greedy_gpt2-xl_256_opt-2.7b_coherence_result.json"
    },
    "wikinews": {
        "diversity_mauve": "C:/Users/Dell/Desktop/Book Gen/wikinews/wikinews_greedy_gpt2-xl_256_diversity_mauve_gen_length_result.json",
        "coherence": "C:/Users/Dell/Desktop/Book Gen/wikinews/wikinews_greedy_gpt2-xl_256_opt-2.7b_coherence_result.json"
    }
}

# Output CSV path
OUTPUT_FILE = "comparison_results.csv"

def process_dataset(dataset_name, paths):
    """
    Process each dataset and collect metrics for all strategies.
    """
    results = {"Dataset": dataset_name}

    # Load diversity & MAUVE results
    diversity_mauve_path = paths["diversity_mauve"]
    coherence_path = paths["coherence"]

    if os.path.exists(diversity_mauve_path):
        with open(diversity_mauve_path, "r") as f:
            diversity_mauve_results = json.load(f)[0]

        results.update({
            "Diversity": diversity_mauve_results.get("prediction_dive", "N/A"),
            "MAUVE": diversity_mauve_results.get("mauve_score", "N/A"),
            "Gen_Length": diversity_mauve_results.get("prediction_gen_len", "N/A"),
        })

    if os.path.exists(coherence_path):
        with open(coherence_path, "r") as f:
            coherence_data = json.load(f)[0]
        results.update({
            "Coherence_Mean": coherence_data.get("coherence_mean", "N/A"),
            "Coherence_STD": coherence_data.get("coherence_std", "N/A"),
        })

    return results

def compare_strategies():
    """
    Compare all strategies across datasets and save results to a CSV file.
    """
    all_results = []
    for dataset_name, paths in DATASETS.items():
        dataset_results = process_dataset(dataset_name, paths)
        all_results.append(dataset_results)

    # Save to CSV
    fieldnames = [
        "Dataset", "Diversity", "MAUVE", "Gen_Length", "Coherence_Mean", "Coherence_STD"
    ]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    print("Starting strategy comparison...")
    compare_strategies()
