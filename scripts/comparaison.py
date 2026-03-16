import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import csv
import sys

# Add the path to ensure access to custom modules
sys.path.append(r"C:\Users\Dell\Desktop\Book Gen")

# Import custom functions and classes
from compute_diversity import measure_diversity
from compute_mauve import measure_mauve
from compute_coherence import CoherenceEvaluator
from epsilon_greedy_search import epsilon_greedy_search

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

# Output folder and file
OUTPUT_FOLDER = "results"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "comparison_results.csv")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model and tokenizer
print("Loading OPT model...")
model_path = r"C:\Users\Dell\Desktop\opt-2.7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded.")

# Load JSON file
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Process dataset metrics
def process_dataset_metrics(dataset_name, paths):
    results = []
    diversity_mauve_data = load_json(paths["diversity_mauve"])
    coherence_data = load_json(paths["coherence"])

    for i, entry in enumerate(diversity_mauve_data[:5]):  # Limit to 5 entries
        try:
            result = {
                "Dataset": dataset_name,
                "Prompt": f"Prompt {i + 1}",
                "Diversity": entry.get("prediction_dive", "N/A"),
                "MAUVE": entry.get("mauve_score", "N/A"),
                "Generated Length": entry.get("prediction_gen_len", "N/A"),
                "Coherence": coherence_data[i].get("coherence_mean", "N/A")
            }
            results.append(result)
        except IndexError:
            print(f"Error: Mismatch in dataset {dataset_name} entry {i}.")
    return results

# Main function
def process_all_datasets():
    all_results = []
    for dataset_name, paths in DATASETS.items():
        print(f"Processing dataset: {dataset_name}")
        dataset_results = process_dataset_metrics(dataset_name, paths)
        all_results.extend(dataset_results)

    # Save results to CSV
    fieldnames = ["Dataset", "Prompt", "Diversity", "MAUVE", "Generated Length", "Coherence"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Processing complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all_datasets()
