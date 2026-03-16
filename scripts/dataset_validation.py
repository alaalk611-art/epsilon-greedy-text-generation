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

# Updated Paths to datasets with precomputed metrics
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

# Hyperparameters to test
HYPERPARAMETERS = {
    "epsilon_values": [0.5],  
    "k_values": [5]  
}

# Output folder and file
OUTPUT_FOLDER = "results"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "dataset_results_hyperparameters.csv")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model and tokenizer
print("Loading GPT-2 model...")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
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

    for i, entry in enumerate(diversity_mauve_data):
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
            print(f"Coherence data mismatch for dataset {dataset_name}, prompt {i + 1}")
    return results

# Main function
def process_all_datasets():
    all_results = []
    for dataset_name, paths in DATASETS.items():
        print(f"Processing dataset: {dataset_name}")
        dataset_results = process_dataset_metrics(dataset_name, paths)
        all_results.extend(dataset_results)

    # Write results to a CSV file
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Dataset", "Prompt", "Diversity", "MAUVE", "Generated Length", "Coherence"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Processing complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all_datasets()
