import os
import csv
import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mauve import compute_mauve

# Add the path to ensure access to epsilon_greedy_search.py
sys.path.append(r"C:\Users\Dell\Desktop\Book Gen")
import os
import json
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mauve import compute_mauve
from epsilon_greedy_search import epsilon_greedy_search

# Paths to datasets
DATASETS = {
    "book": {
        "diversity_mauve": "C:/Users/Dell/Desktop/Book Gen/book/book_greedy_gpt2-xl_256_diversity_mauve_gen_length_result.json",
        "coherence": "C:/Users/Dell/Desktop/Book Gen/book/book_greedy_gpt2-xl_256_opt-2.7b_coherence_result.json"
    },
    "wikinews": {
        "diversity_mauve": "C:/Users/Dell/Desktop/Book Gen/wikinews/wikinews_greedy_gpt2-xl_256_diversity_mauve_gen_length_result.json",
        "coherence": "C:/Users/Dell/Desktop/Book Gen/wikinews/wikinews_greedy_gpt2-xl_256_opt-2.7b_coherence_result.json"
    },
    "wikitext": {
        "diversity_mauve": "C:/Users/Dell/Desktop/Book Gen/wikitext/wikitext_greedy_gpt2-xl_256_diversity_mauve_gen_length_result.json",
        "coherence": "C:/Users/Dell/Desktop/Book Gen/wikitext/wikitext_greedy_gpt2-xl_256_opt-2.7b_coherence_result.json"
    }
}

# Hyperparameters
EPSILON = 0.5
K = 5

# Output folder and file
OUTPUT_FOLDER = "results"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "evaluation_results_hyperparameters.csv")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the GPT-2 model and tokenizer
print("Loading GPT-2 model...")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model loaded.")

# Function to compute MAUVE
def compute_mauve_score(generated_texts, reference_texts):
    if not generated_texts or not reference_texts:
        print("MAUVE computation skipped: Empty references or generated texts.")
        return "N/A"
    try:
        mauve_result = compute_mauve(p_text=generated_texts, q_text=reference_texts)
        return mauve_result.mauve
    except Exception as e:
        print(f"Error computing MAUVE: {e}")
        return "N/A"

# Function to compute coherence
def compute_coherence(prompt, generated_text):
    try:
        inputs = tokenizer.encode(prompt + generated_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
        return torch.exp(outputs.loss).item()
    except Exception as e:
        print(f"Error computing Coherence: {e}")
        return "N/A"

# Evaluate dataset
def evaluate_dataset(dataset_name, paths):
    results = []
    # Load diversity and coherence data
    diversity_mauve_data = load_json(paths["diversity_mauve"])
    coherence_data = load_json(paths["coherence"])

    # Generate reference texts
    reference_texts = [entry.get("generated_text", "") for entry in diversity_mauve_data if entry.get("generated_text")]

    for i, entry in enumerate(diversity_mauve_data):
        prompt = f"Prompt {i + 1}"
        try:
            generated_texts = epsilon_greedy_search(
                model, tokenizer, tokenizer.encode(prompt, return_tensors="pt"),
                max_length=50, epsilon=EPSILON, k=K
            )
            generated_text = generated_texts[0] if generated_texts else "N/A"
            diversity = len(set(generated_text.split())) / len(generated_text.split()) if generated_text else "N/A"
            coherence = compute_coherence(prompt, generated_text) if generated_text else "N/A"
            mauve = compute_mauve_score([generated_text], reference_texts)
            generated_length = len(generated_text.split()) if generated_text else "N/A"

            results.append({
                "Dataset": dataset_name,
                "Prompt": prompt,
                "Epsilon": EPSILON,
                "k": K,
                "Diversity": diversity,
                "MAUVE": mauve,
                "Generated Length": generated_length,
                "Coherence": coherence,
                "Generated_Text": generated_text
            })
        except Exception as e:
            print(f"Error processing {prompt}: {e}")
    return results

# Load JSON file
def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file at {file_path}: {e}")
        return []

# Process all datasets
def process_all_datasets():
    all_results = []
    for dataset_name, paths in DATASETS.items():
        print(f"Processing dataset: {dataset_name}")
        dataset_results = evaluate_dataset(dataset_name, paths)
        all_results.extend(dataset_results)

    # Write results to a CSV file
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Dataset", "Prompt", "Epsilon", "k", "Diversity", "MAUVE", "Generated Length", "Coherence", "Generated_Text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Processing complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all_datasets()
