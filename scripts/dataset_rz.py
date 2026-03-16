import os
import json
import csv
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mauve import compute_mauve

# Add the path to ensure access to epsilon_greedy_search.py
sys.path.append(r"C:\Users\Dell\Desktop\Book Gen")
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

# Hyperparameters to test
HYPERPARAMETERS = {
    "epsilon_values": [0.1,0.3,0.5,0.7,0.9],
    "k_values": [1,3,5,10,20]
}

# Output folder and file
OUTPUT_FOLDER = "results"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "dataset_hyperparameters.csv")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the GPT-2 model and tokenizer
print("Loading GPT-2 model...")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model loaded.")

# Load JSON file
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Compute MAUVE
def compute_mauve_score(generated_texts, reference_texts):
    if not generated_texts or not reference_texts:
        return "N/A"
    try:
        mauve_result = compute_mauve(p_text=generated_texts, q_text=reference_texts)
        return mauve_result.mauve
    except Exception as e:
        print(f"Error computing MAUVE: {e}")
        return "N/A"

# Compute perplexity
def compute_perplexity(text):
    if not text:
        return "N/A"
    try:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()
    except Exception as e:
        print(f"Error computing Perplexity: {e}")
        return "N/A"

# Compute coherence
def compute_coherence(prompt, generated_text):
    if not prompt or not generated_text:
        return "N/A"
    try:
        inputs = tokenizer.encode(prompt + generated_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
        return torch.exp(outputs.loss).item()
    except Exception as e:
        print(f"Error computing Coherence: {e}")
        return "N/A"

# Evaluate hyperparameters for epsilon and k
def evaluate_hyperparameters(dataset_name, prompt, reference_texts):
    results = []
    for epsilon in HYPERPARAMETERS["epsilon_values"]:
        for k in HYPERPARAMETERS["k_values"]:
            try:
                generated_texts = epsilon_greedy_search(model, tokenizer, tokenizer.encode(prompt, return_tensors="pt"), max_length=50, epsilon=epsilon, k=k)
                if generated_texts:
                    diversity = len(set(generated_texts[0].split())) / len(generated_texts[0].split())
                    coherence = compute_coherence(prompt, generated_texts[0])
                    perplexity = compute_perplexity(generated_texts[0])
                    mauve = compute_mauve_score(generated_texts, reference_texts)
                    generated_length = len(generated_texts[0].split())
                else:
                    diversity = coherence = perplexity = mauve = generated_length = "N/A"

                results.append({
                    "Dataset": dataset_name,
                    "Prompt": prompt[:50] + "...",
                    "Epsilon": epsilon,
                    "k": k,
                    "Diversity": diversity,
                    "MAUVE": mauve,
                    "Generated Length": generated_length,
                    "Coherence": coherence,
                    "Perplexity": perplexity,
                    "Generated_Text": generated_texts[0] if generated_texts else "N/A"
                })
            except Exception as e:
                print(f"Error with epsilon={epsilon}, k={k}: {e}")
    return results

# Process dataset metrics
def process_dataset_metrics(dataset_name, paths):
    results = []
    diversity_mauve_data = load_json(paths["diversity_mauve"])
    coherence_data = load_json(paths["coherence"])

    coherence_scores = []
    for item in coherence_data:
        if "coherence_score_list" in item:
            coherence_scores.extend(item["coherence_score_list"])
        else:
            coherence_scores.append("N/A")

    for i, entry in enumerate(diversity_mauve_data):
        result = {
            "Dataset": dataset_name,
            "Prompt": f"Prompt {i + 1}",
            "Diversity": entry.get("prediction_dive", "N/A"),
            "MAUVE": entry.get("mauve_score", "N/A"),
            "Generated Length": entry.get("prediction_gen_len", "N/A"),
            "Coherence": coherence_scores[i] if i < len(coherence_scores) else "N/A",
            "Generated_Text": entry.get("generated_text", "N/A")
        }
        results.append(result)
    return results

# Process all datasets
def process_all_datasets():
    all_results = []
    for dataset_name, paths in DATASETS.items():
        print(f"Processing dataset: {dataset_name}")
        dataset_results = process_dataset_metrics(dataset_name, paths)

        if dataset_results:
            reference_texts = [result["Generated_Text"] for result in dataset_results if result["Generated_Text"] != "N/A"]
            for prompt_data in dataset_results[:5]:
                hyperparameter_results = evaluate_hyperparameters(dataset_name, prompt_data["Prompt"], reference_texts)
                all_results.extend(hyperparameter_results)

        all_results.extend(dataset_results)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Dataset", "Prompt", "Epsilon", "k", "Diversity", "MAUVE", "Generated Length", "Coherence", "Perplexity", "Generated_Text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Processing complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all_datasets()
