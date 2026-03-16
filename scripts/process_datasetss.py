import sys
import os
import json
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append(r"C:\Users\Dell\Desktop\Book Gen")
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

# Hyperparameter ranges
EPSILON_VALUES = [0.5]  # Add more values as needed
K_VALUES = [5]  # Add more values as needed

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

# Function to load JSON
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Function to compute MAUVE
def compute_mauve_score(generated_texts, reference_texts):
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

# Function to evaluate hyperparameters
def evaluate_hyperparameters(dataset_name, prompt, reference_texts):
    results = []
    for epsilon in EPSILON_VALUES:
        for k in K_VALUES:
            try:
                # Generate text using epsilon-greedy strategy
                generated_texts = epsilon_greedy_search(
                    model, tokenizer, tokenizer.encode(prompt, return_tensors="pt"),
                    max_length=50, epsilon=epsilon, k=k
                )
                if generated_texts:
                    diversity = len(set(generated_texts[0].split())) / len(generated_texts[0].split())
                    coherence = compute_coherence(prompt, generated_texts[0])
                    mauve = compute_mauve_score(generated_texts, reference_texts)
                    generated_length = len(generated_texts[0].split())
                else:
                    diversity = coherence = mauve = generated_length = "N/A"

                results.append({
                    "Dataset": dataset_name,
                    "Prompt": prompt[:50] + "...",
                    "Epsilon": epsilon,
                    "k": k,
                    "Diversity": diversity,
                    "MAUVE": mauve,
                    "Generated Length": generated_length,
                    "Coherence": coherence,
                    "Generated_Text": generated_texts[0] if generated_texts else "N/A"
                })
            except Exception as e:
                print(f"Error for epsilon={epsilon}, k={k}: {e}")
    return results

# Function to process dataset metrics
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
        results.append({
            "Dataset": dataset_name,
            "Prompt": f"Prompt {i + 1}",
            "Diversity": entry.get("prediction_dive", "N/A"),
            "MAUVE": entry.get("mauve_score", "N/A"),
            "Generated Length": entry.get("prediction_gen_len", "N/A"),
            "Coherence": coherence_scores[i] if i < len(coherence_scores) else "N/A",
            "Generated_Text": entry.get("generated_text", "N/A")
        })
    return results

# Main function to process all datasets
def process_all_datasets():
    all_results = []
    for dataset_name, paths in DATASETS.items():
        print(f"Processing dataset: {dataset_name}")
        dataset_results = process_dataset_metrics(dataset_name, paths)

        # Evaluate hyperparameters for the first prompt
        if dataset_results:
            reference_texts = [result["Generated_Text"] for result in dataset_results if result["Generated_Text"] != "N/A"]
            for prompt_data in dataset_results[:5]:  # Evaluate on up to 5 prompts per dataset
                hyperparameter_results = evaluate_hyperparameters(dataset_name, prompt_data["Prompt"], reference_texts)
                all_results.extend(hyperparameter_results)

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
