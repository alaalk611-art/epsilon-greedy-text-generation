import os
import json
from generate_text_utils import generate_text
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from mauve import compute_mauve  # Import MAUVE library

# Evaluate generated results
def evaluate_results(prompt, results):
    """
    Evaluate generated results for coherence, diversity, and MAUVE.
    """
    evaluation_scores = {}

    # Coherence Evaluation
    print("Loading coherence evaluation model...")
    coherence_model_name = "facebook/opt-2.7b"
    coherence_model = AutoModelForCausalLM.from_pretrained(coherence_model_name)
    coherence_tokenizer = AutoTokenizer.from_pretrained(coherence_model_name)
    print("Coherence evaluation model loaded.")

    def evaluate_coherence(model, tokenizer, prefix_text, prediction_text):
        context_ids = tokenizer(prefix_text, return_tensors="pt").input_ids
        prediction_ids = tokenizer(prediction_text, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(input_ids=torch.cat([context_ids, prediction_ids], dim=1)).logits
        probabilities = torch.softmax(logits[:, -1, :], dim=-1)
        return torch.mean(torch.log(probabilities[:, prediction_ids[0]])).item()

    coherence_scores = [
        evaluate_coherence(coherence_model, coherence_tokenizer, prompt, text) for text in results
    ]
    evaluation_scores["coherence_mean"] = np.mean(coherence_scores)

    # Diversity (Basic Example Using Unique Tokens)
    all_tokens = [token for text in results for token in text.split()]
    unique_tokens = len(set(all_tokens))
    evaluation_scores["diversity"] = unique_tokens / len(all_tokens)

    # MAUVE Evaluation
    print("Calculating MAUVE score...")
    mauve_result = compute_mauve(p_text=results, q_text=[prompt], device_id=-1)  # Set device_id=0 for GPU
    evaluation_scores["mauve"] = mauve_result.mauve

    return evaluation_scores


if __name__ == "__main__":
    # User-defined inputs
    prompt = input("Enter a prompt: ").strip()
    max_length = 50
    epsilon = 0.5
    k = 5
    num_return_sequences = 3

    if not prompt:
        print("No prompt entered. Exiting...")
    else:
        # Generate text using Epsilon Greedy Search
        print(f"\nGenerating text using Epsilon Greedy Search (ε={epsilon}, k={k}):")
        generated_texts = generate_text(prompt, max_length=max_length, epsilon=epsilon, k=k, num_return_sequences=num_return_sequences)

        print("\nGenerated Texts:")
        for idx, text in enumerate(generated_texts):
            print(f"[{idx + 1}]: {text}")

        # Evaluate generated results
        print("\nEvaluating results...")
        scores = evaluate_results(prompt, generated_texts)
        print(f"Evaluation Scores: {scores}")

        # Save results
        save_path = "epsilon_greedy_results.json"
        with open(save_path, "w") as f:
            json.dump({"prompt": prompt, "generated_texts": generated_texts, "evaluation_scores": scores}, f, indent=4)
        print(f"\nResults saved to {save_path}.")
