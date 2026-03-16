import os
import sys
import csv
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add the path to the parent directory for importing evaluate_text_
sys.path.append("C:/Users/Dell/Desktop/Book Gen")
from evaluate_text_ import generate_text_with_strategy
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from epsilon_greedy_search import epsilon_greedy_search
from evaluate_text_ import generate_text_with_strategy

# Metrics functions
def evaluate_diversity(results):
    all_tokens = [token for text in results for token in text.split()]
    unique_tokens = len(set(all_tokens))
    return unique_tokens / len(all_tokens) if len(all_tokens) > 0 else 0

def evaluate_coherence(model, tokenizer, prefix_text, prediction_text):
    context_ids = tokenizer(prefix_text, return_tensors="pt").input_ids
    prediction_ids = tokenizer(prediction_text, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(input_ids=torch.cat([context_ids, prediction_ids], dim=1)).logits
    probabilities = torch.softmax(logits[:, -1, :], dim=-1)
    return torch.mean(torch.log(probabilities[:, prediction_ids[0]])).item()

def evaluate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    return np.exp(loss)

# Load coherence evaluation model
def load_coherence_model():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# Main function
def evaluate_single_prompt(prompt, max_length=50, num_return_sequences=1, epsilon=0.8, k=5, top_p=0.9, num_beams=3):
    coherence_model, coherence_tokenizer = load_coherence_model()
    strategies = ["greedy", "beam", "nucleus", "epsilon_greedy"]

    results = []
    for strategy in strategies:
        print(f"\nEvaluating {strategy} strategy...")
        if strategy == "epsilon_greedy":
            generated_texts = generate_text_with_strategy(
                prompt, max_length=max_length, num_return_sequences=num_return_sequences,
                strategy=strategy, epsilon=epsilon, k=k
            )
        elif strategy == "nucleus":
            generated_texts = generate_text_with_strategy(
                prompt, max_length=max_length, num_return_sequences=num_return_sequences,
                strategy=strategy, top_p=top_p
            )
        elif strategy == "beam":
            generated_texts = generate_text_with_strategy(
                prompt, max_length=max_length, num_return_sequences=num_return_sequences,
                strategy=strategy, num_beams=num_beams
            )
        elif strategy == "greedy":
            generated_texts = generate_text_with_strategy(
                prompt, max_length=max_length, num_return_sequences=1, strategy=strategy
            )
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        # Calculate metrics
        diversity = evaluate_diversity(generated_texts)
        coherence_scores = [
            evaluate_coherence(coherence_model, coherence_tokenizer, prompt, text)
            for text in generated_texts
        ]
        perplexity = [
            evaluate_perplexity(coherence_model, coherence_tokenizer, text)
            for text in generated_texts
        ]

        results.append({
            "Strategy": strategy,
            "Diversity": diversity,
            "Coherence": np.mean(coherence_scores),
            "Perplexity": np.mean(perplexity),
            "Generated_Texts": generated_texts
        })

    return results

if __name__ == "__main__":
    # Input prompt
    prompt = input("Enter a single prefix: ").strip()
    if not prompt:
        print("No prefix entered. Exiting...")
    else:
        results = evaluate_single_prompt(prompt)
        print("\nEvaluation Results:")
        for result in results:
            print(f"\nStrategy: {result['Strategy']}")
            print(f"Diversity: {result['Diversity']}")
            print(f"Coherence: {result['Coherence']}")
            print(f"Perplexity: {result['Perplexity']}")
            print(f"Generated Texts: {result['Generated_Texts']}")
