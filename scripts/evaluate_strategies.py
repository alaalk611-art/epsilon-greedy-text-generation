import sys
import os
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

# Import functions from the root directory
from compute_coherence import evaluate_batch_coherence
from compute_diversity import measure_diversity
from scripts.compute_mauve import measure_mauve # type: ignore

def generate_text(prompt, model_name="gpt2", max_length=50, num_return_sequences=1, strategy="greedy"):
    """
    Generate text using GPT-2 with specified strategy.
    """
    print(f"Loading {model_name} model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    print("Model loaded.")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if strategy == "greedy":
        outputs = model.generate(input_ids=input_ids, max_length=max_length, num_return_sequences=1, do_sample=False)
    elif strategy == "nucleus":
        outputs = model.generate(input_ids=input_ids, max_length=max_length, num_return_sequences=num_return_sequences, top_p=0.95, do_sample=True)
    elif strategy == "typical":
        outputs = model.generate(input_ids=input_ids, max_length=max_length, num_return_sequences=num_return_sequences, top_p=0.8, do_sample=True)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def evaluate_results(prompt, results):
    """
    Evaluate generated results for coherence, diversity, and MAUVE.
    """
    evaluation_scores = {}

    # Load the coherence evaluation model and tokenizer
    print("Loading coherence evaluation model...")
    coherence_model_name = "facebook/opt-2.7b"
    coherence_tokenizer = AutoTokenizer.from_pretrained(coherence_model_name)
    coherence_model = AutoModelForCausalLM.from_pretrained(coherence_model_name)
    print("Coherence evaluation model loaded.")

    # Coherence
    coherence_scores = []
    for text in results:
        score = evaluate_batch_coherence(
            model=coherence_model,
            tokenizer=coherence_tokenizer,
            prefix_text_list=[prompt],
            prediction_text_list=[text],
            cuda_available=False,
            device=None
        )
        coherence_scores.append(score)
    evaluation_scores["coherence_mean"] = round(sum(coherence_scores) / len(coherence_scores), 4)

    # Diversity
    evaluation_scores["diversity"] = measure_diversity(results)

    # MAUVE
    evaluation_scores["mauve"] = measure_mauve(results)

    return evaluation_scores

if __name__ == "__main__":
    # Define strategies and prompt
    strategies = ["greedy", "nucleus", "typical"]
    user_prompt = input("Enter a prompt: ").strip()
    max_length = 100

    if not user_prompt:
        print("No prompt entered. Exiting...")
    else:
        final_scores = {}

        for strategy in strategies:
            print(f"\nGenerating text with {strategy.capitalize()} strategy...")
            outputs = generate_text(prompt=user_prompt, max_length=max_length, num_return_sequences=3, strategy=strategy)
            print(f"\nOutputs for {strategy.capitalize()} Strategy:")
            for i, text in enumerate(outputs):
                print(f"[{i+1}]: {text}")

            print("\nEvaluating results...")
            scores = evaluate_results(user_prompt, outputs)
            final_scores[strategy] = scores
            print(f"Scores for {strategy.capitalize()} Strategy: {scores}")

        # Save final scores
        with open("evaluation_results.json", "w") as f:
            json.dump(final_scores, f, indent=4)

        print("\nFinal evaluation scores saved to 'evaluation_results.json'.")
