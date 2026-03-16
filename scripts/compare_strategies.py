from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, model_name="gpt2", max_length=50, num_return_sequences=1, strategy="greedy"):
    """
    Generate text using a GPT-2 model with different strategies.
    
    Args:
        prompt (str): The input text to base generation on.
        model_name (str): The name of the GPT-2 model to load (e.g., 'gpt2', 'gpt2-medium').
        max_length (int): The maximum length of the generated text.
        num_return_sequences (int): The number of sequences to generate.
        strategy (str): The generation strategy ("greedy", "nucleus", "typical").
    
    Returns:
        list: Generated text sequences.
    """
    print("Loading model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    print("Model loaded.")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    if strategy == "greedy":
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=False
        )
    elif strategy == "nucleus":
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            temperature=1.0,
            top_p=0.95,
            do_sample=True
        )
    elif strategy == "typical":
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            temperature=1.0,
            top_p=0.8,
            do_sample=True
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

if __name__ == "__main__":
    # Prompt for user input
    user_prompt = input("Enter a prompt: ").strip()
    
    if not user_prompt:
        print("No prompt entered. Exiting...")
    else:
        # Test all strategies
        strategies = ["greedy", "nucleus", "typical"]
        results = {}

        # Loop through each strategy
        for strategy in strategies:
            print(f"\nTesting strategy: {strategy.capitalize()}")
            outputs = generate_text(prompt=user_prompt, max_length=100, num_return_sequences=3, strategy=strategy)
            results[strategy] = outputs

        # Display results
        for strategy, texts in results.items():
            print(f"\n=== {strategy.capitalize()} Strategy ===")
            for i, text in enumerate(texts):
                print(f"[{i+1}]: {text}")
