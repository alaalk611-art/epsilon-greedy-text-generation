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
    # Load the model and tokenizer
    print("Loading model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    print("Model loaded.")
    
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Configure generation based on strategy
    if strategy == "greedy":
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=1,  # Greedy only supports 1 output
            no_repeat_ngram_size=2,  # Prevent repetitive n-grams
            do_sample=False          # Disable sampling for greedy
        )
    elif strategy == "nucleus":
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            temperature=1.0,         # Sampling temperature
            top_p=0.95,              # Nucleus sampling
            do_sample=True           # Enable sampling
        )
    elif strategy == "typical":
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            temperature=1.0,         # Sampling temperature
            top_p=0.8,               # Typical sampling uses smaller top_p
            do_sample=True           # Enable sampling
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    
    # Decode and return the generated sequences
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

if __name__ == "__main__":
    # Prompt for user input
    user_prompt = input("Enter a prompt: ").strip()
    
    if not user_prompt:
        print("No prompt entered. Exiting...")
    else:
        # Prompt for generation strategy
        print("\nSelect a generation strategy:")
        print("1. Greedy")
        print("2. Nucleus Sampling (top_p=0.95)")
        print("3. Typical Sampling (top_p=0.8)")
        strategy_choice = input("Enter 1, 2, or 3: ").strip()
        
        strategy_map = {"1": "greedy", "2": "nucleus", "3": "typical"}
        strategy = strategy_map.get(strategy_choice, "greedy")  # Default to greedy if invalid input
        
        # Generate text with the selected strategy
        results = generate_text(prompt=user_prompt, max_length=100, num_return_sequences=3, strategy=strategy)
        
        # Display the results
        print(f"\nGenerated Texts using {strategy.capitalize()} Strategy:")
        for i, text in enumerate(results):
            print(f"\n[{i+1}]: {text}")
