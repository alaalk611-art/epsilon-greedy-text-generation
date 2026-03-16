from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, model_name="gpt2", max_length=50, num_return_sequences=1, use_sampling=True):
    """
    Generate text using a GPT-2 model.
    
    Args:
        prompt (str): The input text to base generation on.
        model_name (str): The name of the GPT-2 model to load (e.g., 'gpt2', 'gpt2-medium').
        max_length (int): The maximum length of the generated text.
        num_return_sequences (int): The number of sequences to generate.
        use_sampling (bool): Whether to use sampling-based methods (e.g., top-p or top-k).
    
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
    
    # Generate text
    print("Generating text...")
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences if use_sampling else 1,  # Ensure 1 for greedy
        no_repeat_ngram_size=2,  # Prevent repetitive n-grams
        temperature=1.0,         # Sampling temperature
        top_k=50 if use_sampling else 0,  # Use top-k sampling if sampling is enabled
        top_p=0.95 if use_sampling else 1.0,  # Nucleus sampling if sampling is enabled
        do_sample=use_sampling,  # Enable sampling-based generation
    )
    
    # Decode and return the generated sequences
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

if __name__ == "__main__":
    # Prompt for user input
    user_prompt = input("Enter a prompt: ")  # Waits for user input
    
    # Check if the user entered a prompt
    if not user_prompt.strip():
        print("No prompt entered. Exiting...")
    else:
        # Generate text
        results = generate_text(prompt=user_prompt, max_length=100, num_return_sequences=3)
        
        # Display the results
        print("\nGenerated Texts:")
        for i, text in enumerate(results):
            print(f"\n[{i+1}]: {text}")
