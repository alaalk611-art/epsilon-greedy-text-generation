from transformers import GPT2LMHeadModel, GPT2Tokenizer
from epsilon_greedy_search import epsilon_greedy_search

# Load model and tokenizer globally to avoid repeated loading
print("Loading GPT-2 model...")
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure pad token ID is set
model = GPT2LMHeadModel.from_pretrained(model_name)
print("Model loaded.")

def generate_text(prompt, max_length=50, num_return_sequences=1, strategy="epsilon_greedy", epsilon=0.8, k=5, top_p=0.9, num_beams=3):
    """
    Generate text using the specified decoding strategy.
    Args:
        prompt: Input text prompt.
        max_length: Maximum length of the generated text.
        num_return_sequences: Number of sequences to generate.
        strategy: Decoding strategy to use ("epsilon_greedy", "greedy", "nucleus", "beam").
        epsilon: Probability for greedy selection in epsilon greedy search.
        k: Number of top tokens to consider in epsilon greedy search.
        top_p: Nucleus sampling parameter (p-value).
        num_beams: Number of beams for beam search.
    Returns:
        List of generated text sequences.
    """
    # Encode the prompt and create an attention mask
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = input_ids != tokenizer.pad_token_id  # Create attention mask

    if strategy == "epsilon_greedy":
        # Epsilon-greedy search
        outputs = []
        for _ in range(num_return_sequences):
            generated = epsilon_greedy_search(model, tokenizer, input_ids, max_length, epsilon, k)
            outputs.append(generated)
    elif strategy == "greedy":
        # Greedy search
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=False
        )
    elif strategy == "nucleus":
        # Nucleus sampling
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_p=top_p
        )
    elif strategy == "beam":
        # Beam search
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            early_stopping=True
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    # Decode generated sequences to text
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
