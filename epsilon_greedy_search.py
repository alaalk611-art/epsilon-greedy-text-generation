import random
import torch

def epsilon_greedy_search(model, tokenizer, input_ids, max_length, epsilon, k):
    """
    Epsilon Greedy Search for text generation.
    Args:
        model: Pretrained language model.
        tokenizer: Corresponding tokenizer.
        input_ids: Tokenized prompt input.
        max_length: Maximum sequence length.
        epsilon: Probability for greedy selection.
        k: Number of top tokens to consider during sampling.
    Returns:
        Generated sequence as a string.
    """
    generated = input_ids
    for _ in range(max_length):
        outputs = model(input_ids=generated)
        logits = outputs.logits[:, -1, :]  # Get logits of the last token
        probabilities = torch.softmax(logits, dim=-1)

        if random.random() < epsilon:
            # Greedy selection
            next_token = torch.argmax(probabilities, dim=-1)
        else:
            # Sample from top-k tokens
            top_k_probs, top_k_indices = torch.topk(probabilities, k)
            top_k_probs = top_k_probs / torch.sum(top_k_probs)  # Normalize probabilities
            next_token = random.choices(top_k_indices[0].tolist(), weights=top_k_probs[0].tolist())[0]

        next_token = torch.tensor([[next_token]]).to(input_ids.device)
        generated = torch.cat((generated, next_token), dim=1)

        # Stop if end-of-sequence token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)
