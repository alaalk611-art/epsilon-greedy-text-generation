# Epsilon-Greedy Text Generation

Implementation and evaluation of an **epsilon-greedy decoding strategy for neural text generation** using large language models.

This project studies how controlled randomness can improve the **diversity and quality of generated text** compared to classical decoding strategies such as greedy decoding or typical sampling.

---

## Project Overview

Language models usually generate text using decoding strategies like greedy search or sampling methods.

However:

- **Greedy decoding** often produces repetitive text  
- **Sampling methods** increase diversity but may reduce coherence  

The **epsilon-greedy strategy** introduces a balance between exploration and exploitation:

- With probability **1 − ε**, the model selects the most probable token  
- With probability **ε**, it samples alternative tokens  

This mechanism allows exploration while preserving generation stability.

---

## Models Used

Experiments were conducted using the following large language models:
- **OPT-2.7B (Meta AI)**

These models were evaluated under different decoding strategies.

---

## Dataset

Experiments were performed using the **WikiText dataset**, widely used for language modeling benchmarks.

Generated outputs are evaluated against real text samples from the dataset.

---

## Evaluation Metrics

Several metrics are used to evaluate the quality of generated text:

- **Perplexity** – language modeling performance  
- **MAUVE score** – similarity between generated and real text distributions  
- **Diversity metrics** – lexical diversity of generated sequences  
- **Coherence evaluation** – semantic consistency of the output

---

