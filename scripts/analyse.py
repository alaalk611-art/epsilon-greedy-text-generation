import matplotlib.pyplot as plt
import pandas as pd

# Updated dataset with new Epsilon Greedy values
data = {
    "Dataset": ["wikinews"] * 5 + ["wikitext"] * 5 + ["book"] * 5,
    "Method": [
        "Greedy search", "Top-k sampling", "Nucleus sampling", "Typical sampling", "Epsilon Greedy search",
        "Greedy search", "Top-k sampling", "Nucleus sampling", "Typical sampling", "Epsilon Greedy search",
        "Greedy search", "Top-k sampling", "Nucleus sampling", "Typical sampling", "Epsilon Greedy search"
    ],
    "% div ↑": [
        3.55, 91.56, 93.54, 95.37, 3.55,  # Wikinews
        1.77, 87.49, 92.16, 94.82, 1.77,  # Wikitext
        0.86, 91.22, 94.5, 96.29, 0.86   # Book
    ],
    "MAUVE(% ↑)": [
        13.96, 89.86, 89.45, 90.97, 13.96,  # Wikinews
        4.91, 81.0, 86.54, 86.07, 4.91,     # Wikitext
        2.65, 87.49, 91.47, 88.58, 2.65     # Book
    ],
    "coh ↑": [
        -0.47, -2.22, -2.61, -3.26, -0.4674,  # Wikinews
        -0.41, -2.37, -3.03, -3.71, -0.4069,  # Wikitext
        -0.3362, -2.45, -3.02, -3.68, -0.3362 # Book
    ]
}

df = pd.DataFrame(data)

# Generate updated visualizations for each metric and dataset
metrics = ["% div ↑", "MAUVE(% ↑)", "coh ↑"]
datasets = df["Dataset"].unique()

for metric in metrics:
    for dataset in datasets:
        subset = df[df["Dataset"] == dataset]

        plt.figure(figsize=(10, 6))
        colors = ['red' if method == 'Epsilon Greedy search' else 'gray' for method in subset["Method"]]
        plt.bar(subset["Method"], subset[metric], color=colors)
        plt.title(f"{metric} Comparison Across Methods for {dataset} (Epsilon Greedy Highlighted)")
        plt.xlabel("Method")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Annotate values on bars
        for index, value in enumerate(subset[metric]):
            plt.text(index, value + (0.5 if value > 0 else -0.5), f"{value:.2f}", ha='center', fontsize=9)

        plt.show()
