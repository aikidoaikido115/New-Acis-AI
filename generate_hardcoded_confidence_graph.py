import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    # Hardcoded values from the latest evaluation summary
    avg_confidence = {
        "ALLERGY_WARN": 93.47,
        "SAFE": 88.37,
        "MANUAL_REVIEW": 75.61,
    }

    labels = list(avg_confidence.keys())
    values = list(avg_confidence.values())
    colors = ["#d9534f", "#5cb85c", "#f0ad4e"]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor="black", linewidth=1)

    plt.ylim(0, 100)
    plt.ylabel("Average Confidence (%)")
    plt.title("Average Confidence by Status")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        plt.text(x, y + 1, f"{value:.2f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    output_file = "hardcoded_avg_confidence_by_status.png"
    plt.savefig(output_file, dpi=200)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
