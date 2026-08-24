import os
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import fetch_and_prepare_data


def generate_eda_plots():
    df = fetch_and_prepare_data()

    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../static')
    os.makedirs(static_dir, exist_ok=True)

    # Set aesthetic style
    sns.set_theme(style="whitegrid")

    # 1. Borough Distribution Plot
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x='borough', palette='viridis')
    plt.title('Requests by Borough', fontsize=12, fontweight='bold')
    plt.xlabel('Borough')
    plt.ylabel('Count')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'borough_distribution.png'), dpi=150)
    plt.close()

    # 2. Resolution Time Histogram
    plt.figure(figsize=(7, 4))
    sns.histplot(df['resolution_hours'], bins=30, kde=True, color='#0284c7')
    plt.title('Resolution Time Distribution (Hours)', fontsize=12, fontweight='bold')
    plt.xlabel('Hours')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'resolution_histogram.png'), dpi=150)
    plt.close()

    print("EDA charts generated successfully in static folder!")


if __name__ == "__main__":
    generate_eda_plots()