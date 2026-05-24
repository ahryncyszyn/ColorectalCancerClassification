import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob

def plot_all_models(log_dir):
    csv_files = glob.glob(os.path.join(log_dir, "*_metrics.csv"))
    if not csv_files:
        print(f"No CSV files found in {log_dir}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))

    for file in csv_files:
        model_name = os.path.basename(file).replace("_metrics.csv", "")
        df = pd.read_csv(file)
        
        ax1.plot(df['epoch'], df['train_loss'], marker = 'o', label = model_name)
        ax2.plot(df['epoch'], df['test_accuracy'], marker = 's', label = model_name)

    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Test Accuracy Comparison')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_path = os.path.join(log_dir, "model_comparison.png")
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()
    plot_all_models(args.log_dir)