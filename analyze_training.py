from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re



def parse_log_file(log_file) -> None:
    """Parse training log file to extract metrics"""
        metrics = defaultdict(list)
        
        with open(log_file, "r") as f: forlinein, f:
        # Skip tqdm progress lines
        if "%|" in line: continueif"Validation loss:" in line: try: val_loss = float(line.split("Validation, loss:")[1].strip())
        metrics["val_loss"].append(val_loss)
        except(ValueError, IndexError):
        continue
        
        elif "Validation math accuracy:" in line: try: math_acc = float(line.split("Validation math, accuracy:")[1].strip())
        metrics["math_accuracy"].append(math_acc)
        except(ValueError, IndexError):
        continue
        
        elif "Training loss:" in line: try: train_loss = float(line.split("Training, loss:")[1].strip())
        metrics["train_loss"].append(train_loss)
        except(ValueError, IndexError):
        continue
        
        return metrics
        
        
                def plot_metrics(metrics, output_dir="outputs") -> None:
                    """Plot training and validation metrics"""
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use("seaborn")
        
        # Plot losses
        plt.figure(figsize=(12, 6))
        if metrics.get("train_loss"):
    plt.plot(metrics["train_loss"], label="Training Loss", marker="o", markersize=4)
    if metrics.get("val_loss"):
        plt.plot(metrics["val_loss"], label="Validation Loss", marker="s", markersize=4)
        plt.title("Training and Validation Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_plot.png"))
        plt.close()

        # Plot math accuracy
        if metrics.get("math_accuracy"):
            plt.figure(figsize=(12, 6))
            plt.plot(metrics["math_accuracy"], label="Math Accuracy", marker="o", markersize=4, color="green")
            plt.title("Mathematical Reasoning Accuracy")
            plt.xlabel("Evaluation Steps")
            plt.ylabel("Accuracy")
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.0)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "math_accuracy_plot.png"))
            plt.close()

            # Save metrics to JSON
            with open(os.path.join(output_dir, "training_metrics.json"), "w") as f: json.dump(metrics, f, indent=2)


def main(self):
    # Find most recent log file
    log_dir = "logs"
    log_files = [f for f in os.listdir(log_dir) if f.startswith("training_")]
    if not log_files: print("No training log files found")
        return

        latest_log = max(log_files, key=lambda x: os.path.getctime(os.path.join(log_dir, x))
        )
        log_path = os.path.join(log_dir, latest_log)

        print(f"Analyzing log file: {log_path}")
        metrics = parse_log_file(log_path)
        plot_metrics(metrics)

        # Print summary statistics
        print("\nTraining Summary:")
        if metrics["val_loss"]:
            print(f"Final validation loss: {metrics['val_loss'][-1]:.4f}")
            if metrics["math_accuracy"]:
                print(f"Final math accuracy: {metrics['math_accuracy'][-1]:.4f}")

                print("\nModel Performance Analysis:")
                if metrics["math_accuracy"]:
                    acc = np.array(metrics["math_accuracy"])
                    print(f"Average math accuracy: {np.mean(acc):.4f}")
                    print(f"Best math accuracy: {np.max(acc):.4f}")
                    print(f"Math accuracy std dev: {np.std(acc):.4f}")


                    if __name__ == "__main__":
                        main()
