import numpy as np
import pandas as pd
import os
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

# Set global dash style
plt.rcParams['lines.dashed_pattern'] = [2, 6]  # Adjust the values as needed
plt.rcParams['lines.scale_dashes'] = False
# Set global line width
plt.rcParams['lines.linewidth'] = 1.5  # Adjust the value as needed

plotcolours = plt.get_cmap("Set1")
rule_color_map = {"BP":"black",
                  "NP Bernoulli": plotcolours(0),
                  "NP Simulated": plotcolours(2),
                  "NP Measured": plotcolours(1),
                  }
rule_label_map = {"BP":"BP",
                  "NP Bernoulli": "DANP (Bernoulli noise)",
                  "NP Simulated": "DANP (Simulated sMTJ noise for each node)",
                  "NP Measured": "DANP (Real-world single sMTJ noise)",
                  }

folder_path = None
output_path = None


def plot_with_fill_legacy(ax, df, accloss, testtrain, color, ls, alpha=0.5, **kwargs):
    ax.plot(df["Grouped runs - " + testtrain + "/" + accloss], color=color, linestyle=ls, **kwargs)
    ax.fill_between(df["Step"], df["Grouped runs - " + testtrain + "/" + accloss + "__MIN"], df["Grouped runs - " + testtrain + "/" + accloss + "__MAX"], alpha=alpha, color=color, linewidth=0.0)
    return

def plot_with_fill(ax, df, learning_rule, testtrain, color, ls, alpha=0.5, **kwargs):
    ax.plot(df[f"Grouped runs ({learning_rule}) - " + testtrain + "/acc"],
            color=color, linestyle=ls, **kwargs)
    ax.fill_between(df["Step"],
                    df[f"Grouped runs ({learning_rule}) - " + testtrain + "/acc" + "__MIN"],
                    df[f"Grouped runs ({learning_rule}) - " + testtrain + "/acc" + "__MAX"],
                    alpha=alpha, color=color, linewidth=0.0)
    return

def make_legend(ax, ignore_ber=True):
    ax.axis('off')  # Turn off the axis

    # Create custom legend handles with filled rectangles
    legend_elements_color = []
    for rule, color in rule_color_map.items():
        if rule == "NP Bernoulli" and ignore_ber: continue
        legend_elements_color += [
            Patch(facecolor=color, edgecolor='none', label=rule_label_map[rule])
        ]

    legend_elements_linestyle = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Train'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Test', alpha=0.25)
    ]

    # Create the legend
    first_legend = ax.legend(handles=legend_elements_color, loc='upper left', bbox_to_anchor=(0, 1))
    ax.add_artist(first_legend)
    second_legend = ax.legend(handles=legend_elements_linestyle, loc='upper left', bbox_to_anchor=(0, 0.6))
    ax.add_artist(second_legend)
    
    return first_legend, second_legend

def plot_experiment(dataset, ax, ignore_ber=True):
    csv_data = pd.read_csv(folder_path + dataset + '.csv')

    # set title
    ax.set_title(dataset, fontweight='bold')
    # set axis labels
    ax.set_xlabel("Epoch", fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontweight='bold')

    if dataset == 'MNIST':
        color = rule_color_map["BP"]
        old_BP_data = pd.read_csv(folder_path + "BP_old/" 'MNIST_acc.csv')
        plot_with_fill_legacy(ax, old_BP_data, "acc", "train", color, "-")
        plot_with_fill_legacy(ax, old_BP_data, "acc", "test", color, "--", alpha=0.25)

    for rule, color in rule_color_map.items():
        if rule == "BP" and dataset == 'MNIST': continue
        if rule == "NP Bernoulli" and ignore_ber: continue
        plot_with_fill(ax, csv_data, rule, "train", color, "-")
        plot_with_fill(ax, csv_data, rule, "test", color, "--", alpha=0.25)

    if dataset == 'MNIST': ax.set_ylim([90,101])


save_file = True
if __name__ == "__main__":
    assert folder_path is not None, "results data folder path not set"
    fig, axs = plt.subplots(1, 4, dpi=10, figsize=(17.5, 3.5))

    plot_experiment("MNIST", axs[0])
    plot_experiment("CIFAR10", axs[1])
    plot_experiment("CIFAR100", axs[2])
    make_legend(axs[3])

    plt.tight_layout()
    if save_file:
        assert output_path is not None, "Output path not set"
        # save to file and exit
        os.makedirs(output_path, exist_ok=True)
        output_file = output_path +"MTJ_accuracy_plot.pdf"
        plt.savefig(output_file)
        exit()
    plt.show()
    exit()