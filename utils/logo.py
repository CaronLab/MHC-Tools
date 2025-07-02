import logomaker
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt


def draw_logo(sequences, aa_len=9, figure_name=None):
    filtered_sequences = [s for s in sequences if len(s) == aa_len]
    if len(filtered_sequences) == 0:
        return

    # Create logo from peptides
    df = logomaker.alignment_to_matrix(sequences=filtered_sequences, to_type='information')

    # Set figure size
    fig, ax = plt.subplots(figsize=(4, 3))

    # Create logo and get Axes
    logomaker.Logo(df, ax=ax, font_name='DejaVu Sans Mono', color_scheme='chemistry')

    # Set x-axis to start at position 1
    ax.set_xticks(range(df.shape[0]))
    ax.set_xticklabels((range(1, df.shape[0] + 1)))

    # Axis labels
    ax.set_xlabel("Position")
    ax.set_ylabel("Information (bits)")

    plt.tight_layout()
    if figure_name:
        output_folder = Path(__file__).parent.parent / 'output' / 'figures'
        output_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_folder / figure_name)
        plt.clf()
    else:
        plt.show()

if __name__ == '__main__':
    sequences = pd.read_csv('../output/MixMHCpred_non_binders.csv', header=None).values[:, 0].tolist()
    draw_logo(sequences, aa_len=9)