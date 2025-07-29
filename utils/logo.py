import logomaker
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from collections import Counter

aas = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

blosum62_background = {
    'A': 0.0755236,
    'R': 0.0515842,
    'N': 0.0453131,
    'D': 0.0530344,
    'C': 0.0169811,
    'Q': 0.0402483,
    'E': 0.0632002,
    'G': 0.0684442,
    'H': 0.0224067,
    'I': 0.0573156,
    'L': 0.0934327,
    'K': 0.0594192,
    'M': 0.0235696,
    'F': 0.0407819,
    'P': 0.0492775,
    'S': 0.0722465,
    'T': 0.0574747,
    'W': 0.0125173,
    'Y': 0.0319968,
    'V': 0.0652477
}

blosum_matrix = pd.DataFrame([
    [0.2901, 0.0310, 0.0256, 0.0297, 0.0216, 0.0256, 0.0405, 0.0783, 0.0148, 0.0432, 0.0594, 0.0445, 0.0175, 0.0216,
     0.0297, 0.0850, 0.0499, 0.0054, 0.0175, 0.0688],
    [0.0446, 0.3450, 0.0388, 0.0310, 0.0078, 0.0484, 0.0523, 0.0329, 0.0233, 0.0233, 0.0465, 0.1202, 0.0155, 0.0174,
     0.0194, 0.0446, 0.0349, 0.0058, 0.0174, 0.0310],
    [0.0427, 0.0449, 0.3169, 0.0831, 0.0090, 0.0337, 0.0494, 0.0652, 0.0315, 0.0225, 0.0315, 0.0539, 0.0112, 0.0180,
     0.0202, 0.0697, 0.0494, 0.0045, 0.0157, 0.0270],
    [0.0410, 0.0299, 0.0690, 0.3974, 0.0075, 0.0299, 0.0914, 0.0466, 0.0187, 0.0224, 0.0280, 0.0448, 0.0093, 0.0149,
     0.0224, 0.0522, 0.0354, 0.0037, 0.0112, 0.0243],
    [0.0650, 0.0163, 0.0163, 0.0163, 0.4837, 0.0122, 0.0163, 0.0325, 0.0081, 0.0447, 0.0650, 0.0203, 0.0163, 0.0203,
     0.0163, 0.0407, 0.0366, 0.0041, 0.0122, 0.0569],
    [0.0559, 0.0735, 0.0441, 0.0471, 0.0088, 0.2147, 0.1029, 0.0412, 0.0294, 0.0265, 0.0471, 0.0912, 0.0206, 0.0147,
     0.0235, 0.0559, 0.0412, 0.0059, 0.0206, 0.0353],
    [0.0552, 0.0497, 0.0405, 0.0902, 0.0074, 0.0645, 0.2965, 0.0350, 0.0258, 0.0221, 0.0368, 0.0755, 0.0129, 0.0166,
     0.0258, 0.0552, 0.0368, 0.0055, 0.0166, 0.0313],
    [0.0783, 0.0229, 0.0391, 0.0337, 0.0108, 0.0189, 0.0256, 0.5101, 0.0135, 0.0189, 0.0283, 0.0337, 0.0094, 0.0162,
     0.0189, 0.0513, 0.0297, 0.0054, 0.0108, 0.0243],
    [0.0420, 0.0458, 0.0534, 0.0382, 0.0076, 0.0382, 0.0534, 0.0382, 0.3550, 0.0229, 0.0382, 0.0458, 0.0153, 0.0305,
     0.0191, 0.0420, 0.0267, 0.0076, 0.0573, 0.0229],
    [0.0471, 0.0177, 0.0147, 0.0177, 0.0162, 0.0133, 0.0177, 0.0206, 0.0088, 0.2710, 0.1679, 0.0236, 0.0368, 0.0442,
     0.0147, 0.0250, 0.0398, 0.0059, 0.0206, 0.1767],
    [0.0445, 0.0243, 0.0142, 0.0152, 0.0162, 0.0162, 0.0202, 0.0213, 0.0101, 0.1154, 0.3755, 0.0253, 0.0496, 0.0547,
     0.0142, 0.0243, 0.0334, 0.0071, 0.0223, 0.0962],
    [0.0570, 0.1071, 0.0415, 0.0415, 0.0086, 0.0535, 0.0708, 0.0432, 0.0207, 0.0276, 0.0432, 0.2781, 0.0155, 0.0155,
     0.0276, 0.0535, 0.0397, 0.0052, 0.0173, 0.0328],
    [0.0522, 0.0321, 0.0201, 0.0201, 0.0161, 0.0281, 0.0281, 0.0281, 0.0161, 0.1004, 0.1968, 0.0361, 0.1606, 0.0482,
     0.0161, 0.0361, 0.0402, 0.0080, 0.0241, 0.0924],
    [0.0338, 0.0190, 0.0169, 0.0169, 0.0106, 0.0106, 0.0190, 0.0254, 0.0169, 0.0634, 0.1142, 0.0190, 0.0254, 0.3869,
     0.0106, 0.0254, 0.0254, 0.0169, 0.0888, 0.0550],
    [0.0568, 0.0258, 0.0233, 0.0310, 0.0103, 0.0207, 0.0362, 0.0362, 0.0129, 0.0258, 0.0362, 0.0413, 0.0103, 0.0129,
     0.4935, 0.0439, 0.0362, 0.0026, 0.0129, 0.0310],
    [0.1099, 0.0401, 0.0541, 0.0489, 0.0175, 0.0332, 0.0524, 0.0663, 0.0192, 0.0297, 0.0419, 0.0541, 0.0157, 0.0209,
     0.0297, 0.2199, 0.0820, 0.0052, 0.0175, 0.0419],
    [0.0730, 0.0355, 0.0434, 0.0375, 0.0178, 0.0276, 0.0394, 0.0434, 0.0138, 0.0533, 0.0651, 0.0454, 0.0197, 0.0237,
     0.0276, 0.0927, 0.2465, 0.0059, 0.0178, 0.0710],
    [0.0303, 0.0227, 0.0152, 0.0152, 0.0076, 0.0152, 0.0227, 0.0303, 0.0152, 0.0303, 0.0530, 0.0227, 0.0152, 0.0606,
     0.0076, 0.0227, 0.0227, 0.4924, 0.0682, 0.0303],
    [0.0405, 0.0280, 0.0218, 0.0187, 0.0093, 0.0218, 0.0280, 0.0249, 0.0467, 0.0436, 0.0685, 0.0312, 0.0187, 0.1308,
     0.0156, 0.0312, 0.0280, 0.0280, 0.3178, 0.0467],
    [0.0700, 0.0219, 0.0165, 0.0178, 0.0192, 0.0165, 0.0233, 0.0247, 0.0082, 0.1646, 0.1303, 0.0261, 0.0316, 0.0357,
     0.0165, 0.0329, 0.0494, 0.0055, 0.0206, 0.2689]
], columns=aas)

color_map = {
    'A': '#000000',
    'R': '#0000FF',
    'N': '#00D900',
    'D': '#E60000',
    'C': '#000000',
    'Q': '#00D900',
    'E': '#E60000',
    'G': '#00D900',
    'H': '#0000FF',
    'I': '#000000',
    'L': '#000000',
    'K': '#0000FF',
    'M': '#000000',
    'F': '#000000',
    'P': '#000000',
    'S': '#00D900',
    'T': '#00D900',
    'W': '#000000',
    'Y': '#00D900',
    'V': '#000000'
}

def sequence_identity(seq1, seq2):
    """Compute fraction identity between two same-length sequences."""
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / 9

def hobohm1_clustering(sequences, threshold=0.63):
    """Apply Hobohm I clustering to reduce sequence redundancy."""
    unique_seqs = []
    seq_id_map = {}
    for seq in sequences:
        is_redundant = False
        for rep in unique_seqs:
            identity = sequence_identity(seq, rep)
            if identity >= threshold:
                is_redundant = True
                seq_id_map[seq] = seq_id_map[rep]
                break
        if not is_redundant:
            unique_seqs.append(seq)
            seq_id_map[seq] = len(seq_id_map)
    id_counts = Counter(seq_id_map.values())
    seq_weights = [1 / id_counts[id_] for seq, id_ in seq_id_map.items()]
    return unique_seqs, seq_weights

def apply_blosum_pseudocounts(freq_df, blosum_matrix, alpha, beta):
    pseudo_df = pd.DataFrame(index=freq_df.index, columns=freq_df.columns)

    for pos in freq_df.index:
        c = freq_df.loc[pos].fillna(0).values
        smoothed = (alpha * c + beta * np.dot(blosum_matrix.values, c)) / (alpha + beta)
        pseudo_df.loc[pos] = smoothed

    return pseudo_df.astype(float)

def draw_logo(sequences, aa_len=9, figure_name=None):
    filtered_sequences = [s for s in sequences if len(s) == aa_len]
    if len(filtered_sequences) == 0:
        return

    # Shannon
    # df = logomaker.alignment_to_matrix(sequences=filtered_sequences, to_type='information')

    # Hobohm1 + KL + depletion
    unique_seqs, seq_weights = hobohm1_clustering(filtered_sequences)
    count_df = pd.DataFrame(np.zeros((aa_len, len(aas))), index=range(aa_len), columns = aas)
    for seq, weight in zip(filtered_sequences, seq_weights):
        for pos, aa in enumerate(seq):
            count_df.loc[pos, aa] += weight

    alpha = len(unique_seqs) - 1
    beta = 200
    freq_df = count_df.div(count_df.sum(axis=1), axis=0)
    smooth_count_df = apply_blosum_pseudocounts(freq_df, blosum_matrix.T, alpha=alpha, beta=beta)
    prob_df = smooth_count_df.div(smooth_count_df.sum(axis=1), axis=0)

    epsilon = 1e-9
    prob_df_clipped = prob_df.clip(lower=epsilon)

    kld_df = pd.DataFrame(index=prob_df.index, columns=prob_df.columns)
    for aa in prob_df.columns:
        p = prob_df_clipped[aa]
        q = max(blosum62_background.get(aa, epsilon), epsilon)
        kld_df[aa] = p * np.log2(p / q)
    df = (prob_df_clipped * np.sign(kld_df)).mul(kld_df.sum(axis=1), axis=0)
    vsep = (df.max().max() - df.min().min()) * 0.003

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    logomaker.Logo(df, ax=ax, font_name='DejaVu Sans Mono', color_scheme=color_map, flip_below=False, vsep=vsep)
    ax.set_xticks(range(df.shape[0]))
    ax.set_xticklabels((range(1, df.shape[0] + 1)))
    ax.set_ylabel("Bits")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False)

    plt.tight_layout()
    if figure_name:
        output_folder = Path(__file__).parent.parent / 'output' / 'figures'
        output_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_folder / figure_name)
        plt.clf()
    else:
        plt.show()

if __name__ == '__main__':
    sequences = pd.read_csv('../output/sample_peptide_data.txt', header=None).values[:, 0].tolist()
    draw_logo(sequences, aa_len=9)