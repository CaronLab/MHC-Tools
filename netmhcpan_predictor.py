import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from psims.test.mzid_data import peptides

from netmhcpan_helper import NetMHCpanHelper, create_netmhcpan_peptide_index

MIN_SEQ_LEN = 8
MAX_SEQ_LEN = 12

def run_netmhcpan():
    alleles = pd.read_csv('./data/alleles.csv', header=None).values[:, 0].tolist()
    alleles = [allele.replace('_', '-') for allele in alleles]
    peptides = pd.read_csv('./data/peptides.csv', header=None).values[:, 0].tolist()
    peptides_len = len(peptides)

    netmhcpan = NetMHCpanHelper(alleles=alleles, n_threads=os.cpu_count() // 2)
    netmhcpan.min_length = MIN_SEQ_LEN
    netmhcpan.max_length = MAX_SEQ_LEN
    netmhcpan.add_peptides(peptides, MIN_SEQ_LEN, MAX_SEQ_LEN)
    netmhcpan.netmhcpan_peptides = create_netmhcpan_peptide_index(netmhcpan.peptides)
    netmhcpan.predictions = {x: {} for x in netmhcpan.peptides}
    print('Input file prepared.')

    if len(netmhcpan.netmhcpan_peptides) != peptides_len:
        print(f'Peptides length out of [{MIN_SEQ_LEN}, {MAX_SEQ_LEN}] are filtered out. {len(netmhcpan.netmhcpan_peptides)} peptides left. {peptides_len - len(netmhcpan.netmhcpan_peptides)} peptides are ignored.')
    pred_df = netmhcpan.predict_df()
    assert len(pred_df) / len(alleles) == len(netmhcpan.netmhcpan_peptides)
    pred_df.to_csv('./data/netmhcpan_output.csv', index=False)

def analyze_binders():
    results = pd.read_csv('./data/netmhcpan_output.csv')

    stat = pd.DataFrame(columns=['Strong Binders', 'Weak Binders', 'None Binders', 'Strong Ratio', 'Weak Ratio', 'None Ratio'])
    for allele in results['Allele'].drop_duplicates():
        ba = results[results['Allele'] == allele]
        n_strong = np.sum(ba['Binder'] == 'Strong')
        n_weak = np.sum(ba['Binder'] == 'Weak')
        n_none = np.sum(ba['Binder'] == 'Non-binder')
        r_strong = '{:.1f} %'.format(n_strong / len(ba) * 100)
        r_weak = '{:.1f} %'.format(n_weak / len(ba) * 100)
        r_none = '{:.1f} %'.format(n_none / len(ba) * 100)
        stat.loc[allele] = [n_strong, n_weak, n_none, r_strong, r_weak, r_none]
    stat.to_csv('./data/netmhcpan_output_stat.csv', index=True)
    print(stat.to_string())

    total_peptides = len(results['Peptide'].drop_duplicates())
    print(f'Total peptides: {total_peptides}')

    strong_binders = results['Peptide'].loc[results['Binder'] == 'Strong']
    strong_binders_deduplicate = strong_binders.drop_duplicates()
    strong_binders_deduplicate.to_csv('./data/netmhcpan_output_sb.csv', index=False, header=False)
    print(f'Strong binders (peptide): {len(strong_binders_deduplicate)}')

    results = results.loc[-results['Peptide'].isin(strong_binders_deduplicate)]
    weak_binders = results['Peptide'].loc[results['Binder'] == 'Weak']
    weak_binders_deduplicate = weak_binders.drop_duplicates()
    weak_binders_deduplicate.to_csv('./data/netmhcpan_output_wb.csv', index=False, header=False)
    print(f'Weak binders (peptide): {len(weak_binders_deduplicate)}')

    results = results.loc[-results['Peptide'].isin(weak_binders_deduplicate)]
    non_binders = results['Peptide'].loc[results['Binder'] == 'Non-binder']
    non_binders_deduplicate = non_binders.drop_duplicates()
    non_binders_deduplicate.to_csv('./data/netmhcpan_output_nb.csv', index=False, header=False)
    print(f'Non-binders (peptide): {len(non_binders_deduplicate)}')


def draw_binding_affinity():
    results = pd.read_csv('./data/netmhcpan_output.csv')
    alleles = results['Allele'].drop_duplicates()
    peptides = results['Peptide'].drop_duplicates()
    df = pd.DataFrame(index=peptides, columns=alleles)
    for i in range(len(results)):
        row = results.iloc[i]
        df.loc[row['Peptide'], row['Allele']] = row['Aff_nM']
    data = df.to_numpy().astype(np.float64)

    max_bas = np.min(data, axis=1)
    max_cols = np.argmin(data, axis=1)
    strong_peptides = results['Peptide'].loc[results['Binder'] == 'Strong'].drop_duplicates().values
    strong_indices = np.array([peptide in strong_peptides for peptide in peptides])
    result = []
    n_strong = [0]
    for i in range(len(alleles)):
        strong_idx = strong_indices * (max_cols == i)
        result += data[strong_idx].tolist()
        n_strong.append(np.sum(strong_idx) + n_strong[-1])
    result += data[~strong_indices].tolist()
    n_strong.append(len(peptides))
    ax = sns.heatmap(result, cmap='viridis_r', vmin=0, vmax=1000, xticklabels=df.columns)
    ax.set_yticks([0] + (np.array(n_strong[1:]) - 1).tolist())
    ax.set_yticklabels(n_strong)
    plt.show()


if __name__ == '__main__':
    run_netmhcpan()
    analyze_binders()
    draw_binding_affinity()
    print('Have a nice day.')
