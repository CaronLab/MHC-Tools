import tempfile
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

STRONG_BA_THRESHOLD = 500
WEAK_BA_THRESHOLD = 1000

MIN_SEQ_LEN = 8
MAX_SEQ_LEN = 12

def run_mhcflurry():
    alleles = pd.read_csv('./input/alleles.csv', header=None).values[:, 0].tolist()
    peptides = pd.read_csv('./input/peptides.csv', header=None).values[:, 0].tolist()
    n_ignored_peptides = 0
    with tempfile.NamedTemporaryFile('w') as mhcflurry_input:
        mhcflurry_input.write('allele,peptide\n')
        for peptide in peptides:
            if len(peptide) < MIN_SEQ_LEN or len(peptide) > MAX_SEQ_LEN:
                n_ignored_peptides += 1
                continue
            for allele in alleles:
                mhcflurry_input.write(allele + ',' + peptide + '\n')
        print('Input file prepared.')
        if n_ignored_peptides != 0:
            print(f'Peptides length out of [{MIN_SEQ_LEN}, {MAX_SEQ_LEN}] are not filtered out. {n_ignored_peptides} peptides are ignored')

        subprocess.run(f'mhcflurry-predict {mhcflurry_input.name} --out ./output/mhcflurry_output.csv', shell=True)
        print('Output file generated.')


def analyze_binders():
    results = pd.read_csv('./output/mhcflurry_output.csv')
    results['Binder'] = ''
    results.loc[results['mhcflurry_affinity'] <= STRONG_BA_THRESHOLD, 'Binder'] = 'Strong'
    results.loc[(results['mhcflurry_affinity'] > STRONG_BA_THRESHOLD) * (results['mhcflurry_affinity'] <= WEAK_BA_THRESHOLD), 'Binder'] = 'Weak'
    results.loc[results['mhcflurry_affinity'] > WEAK_BA_THRESHOLD, 'Binder'] = 'Non-binder'
    results = results[results.columns[:2].append(results.columns[-1:]).append(results.columns[2:-1])]
    results.to_csv('./output/mhcflurry_output.csv', index=False)

    stat = pd.DataFrame(columns=['Strong Binders', 'Weak Binders', 'None Binders', 'Strong Ratio', 'Weak Ratio', 'None Ratio'])
    for allele in results['allele'].drop_duplicates():
        ba = results[results['allele'] == allele]
        n_strong = np.sum(ba['mhcflurry_affinity'] <= STRONG_BA_THRESHOLD)
        n_weak = np.sum((ba['mhcflurry_affinity'] > STRONG_BA_THRESHOLD) * (ba['mhcflurry_affinity'] <= WEAK_BA_THRESHOLD))
        n_none = np.sum(ba['mhcflurry_affinity'] > WEAK_BA_THRESHOLD)
        r_strong = '{:.1f} %'.format(n_strong / len(ba) * 100)
        r_weak = '{:.1f} %'.format(n_weak / len(ba) * 100)
        r_none = '{:.1f} %'.format(n_none / len(ba) * 100)
        stat.loc[allele] = [n_strong, n_weak, n_none, r_strong, r_weak, r_none]
    stat.to_csv('./output/mhcflurry_output_stat.csv', index=True)
    print(stat.to_string())

    total_peptides = len(results['peptide'].drop_duplicates())
    print(f'Total peptides: {total_peptides}')

    strong_binders = results['peptide'].loc[results['mhcflurry_affinity'] <= STRONG_BA_THRESHOLD]
    strong_binders_deduplicate = strong_binders.drop_duplicates()
    strong_binders_deduplicate.to_csv('./output/mhcflurry_output_sb.csv', index=False, header=False)
    print(f'Strong binders (peptide): {len(strong_binders_deduplicate)}')

    results = results.loc[-results['peptide'].isin(strong_binders_deduplicate)]
    weak_binders = results['peptide'].loc[results['mhcflurry_affinity'] <= WEAK_BA_THRESHOLD]
    weak_binders_deduplicate = weak_binders.drop_duplicates()
    weak_binders_deduplicate.to_csv('./output/mhcflurry_output_wb.csv', index=False, header=False)
    print(f'Weak binders (peptide): {len(weak_binders_deduplicate)}')

    results = results.loc[-results['peptide'].isin(weak_binders_deduplicate)]
    non_binders = results['peptide'].loc[results['mhcflurry_affinity'] > WEAK_BA_THRESHOLD]
    non_binders_deduplicate = non_binders.drop_duplicates()
    non_binders_deduplicate.to_csv('./output/mhcflurry_output_nb.csv', index=False, header=False)
    print(f'Non-binders (peptide): {len(non_binders_deduplicate)}')


def draw_binding_affinity():
    results = pd.read_csv('./output/mhcflurry_output.csv')
    alleles = results['allele'].drop_duplicates()
    peptides = results['peptide'].drop_duplicates()
    df = pd.DataFrame(index=peptides, columns=alleles)
    for i in range(len(results)):
        row = results.iloc[i]
        df.loc[row['peptide'], row['allele']] = row['mhcflurry_affinity']
    data = df.to_numpy().astype(np.float64)

    max_bas = np.min(data, axis=1)
    max_cols = np.argmin(data, axis=1)
    strong_indices = max_bas <= STRONG_BA_THRESHOLD
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
    run_mhcflurry()
    analyze_binders()
    draw_binding_affinity()
    print('Have a nice day.')
