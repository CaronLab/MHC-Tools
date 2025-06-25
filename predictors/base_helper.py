

import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from matplotlib import pyplot as plt

class BaseHelper:
    def __init__(self, tool_name):
        self.tool_name = tool_name
        self.output_folder = Path(__file__).parent.parent / 'output'
        self.pred_df = None

    def predict_df(self):
        pass

    def save(self):
        output_path = self.output_folder / f'{self.tool_name}_output.csv'
        self.pred_df.to_csv(output_path, index=False)
        
    def analyze_binders(self):
        # Peptide, Allele, Binder
        stat = pd.DataFrame(
            columns=['Strong Binders', 'Weak Binders', 'None Binders', 'Strong Ratio', 'Weak Ratio', 'None Ratio'])
        for allele in self.pred_df['Allele'].drop_duplicates().sort_values():
            ba = self.pred_df[self.pred_df['Allele'] == allele]
            n_strong = np.sum(ba['Binder'] == 'Strong')
            n_weak = np.sum(ba['Binder'] == 'Weak')
            n_none = np.sum(ba['Binder'] == 'Non-binder')
            r_strong = '{:.1f} %'.format(n_strong / len(ba) * 100)
            r_weak = '{:.1f} %'.format(n_weak / len(ba) * 100)
            r_none = '{:.1f} %'.format(n_none / len(ba) * 100)
            stat.loc[allele] = [n_strong, n_weak, n_none, r_strong, r_weak, r_none]
        stat.to_csv(f'./output/{self.tool_name}_output_stat.csv', index=True)
        print(stat.to_string())

        total_peptides = len(self.pred_df['Peptide'].drop_duplicates())
        print(f'Total peptides: {total_peptides}')

        strong_binders = self.pred_df['Peptide'].loc[self.pred_df['Binder'] == 'Strong']
        strong_binders_deduplicate = strong_binders.drop_duplicates()
        strong_binders_deduplicate.to_csv(f'./output/{self.tool_name}_strong_binders.csv', index=False, header=False)
        print(f'Strong binders (peptide): {len(strong_binders_deduplicate)}')

        results = self.pred_df.loc[-self.pred_df['Peptide'].isin(strong_binders_deduplicate)]
        weak_binders = results['Peptide'].loc[results['Binder'] == 'Weak']
        weak_binders_deduplicate = weak_binders.drop_duplicates()
        weak_binders_deduplicate.to_csv(f'./output/{self.tool_name}_weak_binders.csv', index=False, header=False)
        print(f'Weak binders (peptide): {len(weak_binders_deduplicate)}')

        results = results.loc[-results['Peptide'].isin(weak_binders_deduplicate)]
        non_binders = results['Peptide'].loc[results['Binder'] == 'Non-binder']
        non_binders_deduplicate = non_binders.drop_duplicates()
        non_binders_deduplicate.to_csv(f'./output/{self.tool_name}_non_binders.csv', index=False, header=False)
        print(f'Non-binders (peptide): {len(non_binders_deduplicate)}')

    def draw_binding_affinity(self):
        # Peptide, Allele, Binder, EL_Rank
        alleles = self.pred_df['Allele'].drop_duplicates().sort_values()
        peptides = self.pred_df['Peptide'].drop_duplicates()
        df = pd.DataFrame(index=peptides, columns=alleles)
        for i in range(len(self.pred_df)):
            row = self.pred_df.iloc[i]
            df.loc[row['Peptide'], row['Allele']] = row['EL_Rank']
        data = df.to_numpy().astype(np.float64)

        max_cols = np.argmin(data, axis=1)
        strong_peptides = self.pred_df['Peptide'].loc[self.pred_df['Binder'] == 'Strong'].drop_duplicates().values
        strong_indices = np.array([peptide in strong_peptides for peptide in peptides])
        result = []
        n_strong = [0]
        for i in range(len(alleles)):
            strong_idx = strong_indices * (max_cols == i)
            result += data[strong_idx].tolist()
            n_strong.append(np.sum(strong_idx) + n_strong[-1])
        result += data[~strong_indices].tolist()
        n_strong.append(len(peptides))
        ax = sns.heatmap(result, cmap='viridis_r', vmin=0, vmax=10, xticklabels=df.columns)
        ax.set_yticks([0] + (np.array(n_strong[1:]) - 1).tolist())
        ax.set_yticklabels(n_strong)
        plt.tight_layout()
        plt.show()