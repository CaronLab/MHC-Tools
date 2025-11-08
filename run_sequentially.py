
import pandas as pd

from predictors.netmhcpan_helper import NetMHCpanHelper
from predictors.mixmhcpred_helper import MixMhcPredHelper
from predictors.mhcflurry_helper import MhcFlurryHelper
from predictors.bigmhc_helper import BigMhcHelper
from utils.peptide import filter_peptides_by_length
from utils.logo import draw_logo



def run(tools=['NetMHCpan', 'MHCflurry'], min_seq_length=8, max_seq_length=15, peptide_file='peptides.csv'):
    if tools is None or len(tools) == 0:
        print('No tools specified.')
        return None
    mhc_class = 'I' if tools[0] in ['NetMHCpan', 'MixMHCpred', 'BigMHC'] else 'II'
    alleles = pd.read_csv(f'input/alleles-{mhc_class}.csv', header=None).values[:, 0].tolist()
    alleles = [allele.replace('_', '-') for allele in alleles]
    peptides = pd.read_csv('./input/' + peptide_file, header=None).values[:, 0].tolist()
    peptides = filter_peptides_by_length(peptides, min_seq_length, max_seq_length)

    result_df = None
    for tool in tools:
        if tool == 'NetMHCpan':
            pred_df = NetMHCpanHelper(peptides, alleles, mhc_class=mhc_class).predict_df()
        elif tool == 'NetMHCIIpan':
            pred_df = NetMHCpanHelper(peptides, alleles, mhc_class=mhc_class).predict_df()
        elif tool == 'MixMHCpred':
            pred_df = MixMhcPredHelper(peptides, alleles, mhc_class=mhc_class).predict_df()
        elif tool == 'MixMHC2pred':
            pred_df = MixMhcPredHelper(peptides, alleles, mhc_class=mhc_class).predict_df()
        elif tool == 'MHCflurry':
            pred_df = MhcFlurryHelper(peptides, alleles).predict_df()
        elif tool == 'BigMHC':
            pred_df = BigMhcHelper(peptides, alleles).predict_df()
        else:
            print(f'Tool {tool} is not supported.')
            continue
        # Group by peptide and find min EL_Rank rows
        min_rank_idx = pred_df.groupby('Peptide')['EL_Rank'].idxmin()
        best_allele_df = pred_df.loc[min_rank_idx].reset_index(drop=True)

        # Store all alleles per peptide
        all_binding_alleles = pred_df[pred_df['Binder'].isin(['Strong', 'Weak'])].groupby('Peptide')['Allele'].agg(
            lambda x: ','.join(x)).reset_index()
        all_binding_alleles.rename(columns={'Allele': 'Alleles'}, inplace=True)
        best_allele_df.rename(columns={'Allele': 'Best_Allele'}, inplace=True)
        best_allele_df = best_allele_df.merge(all_binding_alleles, on='Peptide', how='left')
        if mhc_class == 'I':
            best_allele_df = best_allele_df[['Peptide', 'Best_Allele', 'EL_Rank', 'Binder', 'Alleles']]
        else:
            best_allele_df = best_allele_df[['Peptide', 'Best_Allele', 'Core', 'EL_Rank', 'Binder', 'Alleles']]

        if result_df is None:
            result_df = best_allele_df
            result_df['Tool'] = tool
            continue

        result_nonbinder_mask = ~result_df['Binder'].isin(['Strong', 'Weak'])
        pred_binder_mask = best_allele_df['Peptide'].isin(result_df[result_nonbinder_mask]['Peptide']) & best_allele_df['Binder'].isin(['Strong', 'Weak'])
        if pred_binder_mask.any():
            result_binder_mask = result_df['Peptide'].isin(best_allele_df[pred_binder_mask]['Peptide'].unique())
            result_df.loc[result_binder_mask, result_df.columns[:-1]] = best_allele_df[pred_binder_mask].values
            result_df.loc[result_binder_mask, 'Tool'] = tool
    result_df.sort_values(by='Peptide', inplace=True)
    tools_str = '_'.join(tools)
    result_df.to_csv(f'./output/sequentially_{tools_str}.csv', index=False)
    return result_df

def draw_best_allele_logo(result_df, aa_len=9):
    print(f'Generating motif logos.')
    alleles = result_df['Best_Allele'].unique().tolist()
    strong_df = result_df[result_df['Binder'] == 'Strong']
    weak_df = result_df[result_df['Binder'] == 'Weak']
    non_binder_df = result_df[~result_df['Binder'].isin(['Strong', 'Weak'])]

    for allele in alleles:
        strong_peptides = strong_df[strong_df['Best_Allele'] == allele]['Peptide'].unique().tolist()
        weak_peptides = weak_df[weak_df['Best_Allele'] == allele]['Peptide'].unique().tolist()
        draw_logo(strong_peptides, aa_len=aa_len, figure_name=f'{allele}_strong({len(strong_peptides)}).png')
        draw_logo(weak_peptides, aa_len=aa_len, figure_name=f'{allele}_weak({len(weak_peptides)}).png')
        binder_peptides = strong_peptides + weak_peptides
        draw_logo(binder_peptides, aa_len=aa_len, figure_name=f'{allele}_binder({len(binder_peptides)}).png')

    non_binder_peptides = non_binder_df['Peptide'].unique().tolist()
    draw_logo(non_binder_peptides, aa_len=aa_len, figure_name=f'non_binder({len(non_binder_peptides)}).png')

if __name__ == '__main__':
    # NetMHCpan NetMHCIIpan MixMHCpred MixMHC2pred MHCflurry BigMHC
    tools = ['NetMHCpan']
    result_df = run(tools=tools, min_seq_length=8, max_seq_length=14, peptide_file='intersection_1.txt')
    # tools = ['NetMHCIIpan', 'MixMHC2pred']
    # result_df = run(tools=tools, min_seq_length=9, max_seq_length=25)
    # result_df = pd.read_csv(f'./output/sequentially_MHCflurry_NetMHCpan.csv')
    draw_best_allele_logo(result_df, aa_len=9)

    # peptides = pd.read_csv('./input/peptides.csv', header=None).values[:, 0].tolist()
    # draw_logo(peptides, aa_len=8, figure_name='all_peptides.png')
    print('Have a nice day.')