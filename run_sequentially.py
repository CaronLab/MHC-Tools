
import pandas as pd

from predictors.netmhcpan_helper import NetMHCpanHelper
from predictors.mixmhcpred_helper import MixMhcPredHelper
from predictors.mhcflurry_helper import MhcFlurryHelper
from predictors.bigmhc_helper import BigMhcHelper
from utils.peptide import filter_peptides_by_length



def run(tools=['NetMHCpan', 'MHCflurry'], min_seq_length=8, max_seq_length=15):

    alleles = pd.read_csv('./input/alleles.csv', header=None).values[:, 0].tolist()
    alleles = [allele.replace('_', '-') for allele in alleles]
    peptides = pd.read_csv('./input/peptides.csv', header=None).values[:, 0].tolist()
    peptides = filter_peptides_by_length(peptides, min_seq_length, max_seq_length)

    result_df = None
    for tool in tools:
        if tool == 'NetMHCpan':
            pred_df = NetMHCpanHelper(peptides, alleles, mhc_class='I').predict_df()
        elif tool == 'NetMHCIIpan':
            pred_df = NetMHCpanHelper(peptides, alleles, mhc_class='II').predict_df()
        elif tool == 'MixMHCpred':
            pred_df = MixMhcPredHelper(peptides, alleles, mhc_class='I').predict_df()
        elif tool == 'MixMHC2pred':
            pred_df = MixMhcPredHelper(peptides, alleles, mhc_class='II').predict_df()
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
        best_allele_df = best_allele_df[['Peptide', 'Best_Allele', 'EL_Rank', 'Binder', 'Alleles']]

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

if __name__ == '__main__':
    run(tools=['MHCflurry', 'NetMHCpan'], min_seq_length=8, max_seq_length=15)
    print('Have a nice day.')