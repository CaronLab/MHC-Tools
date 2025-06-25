import numpy as np
import pandas as pd
from mhcnames import normalize_allele_name

from predictors.mixmhcpred_helper import MixMhcPredHelper
from predictors.netmhcpan_helper import NetMHCpanHelper

def run_netmhcpan(peptides, alleles, strong_threshold=1, weak_threshold=5):
    netmhcpan = NetMHCpanHelper(peptides=peptides, alleles=alleles, mhc_class='II')
    pred_df = netmhcpan.predict_df()
    assert len(pred_df) / len(alleles) == len(netmhcpan.netmhcpan_peptides)
    result_df = pred_df.pivot(index='Peptide', columns='Allele', values='EL_Rank')
    result_df = result_df.rename(columns={a: normalize_allele_name(a) for a in result_df.columns})
    result_df['Best_Rank'] = result_df.min(axis=1)
    result_df['Best_Allele'] = result_df.idxmin(axis=1)
    conditions = [
        result_df['Best_Rank'] <= strong_threshold,
        (result_df['Best_Rank'] > strong_threshold) & (result_df['Best_Rank'] <= weak_threshold),
        result_df['Best_Rank'] > weak_threshold
    ]
    choices = ['Strong', 'Weak', 'Non-binder']
    result_df['Binder'] = np.select(conditions, choices)

    col_map = {c: 'NMP_' + c for c in result_df.columns}
    result_df = result_df.rename(columns=col_map)
    return result_df

def run_mixmhc2pred(peptide, alleles, strong_threshold=1, weak_threshold=5):
    mixmhc2pred = MixMhcPredHelper(peptides=peptide, alleles=alleles)
    pred_df = mixmhc2pred.predict_df()
    columns = [col for col in pred_df.columns if col.startswith('%Rank') and not col.endswith('best')]
    result_df = pred_df[['Peptide'] + columns + ['%Rank_best', 'BestAllele']]
    result_df = result_df.set_index('Peptide')
    result_df['BestAllele'] = [normalize_allele_name(a) if a!='NA' else 'NA' for a in result_df['BestAllele']]
    result_df['%Rank_best'] = result_df['%Rank_best'].replace('NA', 100).astype(np.float32)
    conditions = [
        result_df['%Rank_best'] <= strong_threshold,
        (result_df['%Rank_best'] > strong_threshold) & (result_df['%Rank_best'] <= weak_threshold),
        result_df['%Rank_best'] > weak_threshold
    ]
    choices = ['Strong', 'Weak', 'Non-binder']
    result_df['Binder'] = np.select(conditions, choices)

    col_map = {col: normalize_allele_name(col.replace('%Rank_', '')) for col in columns}
    col_map['%Rank_best'] = 'Best_Rank'
    col_map['BestAllele'] = 'Best_Allele'
    col_map['Binder'] = 'Binder'
    col_map = {k: 'MMP_' + v for k, v in col_map.items()}
    result_df = result_df.rename(columns=col_map)
    return result_df

def run(min_seq_length, max_seq_length, strong_threshold=1, weak_threshold=5):
    alleles = pd.read_csv('./input/alleles.csv', header=None).values[:, 0].tolist()
    alleles = [allele.replace('_', '-') for allele in alleles]
    peptides = pd.read_csv('./input/peptides.csv', header=None).values[:, 0].tolist()
    peptides_len = len(peptides)
    peptides = [p for p in peptides if min_seq_length <= len(p) <= max_seq_length]
    if len(peptides) != peptides_len:
        print(f'Peptides length out of [{min_seq_length}, {max_seq_length}] are filtered out. '
              f'{len(peptides)} peptides left. {peptides_len - len(peptides)} peptides are ignored.')

    nmp_df = run_netmhcpan(peptides, alleles, strong_threshold, weak_threshold)
    mmp_df = run_mixmhc2pred(peptides, alleles, strong_threshold, weak_threshold)
    result_df = nmp_df.join(mmp_df, how='outer')

    result_df['Best_Allele'] = None
    result_df['Best_Tool'] = None
    result_df['Binder'] = 'Non-binder'

    nmp_binder_indices = result_df['NMP_Binder'].isin(['Strong', 'Weak'])
    result_df.loc[nmp_binder_indices, 'Best_Allele'] = result_df.loc[nmp_binder_indices, 'NMP_Best_Allele']
    result_df.loc[nmp_binder_indices, 'Best_Tool'] = 'NetMHCIIpan'
    result_df.loc[nmp_binder_indices, 'Binder'] = result_df.loc[nmp_binder_indices, 'NMP_Binder']

    mmp_binder_indices = result_df['MMP_Binder'].isin(['Strong', 'Weak'])
    sequential_binder_indices = mmp_binder_indices & ~nmp_binder_indices
    result_df.loc[sequential_binder_indices, 'Best_Allele'] = result_df.loc[sequential_binder_indices, 'MMP_Best_Allele']
    result_df.loc[sequential_binder_indices, 'Best_Tool'] = 'MixMHC2pred'
    result_df.loc[sequential_binder_indices, 'Binder'] = result_df.loc[sequential_binder_indices, 'MMP_Binder']
    result_df.to_csv('./output/sequential_result.csv', index=True)

def run(order=['NetMHCpan', 'MHCflurry']):



if __name__ == '__main__':
    run(min_seq_length=9, max_seq_length=25, strong_threshold=1, weak_threshold=5)
    print('Have a nice day.')
