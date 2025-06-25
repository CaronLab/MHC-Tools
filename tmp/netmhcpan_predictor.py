
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from predictors.netmhcpan_helper import NetMHCpanHelper

def run_netmhcpan(min_seq_length, max_seq_length, mhc_class='I'):
    alleles = pd.read_csv('./input/alleles.csv', header=None).values[:, 0].tolist()
    alleles = [allele.replace('_', '-') for allele in alleles]
    peptides = pd.read_csv('./input/peptides.csv', header=None).values[:, 0].tolist()
    peptides_len = len(peptides)
    peptides = [p for p in peptides if min_seq_length <= len(p) <= max_seq_length]

    netmhcpan = NetMHCpanHelper(peptides=peptides, alleles=alleles, mhc_class=mhc_class)
    print('Input file prepared.')

    if len(netmhcpan.netmhcpan_peptides) != peptides_len:
        print(f'Peptides length out of [{min_seq_length}, {max_seq_length}] are filtered out. {len(netmhcpan.netmhcpan_peptides)} peptides left. {peptides_len - len(netmhcpan.netmhcpan_peptides)} peptides are ignored.')
    pred_df = netmhcpan.predict_df()
    assert len(pred_df) / len(alleles) == len(netmhcpan.netmhcpan_peptides)
    pred_df.to_csv('./output/netmhcpan_output.csv', index=False)




if __name__ == '__main__':
    run_netmhcpan(8, 15, 'I')
    # run_netmhcpan(9, 25, mhc_class='II')
    analyze_binders()
    draw_binding_affinity()
    print('Have a nice day.')
