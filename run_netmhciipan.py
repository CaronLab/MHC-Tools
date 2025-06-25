
import pandas as pd
from predictors.netmhcpan_helper import NetMHCpanHelper
from utils.peptide import filter_peptides_by_length

def run_netmhcpan(min_seq_length, max_seq_length, mhc_class='I'):
    alleles = pd.read_csv('./input/alleles.csv', header=None).values[:, 0].tolist()
    alleles = [allele.replace('_', '-') for allele in alleles]
    peptides = pd.read_csv('./input/peptides.csv', header=None).values[:, 0].tolist()
    peptides = filter_peptides_by_length(peptides, min_seq_length, max_seq_length)

    netmhcpan = NetMHCpanHelper(peptides=peptides, alleles=alleles, mhc_class=mhc_class)
    pred_df = netmhcpan.predict_df()
    assert len(pred_df) / len(alleles) == len(netmhcpan.netmhcpan_peptides)

    netmhcpan.save()
    netmhcpan.analyze_binders()
    netmhcpan.draw_binding_affinity()


if __name__ == '__main__':
    run_netmhcpan(9, 25, 'II')
    print('Have a nice day.')