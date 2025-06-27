

import pandas as pd
from predictors.mixmhcpred_helper import MixMhcPredHelper
from utils.peptide import filter_peptides_by_length


def run_mixmhcpred(min_seq_length, max_seq_length, mhc_class='I'):
    alleles = pd.read_csv('./input/alleles-II.csv', header=None).values[:, 0].tolist()
    alleles = [allele.replace('_', '-') for allele in alleles]
    peptides = pd.read_csv('./input/peptides.csv', header=None).values[:, 0].tolist()
    peptides = filter_peptides_by_length(peptides, min_seq_length, max_seq_length)

    mixmhcpred = MixMhcPredHelper(peptides=peptides, alleles=alleles, mhc_class=mhc_class)
    pred_df = mixmhcpred.predict_df()
    assert len(pred_df) / len(alleles) == len(peptides)

    mixmhcpred.save()
    mixmhcpred.analyze_binders()
    mixmhcpred.draw_binding_affinity()


if __name__ == '__main__':
    run_mixmhcpred(9, 25, 'II')
    print('Have a nice day.')
