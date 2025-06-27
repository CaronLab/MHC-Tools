

import pandas as pd
from predictors.mhcflurry_helper import MhcFlurryHelper
from utils.peptide import filter_peptides_by_length


def run_mhcflurry(min_seq_length, max_seq_length):
    alleles = pd.read_csv('input/alleles-I.csv', header=None).values[:, 0].tolist()
    alleles = [allele.replace('_', '-') for allele in alleles]
    peptides = pd.read_csv('./input/peptides.csv', header=None).values[:, 0].tolist()
    peptides = filter_peptides_by_length(peptides, min_seq_length, max_seq_length)

    mhcflurry = MhcFlurryHelper(peptides=peptides, alleles=alleles)
    pred_df = mhcflurry.predict_df()
    assert len(pred_df) / len(alleles) == len(peptides)

    mhcflurry.save()
    mhcflurry.analyze_binders()
    mhcflurry.draw_binding_affinity()


if __name__ == '__main__':
    run_mhcflurry(8, 15)
    print('Have a nice day.')
