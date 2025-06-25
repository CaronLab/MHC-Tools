
import pandas as pd
from predictors.bigmhc_helper import BigMhcHelper
from utils.peptide import filter_peptides_by_length


def run_bigmhc(min_seq_length, max_seq_length):
    alleles = pd.read_csv('./input/alleles.csv', header=None).values[:, 0].tolist()
    alleles = [allele.replace('_', '-') for allele in alleles]
    peptides = pd.read_csv('./input/peptides.csv', header=None).values[:, 0].tolist()
    peptides = filter_peptides_by_length(peptides, min_seq_length, max_seq_length)

    bigmhc = BigMhcHelper(peptides=peptides, alleles=alleles)
    pred_df = bigmhc.predict_df()
    assert len(pred_df) / len(alleles) == len(peptides)

    bigmhc.save()
    bigmhc.analyze_binders()
    bigmhc.draw_binding_affinity()


if __name__ == '__main__':
    run_bigmhc(8, 15)
    print('Have a nice day.')
