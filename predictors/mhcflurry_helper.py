
import tempfile
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from predictors.base_helper import BaseHelper
from mhcnames import normalize_allele_name
from utils.allele import get_normalized_allele_name

class MhcFlurryHelper(BaseHelper):
    def __init__(self,
                 peptides: list[str],
                 alleles: list[str]):
        super().__init__('MHCflurry')

        if alleles is None or len(alleles) == 0:
            raise RuntimeError('Alleles are needed for MhcFlurry predictions.')
        if np.max(np.vectorize(len)(peptides)) > 16:
            raise RuntimeError('MhcFlurry cannot make predictions on peptides over length 16.')

        self.peptides = peptides
        self.alleles = self._format_class_I_alleles(alleles)

    def _format_class_I_alleles(self, alleles):
        std_alleles = []
        for allele in set(alleles):
            try:
                std_alleles.append(normalize_allele_name(allele))
            except ValueError:
                print(f'ERROR: Allele {allele} not supported.')
        return [a.replace('*', '').replace(':', '') for a in std_alleles]

    def predict_df(self):
        # we will run MhcFlurry in a separate process so the Tensorflow space doesn't get messed up. I don't know why it
        # happens, but it does, and then either MhcFlurry or TensorFlow Decision Forests stops working.
        # I think perhaps because MhcFlurry uses some legacy code from TFv1 (I think), though this is only
        # a suspicion.
        print('Running MhcFlurry')
        with tempfile.NamedTemporaryFile('w', delete=False) as pep_file:
            pep_file.write('allele,peptide\n')
            for pep in self.peptides:
                for allele in self.alleles:
                    pep_file.write(f'{allele},{pep}\n')
            pep_file_path = pep_file.name
        with tempfile.NamedTemporaryFile('w') as results:
            results_file = results.name

        command = f'mhcflurry-predict --out {results_file} {pep_file_path}'.split()
        subprocess.run(command)

        result_df = pd.read_csv(results_file, index_col=False)
        self.pred_df = pd.DataFrame()
        self.pred_df['Peptide'] = result_df['peptide']
        self.pred_df['Allele'] = [get_normalized_allele_name(a) for a in result_df['allele']]
        self.pred_df['EL_Rank'] = result_df['mhcflurry_presentation_percentile']
        self.pred_df['Binder'] = 'Non-binder'
        self.pred_df.loc[self.pred_df['EL_Rank'] <= 2.0, 'Binder'] = 'Weak'
        self.pred_df.loc[self.pred_df['EL_Rank'] <= 0.5, 'Binder'] = 'Strong'

        return self.pred_df