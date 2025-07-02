
import tempfile
import subprocess
import mhcgnomes


import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from mhcnames import normalize_allele_name

from predictors.base_helper import BaseHelper
from utils.allele import prepare_class_I_alleles, prepare_class_II_alleles
from utils.allele import get_normalized_allele_name


class MixMhcPredHelper(BaseHelper):
    def __init__(self,
                 peptides: list[str],
                 alleles: list[str],
                 mhc_class: str = 'I'):

        if mhc_class == 'I':
            super().__init__('MixMHCpred')
            self.alleles = self._format_class_I_alleles(alleles)
        else:
            super().__init__('MixMHC2pred')
            self.alleles = self._format_class_II_alleles(alleles)
        self.mhc_class = mhc_class
        if alleles is None or len(alleles) == 0:
            raise RuntimeError(f'Alleles are needed for {self.tool_name} predictions.')
        pep_lens = np.vectorize(len)(peptides)
        if np.min(pep_lens) < 8 and np.max(pep_lens) > 25:
            raise RuntimeError(f'{self.tool_name} cannot make predictions on peptides <8 mers or >25 mers.')

        self.peptides = peptides
        self.mixmhcpred_exe_path = Path(__file__).parent.parent/'third_party'/'MixMHCpred-3.0'/'MixMHCpred'
        self.mixmhc2pred_exe_path = Path(__file__).parent.parent/'third_party'/'MixMHC2pred-2.0'/'MixMHC2pred_unix'

    def _format_class_I_alleles(self, alleles: List[str]):
        avail_allele_path = Path(__file__).parent.parent/'third_party'/'MixMHCpred-3.0'/'lib'/'alleles_list.txt'
        avail_alleles = [line.split()[0] for line in open(avail_allele_path).readlines()][1:]

        avail_alleles = [mhcgnomes.parse(allele).to_string() for allele in avail_alleles]
        std_alleles = prepare_class_I_alleles(alleles, avail_alleles)
        return [a.replace('HLA-', '').replace('*', '').replace(':', '') for a in std_alleles]

    def _format_class_II_alleles(self, alleles: List[str]):
        avail_allele_path = Path(__file__).parent.parent/'third_party'/'MixMHC2pred-2.0'/'PWMdef'/'Alleles_list_Human.txt'
        avail_alleles = [line.split()[0].replace('__', '-').replace('_', '') \
                         for line in open(avail_allele_path).readlines() if line.startswith('D')]
        paired_alleles = prepare_class_II_alleles(alleles, avail_alleles)
        for i in range(len(paired_alleles)):
            allele = paired_alleles[i]
            allele = normalize_allele_name(allele)
            if allele.startswith('HLA-DRA1*01:01'):
                allele = allele.split('-')[-1].replace(':', '_').replace('*', '_')
            else:
                allele = '__'.join(allele.split('-')[-2:]).replace(':', '_').replace('*', '_')
            paired_alleles[i] = allele
        return paired_alleles

    def predict_df(self):
        print(f'Running {self.tool_name}')
        with tempfile.NamedTemporaryFile('w', delete=False) as pep_file:
            for pep in self.peptides:
                pep_file.write(f'{pep}\n')
            pep_file_path = pep_file.name
        with tempfile.NamedTemporaryFile('w') as results:
            results_file = results.name

        if self.mhc_class == 'I':
            alleles = ','.join(self.alleles)
            command = f'{self.mixmhcpred_exe_path} -i {pep_file_path} -o {results_file} -a {alleles} --no_context'
        else:
            alleles = ' '.join(self.alleles)
            command = f'{self.mixmhc2pred_exe_path} -i {pep_file_path} -o {results_file} -a {alleles} --no_context'
        subprocess.run(command, shell=True)

        pred = []
        for line in open(results_file, 'r'):
            line = line.strip()
            line = line.split('\t')
            if not line or line[0].startswith('#'):
                continue
            pred.append(line)
        result_df = pd.DataFrame(pred[1:], columns=pred[0])

        df_columns = ['Peptide', 'Allele', 'EL_Rank', 'Binder']
        data = []
        for i in range(len(result_df)):
            for j in range(len(result_df.columns)):
                if not result_df.columns[j].startswith('%Rank') or result_df.columns[j].startswith('%Rank_best'):
                    continue
                peptide = result_df.loc[i, 'Peptide']
                column_name = result_df.columns[j]
                allele = get_normalized_allele_name(column_name[column_name.find('_') + 1:])
                # allele = result_df.columns[j].split('_')[1]
                el_rank = result_df.iloc[i, j]
                if el_rank == 'NA':
                    binder = 'Non-binder'
                else:
                    el_rank = float(el_rank)
                    binder = 'Strong' if el_rank <= 0.5 else 'Weak' if el_rank <= 2.0 else 'Non-binder'
                data.append([peptide, allele, el_rank, binder])
        self.pred_df = pd.DataFrame(data=data, columns=df_columns)
        self.pred_df.loc[self.pred_df['EL_Rank'] == 'NA', 'EL_Rank'] = 100
        return self.pred_df


