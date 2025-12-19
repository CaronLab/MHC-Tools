import os
import random
import tempfile
import mhcgnomes
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List
from itertools import islice
from uuid import uuid4
from mhcnames import normalize_allele_name

from predictors.base_helper import BaseHelper
from utils.allele import prepare_class_II_alleles, prepare_class_I_alleles, get_normalized_allele_name
from utils.peptide import remove_previous_and_next_aa, remove_modifications, replace_uncommon_aas
from utils.job import Job
from tqdm.contrib.concurrent import process_map

from utils.constants import EPSILON

TMP_DIR = str(Path(tempfile.gettempdir(), 'pynetmhcpan').expanduser())
NETMHCPAN = Path(__file__).parent.parent / 'third_party' / 'netMHCpan-4.1' / 'netMHCpan'
NETMHCIIPAN = Path(__file__).parent.parent / 'third_party' / 'netMHCIIpan-4.3' / 'netMHCIIpan'


class NetMHCpanHelper(BaseHelper):
    """
    example usage:
    cl_tools.make_binding_prediction_jobs()
    cl_tools.run_jubs()
    cl_tools.aggregate_netmhcpan_results()
    cl_tools.clear_jobs()
    """
    def __init__(self,
                 peptides: List[str] = None,
                 alleles: List[str] = None,
                 mhc_class: str = 'I',
                 n_threads: int = 0,
                 tmp_dir: str = TMP_DIR,
                 output_dir: str = None):
        """
        Helper class to run NetMHCpan on multiple CPUs from Python. Can annotated a file with peptides in it.
        """
        if mhc_class == 'I':
            super().__init__('NetMHCpan')
        else:
            super().__init__('NetMHCIIpan')

        if alleles is None or len(alleles) == 0:
            raise RuntimeError(f'Alleles are needed for {self.tool_name} predictions.')

        if mhc_class == 'I':
            self.alleles = self._format_class_I_alleles(alleles)
            self.min_length = 8
        else:
            self.alleles = self._format_class_II_alleles(alleles)
            self.min_length = 9

        self.peptides = []
        if peptides is not None:
            self.add_peptides(peptides)
            self.netmhcpan_peptides = replace_uncommon_aas(self.peptides)
        else:
            self.netmhcpan_peptides = dict()
        self.predictions = {x: {} for x in self.peptides}
        self.wd = Path(output_dir) if output_dir else Path(os.getcwd())
        self.temp_dir = Path(tmp_dir) / 'PyNetMHCpan'
        if self.wd and not self.wd.exists():
            self.wd.mkdir(parents=True)
        if not self.temp_dir.exists():
            self.temp_dir.mkdir(parents=True)
        self.predictions_made = False
        self.not_enough_peptides = []
        if n_threads < 1 or n_threads > os.cpu_count():
            self.n_threads = os.cpu_count()
        else:
            self.n_threads = n_threads
        self.jobs = []
        # self.add_peptides(peptides)
        self.mhc_class: str = mhc_class

    def add_peptides(self, peptides: List[str]):
        peptides = remove_previous_and_next_aa(peptides)
        peptides = remove_modifications(peptides)
        for p in peptides:
            if len(p) < self.min_length:
                raise ValueError(f"One or more peptides is shorter than the minimum length of {self.min_length} mers")
        self.peptides += peptides

        self.predictions = {pep: {} for pep in self.peptides}

    def _format_class_I_alleles(self, alleles: List[str]):
        avail_allele_path = Path(__file__).parent.parent/'third_party'/'netMHCpan-4.1'/'Linux_x86_64'/'data'/'MHC_pseudo.dat'
        avail_alleles = [line.split()[0].replace(':', '') for line in open(avail_allele_path).readlines()]

        avail_alleles = [mhcgnomes.parse(allele).to_string().replace('*', '').replace('H2-', 'H-2-') for allele in avail_alleles]
        std_alleles = prepare_class_I_alleles(alleles, avail_alleles)
        return [a.replace(':', '') for a in std_alleles]

    def _format_class_II_alleles(self, alleles: List[str]):
        avail_allele_path = Path(__file__).parent.parent/'third_party'/'netMHCIIpan-4.3'/'data'/'pseudosequence.2023.all.X.dat'
        avail_alleles = [line.split()[0].replace('_', '') for line in open(avail_allele_path).readlines()]
        paired_alleles = prepare_class_II_alleles(alleles, avail_alleles)
        for i in range(len(paired_alleles)):
            allele = paired_alleles[i]
            allele = normalize_allele_name(allele)
            if allele.startswith('HLA-DRA1*01:01'):
                allele = allele.split('-')[-1].replace(':', '').replace('*', '_')
            else:
                allele = allele.replace(':', '').replace('*', '')
            paired_alleles[i] = allele
        return paired_alleles

    def _make_binding_prediction_jobs(self):
        if not self.peptides:
            print("ERROR: You need to add some peptides first!")
            return
        self.jobs = []

        # split peptide list into chunks
        if self.netmhcpan_peptides:
            peptides = list(self.netmhcpan_peptides.values())
        else:
            peptides = self.peptides
        random.shuffle(peptides)  # we need to shuffle them so we don't end up with files filled with peptide lengths that take a LONG time to compute (this actually is a very significant speed up)

        if len(peptides) > 100:
            peptide_iter = iter(peptides)
            chunks = list(iter(lambda: tuple(islice(peptide_iter, 100)), ()))
        else:
            chunks = [peptides]
        job_number = 1
        print(f'Peptide list broken into {len(chunks)} chunks.')

        for chunk in chunks:
            if len(chunk) < 1:
                continue
            fname = Path(self.temp_dir, f'peplist_{job_number}.csv')
            # save the new peptide list, this will be given to netMHCpan
            with open(str(fname), 'w') as f:
                f.write('\n'.join(chunk))
            # run netMHCpan
            if self.mhc_class == 'I':
                command = f'{NETMHCPAN} -p -f {fname} -a {",".join(self.alleles)} -BA'.split(' ')
            else:
                command = f'{NETMHCIIPAN} -inptype 1 -f {fname} -a {",".join(self.alleles)} -BA'.split(' ')

            job = Job(command=command, working_directory=self.temp_dir)
            self.jobs.append(job)
            job_number += 1

    @staticmethod
    def _run_job(job: Job):
        job.run()
        return job

    def _run_jobs(self):
        self.jobs = process_map(self._run_job, self.jobs, max_workers=self.n_threads, chunksize=1)
        for job in self.jobs:
            if job.returncode != 0:
                raise ChildProcessError(f'{job.stdout.decode()}\n\n{job.stderr.decode()}')
            out = (job.stdout.decode() + job.stderr.decode()).split('\n')
            if 'error' in (' '.join(out[-5:])).lower():
                raise ChildProcessError(f'{job.stdout.decode()}\n\n{job.stderr.decode()}')

    def _parse_netmhc_output(self, stdout: str):
        lines = stdout.split('\n')
        reverse_lookup = {value: key for key, value in self.netmhcpan_peptides.items()}
        if self.mhc_class == 'I':
            allele_idx = 1
            peptide_idx = 2
            el_score_idx = 11
            el_rank_idx = 12
            aff_score_idx = 13
            aff_rank_idx = 14
            aff_nM_idx = 15
            strong_cutoff = 0.5
            weak_cutoff = 2.0
        else:
            allele_idx = 1
            peptide_idx = 2
            core_idx = 4
            el_score_idx = 8
            el_rank_idx = 9
            aff_score_idx = 11
            aff_nM_idx = 13
            aff_rank_idx = 12
            strong_cutoff = 2.0
            weak_cutoff = 10.0
        for line in lines:
            line = line.strip()
            line = line.split()
            if not line or line[0] == '#' or not line[0].isnumeric():
                continue
            allele = line[allele_idx].replace('*', '').replace(':', '')
            peptide = line[peptide_idx]
            el_rank = float(line[el_rank_idx])
            el_score = float(line[el_score_idx])
            aff_rank = float(line[aff_rank_idx])
            aff_score = float(line[aff_score_idx])
            aff_nM  = float(line[aff_nM_idx])

            if float(el_rank) <= strong_cutoff:
                binder = 'Strong'
            elif float(el_rank) <= weak_cutoff:
                binder = 'Weak'
            else:
                binder = 'Non-binder'
            if self.mhc_class == 'I':
                self.predictions[reverse_lookup[peptide]][allele] = {
                    'el_rank': el_rank,
                    'el_score': el_score,
                    'aff_rank': aff_rank,
                    'aff_score': aff_score,
                    'aff_nM': aff_nM,
                    'binder': binder}
            else:
                self.predictions[reverse_lookup[peptide]][allele] = {
                    'core': line[core_idx],
                    'el_rank': el_rank,
                    'el_score': el_score,
                    'aff_rank': aff_rank,
                    'aff_score': aff_score,
                    'aff_nM': aff_nM,
                    'binder': binder}


    def _aggregate_netmhcpan_results(self):
        for job in self.jobs:
            if job.returncode != 0:
                print(job.stdout.decode())
                print(job.stderr.decode())
                print('ERROR: There was a problem in NetMHCpan. See the above about for possible information.')
                exit(1)
            self._parse_netmhc_output(job.stdout.decode())

    def _clear_jobs(self):
        self.jobs = []

    def make_predictions(self):
        self.temp_dir = self.temp_dir / str(uuid4())
        self.temp_dir.mkdir(parents=True)
        self._make_binding_prediction_jobs()
        self._run_jobs()
        self._aggregate_netmhcpan_results()
        self._clear_jobs()

    def predict_df(self):
        print(f'Running {self.tool_name}')
        self.make_predictions()
        if self.mhc_class == 'I':
            df_columns = ['Peptide', 'Allele', 'EL_score', 'EL_Rank', 'Aff_Score', 'Aff_Rank', 'Aff_nM', 'Binder']
        else:
            df_columns = ['Peptide', 'Allele', 'Core', 'EL_score', 'EL_Rank', 'Aff_Score', 'Aff_Rank', 'Aff_nM', 'Binder']
        data = []
        for allele in self.alleles:
            normed_allele = get_normalized_allele_name(allele)
            for pep in self.peptides:
                # netmhc_pep = self.netmhcpan_peptides[pep]
                if self.mhc_class == 'I':
                    data.append([pep,
                                 normed_allele,
                                 self.predictions[pep][allele]['el_score'],
                                 self.predictions[pep][allele]['el_rank'],
                                 self.predictions[pep][allele]['aff_score'],
                                 self.predictions[pep][allele]['aff_rank'],
                                 self.predictions[pep][allele]['aff_nM'],
                                 self.predictions[pep][allele]['binder']])
                else:
                    data.append([pep,
                                 normed_allele,
                                 self.predictions[pep][allele]['core'],
                                 self.predictions[pep][allele]['el_score'],
                                 self.predictions[pep][allele]['el_rank'],
                                 self.predictions[pep][allele]['aff_score'],
                                 self.predictions[pep][allele]['aff_rank'],
                                 self.predictions[pep][allele]['aff_nM'],
                                 self.predictions[pep][allele]['binder']])

        self.pred_df = pd.DataFrame(data=data, columns=df_columns)
        return self.pred_df

