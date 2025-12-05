import pandas as pd

def merge_to_ref(ref_path, input_path, output_path):
    ref_df = pd.read_csv(ref_path, sep='\t')
    input_df = pd.read_csv(input_path, sep='\t')

    ref_peps = set(ref_df['ModifiedPeptideSequence'])
    masks = input_df['ModifiedPeptideSequence'].apply(lambda x: x in ref_peps)
    ref_df = pd.concat([ref_df, input_df[~masks]], ignore_index=True)
    ref_df.to_csv(output_path, sep='\t', index=False)

if __name__ == '__main__':
    ref_path = '/mnt/d/workspace/MS-Tools/data/library1.tsv'
    input_path = '/mnt/d/workspace/MS-Tools/data/library2.tsv'
    output_path = '/mnt/d/workspace/MS-Tools/data/library_merged.tsv'
    merge_to_ref(ref_path, input_path, output_path)