from tqdm import tqdm
import pandas as pd

def sh_text(item):
    fasta_file = './fasta/' + item + '.fasta'
    pssm_file = './pssm/' + item + '.pssm'
    s = 'psiblast -query ' + fasta_file + ' -db ./uniref/uniprot_sprot.fasta -num_iterations 3 -out_ascii_pssm ' +  pssm_file + '\n'
    with open('pssm_blast.sh', 'a') as f:
        f.write(s)


data = pd.read_csv("PDB.seq")
sequences = data['Seq'].tolist()

for seq in sequences:
    sh_text(seq[1:5])       # 链名传进了列表，变成了sh_text的item

# for seq in sequences:
#     with open('fasta/' + seq[1:5] + '.fasta', 'w') as f:
#         f.write('>' + seq[1:5] + '\n' + seq[8:])

