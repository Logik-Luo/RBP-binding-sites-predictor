import pandas as pd
import math
import numpy as np

"""
    功能：存放氨基酸的特征向量字典，1-6为氨基酸类别编码，7-11为理化性质编码
    输入：一个氨基酸名称
    返回：该氨基酸的特征向量
"""
def amino_encoding(amino_name):
    amino = {'ALA': [1, 0, 0, 0, 0, 0, 5, 0, 2, 7.00, 1.8],
             'CYS': [0, 0, 1, 0, 0, 0, 6, 0, 2, 7.00, 2.5],
             'ASP': [0, 0, 0, 0, 0, 1, 8, -1, 4, 3.65, -3.5],
             'GLU': [0, 0, 0, 0, 0, 1, 9, -1, 4, 3.22, -3.5],
             'PHE': [0, 1, 0, 0, 0, 0, 11, 0, 2, 7.00, 2.8],
             'GLY': [1, 0, 0, 0, 0, 0, 4, 0, 2, 7.00, -0.4],
             'HIS': [0, 0, 0, 1, 0, 0, 10, 1, 4, 6.00, -3.2],
             'ILE': [0, 1, 0, 0, 0, 0, 8, 0, 2, 7.00, 4.5],
             'LYS': [0, 0, 0, 0, 1, 0, 9, 1, 2, 10.53, 3.9],
             'LEU': [0, 1, 0, 0, 0, 0, 8, 0, 2, 7.00, 3.8],
             'MET': [0, 0, 1, 0, 0, 0, 8, 0, 2, 7.00, 1.9],
             'ASN': [0, 0, 0, 1, 0, 0, 8, 0, 4, 8.18, -3.5],
             'PRO': [0, 1, 0, 0, 0, 0, 7, 0, 2, 7.00, -1.6],
             'GLN': [0, 0, 0, 1, 0, 0, 9, 0, 4, 7.00, -3.5],
             'ARG': [0, 0, 0, 0, 1, 0, 11, 1, 4, 12.48, -4.5],
             'SER': [0, 0, 1, 0, 0, 0, 6, 0, 4, 7.00, -0.8],
             'THR': [0, 0, 1, 0, 0, 0, 7, 0, 4, 7.00, -0.7],
             'VAL': [1, 0, 0, 0, 0, 0, 7, 0, 2, 7.00, 4.2],
             'TRP': [0, 0, 0, 1, 0, 0, 14, 0, 3, 7.00, -0.9],
             'TYR': [0, 0, 1, 0, 0, 0, 12, 0, 3, 10.07, -1.3]}

    return amino[amino_name]


"""
    功能：将氨基酸的三个字母名称转换为一个字母的名称
    输入：一个氨基酸的三个字母名称
    输出：该氨基酸的一个字母名称
"""
def amino_upper2lower(amino_name):
    amino = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
             'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
             'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

    return amino[amino_name]


"""
    功能：生成氨基酸的PSSM矩阵
    输入：（蛋白质的pdb_id，一个氨基酸的名称）
    输出：该氨基酸的PSSM矩阵
"""
def gen_pssm(pdb_id, amino_name):
    reader = open("./pssm/" + pdb_id + '.pssm').readlines()
    for line in reader:
        if line[:2] == '  ' and line[3:5] != '  ':
            if line[6:7] == amino_name:
                feat_pssm = [int(i) for i in line[7:].split('  ')[2:22]]

    return feat_pssm


"""
    功能：生成氨基酸的标签
    输入：（蛋白质的pdb_id，一个氨基酸的名称）
    输出：氨基酸的标签
"""
def gen_label(pdb_id, amino_name):
    pdb = open("./pdb/pdb" + pdb_id.lower() + ".ent", "r").readlines()
    label = 0

    for line_amino in pdb:
        if line_amino[:4] == 'ATOM' and line_amino[17:20] == amino_name:  # 此时为氨基酸的原子
            line2 = [i for i in line_amino.strip()[28:].split(' ') if i != '']
            amino_cord = np.array([float(i) for i in line2[0:3]])  # 得氨基酸原子的坐标
            amino_atom = line_amino.strip()[-1:]  # 得到amino的原子名称

            for line_rna in pdb:
                if line_rna[:4] == 'ATOM':
                    if line_rna[17:19] == '  ':             # 此时为RNA的原子
                        line1 = [i for i in line_rna.strip()[28:].split(' ') if i != '']
                        rna_cord = np.array([float(i) for i in line1[0:3]])                 # 得RNA原子的坐标
                        rna_atom = line_rna.strip()[-1:]                    # 得到RNA的原子名称
                        # 如果氨基酸中的原子和RNA中的原子相同，且两原子间的距离小于6A
                        if amino_atom == rna_atom and np.sqrt(sum(pow(amino_cord - rna_cord, 2))) < 6:
                        #if np.sqrt(sum(pow(amino_cord - rna_cord, 2))) < 6:
                            label = 1
                            break
    return label




"""
    功能：获得氨基酸的特征向量
    输入：存放所有pdb的文件夹路径
    输出：所有的氨基酸的特征向量
"""
def gen_all_feature(pdb_id, sequence):
    amino_feats = []
    amino_names = []
    amino_labels = []
    pdb_file = open('./pdb/pdb' + pdb_id.lower() + '.ent').readlines()
    amino_name_last = 0  # 用来标记该轮循环的氨基酸和上一轮循环氨基酸是否相同

    for line in pdb_file:
        # 得到总的氨基酸和RNA的名称
        amino_name = line.strip()[17:20]
        # 得到氨基酸前11位编码
        if line[:4] == 'ATOM' and amino_name[:1] != ' ':
            feat_11 = amino_encoding(amino_name)                            # 获得氨基酸名称的特征
            feat_pssm = gen_pssm(pdb_id, amino_upper2lower(amino_name))     # 获得氨基酸pssm特征

            if amino_name != amino_name_last:                               # 获得氨基酸的标签
                label = gen_label(pdb_id, amino_name)

                amino_name_last = amino_name
                amino_feats.append(feat_11 + feat_pssm)                         # 将氨基酸的31维特征合并
                amino_names.append(amino_name)                                  # 将氨基酸的名称合并
                amino_labels.append(label)                                      # 将氨基酸的标签合并

    return amino_names, amino_feats, amino_labels




data = pd.read_csv('PDB.seq')
pdb = data['Seq'].tolist()

# for line in pdb:
#     pdb_id = line[1:5]              # 蛋白质的pdb_id
#     pdb_seq = line[8:]              # 蛋白质序列
#
#     # 生成蛋白质的前11维特征向量
#     feat_11 = gen_11_feature(pdb_id)
#
#
#     print(pdb_seq)

name, feat, label = gen_all_feature('1A34', 'TGDNSNVVTMIRAGSYPKVNPTPTWVRAIPFEVSVQSGIAFKVPVGSLFSANFRTDSFTSVTVMSVRAWTQLTPPVNEYSFVRLKPLFKTGDSTEEFEGRASNINTRASVGYRIPTNLRQNTVAADNVCEVRSNCRQVALVISCCFN')

data = {"Name":name, "Feat":feat, "Label":label}
df_data = pd.DataFrame(data)
df_data.to_csv("data.csv")
