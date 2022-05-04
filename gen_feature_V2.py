import pandas as pd
import math
import numpy as np
import json
from tqdm import tqdm


"""
    功能：得到rna链的名称
    输入：蛋白质的pdb_id
    输入：rna链的名称
"""
def get_rna_name(pdb_id):
    data = pd.read_csv('dataset/rna.csv')
    rna = data['rna'].tolist()
    
    for i in rna:
        if pdb_id == i[1:5]:
            # print(pdb_id == i[1:5])
            rna_name = i[6:7]
            break


    return rna_name

"""
    功能：生成氨基酸的子结构特征
    输入：一个氨基酸的子结构序列
    返回：该氨基酸的特征向量
"""
def sub_encoding(sub_seq):
    result = []
    for i in sub_seq:
        if i == 'C':
            result.append([1, 0, 0])
        elif i == 'E':
            result.append([0, 1, 0])
        elif i == 'H':
            result.append([0, 0, 1])

    return result
"""
    功能：生成氨基酸的溶剂可及性特征
    输入：一个氨基酸的名称
    返回：该氨基酸的特征向量
"""
def rsa_encoding(rsa_seq):
    result = []
    for i in rsa_seq:
        if i == 'B':
            result.append([1, 0, 0, 0])
        elif i == 'b':
            result.append([0, 1, 0, 0])
        elif i == 'e':
            result.append([0, 0, 1, 0])
        elif i == 'E':
            result.append([0, 0, 0, 1])

    return result

"""
    功能：生成氨基酸的子结构特征和溶剂可及性特征
    输入：一个氨基酸的可及性序列
    返回：该氨基酸的特征向量
"""
def sub_rsa_encoding(pdb_id, num):
    sub_structure = open("sub-structure/data.csv").readlines()
    name_index = 0      # 行数
    sub_index = 3
    rsa_index = 4

    while name_index <= len(sub_structure) - 5:
        name = sub_structure[name_index]
        sub = sub_structure[sub_index]
        rsa = sub_structure[rsa_index]

        if name[-5:-1] == pdb_id:       # name索引行的最后四列如果是链名
            result_sub = sub_encoding(sub[:-1])     # 编码直到最后一列
            result_rsa = rsa_encoding(rsa[:-1])
        name_index += 5     # 读下一个链名
        sub_index += 5
        rsa_index += 5

    return result_sub[num] + result_rsa[num]
    # 这里的num框在列内，即在data中，序列往右走

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

def amino_lower2upper(amino_name):
    amino = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS',
             'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN',
             'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}

    return amino[amino_name]
"""
    功能：生成氨基酸的PSSM矩阵
    输入：（蛋白质的pdb_id，一个氨基酸的名称）
    输出：该氨基酸的PSSM矩阵
"""
def gen_pssm(pdb_id, num):
    reader = open("./pssm/" + pdb_id + '.pssm').readlines()
    for line_num, line in enumerate(reader):
        if line_num - 3 == num and line[:2] == '  ' and line[4:5] != ' ':       # 这里的num定位是行，在pssm中，序列往下走
            data = line[7:].split(' ')
            for num, i in enumerate(data):
                if i == '':
                    data.remove(i)      # 可能是剪切的意思
            
            feat_pssm = [int(i) for i in data[:20]]

    return feat_pssm


"""
    功能：生成氨基酸的标签
    输入：（蛋白质的pdb_id，一个氨基酸的名称）
    输出：氨基酸的标签
"""
def gen_label(pdb, pdb2, amino_name, protein_name, index, rna_name):
    # print(amino_name + ":" + protein_name + ":" + str(index))
    pdb1 = []
    for line_amino in pdb:
        if line_amino[:4] == 'ATOM' and line_amino[17:20] == amino_name and protein_name == line_amino[21:22]: # 此时为氨基酸的原子
            index2 = int(line_amino[22:26])     #残基标识码
            if index == index2:
                pdb1.append(line_amino)

    for line_amino in pdb1:
        # print(protein_name) # 此时为氨基酸的原子
        line2 = [i for i in line_amino.strip()[28:].split(' ') if i != '']
        amino_cord = np.array([float(i) for i in line2[0:3]])  # 得氨基酸原子的坐标
        for line_rna in pdb2:
            # print(rna_name) # 此时为RNA的原子
            line1 = [i for i in line_rna.strip()[28:].split(' ') if i != '']
            rna_cord = np.array([float(i) for i in line1[0:3]])  # 得RNA原子的坐标

            if np.sqrt(np.sum(np.square(amino_cord - rna_cord))) < 6:
                return 1

    return 0

"""
    功能：获得氨基酸的特征向量
    输入：存放序列的路径，要存在的路径
    输出：保存文件
"""
def gen_all_feature(data_path, save_path):
    data = pd.read_csv(data_path)
    pdb = data['Seq'].tolist()
    amino_feats = []
    amino_names = []        # 名字，特征，标签各成一列
    amino_labels = []
    count_amino = 0
    count_0 = 0
    count_1 = 0

    for line in tqdm(pdb):
        pdb_id = line[1:5]  # 蛋白质的pdb_id
        pdb_seq = line[8:]  # 蛋白质序列
        protein_name = line[6:7]            # 蛋白质链名称，例如 A
        rna_name = get_rna_name(pdb_id)

        pdb = open("./pdb/pdb" + pdb_id.lower() + ".ent", "r").readlines()
        pdb2 = []
        for line_amino in pdb:
            if line_amino[:4] == 'ATOM' and line_amino[17:19] == '  ' and rna_name == line_amino[21:22]:  # 此时为RNA的原子
                pdb2.append(line_amino)

        start = 0
        # 得到氨基酸的名称
        for num, amino_name in enumerate(pdb_seq):
            feat_11 = amino_encoding(amino_lower2upper(amino_name))         # 获得氨基酸理化和类别特征
            feat_sub_ras = sub_rsa_encoding(pdb_id, num)                    # 获得氨基酸sub和rsa特征
            feat_pssm = gen_pssm(pdb_id, num)                               # 获得氨基酸pssm特征

            if num == 0:
                for line_amino in pdb:
                    if line_amino[:4] == 'ATOM' and line_amino[17:20] == amino_lower2upper(amino_name) and protein_name == line_amino[21:22]:  # 此时为氨基酸的原子
                        start = int(line_amino[22:26])      # 开始位置不一的残基标识码
                        print("start:" + str(start))
                        break

            label = gen_label(pdb, pdb2, amino_lower2upper(amino_name), protein_name, num+start, rna_name)        # 获得氨基酸的标签

            amino_feats.append(feat_11 + feat_sub_ras + feat_pssm)          # 将氨基酸的所有特征合并
            amino_names.append(amino_name)                                  # 将氨基酸的名称合并
            amino_labels.append(label)                                      # 将氨基酸的标签合并
            all_feat = {"name": amino_names, "feat": amino_feats, "label": amino_labels}
        
            df_all_feat = pd.DataFrame(all_feat)
            df_all_feat.to_csv(save_path)
      
        #count_amino = count_amino + len(pdb_seq)
        #count_1 = count_1 + amino_labels.count(1)
        #count_0 = count_0 + amino_labels.count(0)

    return count_amino, count_1, count_0

val_count_amino, val_count_1, val_count_0 = gen_all_feature("dataset/val_seq.csv", "dataset/val.csv")

train_count_amino, train_count_1, train_count_0 = gen_all_feature("dataset/train_seq.csv", "dataset/train.csv")

test_count_amino, test_count_1, test_count_0 = gen_all_feature("dataset/test_seq.csv", "dataset/test.csv")

# print("氨基酸总数：", val_count_amino + train_count_amino + test_count_amino)
# print("标签为1：", val_count_1 + train_count_1 + test_count_1)
# print("标签为0：", val_count_0 + train_count_0 + test_count_0)


