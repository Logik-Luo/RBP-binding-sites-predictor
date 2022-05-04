from Bio.PDB import *
import pandas as pd

data = pd.read_csv('PDB.seq')       # PDB.seq就是PDBinter改名
pdb = data['Seq'].tolist()      # 作为列表返回

for line in pdb:
    # 定义pdb_id
    pdb_id = line[1:5]      # 每行锁定名字

    # 开始获取蛋白质pdb文件，并保存至protein文件夹
    pdbl = PDBList()        # 通过互联网访问PDB(例如下载结构)
    pdbl.retrieve_pdb_file(pdb_id, pdir='./pdb', file_format='pdb')     # 指定了文件保存位置
    parser = PDBParser(PERMISSIVE=True, QUIET=True)     # quiet应该设置false的，能屏蔽错误结构，但是permissive已经排除了部分异常

    # 如果pdb文件不存在，则跳过该pdb
    try:
        data = parser.get_structure(pdb_id, 'pdb' + pdb_id.lower() + '.ent')        # 返回model/chain/residue/atom
    except:
        pass
