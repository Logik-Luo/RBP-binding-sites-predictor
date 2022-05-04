任务：在已知RNA-protein配对中获取结合位点，在‘generate your feature vector’文件获取详细特征工程构建流程，然后您可以直接调用我们的‘comparison of DL and ML’中的任一模型

model文件夹存放的是深度学习预实验，有cnn（有/无池化层）,gru,lstm和transformer五个模型，只是看个大概，这个预实验不怎么影响

'comparison of DL and ML'是正式训练模型，val，test，train数据集经过excel处理，合并到了一张表，但没有实际数据改动

要构建您的数据集，请直接运行gen_feature_V2，在那之前需要
·生成您的序列的PSSM矩阵；
·将序列上传到PaleAle4.0获得二级结构和溶剂可及性；
·利用python的pdb工具包获取您的序列的PDB原子坐标文件

关于AutoML平台的使用，请参阅https://github.com/mljar/mljar-supervised

关于PDBinter的源数据，来自http://doc.aporc.org/wiki/PRNAinter

请在链接：https://pan.baidu.com/s/1BI4rraGAEnGltTunn9iLwg（提取码：13vc） 
获取'fasta''pdb''pssm''uniref'文件夹，置入文件夹'RBP-binding-sites-predictor'下


English:
·Task: obtain the binding site in the known RNA protein pairing, obtain the detailed feature engineering construction process in the 'generate your feature vector' file, and then you can directly call any of our 'comparison of DL and ML' models to start prediction

·The 'model' folder stores the pre experiment of deep learning

·'comparison of DL and ml' is a formal training model. val, test and train data sets are processed by Excel and merged into one table

·To build your dataset, run 'gen directly_ feature_ V2', before that:

	Generate PSSM matrix of your sequence using pssm.py;
	Upload the sequence to paleale4.0 to obtain secondary structure and solvent accessibility;
	Use Python's PDB toolkit to obtain the PDB atomic coordinate file of your sequence

·For the use of the AutoML platform, see https://github.com/mljar/mljar-supervised

·About the source data of PDBinter, see http://doc.aporc.org/wiki/PRNAinter

Please link: https://pan.baidu.com/s/1BI4rraGAEnGltTunn9iLwg (extraction code: 13VC)
Get the 'FASTA' 'PDB' 'PSSM' 'uniref' folder and put it under the folder 'RBP binding sites predictor'
