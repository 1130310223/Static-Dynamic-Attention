type1文件夹为第1种hybrid模型，向量加
type1文件夹为第2种hybrid模型，向量拼接

里面的len9文件夹 包含了输入上文长度为9的情况
hybrid_len9_t1_predict.py和hybrid_len9_t1_train.py里的参数max_context_size可以改变输入上文长度

运行时还需修改hybrid_len9_t1_predict.py和hybrid_len9_t1_train.py里的训练语料和w2v文件路径
其余参数无需修改

训练时运行
nohup python3 -u hybrid_len9_t1_train.py >log.txt 2>&1 & 即可

预测时，可以修改hybrid_len9_t1_predict.py中171，172行的输出文件路径，运行
python3 -u hybrid_len9_t1_predict.py --weights seq2seq_parameters_IterEnd即可

