# Static-Dynamic-Attention

## Prerequisites
* Python 3.6
* Pytorch 0.3.0
* CUDA 8.0

## Getting Started

The ```data``` folder contains only training sets, test sets, and word2vec files for the example. <br>
You can use your own dataset, but it must be consistent with the data format in the sample file, if you do not want to make changes to the code. <br>
The w2v file we use is stored as text, if your w2v file is binary storage, you can choose to convert the file or modify the code.
### train static model
```bash
python3 train.py  --type_model 1
```
### train dynamic model
```bash
python3 train.py  --type_model 0
```
### predict 
```bash
python3 predict.py --type_model 1 --weights static_parameters_IterEnd
or 
python3 predict.py --type_model 0 --weights dyna_parameters_IterEnd
```

You can use your own datasets with these commands ```--train_file```,```--test_file``` and ```--w2v_file```. <br>
```bash
python3 train.py  --type_model 0  --train_file your_path/your_train_data  --w2v_file your_path/your_w2v
python3 predict.py  --test_file your_path/your_test_data  --w2v_file your_path/your_w2v  --weights static_parameters_IterEnd 
```

## Others
###Hybrib Attention
A combination of static and dynamic attention, not presented in the coling2018 paper. Refer to ```/hybrid```
###PersonaChat Dataset
ConvAI2 persona-chat dataset:[http://convai.io/#personachat-convai2-dataset](ConvAI2dataset)