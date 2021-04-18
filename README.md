# NLP Assignment

#### Evaluation on Colab

#### Evaluation on Server

Setup

```
conda create -n assignment python=3.7
conda activate assignment
python -m pip install -r requirements.txt
```
Train/Evaluate
```
shell run.sh <mode> <problem> <model>
```
With:
* <mode> is  <train|evaluation>
* <problem> is <sentiment|sequence>
* <problem> is <sentiment> , avaiable models are: <bert|rnn|lstm|gru|cnn|att_add|att_mul|att_dot|att_bilstm_add>
* <problem> is <sequence> , avaiable models are: <bert|bilstm>

Example:
```
shell run.sh train sentiment lstm
```
#### Get pretrain of all model

```
gdown
tar xz
```