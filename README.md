# SimpleDH3
Code for "Simple End-to-end Deep Learning Model for CDR-H3 Loop Structure Prediction" paper accepted at Machine Learning for Structural Biology (MLSB) Workshop at NeurIPS 2021

#### <ins>NB! This repo is in the process, we will update instructions and the code in the beginning of 2022</ins>


### Setup development environment 
 
* Install requirements from `requirements.txt` 

### Data

The data can be created using script `data/create_data.py` by providing summary file from SAbDab.
Two files will be produced: with sequnces and with coordinates.

### Usage


Firstly, embeddings needs to be created:
```
cd modules
python3 embedding/data_to_embedding.py
```

To train model, run:
```
bash train.sh
```

To evaluate model, run:
```
bash predict_test.sh
```

