## Document-level Stance Detection
---

### Requirements
- python 3.6
- pytorch >= 0.4
- pretrained word vectors: [glove.6B.50d.txt](http://nlp.stanford.edu/data/glove.6B.zip)


### Abbreviations
- TE: TEND-T
- THAN: TEND-C
- CE: Conditional Encoding
- HL: Word-to-word Alignment
- HN: TD-HN, Hierarchical RNN
- HAN: TD-HAN, Hierarchical Attention Neural Network



### Load the pretrained models

To run the code, you have to: 

1. download the glove vectors [glove.6B.50d.txt](http://nlp.stanford.edu/data/glove.6B.zip)
2. replace the line 111 `/path/to/GloVe/vector` in stance_code/main.py with your new file path.


The pretrained models are saved in the stance_code/models directory. 

> python main.py --gpu 0 --model model_abbreviation --resume /path/to/model

for example, to load the pretrained TEND-T model from models/ directory:

> python main.py --gpu 0 --model TE  --resume /path/to/model


### To train the model from scratch

> python main.py --gpu 0 --model model_abbreviation