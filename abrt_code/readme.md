## Cumulative Abnormal Return Prediction
--------------
### Requirements
- python 3.6
- pytorch >= 0.4  


### Abbreviations
- TE: TEND-T
- THAN: TEND-C


### Load the pretrained models
The pretrained models are saved in the abrt_code/models directory.

**load TEND-T model**
> python main.py --gpu 0 --model TE  --resume /path/to/model

**load TEND-C model**
> python main.py --gpu 0 --model THAN --mode att --resume /path/to/model

### To train the model from scratch

> python main.py --gpu 0 --model TE

or 

> python main.py --gpu 0 --model THAN --mode att