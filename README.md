# ASR Guided Tacotron

Use LAS model to enhance the performance of tacotron, especially at the lack of the speaker labels.

## Prerequisites
1. Python packages:
    - Python 3.4 or higher
    - Tensorflow r1.8 or higher
    - Numpy

2. Clone this repository:
```shell=
https://github.com/HappyBall/asr_guided_tacotron.git
```

## Dataset

1. multi-speaker dataset mixed from VCTK and LibriSpeech [Download](http://speech.ee.ntu.edu.tw/~yangchiyi/asr_guided_tacotron_dataset.tgz)

## Usage

Before training the whole model includes Tacotron model and LAS model, you need to pre-train both models respectively.

Also, you can download our pre-trained Tacotron models
[Here](http://speech.ee.ntu.edu.tw/~yangchiyi/pretrained_tacotron.tgz) and LAS models [Here](http://speech.ee.ntu.edu.tw/~yangchiyi/pretrained_las.tgz) to skip the pre-train procedures.

### Pre-train LAS model

1. Download the dataset and use `transcript_training_las.txt` as `--train_data_name` in the `hyperparams.py`.

2. Set up the correct path of the dataset and other hyperparameters in `hyperparams.py`.

Run:
`python train_las.py --keep_train False`

Parameter `--keep_train` determines either start a new training or continue
training with the existed model which the path should be correctly set up in `hyperparams.py`.

### Pre-train Tacotron model

1. Download the dataset and use `transcript_training_tacotron_seqlen50.txt` as `--train_data_name` in the `hyperparams.py`.

2. Set up the correct path of the dataset and other hyperparameters in `hyperparams.py`.

Run:
`python train_origin_tacotron.py --keep_train False`

### Train Tacotron model with the guidance of LAS

1. Download the dataset and use `transcript_training_tacotron_seqlen50.txt` as `--train_data_name` in the `hyperparams.py`.

2. Set up the correct paths of the pre-trained models and other hyperparameters in `hyperparams.py`.

Run:
`python train_tacotron.py --keep_train True`

The program will automatically load the pre-trained models and start training with the guidance of ASR.

While training, only Tacotron model will be updated.

### Synthesis

1. Set up the correct path of the existed model and the reference audio file which you want to encode prosody from in `hyperparams.py`.

2. Add English input sequences you want to synthesize into `test_sentenses.txt`.

Run:
`python synthesize.py`

### Hyperparameters of the hyperparams.py
`--data`: the path of the data directory which contains the wav files  
`--prepro_path`: the path of the preprocessed data directory  
`--test_data`: the path of the text file which contains input text sequences to synthesize speech  
`--ref_wavfile`: the path of the reference audio file which you want to encode prosody from  
`--train_data_name`: the transcription file name using for training  
`--taco_logdir`: the path of the directory to save or load Tacotron models  
`--taco_logfile`: the path of Tacotron training log file  
`--las_logdir`: the path of the directory to save or load LAS models  
`--sampledir`: the path of the directory to save speech files when synthesizing  
`--attention_mechanism`: choose the type of attention mechanism (original or dot)  
`taco_consis_weight`: determines how much would the attention consistency influence the loss function (decimal number from 0 to 1)  
`--n_iter`: iteration number of Griffin Lim algorithm  
`--lr`: initial learning rate  

## Files in this project

### Folders:
`las/`: modules and networks of las model  
`tacotron/`: modules and networks of tacotron model

### Files:
`data_load.py`: data loader for training data and testing data  
`evaluate_las.py`: calculate the character error rate (CER) of las model  
`graph.py`: define model graph  
`hyperparams.py`: set up training hyperparameters and directory for saving models  
`prepro.py`: preprocess data  
`synthesize.py`: synthesize speech conditioned on input text sequences in the test sentence file and the reference audio file  
`train_las.py`: pre-train LAS models  
`train_origin_tacotron.py`: pre-train Tacotron models without guidance of ASR  
`train_tacotron.py`: train to improve existed Tacotron models with guidance of ASR  
