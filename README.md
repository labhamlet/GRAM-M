# GRAM-M

This repository hosts the code for training GRAM-M on AudioSet with naturalistic scenes. As well as the listen-eval-kit, which is basically fork of the hear-eval-kit with added functionality for executing sound localization tasks. We utilize pytorch-lightning and hydra with tensorboard logging to make the framework as extensible as possible for hyperparameter tuning.


## Installation

We only tested this repository using python 3.10 with torch 2.1.2. 

To create a running repository follow the steps below.


### Training
```bash
conda create -n gram-m python=3.10 -y
conda activate gram-m

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# install gram-m specific requirements

pip install -r requirements.txt

# install causal-conv1d
pip install git+https://github.com/Dao-AILab/causal-conv1d.git@v1.1.3.post1

# install mamba-ssm
pip install git+https://github.com/state-spaces/mamba.git@v1.1.3.post1
``` 


### Evaluation
For the evaluation, we use a different specification of requirements.

```bash
conda create -n gram-m-eval python=3.10 -y
conda activate gram-m-eval

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# install gram-m-eval specific requirements

pip install -r requirements_eval.txt

# install causal-conv1d
pip install git+https://github.com/Dao-AILab/causal-conv1d.git@v1.1.3.post1

# install mamba-ssm
pip install git+https://github.com/state-spaces/mamba.git@v1.1.3.post1
``` 

## Running the training 

To run the training, we use the train.py file.

GRAM-M-Time model can be trained using

```bash
python3 train.py data=audioset data.sr=32000 patching=time data.mask_patch=80 trainer.batch_size=32 trainer.steps=200000 
``` 

GRAM-M-Patch model can be trained using

```bash
python3 train.py data=audioset data.sr=32000 patching=frame data.mask_patch=100 trainer.batch_size=32 trainer.steps=200000 
``` 

On A100 GPU training should take around ~72 hours.

This saves the models and the tensorboard logs on cfg.save_dir.


## Running the down stream evaluation on HEAR benchmark


### Get 32000 Hz data from hear
* Follow https://hearbenchmark.com/hear-tasks.html to get data. By default, data on HEAR's zenodo page is 48000 Hz.
* We recommend downloading data directly from HEAR's [GCS bucket](gs://hear2021-archive/tasks/), where you can find preprocessed 32000 Hz data.
* Extract all the files to a folder `$TASKS_DIR`

### Get pretrained weights

* Pre-trained can be downloaded from XXX
* Download the entire folder and export that folder as `$MODEL_DIR`

### Extract features and execute downstream experiments

```shell
cd listen-eval-kit
MODEL_DIR=/path/to/pretrained_weights

embeddings_dir=/embeddings/to/save

tasks_dir=$TASKS_DIR
task_name=dcase2016_task2-hear2021-full

weights=$MODEL_DIR
model_name=hear_configs.GRAMM
strategy=raw
use_mwmae_decoder=true
in_channels=2
model_options="{\"strategy\": \"$strategy\",\"use_mwmae_decoder\": \"$use_mwmae_decoder\", \"in_channels\": $in_channels}"

#This extracts the features
python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir --model-options "$model_options" --model $weights 

# This runs the experiments on the task_name
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name
```


## Running the naturalistic down stream evaluation on Nat-HEAR benchmark

### Get 32000 Hz data from Nat-HEAR
* We provide the Nat-HEAR on the mirror XXX.
* Extract all the files to a folder `$NATURALISTIC_TASKS_DIR`

### Get pretrained weights

* Pre-trained can be downloaded from XXX
* Download the entire folder and export that folder as `$MODEL_DIR`

### Extract features and execute downstream experiments

```shell
cd listen-eval-kit
MODEL_DIR=/path/to/pretrained_weights

embeddings_dir=/embeddings/to/save

tasks_dir=$NATURALISTIC_TASKS_DIR
task_name=dcase2016_task2-hear2021-full

weights=$MODEL_DIR
model_name=hear_configs.GRAMM
strategy=raw
use_mwmae_decoder=true
in_channels=2
model_options="{\"strategy\": \"$strategy\",\"use_mwmae_decoder\": \"$use_mwmae_decoder\", \"in_channels\": $in_channels}"

#This extracts the features
python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir --model-options "$model_options" --model $weights 

# This runs the experiments on the task_name
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name
```



## Running the naturalistic sound localization on Nat-HEAR benchmark

### Get 32000 Hz data from Nat-HEAR-localization
* We provide the Nat-HEAR-localization on the mirror XXX.
* Extract all the files to a folder `$LOCALIZATION_TASKS_DIR`

### Get pretrained weights

* Pre-trained can be downloaded from XXX
* Download the entire folder and export that folder as `$MODEL_DIR`


```shell
cd listen-eval-kit
MODEL_DIR=/path/to/pretrained_weights

embeddings_dir=/embeddings/to/save

tasks_dir=$NATURALISTIC_TASKS_DIR
task_name=dcase2016_task2-hear2021-full

weights=$MODEL_DIR
model_name=hear_configs.GRAMM
strategy=mean
use_mwmae_decoder=true
in_channels=2
model_options="{\"strategy\": \"$strategy\",\"use_mwmae_decoder\": \"$use_mwmae_decoder\", \"in_channels\": $in_channels}"

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir --model-options "$model_options" --model $weights 
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name --localization cartesian-regression
```




