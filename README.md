## Usage

### Overview
```
.
├── main_CL.py                      # This this the python file to be executed for running all experiments
├── utils                               # This folder contains all basic files for incremental learning 
│   ├── backbone.py                     # This file loads backbone models from the transformers library
│   ├── buffer.py                       # This file defines the replay buffer
│   ├── classifier.py                   # This file loads Linear/CosineLinear classifiers
│   ├── wrapmodel.py                    # This file wrap the model for using DeepSpeed with accelerate
│   ├── dataformat_preprocess.py        # This file preprocess the raw datasets to the continual learning dataset
│   ├── dataloader.py                   # This file prepare the input for languge models
│   ├── dataset.py                      # This file defines the format for different datasets for continual learning
│   ├── download_backbones.py           # This file downloads models in advance to avoid network problem.
│   ├── evaluation.py                   # This file defines the evaluation process for various tasks
│   ├── factory.py                      # This file loads the various models from the ./models folder
│   ├── logger.py                       # This file defines the logger
│   ├── metric.py                       # This file defines the evaluation metric for continual learning
│   ├── optimizer.py                    # This file defines the optimizer for different models
│   └── config.py                       # This file defines general parameters and settings for the experiments
├── config                          # This folder contains the hyper-parameters for each methods in each datasets
├── dataset                         # This folder contains datasets for continual learning
├── models                          # This folder contains models for continual learning
└── experiments                     # This folder contains log data for each run                 
```

### Quick Start

#### Step 1: prepare the environment
```
pip install -r requirement.txt
```

#### Step 2: prepare the dataset
Check the *support_dataset_list* in *utils/dataformat_preprocess.py* and select the dataset you want for experiment.

Next, exceute the file *utils/dataformat_preprocess.py* to convert the raw dataset to the data for continual learning.
This process will create a new target folder *dataset/{dataset-for-continual-learning-name}*.
In the target folder, two json files *continual_data.json* and *continual_config.json* will be saved.
For example, you can prepare ontonotes5 dataset by running
```
python utils/dataformat_preprocess.py --dataset ontonotes5 --seed 1 --base_task_entity 8 --incremental_task_entity 2 --seen_all_labels False
``` 
The program will create a target folder *dataset/ontonotes5_task6_base8_inc2*.
We note that fixing the random seed enables that exctaly the same datasets can be generated on different devices.
Finally, the post-precessed dataset *ontonotes5_task6_base8_inc2* are ready for continual learning!


### Step 3: execute the main_CL.py
For example, you can run Finetune method on ontonotes5_task6 dataset with bert-base-cased using the following command:

If you want to use accelerate for data/model parallel (see [here](https://huggingface.co/docs/accelerate/quicktour) for more help):
```
accelerate launch --config_file {your-accelerate-config-file} main_CL.py --is_wandb True --wandb_project {your-project-name} --wandb_entity {your-entity-name} --exp_prefix {your-experiment-name} --cfg './config/ontonotes5_task6_base8_inc2/FT.yaml' --backbone bert-base-cased --classifier Linear --training_epochs 5 
```
or
```
bash shell/IS3/ontonotes5_task6_base8_inc2.sh
```



