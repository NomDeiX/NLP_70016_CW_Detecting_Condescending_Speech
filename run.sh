#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL     # required to send email notifcations
#SBATCH --mail-user=ms3319              # required to send email notifcations - please replace <your_username> with your college login name or email address
#SBATCH --job-name=train_model_no_augmentation_normal_sampling             # Name of the job

export PATH=/vol/bitbucket/${USER}/NLP_70016_CW_Detecting_Condescending_Speech/myenv/bin/:$PATH

# Activate your virtual environment
source activate

# Run your script
python main.py none
