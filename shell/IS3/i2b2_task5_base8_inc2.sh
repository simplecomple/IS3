config_setting='--config_file your_dir/.cache/huggingface/accelerate/your_yaml.yaml'
wandb_setting='--is_wandb True --wandb_project your_wandb_project --wandb_entity your_wandb_name' 
for i in {1..1}
do 
    accelerate launch $config_setting main_CL.py $wandb_setting \
    --exp_prefix i2b2_task5_base8_inc2 --cfg './config/i2b2_task5_base8_inc2/IS3.yaml'\
    --backbone bert-base-cased --classifier Linear --training_epochs 5
done
