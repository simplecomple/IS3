config_setting='--config_file your_dir/.cache/huggingface/accelerate/your_yaml.yaml'
wandb_setting='--is_wandb True --wandb_project your_wandb_project --wandb_entity your_wandb_name' 
for i in {1..5}
do 
    accelerate launch $config_setting main_CL.py $wandb_setting \
    --exp_prefix ontonotes5_task6_base8_inc2 --cfg './config/ontonotes5_task6_base8_inc2/FT.yaml'\
    --backbone bert-base-cased --classifier Linear --training_epochs 5

    accelerate launch $config_setting main_CL.py $wandb_setting \
    --exp_prefix ontonotes5_task6_base8_inc2 --cfg './config/ontonotes5_task6_base8_inc2/SelfTrain.yaml'\
    --backbone bert-base-cased --classifier Linear --training_epochs 5

    accelerate launch $config_setting main_CL.py $wandb_setting \
    --exp_prefix ontonotes5_task6_base8_inc2 --cfg './config/ontonotes5_task6_base8_inc2/ExtendNER.yaml'\
    --backbone bert-base-cased --classifier CosineLinear --training_epochs 5

    accelerate launch $config_setting main_CL.py $wandb_setting \
    --exp_prefix ontonotes5_task6_base8_inc2 --cfg './config/ontonotes5_task6_base8_inc2/CFNER.yaml'\
    --backbone bert-base-cased --classifier CosineLinear --training_epochs 5

    accelerate launch $config_setting main_CL.py $wandb_setting \
    --exp_prefix ontonotes5_task6_base8_inc2 --cfg './config/ontonotes5_task6_base8_inc2/DLD.yaml'\
    --backbone bert-base-cased --classifier CosineLinear --training_epochs 5

    accelerate launch $config_setting main_CL.py $wandb_setting \
    --exp_prefix ontonotes5_task6_base8_inc2 --cfg './config/ontonotes5_task6_base8_inc2/RDP.yaml'\
    --backbone bert-base-cased --classifier CosineLinear --training_epochs 5

    accelerate launch $config_setting main_CL.py $wandb_setting \
    --exp_prefix ontonotes5_task6_base8_inc2 --cfg './config/ontonotes5_task6_base8_inc2/OCILNER.yaml'\
    --backbone bert-base-cased --classifier Linear --training_epochs 5

    accelerate launch $config_setting main_CL.py $wandb_setting \
    --exp_prefix ontonotes5_task6_base8_inc2 --cfg './config/ontonotes5_task6_base8_inc2/ICE_PLO.yaml'\
    --backbone bert-base-cased --classifier Linear --training_epochs 5

    accelerate launch $config_setting main_CL.py $wandb_setting \
    --exp_prefix ontonotes5_task6_base8_inc2 --cfg './config/ontonotes5_task6_base8_inc2/ICE_O.yaml'\
    --backbone bert-base-cased --classifier Linear --training_epochs 5

    accelerate launch $config_setting main_CL.py $wandb_setting \
    --exp_prefix ontonotes5_task6_base8_inc2 --cfg './config/ontonotes5_task6_base8_inc2/CPFD.yaml'\
    --backbone bert-base-cased --classifier CosineLinear --training_epochs 5
done
