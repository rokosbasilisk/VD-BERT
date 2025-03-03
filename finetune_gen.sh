visdial_v=1.0
loss_type=mlm
split=train
len_vis_input=18
use_num_imgs=18
bs=30

checkpoint_output=v${visdial_v}_${loss_type}_gen

WORK_DIR=./

model_path=./saved_models/model_40.bin

python train.py \
    --output_dir checkpoints/${checkpoint_output} \
    --model_recover_path ${model_path} --len_vis_input ${len_vis_input}  \
    --do_train --new_segment_ids --enable_butd --visdial_v ${visdial_v} \
    --data_path ./data/corpus/saved_cwc_datasets/train-samples.pkl \
    --s2s_prob 1 --bi_prob 0 --loss_type ${loss_type} --max_pred 1 --neg_num 0 --multiple_neg 0 \
    --inc_full_hist 1  --max_len_hist_ques 200 --max_len_ans 10 --only_mask_ans 1 \
    --num_workers 3 --train_batch_size ${bs}  --use_num_imgs 1 --num_train_epochs 40 \
    --local_rank -1 --global_rank -1 --world_size 1



