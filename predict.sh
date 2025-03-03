visdial_v=1.0
loss_type=mlm
split=train
len_vis_input=18
use_num_imgs=18
bs=30

checkpoint_output=v${visdial_v}_${loss_type}_gen

WORK_DIR=./

model_path=./saved_models/model_40.bin

python predict.py \
    --model_recover_path ${model_path} --len_vis_input ${len_vis_input}  \
    --new_segment_ids \
    --data_path ./data/corpus/saved_cwc_datasets/train-samples.pkl \
    --max_pred 1 --neg_num 0 --multiple_neg 0 \
    --inc_full_hist 1  --max_len_hist_ques 200 --max_len_ans 10 --only_mask_ans 1 \
    --num_workers 3  --use_num_imgs 1 \
    --local_rank -1 --global_rank -1 --world_size 1



