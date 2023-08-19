models=(cogvideo text2video-zero videofusion ground-truth)
devices=(0 1 2 3)

for i in 0 1 2 3
do
        model=${models[$i]}
        CUDA_VISIBLE_DEVICES=${devices[$i]} python otter_video_dataset.py \
                --eval_data_path ../../data/fetv_data.json \
                --videoqa_file ../../data/question_yesno_300_399.json \
                --multi_round 5 \
                --video_model $model    &
done