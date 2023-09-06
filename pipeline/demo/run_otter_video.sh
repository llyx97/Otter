models=(cogvideo text2video-zero videofusion ground-truth)
devices=(3 4 3 4)

for i in 0 1
do
        model=${models[$i]}
        CUDA_VISIBLE_DEVICES=${devices[$i]} python otter_video_dataset.py \
                --eval_data_path ../../data/fetv_data.json \
                --videoqa_file ../../data/question_yesno_.json \
                --multi_round 1 \
                --return_logits true \
                --video_model $model    &
done