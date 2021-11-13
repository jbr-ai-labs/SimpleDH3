#!/bin/sh
targets_path="../../results/test_results"
preds_path="../../results/test_results"
preds_corrected_path1="../../results/test_results_dist"
preds_corrected_path2="../../results/test_results_rot"
frames_path="../../results/frames"
videos_path="../../results/videos"

loop_name="6ct7"
python3 make_video.py --targets_path "$(echo $targets_path)/$(echo $loop_name)_target.pt" \
--preds_path "$(echo $preds_path)/$(echo $loop_name)_pred.pt" \
--frames_path "$(echo $frames_path)" \
--videos_path "$(echo $videos_path)" \
--video_name "$(echo $loop_name)"

#--preds_corrected_path2 "$(echo $preds_corrected_path2)/$(echo $loop_name)_pred.pt" \
#--preds_corrected_path1 "$(echo $preds_corrected_path1)/$(echo $loop_name)_pred.pt" \

