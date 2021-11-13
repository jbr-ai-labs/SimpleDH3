#!/bin/sh

rm -rf ../../results/test_results_rot

python3 predict_test_data.py --model simple_coordinates \
--input_size 1024 \
--output_size 9 \
--hidden_dim 256 \
--n_layers 1 \
--saved_model_path ../results/best_model.sav \
--data_path_seq ../data/cdr_h3_seq_n \
--data_path_coord ../results/coord_data_emb \
--test_data_ids_path ../results/test_dataset_ids.txt \
--data_path_gdt_ts ../results/gdt_ts.txt \
--data_path_tm_score ../results/tm_score.txt \
--data_path_rmsd ../results/rmsd_score.txt \
--use_corrector False \
--corrector rotate_normal
# rotate_normal distances
