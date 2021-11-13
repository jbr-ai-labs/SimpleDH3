#!/bin/sh
python3 cross-valid-new.py --model simple_coordinates \
--model_input_size=1024 \
--model_output_size=9 \
--model_hidden_dim=256 \
--learning_rate=0.001 \
--n_layers=1 \
--batch_size=64 \
--epochs=2000 \
--save_prefix=best_model \
--iterations_cv=10 \
--data_path_seq ../data/cdr_h3_seq_normal \
--data_path_coord ../results/coord_data_emb \
--test_data_ids_path ../results/test_dataset_ids.txt
--use_corrector True \
--corrector rotate_normal

# rotate_normal distances
