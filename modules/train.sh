#!/bin/sh
rm ../results/distance_loss ../results/torsion_loss ../results/coord_loss ../results/angles_loss
python3 main.py --model simple_coordinates \
--model_input_size=1024 \
--model_output_size=9 \
--model_hidden_dim=256 \
--learning_rate=0.001 \
--n_layers=1 \
--batch_size=64 \
--epochs=2000 \
--save_prefix=best_model \
--early_stopping_steps=500 \
--cross_valid=False \ 
--no_test=True \
