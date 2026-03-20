# !/bin/bash

python train_gtsrb.py --epochs 30 --lr 0.005 &&
CUDA_VISIBLE_DEVICES=0 python train_gtsrb_trades.py \
  --adv_epsilon 0.01 --adv_alpha 0.003 --adv_steps 10 --beta 6.0 \
  --epochs 60 --lr 0.001 --batch_size 128 --use_scheduler \
  --checkpoint checkpoints/gtsrb/gtsrb_cnn.pt --target_acc 0.89