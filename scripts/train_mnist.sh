

CUDA_VISIBLE_DEVICES=1 python train_mnist.py --use_adversarial --adv_epsilon 0.01 --epochs 20 --target_acc 0.92