# 1 GPU
CUDA_VISIBLE_DEVICES=1 python train.py --opt_path option_ovqe.yml

# 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12354 train.py --opt_path option_ovqe.yml

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 train.py --opt_path option_ovqe.yml





