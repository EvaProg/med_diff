torchrun --nnodes=1 --nproc_per_node=1 train.py \
    --model SiT-S/2 \
    --dataset nih_chestxray \
    --data-path /data/evan/NIH_CXR \
    --ckpt /home/evan/SiT/pretrained_models/SiT-S-2-256.pt \
    --finetune \
    --global-batch-size 32 \
    --epochs 100 \
    --ckpt-every 5000 \
    --sample-every 5000