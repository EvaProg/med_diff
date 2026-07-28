torchrun --nnodes=1 --nproc_per_node=1 sample_ddp.py ODE \
    --model SiT-S/2 \
    --ckpt /home/evan/SiT/results/002-SiT-S-2-Linear-velocity-None/checkpoints/0045000.pt \
    --num-classes 1 \
    --class-dropout-prob 0.0 \
    --num-fid-samples 10000 \
    --per-proc-batch-size 32 \
    --sample-dir /data/evan/MEDICAL_GENERATED_DATA/CHEXPERT_CXR_S2_10E