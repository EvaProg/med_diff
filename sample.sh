torchrun --nnodes=1 --nproc_per_node=1 sample_ddp.py ODE \
    --model SiT-S/2 \
    --ckpt /home/evan/SiT/results/003-SiT-S-2-Linear-velocity-None/checkpoints/0014000.pt \
    --num-classes 1 \
    --class-dropout-prob 0.0 \
    --num-fid-samples 5000 \
    --per-proc-batch-size 32 \
    --sample-dir /data/evan/MEDICAL_GENERATED_DATA/NIH_CXR_S2_100E