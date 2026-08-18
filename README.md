## Training/Finetuning

**Fine-tuning on CheXpert.** To fine-tune a pretrained SiT-S/2 checkpoint, conditioned on CheXpert pathology
labels, on [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) (`--data-path` should point
at the folder that directly contains `train.csv`, e.g. the extracted `CheXpert-v1.0-small` directory):

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-S/2 --dataset chexpert \
    --data-path /path/to/CheXpert-v1.0-small --ckpt /path/to/SiT-S-2-pretrained.pt --finetune
```

`--dataset chexpert` forces `--num-classes 14`, one class per pathology column in CheXpert's `train.csv`
(see `datasets/chexpert_dataset.py:CHEXPERT_CLASSES`). Each image's multi-label row is collapsed to a single
class for conditioning: the first pathology column with a confident positive (`1.0`) wins, falling back to
"No Finding" if none are positive; images with no confident positive label at all (all blank/uncertain/`0.0`)
are dropped. Classifier-free guidance is on by default (`--class-dropout-prob 0.1`, `--cfg-scale` at sample
time). `--finetune` loads only the backbone weights from `--ckpt` into a freshly-initialized optimizer,
skipping any mismatched tensors (e.g. the class-label embedding table, which is always reinitialized since
CheXpert's 14 classes never match the source checkpoint's class count) instead of restoring the exact
training state as plain `--ckpt` resume does.

**Sanity-checking the pipeline on a smaller dataset.** CheXpert is large (~11GB+ of images plus the license
gate), so before pointing at it you can validate the exact same fine-tuning pipeline against the much smaller
[NIH ChestX-ray14 Kaggle mirror](https://www.kaggle.com/datasets/nih-chest-xrays/data)
(`--data-path` should point at the folder you extracted the Kaggle download into, e.g. containing
`images_001/`, `images_002/`, ... — the loader globs for every `.png` under it, so any of that mirror's
folder layouts work):

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-S/2 --dataset nih_chestxray \
    --data-path /path/to/nih-chest-xrays --ckpt /path/to/SiT-S-2-pretrained.pt --finetune

## Evaluation (FID, Inception Score, etc.)

We include a [`sample_ddp.py`](sample_ddp.py) script which samples a large number of images from a SiT model in parallel. This script 
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 50K images from our pre-trained SiT-XL/2 model over `N` GPUs under default ODE sampler settings, run:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py ODE --model SiT-XL/2 --num-fid-samples 50000
```

**Likelihood.** Likelihood evaluation is supported. To calculate likelihood, you can add the `--likelihood` flag to ODE sampler:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py ODE --model SiT-XL/2 --likelihood
```

Notice that only under ODE sampler likelihood can be calculated; see [`sample_ddp.py`](sample_ddp.py) for more details and settings. 

## Chexpert integration issues

* SiT is for RGB 3-channel, medical is grayscale 1-channel
* started with no conditioning; now single-label pathology-conditional (see `datasets/chexpert_dataset.py`)

## Evaluation Metrics 

FID, FRD, RadFID etc. according to https://github.com/mazurowski-lab/medical-image-similarity-metrics