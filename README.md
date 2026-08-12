# Reasoning-Aware Efficient QAT with Progressive Quantization and ASAM

This repository extends EfficientQAT for low-bit quantization-aware training of
Qwen3 models. The pipeline combines:

- reasoning-aware calibration data;
- block-wise reconstruction;
- group-wise progressive weight quantization from W8 to W2;
- Adaptive Sharpness-Aware Minimization (ASAM) in block-wise training;
- end-to-end ASAM optimization of quantization scales;
- packed W2G128 checkpoints for deployment and evaluation.

The primary experiments support Qwen3-1.7B, Qwen3-4B, and Qwen3-8B.

## Method

~~~text
Qwen3 FP16 checkpoint
        |
        v
sweep_0.8 calibration
  - 80% OpenThoughts reasoning samples
  - 20% FineWeb-Edu samples
        |
        v
Block-wise QAT
  - W8 -> W2 group-wise progressive quantization
  - ASAM + AdamW
  - FP block outputs as reconstruction targets
  - quantized upstream activations as inputs
        |
        v
Final target-only W2 reconstruction epoch
        |
        v
Packed W2G128 checkpoint
        |
        v
End-to-end QAT with ASAM
~~~

Progressive fake quantization uses:

~~~text
W_mix = W_8 + r * (W_2 - W_8)
~~~

The group-wise transition ratio is updated using the directional gradient. The
last block epoch fixes the ratio to one so that the training forward matches the
deployed W2 model.

## Repository structure

~~~text
.
├── block_main.py                 # Block-wise QAT entry point
├── e2e_main.py                   # End-to-end QAT entry point
├── datautil_block.py             # Calibration data and PPL evaluation
├── datautil_e2e.py               # End-to-end datasets
├── quantize/
│   ├── block_ap.py               # Block reconstruction loop
│   ├── fake_linear.py            # Progressive fake-quantized linear layer
│   ├── real_linear.py            # Packed low-bit linear layer
│   ├── quantizer.py              # Affine quantization
│   └── triton_utils/             # Low-bit Triton kernels
├── optim/                        # ASAM and related optimizers
├── model_transfer/               # Checkpoint conversion
├── script/                       # Full experiment launchers
└── test/                         # Quantizer diagnostics
~~~

## Environment

The tested reasoningqat Conda environment contains:

- PyTorch 2.3.1 with CUDA 12.1;
- Transformers 4.57.3;
- Accelerate 1.13.0.

Use the existing environment:

~~~bash
conda activate reasoningqat
~~~

Or create it from the repository specification:

~~~bash
conda env create -f environment.yml
conda activate reasoningqat
~~~

Datasets are loaded through Hugging Face Datasets. The machine therefore needs
access to the required datasets and enough cache storage.

## Full Qwen3 experiments

Each launcher runs block-wise QAT and then starts end-to-end QAT after the packed
block checkpoint has been saved successfully.

~~~bash
# Qwen3-1.7B
GPU=0 ./script/qwen3-1.7b-full.sh

# Qwen3-4B
GPU=0 ./script/qwen3-4b-full.sh

# Qwen3-8B
GPU=0 ./script/qwen3-8b-full.sh
~~~

Default model locations are:

~~~text
/share/Qwen3-1.7B
/share/Qwen3-4B
/share/Qwen3-8B
~~~

Override any model location without editing a script:

~~~bash
GPU=0 MODEL_PATH=/path/to/Qwen3-4B ./script/qwen3-4b-full.sh
~~~

## Default configuration

### Block-wise stage

| Parameter | Default |
|---|---:|
| Calibration dataset | sweep_0.8 |
| Training/validation samples | 4096 / 64 |
| Sequence length | 2048 |
| Source and target bits | W8 -> W2 |
| Group size | 128 |
| Epochs per block | 10 |
| Weight learning rate | 2e-5 |
| Quantizer learning rate | 1e-4 |
| Progressive ratio shape | group-wise |
| Progressive rho | 0.05 |
| Final target-only epochs | 1 |
| ASAM rho | 0.05 |

### End-to-end stage

| Parameter | Default |
|---|---:|
| Dataset | RedPajama |
| Context length | 2048 |
| Maximum optimizer steps | 10000 |
| Per-device batch size | 1 |
| Gradient accumulation | 16 |
| Learning rate | 2e-5 |
| ASAM rho | 0.05 |
| Precision | BF16 |
| Checkpoint interval | 250 steps |

## Running one stage

Run block-wise QAT only:

~~~bash
GPU=0 SKIP_E2E=1 ./script/qwen3-4b-full.sh
~~~

Run end-to-end QAT from an existing packed checkpoint:

~~~bash
GPU=0 \
SKIP_BLOCK=1 \
BLOCK_SAVE_DIR=/path/to/qwen3-4b-w2g128-block-checkpoint \
./script/qwen3-4b-full.sh
~~~

Resume end-to-end training:

~~~bash
GPU=0 \
SKIP_BLOCK=1 \
BLOCK_SAVE_DIR=/path/to/qwen3-4b-w2g128-block-checkpoint \
E2E_RESUME_FROM_CHECKPOINT=/path/to/e2e/checkpoint-1000 \
./script/qwen3-4b-full.sh
~~~

## Custom experiments

The common launcher at script/qwen3-full-pipeline.sh accepts environment
variables. For example:

~~~bash
GPU=1 \
TRAIN_SIZE=8192 \
VAL_SIZE=128 \
