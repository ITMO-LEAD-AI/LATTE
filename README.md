<<<<<<< HEAD
# LATTE for DETR and RT-DETR

This repository applies LATTE (Linearized Attention with Tunable Taylor
Expansion) to the decoder self-attention layers of pretrained DETR and
RT-DETR. Each linear layer is initialized from the original projections and
calibrated on a small COCO subset without full model retraining.

## Installation

The reference environment uses Python 3.10.19, PyTorch 2.10.0, TorchVision
0.25.0, and Transformers 5.2.0.

```bash
git clone https://github.com/your-account/LATTE.git
cd LATTE
conda env create -f environment.yml
conda activate latte
```

To use an existing compatible environment:

```bash
pip install -e .
```

## Dataset paths

The calibration and evaluation commands expect a local COCO 2017 structure:

```text
coco/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

Replace the placeholder paths in:

- `configs/rtdetr_debug.yaml`
- `configs/rtdetr_calibration.yaml`
- `configs/rtdetr_evaluation.yaml`
- `configs/detr_debug.yaml`
- `configs/detr_calibration.yaml`
- `configs/detr_evaluation.yaml`

## RT-DETR calibration

Start with the small one-layer debug run:

```bash
run_latte_rtdetr --config configs/rtdetr_debug.yaml
```

Then run the complete calibration:

```bash
run_latte_rtdetr --config configs/rtdetr_calibration.yaml
```

`run_latte` is a shorter alias for the same RT-DETR calibration command.

The pipeline performs the following steps:

1. Loads `PekingU/rtdetr_r50vd`.
2. Loads the configured COCO training subset.
3. Captures hidden states, positional embeddings, and original attention
   outputs for a small chunk of decoder layers.
4. Initializes LATTE from the corresponding pretrained projection weights.
5. Optimizes the selected parameters with cosine and MSE losses.
6. Saves calibrated states and final per-layer losses.

## DETR calibration

Start with the DETR debug configuration:

```bash
run_latte_detr --config configs/detr_debug.yaml
```

Then run the full DETR calibration:

```bash
run_latte_detr --config configs/detr_calibration.yaml
```

The DETR pipeline uses `facebook/detr-resnet-50` and has its own attention,
calibration and evaluation implementation, following the supplied original
DETR script. RT-DETR implementation files are unchanged from the working
RT-DETR-only release.

DETR deliberately uses `RTDetrImageProcessor("PekingU/rtdetr_r50vd")` for
calibration and `DetrImageProcessor("facebook/detr-resnet-50")` for evaluation,
as in the original script. Calibration uses decoder hidden states, omits
positional embeddings in the local optimization, and stores the training-average
cosine loss plus the final shuffled-pass MSE. Its attention preserves the original
direct head-output reshape and `ones_scale1` parameter name.

DETR checkpoints from the earlier combined repo are not equivalent and are
rejected. Recalibrate, or set `checkpoint:` in the evaluation YAML to your
original `DETR_LATTE_Repo.pth`; that original checkpoint format is supported.
Existing RT-DETR checkpoints remain supported.

This release does not silently skip unknown predicted labels. The earlier
`KeyError: 91` explanation was not established: the original code uses the same
strict label lookup. If it recurs, compare the original script and this release
on the same image and original checkpoint before changing postprocessing.

## Evaluation

RT-DETR:

```bash
eval_latte_rtdetr --config configs/rtdetr_evaluation.yaml
```

DETR:

```bash
eval_latte_detr --config configs/detr_evaluation.yaml
```

Layers are ranked using:

```text
score = cosine_weight × cosine_loss + mse_weight × mse_loss
```

The default configuration uses `cosine_weight: 1.0` and `mse_weight: 10.0`.
For every requested linearization depth, both evaluators report COCO AP and
AP50. Temporary COCO prediction data is deleted after each evaluation. The
only retained evaluation artifacts are:

```text
outputs/rtdetr_coco_results.json
outputs/detr_coco_results.json
```

Example result record:

```json
{
  "num_layers": 4,
  "linearized_layers": [0, 1, 3, 5],
  "metrics": {
    "AP@[IoU=0.50:0.95]": 0.0,
    "AP@[IoU=0.50]": 0.0
  }
}
```

## Use a calibrated checkpoint

RT-DETR:

```bash
python examples/linearize_rtdetr.py \
  --checkpoint checkpoints/rtdetr_r50vd_latte.pth \
  --num-layers 4 \
  --lambda-mse 10
```

DETR:

```bash
python examples/linearize_detr.py \
  --checkpoint checkpoints/detr_resnet50_latte.pth \
  --num-layers 4 \
  --lambda-mse 10
```

To specify exact decoder layers:

```bash
python examples/linearize_rtdetr.py \
  --checkpoint checkpoints/rtdetr_r50vd_latte.pth \
  --layers 0 1 3 5
```

Inspect the saved calibration losses:

```bash
python examples/inspect_checkpoint.py checkpoints/rtdetr_r50vd_latte.pth
```

## Repository structure

```text
LATTE/
├── latte/
│   ├── attention.py        # Unchanged RT-DETR LATTE attention
│   ├── detr_attention.py   # Original DETR attention
│   ├── detr_calibration.py # Original DETR optimizer and loss semantics
│   ├── calibration.py      # Activation capture and optimization
│   ├── detr_utils.py       # DETR layer creation and replacement
│   ├── rtdetr_utils.py     # Layer creation, selection, and replacement
│   ├── detr_pipeline.py    # DETR calibration command
│   ├── rtdetr_pipeline.py  # Calibration command
│   ├── detr_evaluator.py   # DETR COCO evaluation
│   └── rtdetr_evaluator.py # RT-DETR COCO evaluation
├── configs/
│   ├── detr_debug.yaml
│   ├── detr_calibration.yaml
│   ├── detr_evaluation.yaml
│   ├── rtdetr_debug.yaml
│   ├── rtdetr_calibration.yaml
│   └── rtdetr_evaluation.yaml
├── examples/
│   ├── linearize_rtdetr.py
│   ├── linearize_detr.py
│   └── inspect_checkpoint.py
├── checkpoints/
├── tests/
├── environment.yml
├── requirements.txt
└── setup.py
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

The unit tests do not download DETR, RT-DETR, or COCO.

## Citation

If you use this repository, please cite the LATTE ECCV 2026 paper. The final
proceedings BibTeX entry can be added here when available.

## License

The repository currently contains an MIT license. Confirm the intended license
with all authors or the owning organization before publication.
=======
# LATTE
# The code will be released soon
>>>>>>> fa2643b56e1abc8195fa0e22e53f80450d629852
