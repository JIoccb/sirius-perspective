# Document Perspective Keypoints

UNet + integral regression pipeline for predicting four document corners and a document mask. The predicted normalized keypoints are intended for perspective correction with `cv2.getPerspectiveTransform` / `cv2.warpPerspective`.

## Environment

Use Python 3.11. Install PyTorch separately for the local GPU, then install the project dependencies:

```powershell
pip install -r requirements-torch-cu126.txt
pip install -r requirements.txt
```

If your CUDA driver does not support CUDA 12.6 wheels, install a matching PyTorch build from the official PyTorch selector and then run only:

```powershell
pip install -r requirements.txt
```

## Expected Data Layout

The default training config expects a generated synthetic dataset in the repository root:

```text
synthetic_dataset/
  annotations.csv
  images/
  masks/
```

`annotations.csv` should contain at least:

- `filepath`: image path, absolute or relative to the CSV directory
- `coords`: four normalized `(x, y)` document corners
- `visibility`: optional keypoint visibility flags

When `annotations.csv` is present, `src/train.py` creates `df_train.csv`, `df_valid.csv`, and `df_test.csv` with image paths, mask paths, and keypoint targets.

## Synthetic Sample

From the repository root, generate a 30-image smoke-test dataset:

```powershell
python sirius-students-perspective/synth-gen/run_synthetic_sample.py
```

The script uses clean WarpDoc renders from `raw_dataset/WarpDoc/WarpDoc/digital/perspective`, COCO backgrounds from `raw_dataset/val2014/val2014`, and writes to `synthetic_dataset/`.

## Training

```powershell
$env:PYTHONPATH='.'
python src/train.py
```

Default local outputs:

- checkpoints: `checkpoints/`
- experiment logs: `runs/`
- predictions: `outputs/`

For an RTX 3060, start with the default local `batch_size: 8` in `configs/config.yaml` and reduce it if CUDA memory is tight.

## Inference

Set `real_path` in `configs/config.yaml` to an annotations CSV with a `filepath` column, then run:

```powershell
$env:PYTHONPATH='.'
python src/test.py
```

The current inference script saves masks/RLE and normalized keypoints. The perspective rectification step is still a TODO.
