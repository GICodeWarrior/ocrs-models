# ocrs-models

This project contains tools for training PyTorch models for use with the
[**Ocrs**](https://github.com/robertknight/ocrs/) OCR engine.

## About the models

The ocrs engine splits text detection and recognition into three phases, each
of which corresponds to a different model in this repository:

1. **Text detection**: This is a semantic segmentation model which classifies
   each pixel in a greyscale input image as text/non-text. Consumers then
   post-process clusters of text pixels to get oriented bounding boxes for
   words.
2. **Layout analysis (VERY WIP)**: This is a graph model which takes word
   bounding boxes as input nodes and classifies each node's relation to nearby
   nodes (eg. start / middle / end of line)
3. **Text recognition**: This is a CRNN model that takes a greyscale image of a
   text line as input and returns a sequence of characters.

All models can be exported to ONNX for downstream use.

## Datasets

The models are trained exclusively on datasets which are a) open and b) have non-restrictive licenses. This currently includes:
- [HierText](https://github.com/google-research-datasets/hiertext) (CC-BY-SA 4.0)

## Pre-trained models

Pre-trained models are available from [Hugging
Face](https://huggingface.co/robertknight/ocrs) as PyTorch checkpoints,
[ONNX](https://onnx.ai) and [RTen](https://github.com/robertknight/rten) models.

## Training custom models

See the [Training guide](docs/training.md) for a walk-through of the process to
train models from scratch or fine-tune existing models.

## Custom font finetuning

poetry run python -m ocrs_models.train_rec synthetic_font . --font-ttf <fontpath> --checkpoint text-rec-checkpoint-s52qdbqt.pt --batch-size 2048 --max-epochs 5 --finetune --lr 3e-4
TORCH_LOGS=recompiles poetry run python -m ocrs_models.train_rec synthetic_font . --font-ttf ~/foxhole/Renner-Book.ttf --checkpoint text-rec-checkpoint-s52qdbqt.pt --batch-size 1024 --max-epochs 5 --finetune --lr 3e-4 --compile --tf32

poetry run python -m ocrs_models.train_rec synthetic_font .   --font-ttf ~/foxhole/Renner-Book.ttf   --checkpoint text-rec-checkpoint.pt   --export text-rec-finetuned-2.onnx

rten-convert text-rec-finetuned-2.onnx text-rec-finetuned-2.rten

ocrs --rec-model text-rec-finetuned-2.rten <image>


TORCH_LOGS=recompiles poetry run python -m ocrs_models.train_rec synthetic_font . --font-ttf ~/foxhole/Renner-Book.ttf --checkpoint text-rec-checkpoint-s52qdbqt.pt --batch-size 1024 --max-epochs 5 --finetune --lr 1e-5 --compile --tf32
poetry run python -m ocrs_models.train_rec synthetic_font .   --font-ttf ~/foxhole/Renner-Book.ttf   --checkpoint text-rec-checkpoint.pt   --export text-rec-finetuned-5.onnx
