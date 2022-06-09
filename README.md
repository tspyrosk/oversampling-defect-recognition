# oversampling-defect-recognition
On-the-fly Image-level Oversampling for Imbalanced Datasets of Manufacturing Defects

### Contents

`src/oversampling-defect-recognition.ipynb`: is the Jupyter Notebook containing the CNN model definition and training, the confidence-aware augmentation and final evaluation.

`input/biggan-generator/generate_images.py`: contains the code responsible for image generation including tiling and best images selection

`input/biggan-generator/biggan_generator.py`: defines classes related to BigGAN and the adaptation of its batch-norm layers

The last two files borrow from the following repositories:
https://github.com/ajbrock/BigGAN-PyTorch/,
https://github.com/nogu-atsu/small-dataset-image-generation,
https://github.com/apple2373/MetaIRNet

### Main Dependencies

python >= 3.7,
tensorflow >= 2.6.4,
keras >= 2.6.0,
torch >= 1.11.0

### Instructions

Download the 128x128 pre-trained weights from the BigGAN repository: https://github.com/ajbrock/BigGAN-PyTorch#pretrained-models.
Place the file named "138k" containing the weights in `input/biggan-weights`.

A dummy input dataset can be found under: `input/dummy-logo-prints`. Unziping the file in place should yield the dataset under `input/dummy-logo-prints/shaver-shell-full`

To run the notebook `src/oversampling-defect-recognition.ipynb` use of one GPU is highly recommended (otherwise synthetic data generation will be really slow).
