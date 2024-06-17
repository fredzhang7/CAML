# CAML: Context-Aware Meta-Learning

This repository contains the official code for CAML, an in-context learning algorithm for few-shot image classification, **maintained & refactored by Fred Zhang @ UBC CS**.

#### Updates:

1. **March 22, 2024:** Initial release.

2. **June 15, 2024**: Bug fixes for eval. Added package installation via pip.

3. **June 16, 2024**: Refactored codebase from `torch` version `1.13.1` to `2.1.2+cu121`. Updated important dependencies to the latest versions to achieve no dependency conflicts with other ML projects (e.g. `numpy==1.21.6` -> `numpy==2.0.0`, `pandas==1.3.5` -> `2.2.2`, etc.). Added code to accept newer image feature extractors and custom datasets.


CAML is designed for the universal meta-learning setting.

Universal meta-learning measures a model's capacity to quickly
learn new image classes. It evaluates models across a diverse set of meta-learning benchmarks spanning many different
image classification tasks without meta-training on any of the benchmarks' training sets or fine-tuning on the
support set during inference.

**Context-Aware Meta-Learning**  
Christopher Fifty, Dennis Duan, Ronald G. Junkins,\
Ehsan Amid, Jure Leskovec, Christopher Ré, Sebastian Thrun\
ICLR 2024\
[arXiv](https://arxiv.org/abs/2310.10971)

## Approach

CAML learns new visual concepts during inference without meta-training on related concepts or fine-tuning on the support
set during inference.
It is competitive with state-of-the-art meta-learning algorithms that meta-train on the training set of each benchmark
in our testing framework.

![CAML method](assets/method.jpg)

## Code environment

This code requires PyTorch 2.1.2 or higher with cuda support. It has been tested on Ubuntu 20.04 and Windows 11.

You can create a conda environment with the correct dependencies using the following command lines:

```
cd CAML
conda env create -f environment.yml
conda activate caml
```

or install required dependencies using pip package manager:

```
pip install -r requirements.txt
```

## Setup

The directory structure for this project should look like:

```
Outer_Directory
│
│───caml_pretrained_models/
│   │   CAML_CLIP/
│   │   ...
│
│───caml_train_datasets/
│   │   fungi/
│   │   ...
│
│───caml_universal_eval_datasets/
│   │   Aircraft_fewshot/
│   │   ...
│
└───CAML
│   │   assets/
│   │   qpth_local/
│   │   src/
│   │   .gitignore
│   │   .enrionment.yml
│   │   .README.md
```

### Downloading The Datasets and Pre-Trained Models

We offer four downloads:

1. The pre-trained model
   checkpoints. [[Download Link]](https://drive.google.com/file/d/1oG-XO6w2Q73ZbofXH3kTsuR0rj2ptOc2/view?usp=sharing)
2. The pre-training datasets as CLIP, Laion-2b, and ResNet34 image
   embeddings. [[Download Link]](https://drive.google.com/file/d/1oISLcISDOUeFyZxgjOWLxWhOxmK-5wmM/view?usp=sharing)
    * Using image embeddings makes the training **substantively** faster.
3. The universal eval
   datasets. [[Download Link]](https://drive.google.com/file/d/1FCjJeoZzunqdhrEOI3gWyJdVXMZ2kQ8j/view?usp=sharing)
4. qpth_local.zip: this is needed to train/test
   MetaOpt. [[Download Link]](https://drive.google.com/file/d/1ZOP4CB9l0XDPiPfi6vTeX3dZDwNP7rwx/view?usp=sharing)

After unzipping a file, move it to the location specified by the directory diagram in Setup.

## Training a Model

To pre-train CAML with a CLIP image encoder, ELMES label encoder, run the following:

```commandline
python src/train.py \
     --opt adam \
     --lr 1e-5 \
     --epoch 100 \
     --val_epoch 1 \
     --batch_sizes 525 \
     --detailed_name \
     --encoder_size large \
     --dropout 0.0  \
     --fe_type cache:timm:vit_base_patch16_clip_224.openai:768 \
     --schedule custom_cosine \
     --fe_dtype float32 \
     --model CAML \
     --label_elmes \
     --save_dir test_CAML_repo \
     --gpu 0
```

Additional training details are located in a comment at the top of `CAML/src/train.py`.

## Evaluating a Model in the Universal Setting

To evaluate CAML in the *universal setting* on Aircraft:

```commandline
python src/evaluation/test.py --model CAML --gpu 0 --eval_dataset Aircraft  \
--fe_type timm:vit_base_patch16_clip_224.openai:768
```

You can also evaluate on any of the following datasets:

```commandline
[Aircraft, pascal_paintings, mini_ImageNet, meta_iNat, ChestX, tiered_ImageNet, CUB_fewshot, 
tiered_meta_iNat, cifar, paintings, pascal]
```

Additional evaluation details are located in a comment at the top of `CAML/src/evaluation/test.py`.

## Custom Datasets and Feature Extractors

You can evaluate on a custom dataset and `fe_type` (feature extractor) by adding the correct input image resolution under a new `transform_type`. Here's [an example](https://github.com/fredzhang7/CAML/blob/master/src/evaluation/datasets/transform_manager.py#L66).


## Citation

This work builds on the codebase of
[Few-Shot Classification with Feature Map Reconstruction Networks](https://github.com/Tsingularity/FRN), and uses the
following datasets for pre-training.

1. [ImageNet](https://www.image-net.org/)
2. [MSCOCO](https://cocodataset.org/#home)
3. [Fungi](https://github.com/visipedia/fgvcx_fungi_comp)
4. [WikiArt](https://huggingface.co/datasets/huggan/wikiart)

We release CLIP, Laion-2b, and ResNet34 image embeddings as a download link
under the "Downloading The Datasets and Pre-Trained Models" section.

The following datasets are used to evaluate *universal meta-learning* performance.

1. [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
2. [CIFAR-FS](https://www.cs.toronto.edu/~kriz/cifar.html)
3. [ChestX](https://nihcc.app.box.com/v/ChestXray-NIHCC)
4. [CUB Fewshot](https://www.vision.caltech.edu/datasets/)
5. [Meta iNat](https://github.com/visipedia/inat_comp/tree/master/2017)
6. [Tiered Meta iNat](https://github.com/visipedia/inat_comp/tree/master/2017)
7. [Mini ImageNet](https://github.com/twitter-research/meta-learning-lstm)
8. [Tiered Mini ImageNet](https://github.com/icoz69/DeepEMD)
9. [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
10. [Paintings](https://www.robots.ox.ac.uk/~vgg/data/paintings/)

We release a zip file containing the test set as a download link under the "Downloading The Datasets and Pre-Trained
Models" section.

If you use this codebase or otherwise found the ideas useful, please reach out to let us know. You can contact Chris
at [fifty@cs.stanford.edu](mailto:fifty@cs.stanford.edu).

You can also cite our work:

```
@inproceedings{fifty2023context,
  title={Context-Aware Meta-Learning},
  author={Fifty, Christopher and Duan, Dennis and Junkins, Ronald Guenther and Amid, Ehsan and Leskovec, Jure and Re, Christopher and Thrun, Sebastian},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
