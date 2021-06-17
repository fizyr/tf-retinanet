# TF RetinaNet

Tensorflow Keras implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

## Disclaimer

The repository is still work in progress. Same results as ``keras-retinanet`` are not yet achieved in this repository. Any help will be welcomed.

### TODO's

- [ ] Train properly in order to achieve the same results as ``keras-retinanet``.
- [ ] Update jupyter notebook.
- [ ] Benchmark network speed.

## Components

The ``tf-retinanet`` project has been designed to be modular. The following components are part of the project:
* **Backbones**:
   * [ResNet](https://github.com/fizyr/tf-retinanet-backbones-resnet)
   * [ResNet50v2](https://github.com/jakedismo/tf-retinanet-backbones-resnet50v2)
   * [ResNet101v2](https://github.com/jakedismo/tf-retinanet-backbones-resnet101v2)
   * [ResNet152v2](https://github.com/jakedismo/tf-retinanet-backbones-resnet152v2)
* **Generators**:
   * [COCO](https://github.com/fizyr/tf-retinanet-generators-coco)
   * [CSV](https://github.com/jakedismo/tf-retinanet-generators-csv)

## Installation

1) Clone this repository.
2) Ensure numpy is installed using `pip install numpy --user`
3) In the repository, execute `pip install . --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
4) Alternatively, you can run the code directly from the cloned  repository, however you need to run `python setup.py build_ext --inplace` to compile Cython code first.
5) Optionally, install `pycocotools` if you want to train / test on the MS COCO dataset by running `pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI`.

## Testing

In general, inference of the network works as follows:
```python
boxes, scores, labels = model.predict_on_batch(inputs)
```

Where `boxes` are shaped `(None, None, 4)` (for `(x1, y1, x2, y2)`), scores is shaped `(None, None)` (classification score) and labels is shaped `(None, None)` (label corresponding to the score). In all three outputs, the first dimension represents the shape and the second dimension indexes the list of detections.

Loading models can be done in the following manner:
```python
from tf_retinanet.models import load_model
model = load_model('/path/to/model.h5', backbone=backbone)
```

### Converting a training model to inference model
The training procedure of `tf-retinanet` works with *training models*. These are stripped down versions compared to the *inference model* and only contains the layers necessary for training (regression and classification values). If you wish to do inference on a model (perform object detection on an image), you need to convert the trained model to an inference model. This is done as follows:

```shell
# Running directly from the repository:
tf_retinanet/bin/convert_model.py /path/to/training/model.h5 /path/to/save/inference/model.h5 --config /path/to/config_file.yaml

# Using the installed script:
retinanet-convert-model /path/to/training/model.h5 /path/to/save/inference/model.h5 --config /path/to/config_file.yaml
```

Most scripts (like `retinanet-evaluate`) also support converting on the fly, using the `--convert-model` argument.


## Training
`tf-retinanet` can be trained using [this](https://github.com/fizyr/tf-retinanet/blob/master/tf_retinanet/bin/train.py) script.
Note that the train script uses relative imports since it is inside the `tf_retinanet` package.
If you want to adjust the script for your own use outside of this repository,
you will need to switch it to use absolute imports.

If you installed `tf-retinanet` correctly, the train script will be installed as `retinanet-train`.
However, if you make local modifications to the `tf-retinanet` repository, you should run the script directly from the repository.
That will ensure that your local changes will be used by the train script.

f.ex. with resnet50v2 backbone and csv generator `python3 ./tf-retinanet/tf_retinanet/bin/train.py --freeze-backbone --backbone resnet50v2 --random-transform --batch-size 2 --steps 10 --epochs 10 --generator csv csv "/PATH_TO_DATA/annotations.csv", "/PATH_TO_DATA/classes.csv"`

### Usage

## Anchor optimization

In some cases, the default anchor configuration is not suitable for detecting objects in your dataset, for example, if your objects are smaller than the 32x32px (size of the smallest anchors).
In this case, it might be suitable to modify the anchor configuration, this can be done automatically by following the steps in the [anchor-optimization](https://github.com/martinzlocha/anchor-optimization/) repository.
To use the generated configuration check the sample `train.yaml` file.

## Debugging
Creating your own dataset does not always work out of the box. There is a [`debug.py`](https://github.com/fizyr/tf-retinanet/blob/master/tf_retinanet/bin/debug.py) tool to help find the most common mistakes.

Particularly helpful is the `--annotations` flag which displays your annotations on the images from your dataset. Annotations are colored in green when there are anchors available and colored in red when there are no anchors available. If an annotation doesn't have anchors available, it means it won't contribute to training. It is normal for a small amount of annotations to show up in red, but if most or all annotations are red there is cause for concern. The most common issues are that the annotations are too small or too oddly shaped (stretched out).

## Results

### MS COCO

## Status
Example output images using `tf-retinanet` are shown below.

<p align="center">
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco1.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco2.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco3.png" alt="Example result of RetinaNet on MS COCO"/>
</p>

### Projects using keras-retinanet, the ancestor of tf-retinanet.
* [Improving RetinaNet for CT Lesion Detection with Dense Masks from Weak RECIST Labels](https://arxiv.org/abs/1906.02283). Research project for detecting lesions in CT using keras-retinanet.
* [NudeNet](https://github.com/bedapudi6788/NudeNet). Project that focuses on detecting and censoring of nudity.
* [Individual tree-crown detection in RGB imagery using self-supervised deep learning neural networks](https://www.biorxiv.org/content/10.1101/532952v1). Research project focused on improving the performance of remotely sensed tree surveys.
* [ESRI Object Detection Challenge 2019](https://github.com/kunwar31/ESRI_Object_Detection). Winning implementation of the ESRI Object Detection Challenge 2019.
* [Lunar Rockfall Detector Project](https://ieeexplore.ieee.org/document/8587120). The aim of this project is to map lunar rockfalls on a global scale using the available > 1.6 million satellite images.
* [NATO Innovation Challenge](https://medium.com/data-from-the-trenches/object-detection-with-deep-learning-on-aerial-imagery-2465078db8a9). The winning team of the NATO Innovation Challenge used keras-retinanet to detect cars in aerial images ([COWC dataset](https://gdo152.llnl.gov/cowc/)).
* [Microsoft Research for Horovod on Azure](https://blogs.technet.microsoft.com/machinelearning/2018/06/20/how-to-do-distributed-deep-learning-for-object-detection-using-horovod-on-azure/). A research project by Microsoft, using keras-retinanet to distribute training over multiple GPUs using Horovod on Azure.
* [Anno-Mage](https://virajmavani.github.io/saiat/). A tool that helps you annotate images, using input from the keras-retinanet COCO model as suggestions.
* [Telenav.AI](https://github.com/Telenav/Telenav.AI/tree/master/retinanet). For the detection of traffic signs using keras-retinanet.
* [Towards Deep Placental Histology Phenotyping](https://github.com/Nellaker-group/TowardsDeepPhenotyping). This research project uses keras-retinanet for analysing the placenta at a cellular level.
* [4k video example](https://www.youtube.com/watch?v=KYueHEMGRos). This demo shows the use of keras-retinanet on a 4k input video.
* [boring-detector](https://github.com/lexfridman/boring-detector). I suppose not all projects need to solve life's biggest questions. This project detects the "The Boring Company" hats in videos.
* [comet.ml](https://towardsdatascience.com/how-i-monitor-and-track-my-machine-learning-experiments-from-anywhere-described-in-13-tweets-ec3d0870af99). Using keras-retinanet in combination with [comet.ml](https://comet.ml) to interactively inspect and compare experiments.
* [Weights and Biases](https://app.wandb.ai/syllogismos/keras-retinanet/reports?view=carey%2FObject%20Detection%20with%20RetinaNet). Trained keras-retinanet on coco dataset from beginning on resnet50 and resnet101 backends.
* [Google Open Images Challenge 2018 15th place solution](https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018). Pretrained weights for keras-retinanet based on ResNet50, ResNet101 and ResNet152 trained on open images dataset. 
* [poke.AI](https://github.com/Raghav-B/poke.AI). An experimental AI that attempts to master the 3rd Generation Pokemon games. Using keras-retinanet for in-game mapping and localization.
* [retinanetjs](https://github.com/faustomorales/retinanetjs) A wrapper to run RetinaNet inference in the browser / Node.js. You can also take a look at the [example app](https://faustomorales.github.io/retinanetjs-example-app/).

If you have a project based on `tf-retinanet` or `keras-retinanet` and would like to have it published here, shoot me a message on Slack.

### Notes
* This repository requires Tensorflow 2.0 or higher.

Contributions to this project are welcome.

### Discussions
Feel free to join the `#keras-retinanet` [Keras Slack](https://keras-slack-autojoin.herokuapp.com/) channel for discussions and questions.

## FAQ
* **I get the warning `UserWarning: No training configuration found in save file: the model was not compiled. Compile it manually.`, should I be worried?** This warning can safely be ignored during inference.
* **I get the error `ValueError: not enough values to unpack (expected 3, got 2)` during inference, what to do?**. This is because you are using a train model to do inference. See https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model for more information.
* **How do I do transfer learning?** The easiest solution is to use the `--weights` argument when training. Keras will load models, even if the number of classes don't match (it will simply skip loading of weights when there is a mismatch). Run for example `retinanet-train --weights snapshots/some_coco_model.h5 pascal /path/to/pascal` to transfer weights from a COCO model to a PascalVOC training session. If your dataset is small, you can also use the `--freeze-backbone` argument to freeze the backbone layers.
* **How do I change the number / shape of the anchors?** The train tool allows to pass a configuration file, where the anchor parameters can be adjusted. Check [here](https://github.com/fizyr/keras-retinanet-test-data/blob/master/config/config.ini) for an example config file.
* **I get a loss of `0`, what is going on?** This mostly happens when none of the anchors "fit" on your objects, because they are most likely too small or elongated. You can verify this using the [debug](https://github.com/fizyr/keras-retinanet#debugging) tool.
* **I have an older model, can I use it after an update of tf-retinanet?** This depends on what has changed. If it is a change that doesn't affect the weights then you can "update" models by creating a new retinanet model, loading your old weights using `model.load_weights(weights_path, by_name=True)` and saving this model. If the change has been too significant, you should retrain your model (you can try to load in the weights from your old model when starting training, this might be a better starting position than ImageNet).
* **I get the error `ModuleNotFoundError: No module named 'tf_retinanet.utils.compute_overlap'`, how do I fix this?** Most likely you are running the code from the cloned repository. This is fine, but you need to compile some extensions for this to work (`python setup.py build_ext --inplace`).
* **How do I train on my own dataset?** The steps to train on your dataset are roughly as follows:
* 1. Prepare your dataset in the CSV format (a training and validation split is advised).
* 2. Check that your dataset is correct using `retinanet-debug`.
* 3. Train retinanet, preferably using the pretrained COCO weights (this gives a **far** better starting point, making training much quicker and accurate). You can optionally perform evaluation of your validation set during training to keep track of how well it performs (advised).
* 4. Convert your training model to an inference model.
* 5. Evaluate your inference model on your test or validation set.
* 6. Profit!
