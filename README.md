# MixPro
MixPro: Data Augmentation with MaskMix and Progressive Attention Labeling for Vision Transformer [Official, ICLR 2023]

## Code will release soon 

### Install

We recommend using the pytorch docker `nvcr>=21.05` by
nvidia: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch.


- Create a conda virtual environment and activate it:

```bash
conda create -n mixpro python=3.7 -y
conda activate mixpro
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.8.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
```

- Install `timm==0.4.12`:

```bash
pip install timm==0.4.12
```

- Install other requirements:

```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
```

### Data preparation

For ImageNet-1K dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...

  ```
- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:
    - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
    - `train_map.txt`, `val_map.txt`: which store the relative path in the corresponding zip file and ground truth
      label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip

  $ head -n 5 data/ImageNet-Zip/val_map.txt
  ILSVRC2012_val_00000001.JPEG  65
  ILSVRC2012_val_00000002.JPEG  970
  ILSVRC2012_val_00000003.JPEG  230
  ILSVRC2012_val_00000004.JPEG  809
  ILSVRC2012_val_00000005.JPEG  516

  $ head -n 5 data/ImageNet-Zip/train_map.txt
  n01440764/n01440764_10026.JPEG        0
  n01440764/n01440764_10027.JPEG        0
  n01440764/n01440764_10029.JPEG        0
  n01440764/n01440764_10040.JPEG        0
  n01440764/n01440764_10042.JPEG        0
  ```
### Training from scratch on ImageNet-1K

To train a MixPro with `Vision Transformer` on ImageNet from scratch, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \
--cfg <config-file> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```


**Notes**:

- To use zipped ImageNet instead of folder dataset, add `--zip` to the parameters.
    - To cache the dataset in the memory instead of reading from files every time, add `--cache-mode part`, which will
      shard the dataset into non-overlapping pieces for different GPUs and only load the corresponding one for each GPU.
- When GPU memory is not enough, you can try the following suggestions:
    - Use gradient accumulation by adding `--accumulation-steps <steps>`, set appropriate `<steps>` according to your need.
    - Use gradient checkpointing by adding `--use-checkpoint`, e.g., it saves about 60% memory when training `DeiT-B`.
      Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.
    - We recommend using multi-node with more GPUs for training very large models, a tutorial can be found
      in [this page](https://pytorch.org/tutorials/intermediate/dist_tuto.html).
- To change config options in general, you can use `--opts KEY1 VALUE1 KEY2 VALUE2`, e.g.,
  `--opts TRAIN.EPOCHS 100 TRAIN.WARMUP_EPOCHS 5` will change total epochs to 100 and warm-up epochs to 5.
- For additional options, see [config](config.py) and run `python main.py --help` to get detailed message.

For example, to train `MixPro DeiT Transformer` with 8 GPU on a single node for 300 epochs, run:

`MixPro DeiT-T`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/deit/deit_tiny.yaml --data-path <imagenet-path> --batch-size 128
```

`MixPro DeiT-S`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/deit/deit_small.yaml --data-path <imagenet-path> --batch-size 128
```

`MixPro DeiT-B`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/deit_base.yaml --data-path <imagenet-path> --batch-size 64 \




