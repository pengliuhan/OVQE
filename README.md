# *OVQE: Omniscient Network for Compressed Video Quality Enhancement*


The *PyTorch* implementation for the [OVQE: Omniscient Network for Compressed
Video Quality Enhancement](https://ieeexplore.ieee.org/document/9908169) which is accepted by [IEEE TBC].

Task: Video Quality Enhancement / Video Artifact Reduction.



## 1. Pre-request

### 1.1. Environment
Suppose that you have installed CUDA 11.0, then:
```bash
conda create -n cvlab python=3.8 -y  
conda activate cvlab
git clone --depth=1 https://github.com/pengliuhan/OVQE && cd OVQE/
python -m pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install tqdm lmdb pyyaml opencv-python scikit-image thop
```

### 1.2. Dataset

Please check [here](https://github.com/ryanxingql/mfqev2.0/wiki/MFQEv2-Dataset).

### 1.3. Create LMDB
We now generate LMDB to speed up IO during training.
```bash
python create_lmdb_ovqe.py
```

## 2. Train

We utilize 2 NVIDIA GeForce RTX 3090 GPUs for training.
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12354 train.py --opt_path option_ovqe.yml
```

## 3. Test         
Pretrained models can be found here: [[GoogleDisk]](https://drive.google.com/file/d/1V0kJ63dr1JzQDIoBVbJ8FBp_cdvsnTUK/view?usp=sharing) and [[百度网盘]](https://pan.baidu.com/s/13m5zSRs4FxYPH_MN77lhDw?pwd=ovqe )

We utilize 1 NVIDIA GeForce RTX 3090 GPU for testing.
```bash
python test_one_video.py
```

## Citation
If you find this project is useful for your research, please cite:
```bash
@article{peng2022ovqe,
  title={OVQE: Omniscient Network for Compressed Video Quality Enhancement},
  author={Peng, Liuhan and Hamdulla, Askar and Ye, Mao and Li, Shuai and Wang, Zengbin and Li, Xue},
  journal={IEEE Transactions on Broadcasting},
  year={2022},
  publisher={IEEE}
}
```

## Acknowledgements
This work is based on [STDF-Pytoch](https://github.com/RyanXingQL/STDF-PyTorch). Thank [RyanXingQL](https://github.com/RyanXingQL)  for sharing the codes.
