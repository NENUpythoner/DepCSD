# DepCSD: Depth-Conditioned Self-Supervised Dehazing for Real Scenes

**Xingkai Wen¹, Yingjie Liu¹, Ryan Wen Liu², Liang-Jian Deng³, Hao Liu⁴, Jun Liu¹**

¹ Northeast Normal University, Changchun, China  
² Wuhan University of Technology, Wuhan, China  
³ University of Electronic Science and Technology of China, Chengdu, China  
⁴ Hong Kong Baptist University, Hong Kong, China

## Repository structure

| Path | Description |
| --- | --- |
| `main.py` | Main restoration program and original DepCSD optimization. |
| `depcsd_runtime.py` | Folder processing, depth interface, path handling and result saving. |
| `net/` | Network modules and their required imports. |
| `Rop_plus/` | ROP+ initialization based on Jun Liu et al., [Rank-One Prior: Real-Time Scene Recovery](https://doi.org/10.1109/TPAMI.2022.3226276), IEEE TPAMI, 2023. [GitHub](https://github.com/junliumath/ROP). |
| `utils/` | Image I/O, resizing, coordinate transforms and numerical utilities. |
| `DepthAnything/` | Depth prediction using Lihe Yang et al., [Depth Anything V2](https://arxiv.org/abs/2406.09414), NeurIPS 2024. [GitHub](https://github.com/DepthAnything/Depth-Anything-V2). Download separately. |
| `requirements.txt` | Pinned dependencies for the supplied inference sources. |

## DepthAnything setup

**Download:** cloud-storage link to be provided by the authors.

Download and extract the complete `DepthAnything/` folder into `DepCSD/DepthAnything/`.

## Installation

Target environment: **Windows x64, Python 3.10.16 and an NVIDIA CUDA GPU**. Full DepCSD restoration requires CUDA; CPU inference is not supported.

Create the environment and enter the project directory. Replace the example project path with your actual location:

```powershell
conda create -n depcsd python=3.10.16 pip -y
conda activate depcsd
Set-Location "D:\Projects\DepCSD"
```

Install PyTorch from its official CUDA 12.1 index, then resolve all remaining dependencies from PyPI:

```powershell
python -m pip install "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" --index-url https://download.pytorch.org/whl/cu121 --no-deps
python -m pip install -r .\requirements.txt --index-url https://pypi.org/simple
python -m pip check
```

The first pip command alone is incomplete; run the second command as well. An existing matching environment can be reused. For slow PyPI access, replace `https://pypi.org/simple` in the second command with `https://pypi.tuna.tsinghua.edu.cn/simple`; no additional requirements file is needed.

## Run

From the directory containing the main Python file, provide only the input and output folder paths:

```powershell
python .\main.py --input "D:\Data\DepCSD_input" --output "D:\Results\DepCSD_outputs"
```

For folders located inside the project directory:

```powershell
python .\main.py --input ".\DepCSD_input" --output ".\DepCSD_outputs"
```
