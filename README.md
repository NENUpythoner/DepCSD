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

Requirements: **Python 3.10.16, an NVIDIA CUDA GPU and a compatible NVIDIA driver**. The supplied environment configuration targets Windows x64; CPU inference is not supported.

Download and extract this repository to any location. Open a terminal in the project directory containing `main.py` and `requirements.txt`. On Windows, open that directory in File Explorer, type `powershell` in the address bar and press Enter. Use a terminal with Conda initialized.

Create and activate the environment using the Tsinghua Anaconda mirror:

```powershell
conda create -n depcsd python=3.10.16 pip -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda activate depcsd
```

Install all Python dependencies with one command:

```powershell
python -m pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu121 --timeout 120 --retries 3
python -m pip check
```

This command uses the Tsinghua PyPI mirror together with the official PyTorch CUDA 12.1 wheel index required by the pinned `+cu121` packages. No separate PyTorch installation or edited requirements file is needed. An existing environment with matching dependencies can be reused.

## Run

Activate the environment and run from the project directory. Replace `PATH_TO_INPUT_FOLDER` and `PATH_TO_OUTPUT_FOLDER` with your own folder paths; keep quotation marks if a path contains spaces. Absolute paths and paths relative to the current terminal directory are accepted.

```powershell
conda activate depcsd
python main.py --input "PATH_TO_INPUT_FOLDER" --output "PATH_TO_OUTPUT_FOLDER"
```

For example, to read `DepCSD_input/` and save to `DepCSD_outputs/` inside the project directory:

```powershell
python main.py --input "./DepCSD_input" --output "./DepCSD_outputs"
```

Only `--input` and `--output` are required; all restoration settings use the original defaults. The input directory must exist and contain PNG, JPG or JPEG images. The output directory is created automatically. Restored images are saved as PNG files with their original filename stems and input dimensions.
