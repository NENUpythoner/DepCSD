DepCSD: Depth-Conditioned Self-Supervised Dehazing for Real Scenes
Xingkai Wen¹, Yingjie Liu¹, Ryan Wen Liu², Liang-Jian Deng³, Hao Liu⁴, Jun Liu¹
¹ Northeast Normal University, Changchun, China  
² Wuhan University of Technology, Wuhan, China  
³ University of Electronic Science and Technology of China, Chengdu, China  
⁴ Hong Kong Baptist University, Hong Kong, China
Repository structure
Path	Description
`dehazing_7_t1d1_hazy_plus.py`	Main restoration program and original DepCSD optimization.
`depcsd_runtime.py`	Folder processing, depth interface, path handling and result saving.
`net/`	Network modules and their required imports.
`Rop_plus/`	ROP+ initialization and its image-resizing implementation.
`utils/`	Image I/O, resizing, coordinate transforms and numerical utilities.
`DepthAnything/`	External depth model and pretrained weights; download separately.
`requirements.txt`	Pinned dependencies for the supplied inference sources.
DepthAnything setup
Download the complete `DepthAnything/` folder, including the matching pretrained checkpoint, from the authors' separately supplied cloud storage. This folder is not included in the GitHub repository.
Cloud download link: to be supplied by the authors.
Place the downloaded folder directly inside `DepCSD/`. These paths must exist:
```text
DepCSD/DepthAnything/run.py
DepCSD/DepthAnything/depth_anything_v2/dpt.py
DepCSD/DepthAnything/checkpoints/depth_anything_v2_vitl.pth
```
Keep the complete model source tree and its original notices. Do not add an extra nested `DepthAnything/` directory. Use the corrected author-provided `run.py`, which exposes `depthanything(J_rop)` and does not consume the main program's `--input` and `--output` arguments. Its original defaults are `vitl` and input size `518`.
Installation
Target environment: Windows x64, Python 3.10.16 and an NVIDIA CUDA GPU. Full DepCSD restoration requires CUDA; CPU inference is not supported.
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
The separately distributed depth bundle must match the original implementation and supply any additional dependencies it requires.
Run
From the directory containing the main Python file, provide only the input and output folder paths:
```powershell
python .\dehazing_7_t1d1_hazy_plus.py --input "D:\Data\DepCSD_input" --output "D:\Results\DepCSD_outputs"
```
For folders located inside the project directory:
```powershell
python .\dehazing_7_t1d1_hazy_plus.py --input ".\DepCSD_input" --output ".\DepCSD_outputs"
```
Input and output
Process `.png`, `.jpg` and `.jpeg` images at the current level of the input folder. Subdirectories are not scanned.
Use single-frame 8-bit RGB/RGBA images; alpha is dropped.
The program automatically performs ROP+ initialization, depth prediction and DepCSD optimization. Original defaults remain: 3000 Adam iterations per image, learning rate 0.001 and diagnostic saving every 100 iterations.
Main results retain the original filename stem and use PNG: `scene.jpg` becomes `scene.png`.
Main output pixel width and height match the input. The original center crop is retained, and selected RGB results are resized at export; cropped borders are not recovered. Other diagnostic maps retain their working/cropped dimensions.
Intermediate images are saved under `_debug/<original-filename>/`. `run_manifest.json` records the latest run's image status and dimensions. Re-running recomputes and replaces corresponding outputs.
Keep input and output folders separate and non-nested. Separate same-stem inputs such as `scene.jpg` and `scene.png`, which would otherwise produce the same output name.
