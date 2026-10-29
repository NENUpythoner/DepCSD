"""Batch I/O, stable paths and the original external depth interface."""
from pathlib import Path
from contextlib import contextmanager
import argparse
import gc
import importlib
import importlib.metadata
import json
import os
import sys
import traceback
import warnings

import numpy as np
from PIL import Image
from utils.numerics import require_finite

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.png', '.jpg', '.jpeg'}
DEFAULT_NUM_ITER = 3000
DEFAULT_SHOW_EVERY = 100


@contextmanager
def repository_cwd():
    """Scope legacy relative weight/config paths to the repository; CLI is sequential."""
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        yield
    finally:
        os.chdir(previous)


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('DepCSD requires a usable NVIDIA CUDA GPU and GPU-enabled PyTorch; CPU fallback is not implemented')
    # Exercise real allocation and arithmetic, not just the availability flag.
    value = (torch.ones(8, device='cuda') * 2).sum().item()
    if value != 16:
        raise RuntimeError('CUDA tensor computation failed')


def load_depth_function():
    try:
        with repository_cwd():
            module = importlib.import_module('DepthAnything.run')
    except Exception as exc:
        raise RuntimeError('Restore your original DepthAnything folder, dependencies and checkpoint paths. Import failed: '+str(exc)) from exc
    fn = getattr(module, 'depthanything', None)
    if not callable(fn):
        raise RuntimeError('DepthAnything.run must provide callable depthanything(J_rop)')
    return fn


def depthanything(image):
    # Preserve RGB uint8 input, the call order and the actual original model.
    with repository_cwd():
        result = load_depth_function()(image)
    require_finite(result, 'DepthAnything output')
    if result.ndim != 3 or result.shape[:2] != image.shape[:2] or result.shape[-1] < 1:
        raise ValueError('Original depth interface must return H x W x C (C >= 1) at matching size; no silent resize or model replacement is applied')
    return result


def save_output(name, image, image_name, output_file, debug_dir, input_size_wh):
    from utils.image_io import save_image
    # Resize only the requested exported RGB results; never feed them back to optimization.
    resize_wh = input_size_wh if (
        name == image_name + '_' or name == image_name + '_J_rop'
        or name.endswith(('_clean_mid', '_clean_mid_gamma', '_clean_best'))
    ) else None
    if name == image_name + '_clean_best':
        destination = Path(output_file) if output_file is not None else ROOT/'output'/(name+'.png')
        save_image(destination.stem, image, destination.parent, resize_wh=resize_wh)
    elif debug_dir is not None:
        save_image(name, image, debug_dir, resize_wh=resize_wh)


def read_rgb(path):
    from utils.image_io import pil_to_np
    with Image.open(path) as image:
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Multi-frame images are unsupported')
        if image.mode not in ('RGB', 'RGBA'):
            raise ValueError(f'Expected 8-bit RGB/RGBA input, got {image.mode}; convert explicitly')
        # No prepare_image call: the original helper center-crops before returning.
        array = pil_to_np(image)
        size = image.size
    require_finite(array, 'RGB input')
    return array, size


def plan_files(source, destination, recursive=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise ValueError(f'Input directory does not exist: {source}')
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Input/output directories must be separate and not nested')
    paths = source.rglob('*') if recursive else source.iterdir()
    images = sorted((p for p in paths if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p:p.relative_to(source).as_posix().casefold())
    if not images:
        raise ValueError('No supported image files found')
    jobs, seen = [], set()
    for image in images:
        if image.is_symlink():
            raise ValueError(f'Image symlinks are unsupported: {image}')
        rel = image.relative_to(source)
        output = destination/rel.parent/(rel.stem+'.png')
        key = output.relative_to(destination).as_posix().casefold()
        if key in seen:
            raise ValueError(f'Images share an output name (case-insensitive): {output}. Separate same-stem inputs such as a.jpg and a.png before running; no file has been processed.')
        if output.exists() and not output.is_file():
            raise ValueError(f'Output path exists but is not a file: {output}')
        seen.add(key)
        jobs.append((image, output))
    return jobs


def build_parser():
    parser = argparse.ArgumentParser(
        description='DepCSD hazy_plus: PNG/JPG/JPEG folder restoration; original defaults',
        allow_abbrev=False)
    parser.add_argument('--input', type=Path, required=True, help='Input image folder')
    parser.add_argument('--output', type=Path, required=True, help='Output PNG folder')
    return parser


def preflight():
    import torch
    require_cuda()
    load_depth_function()
    print(f'CUDA usable: {torch.cuda.get_device_name(torch.cuda.current_device())}; torch {torch.__version__}, CUDA runtime {torch.version.cuda}')
    print('Depth callable found; this check does not run depth prediction or verify checkpoint loading')


def write_manifest(path, args, records):
    packages = ('torch','torchvision','numpy','scipy','Pillow','matplotlib','scikit-image','opencv-python','scikit-video','kornia','kornia-rs','einops')
    versions = {}
    for package in packages:
        try:versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package] = None
    payload = {'python':sys.version,'versions':versions,'geometry':'original_center_crop_then_resize_selected_exports','export_resample':'Pillow BICUBIC after original uint8 conversion','num_iter':DEFAULT_NUM_ITER,'show_every':DEFAULT_SHOW_EVERY,'save_debug':True,'overwrite':True,'recursive':False,'input':str(args.input),'output':str(args.output),'records':records}
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def run_cli(model_class, argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.input, args.output = args.input.resolve(), args.output.resolve()
    try:jobs = plan_files(args.input,args.output)
    except ValueError as exc:parser.error(str(exc))
    records, failed = [], False
    import torch
    try:
        with repository_cwd():
            preflight()
            args.output.mkdir(parents=True, exist_ok=True)
            for index, (source,destination) in enumerate(jobs,1):
                record = {'input':str(source),'output':str(destination)}
                model = None
                print(f'[{index}/{len(jobs)}] {source}',flush=True)
                try:
                    image,size = read_rgb(source)
                    relative = source.relative_to(args.input)
                    debug_dir = args.output/'_debug'/relative.parent/relative.name
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model = model_class(relative.stem,image,num_iter=DEFAULT_NUM_ITER,plot_during_training=True,show_every=DEFAULT_SHOW_EVERY,clip=True,output_file=destination,debug_dir=debug_dir)
                        model.optimize()
                        model.finalize()
                    expected = size
                    with Image.open(destination) as restored:
                        restored.load()
                        if restored.format != 'PNG' or restored.mode != 'RGB':
                            raise RuntimeError('Expected a fully decodable RGB PNG output')
                        if restored.size != expected:
                            raise RuntimeError(f'Output size {restored.size} differs from expected {expected}')
                    score = float(model.best_result.psnr)
                    record.update(status='ok',input_size_wh=list(size),cropped_size_wh=[model.image.shape[2],model.image.shape[1]],working_size_wh=list(reversed(model.working_shape)),output_size_wh=list(expected),best_reconstruction_psnr=score if np.isfinite(score) else str(score),warnings=list(dict.fromkeys(str(w.message) for w in caught)))
                except KeyboardInterrupt:
                    record.update(status='interrupted')
                    raise
                except Exception as exc:
                    record.update(status='error',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
                    failed = True
                    print(record['error'],file=sys.stderr,flush=True)
                finally:
                    model = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    records.append(record)
                    write_manifest(args.output/'run_manifest.json',args,records)
    except Exception as exc:
        print(f'Run failed: {exc}',file=sys.stderr)
        return 1
    print(f'Finished: {sum(r["status"]=="ok" for r in records)} restored, {sum(r["status"]=="skipped_existing" for r in records)} skipped, {sum(r["status"]=="error" for r in records)} failed')
    return 1 if failed else 0
