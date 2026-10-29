from collections import namedtuple
import cv2
# from cv2.ximgproc import guidedFilter
from net import *
from net.losses import StdLoss
from net.skip_transmission_with_all import SkipTransmissionWithAll
from net.skip_transmission_with_ambition import SkipTransmissionWithAmb
from utils.imresize import np_imresize
from net.noise import get_noise
from utils.image_io import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Rop_plus.ROP import ROP_main as rop
from matplotlib.gridspec import GridSpec
from depcsd_runtime import depthanything, require_cuda, save_output
from utils.geometry import center_crop_input, inverse_augmentation
from utils.numerics import minmax_normalize, percentile_stretch, require_finite
import kornia
from net.skip_transmission_with_depth import SkipTransmissionWithDepth
import torch
import torch.nn.functional as F
from net.skip_transmission_with_depth_Unet import Mask_Unet


DehazeResult = namedtuple("DehazeResult", ['learned', 't', 'a', 'psnr'])

# 新增函数：计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Dehaze(object):
    def __init__(self, image_name, image, num_iter=0, plot_during_training=True, show_every=50, clip=True, *, output_file=None, debug_dir=None):
        require_cuda()
        if num_iter < 1 or show_every < 1:
            raise ValueError("num_iter and show_every must be positive")
        self.output_file = output_file
        self.debug_dir = debug_dir
        self.image_name = image_name
        self.image = center_crop_input(image)
        self.input_size_wh = (int(image.shape[2]), int(image.shape[1]))
        self.source_image = self.image.copy()
        self.num_iter = num_iter
        self.plot_during_training = plot_during_training
        self.show_every = show_every
        self.image_net = None
        self.mask_net = None
        self.ambient_net = None
        self.mse_loss = None
        self.learning_rate = 0.001
        self.parameters = None
        self.current_result = None
        self.clip = clip
        self.blur_loss = None
        self.blur_t_out = None
        self.blur_a_out = None
        self.blur_I_out = None
        self.best_result = None
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.done = False
        self.ambient_out = None
        self.total_loss = None
        self.post = None
        self.param = {'omega': 0.99, 'mi': 1, 'ma': 95, 'gamma': 5e1, 'iternum': 10}
        self._init_all()

    def _save_image(self, name, image):
        save_output(name, image, self.image_name, self.output_file, self.debug_dir, self.input_size_wh)

    def _restore_map(self, image):
        return np_imresize(image, output_shape=self.original_image.shape[1:])

    def _init_all(self):
        self._init_images()
        self._init_nets()

        # 计算并打印 self.mask_net 和 self.ambient_net 的参数量
        mask_net_params = count_parameters(self.mask_net)
        ambient_net_params = count_parameters(self.ambient_net)
        print(f"self.mask_net 的参数量: {mask_net_params}")
        print(f"self.ambient_net 的参数量: {ambient_net_params}")

        self._init_inputs()
        self._init_parameters()
        self._init_loss()

    def _init_images(self):
        self.original_image = self.image.copy()
        factor = 1
        image = self.image
        while image.shape[1] >= 2000 or image.shape[2] >= 2000:
            new_shape_x, new_shape_y = self.image.shape[1] / factor, self.image.shape[2] / factor
            new_shape_x -= (new_shape_x % 32)
            new_shape_y -= (new_shape_y % 32)
            if min(new_shape_x, new_shape_y) < 64:
                raise ValueError("The original large-image resize policy produces a side below 64; input aspect ratio is unsupported")
            image = np_imresize(self.image, output_shape=(new_shape_x, new_shape_y))
            factor += 1
        self.working_shape = image.shape[1:]
        self.images = create_augmentations(image)
        self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.images]

        J_rop, t_rop, a_rop = rop(image.transpose(1, 2, 0), self.param)
        for label, value in (("ROP J", J_rop), ("ROP t", t_rop), ("ROP A", a_rop)):
            require_finite(value, label)
            if value.shape != image.transpose(1, 2, 0).shape:
                raise ValueError(label + " must have shape H x W x 3")
        self._save_image(self.image_name + "_t_rop_3", t_rop.transpose(2, 0, 1))
        d_rop = np.mean(depthanything(J_rop), axis=-1, keepdims=True)
        d_rop = minmax_normalize(d_rop, "depth")
        J_rop = minmax_normalize(J_rop, "ROP J display/cache")
        t_rop = np.mean(t_rop, axis=-1, keepdims=True)

        self._save_image(self.image_name + "_J_rop", J_rop.transpose(2, 0, 1))
        self._save_image(self.image_name + "_t_rop_1", t_rop.transpose(2, 0, 1))
        self._save_image(self.image_name + "_a_rop", a_rop.transpose(2, 0, 1))
        self._save_image(self.image_name + "_d_rop", d_rop.transpose(2, 0, 1))

        self.J_rops_torch = [np_to_torch(aug_img).type(torch.cuda.FloatTensor).detach()
                       for aug_img in create_augmentations(J_rop.transpose(2, 0, 1))]
        self.t_rops_torch = [np_to_torch(aug_img).type(torch.cuda.FloatTensor).detach()
                       for aug_img in create_augmentations(t_rop.transpose(2, 0, 1))]
        self.d_rops_torch = [np_to_torch(aug_img).type(torch.cuda.FloatTensor).detach()
                       for aug_img in create_augmentations(d_rop.transpose(2, 0, 1))]
        self.a_rops_torch = [np_to_torch(aug_img).type(torch.cuda.FloatTensor).detach()
                       for aug_img in create_augmentations(a_rop.transpose(2, 0, 1))]

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor

        self.image_net = skip(
            8, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU'
        ).type(data_type)

        self.mask_net = SkipTransmissionWithDepth(
            noise_channels=8,
            depth_channels=1,
            num_output_channels=1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU',
        ).type(data_type)

        self.ambient_net = SkipTransmissionWithAll(
            noise_channels=8,
            amb_channels=3,
            depth_channels=1,
            num_output_channels=3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU',
        ).type(data_type)

    def _init_parameters(self):
        parameters = [p for p in self.image_net.parameters()] + \
                     [p for p in self.mask_net.parameters()] + \
                     [p for p in self.ambient_net.parameters()]
        self.parameters = parameters

    def _init_loss(self):
        def tv_loss(x):
            return torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) + \
                torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        def l2_loss(a_out, a_rop):
            return F.mse_loss(a_out, a_rop)
        def rec_loss(output, target, alpha):
            mse_loss = F.mse_loss(output, target)
            l1_loss = F.l1_loss(output, target)
            return alpha * mse_loss + (1 - alpha) * l1_loss
        def tex_loss(a, d):
            # 如果 a 和 d 是 3 通道图像，则将它们转换为灰度图
            a_gray = 0.2989 * a[:, 0:1, :, :] + 0.5870 * a[:, 1:2, :, :] + 0.1140 * a[:, 2:3, :, :] if a.shape[
                                                                                                           1] == 3 else a
            d_gray = 0.2989 * d[:, 0:1, :, :] + 0.5870 * d[:, 1:2, :, :] + 0.1140 * d[:, 2:3, :, :] if d.shape[
                                                                                                           1] == 3 else d

            # 计算水平方向梯度差异（绝对差）
            grad_a_x = torch.abs(a_gray[:, :, :, 1:] - a_gray[:, :, :, :-1])
            grad_d_x = torch.abs(d_gray[:, :, :, 1:] - d_gray[:, :, :, :-1])

            # 计算竖直方向梯度差异
            grad_a_y = torch.abs(a_gray[:, :, 1:, :] - a_gray[:, :, :-1, :])
            grad_d_y = torch.abs(d_gray[:, :, 1:, :] - d_gray[:, :, :-1, :])

            # 分别计算水平方向和竖直方向的梯度差异
            loss_x = torch.mean(torch.abs(grad_a_x - grad_d_x))
            loss_y = torch.mean(torch.abs(grad_a_y - grad_d_y))

            # 返回两者的总损失
            return loss_x + loss_y
        data_type = torch.cuda.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.blur_loss = StdLoss().type(data_type)
        self.tv_loss = tv_loss
        self.rec_loss = rec_loss
        self.l2_loss = l2_loss
        self.tex_loss = tex_loss

    def _init_inputs(self):
        original_noises = create_augmentations(torch_to_np(get_noise(8, 'noise',
        (self.images[0].shape[1], self.images[0].shape[2]), var=1/10.).type(torch.cuda.FloatTensor).detach()))
        self.image_net_inputs = [np_to_torch(original_noise).type(torch.cuda.FloatTensor).detach()
                                 for original_noise in original_noises]

        original_noises = create_augmentations(torch_to_np(get_noise(8, 'noise',
        (self.images[0].shape[1], self.images[0].shape[2]), var=1/10.).type(torch.cuda.FloatTensor).detach()))
        self.mask_net_inputs = [np_to_torch(original_noise).type(torch.cuda.FloatTensor).detach()
                                for original_noise in original_noises]

        original_noises = create_augmentations(torch_to_np(get_noise(8, 'meshgrid',
        (self.images[0].shape[1], self.images[0].shape[2])).type(torch.cuda.FloatTensor).detach()))
        self.ambient_net_inputs = [np_to_torch(original_noise).type(torch.cuda.FloatTensor).detach()
                                for original_noise in original_noises]

    def _get_augmentation(self, iteration):
        if iteration % 4 in [1, 2, 3]:
            return 0
        iteration //= 2
        return iteration % 8

    def _clip_t(self, t_map):
        if self.clip:
            return np.clip(t_map, 0.1, 1)
        else:
            return np.clip(t_map, 0, 1)

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            if self.plot_during_training:
                self._plot_closure(j)
            if self.done:
                return
            optimizer.step()

    def _optimization_closure(self, step):
        if step == self.num_iter - 1:
            aug = 0
            reg_std = 0
        else:
            aug = self._get_augmentation(step)
            reg_std = 1 / 500

        image_net_input = self.image_net_inputs[aug] + (self.image_net_inputs[aug].clone().normal_() * reg_std)
        self.image_out = self.image_net(image_net_input)

        ambient_net_input = self.ambient_net_inputs[aug] + (self.ambient_net_inputs[aug].clone().normal_() * reg_std)
        self.ambient_out = torch.clamp(self.ambient_net(ambient_net_input, self.a_rops_torch[aug], self.d_rops_torch[aug]), min=0, max=0.95)
        # self.ambient_out = torch.clamp(self.ambient_net(ambient_net_input, self. a_rops_torch[aug], self.d_rops_torch[aug]), min=0, max=0.95)

        mask_net_input = self.mask_net_inputs[aug] + (self.mask_net_inputs[aug].clone().normal_() * reg_std)
        self.mask_out = torch.clamp(self.mask_net(mask_net_input, self.d_rops_torch[aug]), min=0.15, max=1)
        # self.mask_out = torch.clamp(self.mask_net(mask_net_input, self.d_rops_torch[aug]), min=0.15, max=1)

        weightt_bl = 0.005
        weighta_bl = 0.01
        weightr = 1

        self.losst_bl = self.blur_loss(self.mask_out)
        self.lossa_bl = self.blur_loss(self.ambient_out)
        self.losst = self.l2_loss(self.mask_out, self.t_rops_torch[aug])
        self.lossa = self.l2_loss(self.ambient_out, self.a_rops_torch[aug])
        self.lossr = self.rec_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_out, self.images_torch[aug], alpha=0.5)

        if step < 1000:
            weightt = 0.01
            weighta = 1
            self.total_loss = weightt_bl * self.losst_bl + weightt * self.losst + weighta_bl * self.lossa_bl + weighta * self.lossa + self.lossr * weightr
        else:
            # weighta = 0
            weightt = 1
            weighta = 0.01
            self.total_loss = weightt_bl * self.losst_bl + weightt * self.losst + weighta_bl * self.lossa_bl + weighta * self.lossa + self.lossr * weightr

        if not torch.isfinite(self.total_loss).item():
            raise FloatingPointError("Non-finite loss; stopping this image")
        self.total_loss.backward(retain_graph=True)

        image_out_np = np.clip(torch_to_np(self.image_out), 0, 1)
        mask_out_np = np.clip(torch_to_np(self.mask_out),0, 1)
        ambient_out_np = np.clip(torch_to_np(self.ambient_out), 0, 1)
        reconstruction = mask_out_np * image_out_np + (1 - mask_out_np) * ambient_out_np
        # Score in the same augmented coordinates as this forward pass.
        psnr = compare_psnr(self.images[aug].astype(np.float32), reconstruction.astype(np.float32), data_range=1.0)
        # Store every candidate in canonical working-image coordinates.
        self.current_result = DehazeResult(
            learned=inverse_augmentation(image_out_np, aug),
            t=inverse_augmentation(mask_out_np, aug),
            a=inverse_augmentation(ambient_out_np, aug), psnr=psnr)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result

    def _plot_closure(self, step):
        print('\nIteration:%05d    total_loss:%f    lossr:%f    losst:%f    losst_bl:%f   lossa_bl:%f    current_psnr:%f    max_psnr:%f'
              % (step, self.total_loss.item(), self.lossr.item(), self.losst.item(), self.losst_bl.item(),  self.lossa_bl.item(),
                 self.current_result.psnr, self.best_result.psnr), end='')

        if step % self.show_every == self.show_every - 1:
            self.mid_t = self._restore_map(self.best_result.t)
            self.mid_a = self._restore_map(self.best_result.a)
            self.mid_clean = np.clip((self.source_image - ((1 - self.mid_t) * self.mid_a)) / self.mid_t, 0, 1)

            self._save_image(self.image_name + "_", np.clip(self.image, 0, 1))
            self._save_image(self.image_name + "_" + str(step + 1) + "_clean_mid", self.mid_clean)
            ambient_resized = np_imresize(self.mid_a, output_shape=self.best_result.learned.shape[1:])
            self._save_image(self.image_name + "_" + "_a_mid", ambient_resized)
            self._save_image(self.image_name + "_" + "_t_mid", self.mid_t)

            self.mid_clean_gamma = self.gamma(self.mid_clean, self.param)
            self._save_image(self.image_name + "_" + str(step + 1) + "_clean_mid_gamma", self.mid_clean_gamma)

    def finalize(self):
        self.final_image = self._restore_map(self.best_result.learned)
        self.final_t_map = self._restore_map(self.best_result.t)
        self.final_a = self._restore_map(self.best_result.a)

        final_t_map = self._clip_t(self.final_t_map)
        self.post = np.clip((self.source_image - ((1 - final_t_map) * self.final_a)) / final_t_map, 0, 1)
        self.post = self.gamma(self.post, self.param)

        self._save_image(self.image_name + "_t_best", final_t_map)
        self._save_image(self.image_name + "_clean_best", self.post)
        self._save_image(self.image_name + "_a_best", np.clip(self.final_a, 0, 1))
        print()

    def gamma(self, img, param):
        # 调整百分位
        img = np.transpose(img, (1, 2, 0))
        mi = np.percentile(img, param['mi'], axis=(0, 1))
        ma = np.percentile(img, param['ma'], axis=(0, 1))
        img = percentile_stretch(img, mi, ma, "final RGB percentile stretch")
        img = np.round((img * 255))
        img[img > 255] = 255
        img[img < 0] = 0
        img = img.astype(np.uint8)

        # gamma矫正
        i = np.arange(256)
        f = ((i + 0.5) / 256) ** (2/3)  #  1/3
        LUT = np.uint8(np.round(f * 256 - 0.5))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        img[:, :, 0] = LUT[img[:, :, 0]]
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
        img = np.clip(img / 255.0, 0, 1)
        img = np.transpose(img, (2, 0, 1))
        return img


def dehaze(image_name, image, num_iter=3000, plot_during_training=True, show_every=100, *, output_file=None, debug_dir=None):
    dh = Dehaze(image_name, image, num_iter, plot_during_training, show_every=show_every, clip=True, output_file=output_file, debug_dir=debug_dir)
    dh.optimize()
    dh.finalize()



def main(argv=None):
    from depcsd_runtime import run_cli
    return run_cli(Dehaze, argv)


if __name__ == "__main__":
    raise SystemExit(main())
