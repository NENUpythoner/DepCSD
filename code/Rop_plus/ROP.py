from utils.numerics import percentile_stretch, relative_change, require_finite
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2
from Rop_plus.ROP_imresize import imresize
import torch
from PIL import Image


def im2processing():
    # 建立不同类型图像文件路径
    if not os.path.isdir('Hazy'):
        os.mkdir('Hazy')
    if not os.path.isdir('Sandstorm'):
        os.mkdir('Sandstorm')
    if not os.path.isdir('Underwater'):
        os.mkdir('Underwater')
    if not os.path.isdir('Landscape'):
        os.mkdir('Landscape')

    print(' ')
    print('  ********** Recovery Test Based on Rank1+ **********')
    print(' ')
    print(' Please select the problem to be dealt with :')
    print(' ')
    print('  {}. {}'.format(1, 'Hazy image;'))
    print('  {}. {}'.format(2, 'Sandstorm image;'))
    print('  {}. {}'.format(3, 'Underwater image;'))
    print('  {}. {}'.format(4, 'Landscape.'))
    print(' ')

    item = 0
    # while not isinstance(item, int) or item < 1 or item > 4:
    #     try:
    #         item = int(input('  Please choose (1-4): '))
    #     except ValueError:
    #         print('输入错误！')
    #         break
    while not isinstance(item, int) or item < 1 or item > 4:
        item = int(input('  Please choose (1-4): '))

    parameters, road = settings(item)

    print(' ')
    print("Loading path ... (^o^)/")
    # 获取当前脚本所在路径
    script_path = os.path.realpath(__file__)
    pathstr, _ = os.path.split(script_path)

    # 拼接路径
    dirname = os.path.join(pathstr, road)

    # 列出文件夹中的文件
    imglist = os.listdir(dirname)

    print(" ")
    print(f'There are {len(imglist)} images in the folder "{road}".')
    print(" ")
    return imglist, road, pathstr, parameters

def settings(item):
    parameters = {
        'omega': 0,
        'mi': 0,
        'ma': 0,
        'gamma': 0,
        'iternum': 0
    }

    road = ''

    if item == 1:
        parameters['omega'] = 0.99
        parameters['mi'] = 1
        parameters['ma'] = 95
        parameters['gamma'] = 5e1
        road = 'Hazy'
    elif item == 2:
        # Although the sandstorm image looks dim,
        # it is bright enough.
        # The consistency of the spectrum is good
        parameters['omega'] = 0.8
        parameters['mi'] = 1
        parameters['ma'] = 95
        parameters['gamma'] = 3e1
        road = 'Sandstorm'
    elif item == 3:
        # The underwater image is very dark,
        # but the consistency of the spectrum is good.
        parameters['omega'] = 0.75
        parameters['mi'] = 1
        parameters['ma'] = 95
        parameters['gamma'] = 5e1
        road = 'Underwater'
    elif item == 4:
        parameters['omega'] = 0.7
        parameters['mi'] = 2
        parameters['ma'] = 99.6
        parameters['gamma'] = 1e2
        road = 'Landscape'

    parameters['iternum'] = 10

    return parameters, road

def get_atmosphere(image, scatterlight):
    for i in range(1):
        scatter_est = np.sum(scatterlight, axis=2)
        n_pixels = np.prod(scatter_est.shape)
        n_search_pixels = int(np.floor(n_pixels * 0.001))
        image_vec = image.reshape(n_pixels, 3, order='F')
        indices = np.argsort(scatter_est.T.flatten())[::-1]
        atmosphere = np.mean(image_vec[indices[:n_search_pixels], :], axis=0)

        atmos = np.zeros((1, 1, 3))
        atmos[0, 0, :] = atmosphere
        atmosphere = np.tile(atmos, (*scatter_est.shape, 1))

        sek = scatter_est.T.flatten()[indices[n_search_pixels]]

        scatterlight = scatterlight * np.tile((scatter_est <= sek)[:, :, np.newaxis], (1, 1, 3)) + \
                       (2 / 3 * sek - scatterlight) * np.tile((scatter_est > sek)[:, :, np.newaxis], (1, 1, 3))

        return atmosphere, scatterlight

def scattering_mask_sample(I):
    mSize = min(I.shape[0], I.shape[1])
    if mSize < 800:
        r = 0.02
    elif 800 <= mSize < 1500:
        r = 0.01
    else:
        r = 0.005

    # newh = int(np.ceil(I.shape[0] * r))
    # neww = int(np.ceil(I.shape[1] * r))
    # I0 = cv2.resize(I, (neww, newh), interpolation=cv2.INTER_CUBIC)
    # F = cv2.resize(I0, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_CUBIC)

    # I0 = imresize(I, output_shape=(newh, neww))
    I0 = imresize(I, scalar_scale=r)
    F = imresize(I0, output_shape=(I.shape[0], I.shape[1]))
    return F

def gamma0(img):
    i = np.arange(256)
    f = ((i + 0.5) / 256) ** (5 / 6)
    LUT = np.uint8(np.round(f * 256 - 0.5))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img[:, :, 0] = LUT[img[:, :, 0]]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

    return img

def Rank1Plus(img, param):
    imgvec = img.reshape(-1, 3, order='F')  # unified spectrum
    # To update unified spectrum
    # selectedpixel = np.ones_like(imgvec)
    selectedpixel = np.ones((imgvec.shape[0],), dtype=bool)
    previous_basis = np.ones((1, 1, 3))
    x_RGB = np.zeros((1, 1, 3))
    for step in range(1, 21):
        # if step == 1:
        #     x_RGB[0, 0, 0:3] = imgvec[0, :]  # unified spectrum
        # else:
        #     x_RGB[0, 0, 0:3] = np.mean(imgvec[selectedpixel], axis=0)
        if not np.any(selectedpixel):
            raise ValueError("ROP selected an empty spectrum set; no fallback spectrum is invented")
        x_RGB[0, 0, 0:3] = np.mean(imgvec[selectedpixel], axis=0)

        # Create the unified spectrum in each pixel
        x_mean = np.tile(x_RGB, (img.shape[0], img.shape[1], 1))

        # Normalization
        spec_basis = x_mean / np.maximum(np.sqrt(np.sum(x_mean ** 2, axis=2, keepdims=True)), 0.001)

        # Normalization
        img_normalized = img / np.maximum(np.sqrt(np.sum(img ** 2, axis=2, keepdims=True)), 0.001)

        # Projection similarity
        proj_sim = np.tile(np.sum(spec_basis * img_normalized, axis=2, keepdims=True), (1, 1, 3))

        # scattering_light_estimation
        if np.sum(np.abs(spec_basis[0, 0, 0:3] - previous_basis)) != 0:
            previous_basis = spec_basis[0, 0, 0:3]

            boots = proj_sim.reshape(img.shape[0] * img.shape[1], 3, order='F')

            selectedpixel = boots[:, 0] >= 0.99
        else:
            break

    uspec_light = proj_sim * (np.sum(img, axis=2) / np.maximum(np.sum(x_mean, axis=2), 0.001))[:, :,
                             np.newaxis] * x_mean

    initial_img = img
    # get_atmosphere
    atmosphere, uspec_light = get_atmosphere2(initial_img, uspec_light)
    # T = 1 - omega * \tilde{t}
    T_ini = 1 - param['omega'] * uspec_light

    ###############################################################
    # Rank1 +: Transmission Optimization
    Jr, tildeT = Trans_V8(atmosphere, initial_img, 1 - x_mean, T_ini, param)
    ###############################################################
    # Postprocessing;  module for luminance adjustment,
    # For some thick-fog scenes, this operation is not recommended;
    mi = np.percentile(Jr, param['mi'], axis=(0, 1))
    ma = np.percentile(Jr, param['ma'], axis=(0, 1))
    Jr = percentile_stretch(Jr, mi, ma, "ROP RGB percentile stretch")

    Jr = np.round((Jr * 255))
    Jr[Jr > 255] = 255
    Jr[Jr < 0] = 0
    Jr = gamma0(Jr.astype(np.uint8))

    return Jr, tildeT, atmosphere

def get_atmosphere2(image, uspec_light):
    scatter_est = np.sum(uspec_light, axis=2)
    n_pixels = np.prod(scatter_est.shape)
    n_search_pixels = int(np.floor(n_pixels * 0.02))
    image_vec = image.reshape(n_pixels, 3, order='F')
    indices = np.argsort(scatter_est.T.flatten())[::-1]
    atmosphere = np.mean(image_vec[indices[:n_search_pixels], :], axis=0)

    atmos = np.zeros((1, 1, 3))
    atmos[0, 0, :] = atmosphere
    atmosphere = np.tile(atmos, (*scatter_est.shape, 1))

    sek = scatter_est.T.flatten()[indices[n_search_pixels]]
    sek_vec = np.tile(sek * uspec_light[0, 0, :] / np.maximum(scatter_est[0, 0], 0.001),
                      (scatter_est.shape[0], scatter_est.shape[1], 1))
    uspec_light = uspec_light * np.tile((scatter_est <= sek)[:, :, np.newaxis], (1, 1, 3)) + \
                  (2 * sek_vec - uspec_light) * np.tile((scatter_est > sek)[:, :, np.newaxis], (1, 1, 3))

    return atmosphere, uspec_light

def Trans_V8(atmosphere, I, sc, t0, param):
    # Initializing
    C = getC(t0)
    D, Dt = defDDt()
    MaxIter = param['iternum']
    gamma = param['gamma']

    lambda1 = 5  # lambda1
    lambda2 = 5e-1  # lambda2
    lambda3 = 0.5  # lambda3
    beta = 10  # beta1 & beta2

    # Lagrange Multipliers
    m, n, ch = t0.shape
    Lxi1 = np.zeros((m, n, ch))
    Lxi2 = np.zeros((m, n, ch))
    Leta1 = np.zeros((m, n))
    Leta2 = np.zeros((m, n))

    SolRE = 5e-4
    tao = 1.618

    # diff
    D1I, D2I = D(I)
    ts = np.mean(t0, axis=2)
    D1tG, D2tG = D(ts)
    D1t = sc * np.repeat(D1tG[:, :, np.newaxis], 3, axis=2)
    D2t = sc * np.repeat(D2tG[:, :, np.newaxis], 3, axis=2)

    t = ts
    cont = 1
    k = 0

    while cont:
        k += 1
        # X - subproblem
        Xterm1 = D1t - D1I + Lxi1 / beta
        Xterm2 = D2t - D2I + Lxi2 / beta
        Xterm = np.sqrt(Xterm1 ** 2 + Xterm2 ** 2)
        W = np.exp(-gamma * (np.abs(D1I) + np.abs(D2I)))
        Xterm = np.maximum(Xterm - (W * lambda1) / beta, 0) / (Xterm + np.finfo(float).eps)
        Xmg1 = Xterm1 * Xterm
        Xmg2 = Xterm2 * Xterm

        # Z - subproblem
        Zterm1 = D1tG + Leta1 / beta
        Zterm2 = D2tG + Leta2 / beta
        Zterm = np.sqrt(Zterm1 ** 2 + Zterm2 ** 2)
        Zterm = np.maximum(Zterm - lambda2 / beta, 0) / (Zterm + np.finfo(float).eps)
        Zmg1 = Zterm1 * Zterm
        Zmg2 = Zterm2 * Zterm

        # t - subproblem
        zeta1X = Xmg1 + D1I - Lxi1 / beta
        zeta1Y = Xmg2 + D2I - Lxi2 / beta
        zeta2X = Zmg1 - Leta1 / beta
        zeta2Y = Zmg2 - Leta2 / beta

        # https: // blog.csdn.net / weixin_38316806 / article / details / 104951212
        Dt_sc_zeta = Dt(sc * zeta1X, sc * zeta1Y)
        fft2_Dt_sc_zeta = np.empty_like(Dt_sc_zeta, dtype=complex)
        for c in range(Dt_sc_zeta.shape[2]):
            fft2_Dt_sc_zeta[:, :, c] = fft2(Dt_sc_zeta[:, :, c])

        # ttem = fft2(lambda3 * ts) + \
        #        beta * np.sum(fft2(Dt(sc * zeta1X, sc * zeta1Y)), axis=2) + \
        #        beta * fft2(Dt(zeta2X, zeta2Y))
        ttem = fft2(lambda3 * ts) + \
               beta * np.sum(fft2_Dt_sc_zeta, axis=2) + \
               beta * fft2(Dt(zeta2X, zeta2Y))
        ttemp = lambda3 + beta * (np.sum(sc ** 2 * C['eigsDtD'], axis=2) + C['eigsDtD2'])
        tnew = np.real(ifft2(ttem / (ttemp + np.finfo(float).eps)))

        D1tG, D2tG = D(tnew)
        D1t = sc * np.repeat(D1tG[:, :, np.newaxis], 3, axis=2)
        D2t = sc * np.repeat(D2tG[:, :, np.newaxis], 3, axis=2)

        Lxi1 = Lxi1 - tao * beta * (Xmg1 - (D1t - D1I))
        Lxi2 = Lxi2 - tao * beta * (Xmg2 - (D2t - D2I))
        Leta1 = Leta1 - tao * beta * (Zmg1 - D1tG)
        Leta2 = Leta2 - tao * beta * (Zmg2 - D2tG)

        re = relative_change(tnew, t)
        t = tnew
        cont = (k < MaxIter) and (re > SolRE)

    T = sc * np.repeat(t[:, :, np.newaxis], 3, axis=2)
    Jr = (I - atmosphere) / np.maximum(T, 0.001) + atmosphere

    return Jr, T

def getC(t0):
    sizeF = t0.shape

    C = {}
    C['eigsD1'] = psf2otf(np.array([[[1], [-1]]]), sizeF)  # ▽x
    C['eigsD2'] = psf2otf(np.array([[[1]], [[-1]]]), sizeF)  # ▽y
    C['eigsDtD'] = np.abs(C['eigsD1']) ** 2 + np.abs(C['eigsD2']) ** 2  # ▽t*▽

    C['eigsD21'] = psf2otf(np.array([[1, -1]]), [sizeF[0], sizeF[1]])  # ▽x
    C['eigsD22'] = psf2otf(np.array([[1], [-1]]), [sizeF[0], sizeF[1]])
    C['eigsDtD2'] = np.abs(C['eigsD21']) ** 2 + np.abs(C['eigsD22']) ** 2

    return C

def psf2otf(psf, size):
    '''
    参考文章：
    https://blog.csdn.net/weixin_44487231/article/details/124594741
    '''
    if not (0 in psf):
        psf = np.double(psf)
        psfshape = np.shape(psf)
        padsize = size - np.array(psfshape)

        # 确保填充的维度与PSF的维度相匹配
        padsize = [(0, padsize[i]) for i in range(len(padsize))]

        psf = np.pad(psf, padsize, 'constant')
        # 计算中心点的偏移量
        center_offsets = [-int(np.floor(psfshape[i] / 2)) for i in range(len(psfshape))]

        # 对PSF进行循环移位
        psf = np.roll(psf, center_offsets, axis=tuple(range(len(psfshape))))

        otf = np.fft.fftn(psf, axes=tuple(range(len(psfshape))))
        nElem = np.prod(psfshape, axis=0)
        nOps = 0
        for k in range(len(psfshape)):
            nffts = nElem / psfshape[k]
            nOps = nOps + psfshape[k] * np.log2(psfshape[k]) * nffts
        mx1 = abs(np.imag(otf.reshape(-1))).max(0)
        mx2 = abs(otf.reshape(-1)).max(0)
        if mx2 == 0 or mx1 / mx2 <= nOps * np.finfo(float).eps:
            otf = np.real(otf)
    else:
        otf = np.zeros(size)

    return otf

def defDDt():
    '''
    defines finite difference operator D
    and its transpose operator
    referring to FTVD code
    '''
    D = lambda U: ForwardD(U)
    Dt = lambda X, Y: Dive(X, Y)

    return D, Dt

def ForwardD(U):
    # Forward finite difference operator.
    Dux = np.hstack((np.diff(U, axis=1), U[:, :1] - U[:, -1:]))
    Duy = np.vstack((np.diff(U, axis=0), U[:1, :] - U[-1:, :]))

    return Dux, Duy

def Dive(X, Y):
    DtX = np.hstack((X[:, -1:] - X[:, :1], -np.diff(X, axis=1)))
    DtY = np.vstack((Y[-1:, :] - Y[:1, :], -np.diff(Y, axis=0)))
    DtXY = DtX + DtY

    return DtXY

def ROP_main(img, param):
    img = np.array(img).astype(np.float32)
    require_finite(img, "ROP input")
    J_rop, t_rop, a_rop = Rank1Plus(img, param)
    for label, value in (("ROP J", J_rop), ("ROP t", t_rop), ("ROP A", a_rop)):
        require_finite(value, label)
    return J_rop, t_rop, a_rop

