import numpy as np
import skimage.io as io, skimage.color as skiCol, skimage.filters as skiFilt
import matplotlib.pyplot as plt
import random


def generate_sample():
    img = io.imread("download.jpg")
    height, width, channel = img.shape
    print(img.dtype, np.max(img))
    img_stich = np.zeros((height, 3 * width, channel), dtype=np.uint8)
    img_stich[:, :width, :] = img
    img_stich[:, width : 2 * width, :] = img
    img_stich[:, 2 * width : 3 * width, :] = img
    io.imsave("stiched_sample.jpg", img_stich)


def analyse_sample():
    img = io.imread("stiched_sample.jpg")
    img_yuv = skiCol.rgb2yuv(img)
    plt.imshow(img_yuv[:, :, 2])
    plt.show()


def get_split_index(grad: np.array):
    args = np.argsort(grad)
    res = [args[0]]
    for i in range(1, args.shape[0]):
        if args[0] - args[i] > 20:
            res.append(args[i])
            break
    res = np.sort(res)
    return res


def get_stitch_split(val_ch: np.array, sample_num: int = 20, kernel_width: int = 30):
    height, width = val_ch.shape
    sample_ind = random.sample(range(15, height - 15), sample_num)
    kernel_list = [1.0 for i in range(kernel_width // 2)]
    kernel_list.extend([-1.0 for i in range(kernel_width // 2)])
    kernel = np.array(kernel_list)
    split_index_list = []
    for i in range(sample_num):
        conv = np.convolve(val_ch[sample_ind[i], :], kernel, "valid")
        split_index = get_split_index(conv)
        split_index += kernel_width // 2
        split_index_list.append(split_index)
    split_index_arr = np.stack(split_index_list)
    res_ind = [0]
    uni, cnt = np.unique(split_index_arr[:, 0], return_counts=True)
    res_ind.append(uni[np.argsort(cnt)[-1]])
    uni, cnt = np.unique(split_index_arr[:, 1], return_counts=True)
    res_ind.append(uni[np.argsort(cnt)[-1]])
    res_ind.append(width)
    return np.sort(res_ind)


def vignet_correction(img: np.array):
    return img


def piecewise_vignet(img: np.array):
    _, width, _ = img.shape
    img_yuv = skiCol.rgb2yuv(img)
    val_ch = img_yuv[:, :, 2]
    val_ch = abs(val_ch)
    val_ch = skiFilt.gaussian(val_ch)
    split_index = get_stitch_split(val_ch)
    print(split_index)
    for i in range(3):
        val_ind = val_ch[:, split_index[i] : split_index[i + 1]]
        val_ind = vignet_correction(val_ind)
        val_ch[:, split_index[i] : split_index[i + 1]] = val_ind
    img_yuv[:, :, 2] = -val_ch
    img_rgb = skiCol.yuv2rgb(img_yuv)
    print(np.sum(img_rgb - img))
    io.imsave("recons.jpg", img_rgb)


# analyse_sample()
# generate_sample()
img = io.imread("stiched_sample.jpg")
piecewise_vignet(img)
