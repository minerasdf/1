import time

import numpy as np
from scipy import integrate
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import minimize


def read_ascan_x(image):
    image_row_data = {}  # 存储每行的像素值
    width, height = image.shape[0], image.shape[0]  # 获取图片的宽度和高度
    for y in range(height):
        row_data = []
        for x in range(width):
            row_data.append(image[y, x])  # 存入当前行的数据中
        image_row_data[y] = row_data  # 按行索引存储
    return image_row_data


def Ascan_x(initial_ascan, position_x, position_y):
    return initial_ascan[int(position_y)][int(position_x)]


def read_ascan_y(image):
    image_column_data = {}  # 存储每列的像素值
    width, height = image.shape[0], image.shape[1]
    for y in range(width):
        column_data = []
        for x in range(height):
            column_data.append(image[x, y])
        image_column_data[y] = column_data
    return image_column_data


def Ascan_y(initial_ascan, position_x, position_y):
    return initial_ascan[int(position_x)][int(position_y)]


def clip_coordinate(value, min_val, max_val):
    """
    将坐标裁剪到[min_val, max_val]范围内。
    """
    return max(min_val, min(value, max_val))


def cubic_hermite_interpolation(p0, p1, m0, m1, t):
    """三次厄米特样条插值"""
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t ** 2 * (3 - 2 * t)
    h11 = t ** 2 * (t - 1)

    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def resampling_at_time_0_x(initial_ascan_x, position_x, position_y, disp_x):
    new_x = position_x + disp_x
    new_y = position_y
    # 将new_x,new_y拉回合法坐标范围
    new_x = clip_coordinate(new_x, 0, 499)
    new_y = clip_coordinate(new_y, 0, 499)
    if 499 >= new_x >= 0 == new_x % 1 and 499 >= new_y >= 0 == new_y % 1:
        return Ascan_x(initial_ascan_x, new_x, new_y)
    else:
        x0 = int(np.floor(new_x))
        y0 = int(np.floor(new_y))
        x1 = min(x0 + 1, 499)
        y1 = min(y0 + 1, 499)
        Ascan_vals = [
            np.array(Ascan_x(initial_ascan_x, x0, y0)),
            np.array(Ascan_x(initial_ascan_x, x1, y0)),
            np.array(Ascan_x(initial_ascan_x, x0, y1)),
            np.array(Ascan_x(initial_ascan_x, x1, y1))
        ]
        # 中心差分估计方向导数
        dx_val = [(Ascan_vals[1] - Ascan_x(initial_ascan_x, max(x0 - 1, 0), y0)) / 2,
                  (Ascan_vals[3] - Ascan_x(initial_ascan_x, max(x0 - 1, 0), y1)) / 2]
        dy_val = [(Ascan_vals[2] - Ascan_x(initial_ascan_x, x0, max(y0 - 1, 0))) / 2,
                  (Ascan_vals[3] - Ascan_x(initial_ascan_x, x1, max(y0 - 1, 0))) / 2]
        t_x = new_x - x0
        t_y = new_y - y0

        # 在 x 方向上使用三次厄米特样条插值
        interp_y0 = cubic_hermite_interpolation(Ascan_vals[0], Ascan_vals[1], dx_val[0], dx_val[1], t_x)
        interp_y1 = cubic_hermite_interpolation(Ascan_vals[2], Ascan_vals[3], dx_val[0], dx_val[1], t_x)

        # 在 y 方向上使用三次厄米特样条插值
        final_interp = cubic_hermite_interpolation(interp_y0, interp_y1, dy_val[0], dy_val[1], t_y)
        return final_interp


def all_resampling_x_fast(initial_ascan_x, Disp):
    volume_1 = np.zeros(250000)
    for i in range(500):
        for j in range(500):
            volume_1[i * 500 + j] = resampling_at_time_0_x(initial_ascan_x, j, i, Disp[i * 500 + j])

    return volume_1


def resampling_at_time_0_y(initial_ascan_y, position_x, position_y, disp_y):
    new_x = position_x
    new_y = position_y + disp_y
    # 将new_x,new_y拉回合法坐标范围
    new_x = clip_coordinate(new_x, 0, 499)
    new_y = clip_coordinate(new_y, 0, 499)
    if 499 >= new_x >= 0 == new_x % 1 and 499 >= new_y >= 0 == new_y % 1:
        return Ascan_y(initial_ascan_y, new_x, new_y)
    else:
        x0 = int(np.floor(new_x))
        y0 = int(np.floor(new_y))
        x1 = min(x0 + 1, 499)
        y1 = min(y0 + 1, 499)
        Ascan_vals = [
            np.array(Ascan_y(initial_ascan_y, x0, y0)),
            np.array(Ascan_y(initial_ascan_y, x1, y0)),
            np.array(Ascan_y(initial_ascan_y, x0, y1)),
            np.array(Ascan_y(initial_ascan_y, x1, y1))
        ]
        # 中心差分估计方向导数
        dx_val = [(Ascan_vals[1] - Ascan_y(initial_ascan_y, max(x0 - 1, 0), y0)) / 2,
                  (Ascan_vals[3] - Ascan_y(initial_ascan_y, max(x0 - 1, 0), y1)) / 2]
        dy_val = [(Ascan_vals[2] - Ascan_y(initial_ascan_y, x0, max(y0 - 1, 0))) / 2,
                  (Ascan_vals[3] - Ascan_y(initial_ascan_y, x1, max(y0 - 1, 0))) / 2]
        t_x = new_x - x0
        t_y = new_y - y0

        # 在 x 方向上使用三次厄米特样条插值
        interp_y0 = cubic_hermite_interpolation(Ascan_vals[0], Ascan_vals[1], dx_val[0], dx_val[1], t_x)
        interp_y1 = cubic_hermite_interpolation(Ascan_vals[2], Ascan_vals[3], dx_val[0], dx_val[1], t_x)

        # 在 y 方向上使用三次厄米特样条插值
        final_interp = cubic_hermite_interpolation(interp_y0, interp_y1, dy_val[0], dy_val[1], t_y)
        return final_interp


def all_resampling_y_fast(initial_ascan_y, Disp):
    volume_2 = np.zeros(250000)
    for i in range(500):
        for j in range(500):
            volume_2[i * 500 + j] = resampling_at_time_0_y(initial_ascan_y, j, i, Disp[500 ** 2 + 499 + 500 * j - i])

    return volume_2


def similarity_term(initial_ascan_x, initial_ascan_y, Disp):
    v_1 = all_resampling_x_fast(initial_ascan_x, Disp)
    v_2 = all_resampling_y_fast(initial_ascan_y, Disp)
    count = np.sum((v_1 - v_2) ** 2)

    return count


def regularity_term(Disp):
    t_1 = np.linspace(0, 249999, 250000)
    t_2 = np.linspace(250000, 499999, 250000)

    # 直接分割数组，不需要二维索引
    disp1 = Disp[0:250000]
    disp2 = Disp[250000:500000]

    # 计算第一部分的导数
    fx_derivative = np.gradient(disp1, t_1) ** 2

    # 计算第二部分的导数
    fy_derivative = np.gradient(disp2, t_2) ** 2

    # 计算积分
    f_integral = (integrate.cumulative_trapezoid(fx_derivative, t_1, initial=0)[-1] +
                  integrate.cumulative_trapezoid(fy_derivative, t_2, initial=0)[-1])

    return f_integral


def objective_function(initial_ascan_x, initial_ascan_y, Disp, alpha):
    return similarity_term(initial_ascan_x, initial_ascan_y, Disp) + alpha * regularity_term(Disp)


def normalize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    normalized_image = (image - mean) / (std + 1e-8)  # Avoid division by zero
    return normalized_image


def read_and_normalize_images(image_path1, image_path2):
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # Normalize each image
    norm_img1 = normalize_image(img1)
    norm_img2 = normalize_image(img2)

    return norm_img1, norm_img2


def create_pyramid(image, levels):
    """创建图像金字塔"""
    pyramid = [image]
    for i in range(levels - 1):
        # 使用高斯模糊进行降采样，避免混叠
        blurred = cv2.GaussianBlur(pyramid[-1], (3, 3), 0)
        downsampled = cv2.resize(blurred,
                                 (pyramid[-1].shape[1] // 2, pyramid[-1].shape[0] // 2),
                                 interpolation=cv2.INTER_AREA)
        pyramid.append(downsampled)
    return pyramid[::-1]  # 从粗到细返回


def resampling_at_time_0_x_level4(initial_ascan_x, position_x, position_y, disp_x):
    new_x = position_x + disp_x
    new_y = position_y
    # 将new_x,new_y拉回合法坐标范围
    new_x = clip_coordinate(new_x, 0, 249)
    new_y = clip_coordinate(new_y, 0, 249)
    if 249 >= new_x >= 0 == new_x % 1 and 249 >= new_y >= 0 == new_y % 1:
        return Ascan_x(initial_ascan_x, new_x, new_y)
    else:
        x0 = int(np.floor(new_x))
        y0 = int(np.floor(new_y))
        x1 = min(x0 + 1, 249)
        y1 = min(y0 + 1, 249)
        Ascan_vals = [
            np.array(Ascan_x(initial_ascan_x, x0, y0)),
            np.array(Ascan_x(initial_ascan_x, x1, y0)),
            np.array(Ascan_x(initial_ascan_x, x0, y1)),
            np.array(Ascan_x(initial_ascan_x, x1, y1))
        ]
        # 中心差分估计方向导数
        dx_val = [(Ascan_vals[1] - Ascan_x(initial_ascan_x, max(x0 - 1, 0), y0)) / 2,
                  (Ascan_vals[3] - Ascan_x(initial_ascan_x, max(x0 - 1, 0), y1)) / 2]
        dy_val = [(Ascan_vals[2] - Ascan_x(initial_ascan_x, x0, max(y0 - 1, 0))) / 2,
                  (Ascan_vals[3] - Ascan_x(initial_ascan_x, x1, max(y0 - 1, 0))) / 2]
        t_x = new_x - x0
        t_y = new_y - y0

        # 在 x 方向上使用三次厄米特样条插值
        interp_y0 = cubic_hermite_interpolation(Ascan_vals[0], Ascan_vals[1], dx_val[0], dx_val[1], t_x)
        interp_y1 = cubic_hermite_interpolation(Ascan_vals[2], Ascan_vals[3], dx_val[0], dx_val[1], t_x)

        # 在 y 方向上使用三次厄米特样条插值
        final_interp = cubic_hermite_interpolation(interp_y0, interp_y1, dy_val[0], dy_val[1], t_y)
        return final_interp


def all_resampling_x_fast_level4(initial_ascan_x, Disp):
    volume_1 = np.zeros(62500)
    for i in range(250):
        for j in range(250):
            volume_1[i * 250 + j] = resampling_at_time_0_x_level4(initial_ascan_x, j, i, Disp[i * 250 + j])

    return volume_1


def resampling_at_time_0_y_level4(initial_ascan_y, position_x, position_y, disp_y):
    new_x = position_x
    new_y = position_y + disp_y
    # 将new_x,new_y拉回合法坐标范围
    new_x = clip_coordinate(new_x, 0, 249)
    new_y = clip_coordinate(new_y, 0, 249)
    if 249 >= new_x >= 0 == new_x % 1 and 249 >= new_y >= 0 == new_y % 1:
        return Ascan_y(initial_ascan_y, new_x, new_y)
    else:
        x0 = int(np.floor(new_x))
        y0 = int(np.floor(new_y))
        x1 = min(x0 + 1, 249)
        y1 = min(y0 + 1, 249)
        Ascan_vals = [
            np.array(Ascan_y(initial_ascan_y, x0, y0)),
            np.array(Ascan_y(initial_ascan_y, x1, y0)),
            np.array(Ascan_y(initial_ascan_y, x0, y1)),
            np.array(Ascan_y(initial_ascan_y, x1, y1))
        ]
        # 中心差分估计方向导数
        dx_val = [(Ascan_vals[1] - Ascan_y(initial_ascan_y, max(x0 - 1, 0), y0)) / 2,
                  (Ascan_vals[3] - Ascan_y(initial_ascan_y, max(x0 - 1, 0), y1)) / 2]
        dy_val = [(Ascan_vals[2] - Ascan_y(initial_ascan_y, x0, max(y0 - 1, 0))) / 2,
                  (Ascan_vals[3] - Ascan_y(initial_ascan_y, x1, max(y0 - 1, 0))) / 2]
        t_x = new_x - x0
        t_y = new_y - y0

        # 在 x 方向上使用三次厄米特样条插值
        interp_y0 = cubic_hermite_interpolation(Ascan_vals[0], Ascan_vals[1], dx_val[0], dx_val[1], t_x)
        interp_y1 = cubic_hermite_interpolation(Ascan_vals[2], Ascan_vals[3], dx_val[0], dx_val[1], t_x)

        # 在 y 方向上使用三次厄米特样条插值
        final_interp = cubic_hermite_interpolation(interp_y0, interp_y1, dy_val[0], dy_val[1], t_y)
        return final_interp


def all_resampling_y_fast_level4(initial_ascan_y, Disp):
    volume_2 = np.zeros(62500)
    for i in range(250):
        for j in range(250):
            volume_2[i * 250 + j] = resampling_at_time_0_y_level4(initial_ascan_y, j, i,
                                                                  Disp[250 ** 2 + 249 + 250 * j - i])

    return volume_2


def similarity_term_level4(initial_ascan_x, initial_ascan_y, Disp):
    v_1 = all_resampling_x_fast_level4(initial_ascan_x, Disp)
    v_2 = all_resampling_y_fast_level4(initial_ascan_y, Disp)
    count = np.sum((v_1 - v_2) ** 2)

    return count


def regularity_term_level4(Disp):
    t_1 = np.linspace(0, 62499, 62500)
    t_2 = np.linspace(62500, 124999, 62500)

    # 直接分割数组，不需要二维索引
    disp1 = Disp[0:62500]
    disp2 = Disp[62500:125000]

    # 计算第一部分的导数
    fx_derivative = np.gradient(disp1, t_1) ** 2

    # 计算第二部分的导数
    fy_derivative = np.gradient(disp2, t_2) ** 2

    # 计算积分
    f_integral = (integrate.cumulative_trapezoid(fx_derivative, t_1, initial=0)[-1] +
                  integrate.cumulative_trapezoid(fy_derivative, t_2, initial=0)[-1])

    return f_integral


def objective_function_level4(initial_ascan_x, initial_ascan_y, Disp, alpha):
    return similarity_term_level4(initial_ascan_x, initial_ascan_y, Disp) + alpha * regularity_term_level4(Disp)


def resampling_at_time_0_x_level3(initial_ascan_x, position_x, position_y, disp_x):
    new_x = position_x + disp_x
    new_y = position_y
    # 将new_x,new_y拉回合法坐标范围
    new_x = clip_coordinate(new_x, 0, 124)
    new_y = clip_coordinate(new_y, 0, 124)
    if 124 >= new_x >= 0 == new_x % 1 and 124 >= new_y >= 0 == new_y % 1:
        return Ascan_x(initial_ascan_x, new_x, new_y)
    else:
        x0 = int(np.floor(new_x))
        y0 = int(np.floor(new_y))
        x1 = min(x0 + 1, 124)
        y1 = min(y0 + 1, 124)
        Ascan_vals = [
            np.array(Ascan_x(initial_ascan_x, x0, y0)),
            np.array(Ascan_x(initial_ascan_x, x1, y0)),
            np.array(Ascan_x(initial_ascan_x, x0, y1)),
            np.array(Ascan_x(initial_ascan_x, x1, y1))
        ]
        # 中心差分估计方向导数
        dx_val = [(Ascan_vals[1] - Ascan_x(initial_ascan_x, max(x0 - 1, 0), y0)) / 2,
                  (Ascan_vals[3] - Ascan_x(initial_ascan_x, max(x0 - 1, 0), y1)) / 2]
        dy_val = [(Ascan_vals[2] - Ascan_x(initial_ascan_x, x0, max(y0 - 1, 0))) / 2,
                  (Ascan_vals[3] - Ascan_x(initial_ascan_x, x1, max(y0 - 1, 0))) / 2]
        t_x = new_x - x0
        t_y = new_y - y0

        # 在 x 方向上使用三次厄米特样条插值
        interp_y0 = cubic_hermite_interpolation(Ascan_vals[0], Ascan_vals[1], dx_val[0], dx_val[1], t_x)
        interp_y1 = cubic_hermite_interpolation(Ascan_vals[2], Ascan_vals[3], dx_val[0], dx_val[1], t_x)

        # 在 y 方向上使用三次厄米特样条插值
        final_interp = cubic_hermite_interpolation(interp_y0, interp_y1, dy_val[0], dy_val[1], t_y)
        return final_interp


def all_resampling_x_fast_level3(initial_ascan_x, Disp):
    volume_1 = np.zeros(15625)
    for i in range(125):
        for j in range(125):
            volume_1[i * 125 + j] = resampling_at_time_0_x_level3(initial_ascan_x, j, i, Disp[i * 125 + j])

    return volume_1


def resampling_at_time_0_y_level3(initial_ascan_y, position_x, position_y, disp_y):
    new_x = position_x
    new_y = position_y + disp_y
    # 将new_x,new_y拉回合法坐标范围
    new_x = clip_coordinate(new_x, 0, 124)
    new_y = clip_coordinate(new_y, 0, 124)
    if 124 >= new_x >= 0 == new_x % 1 and 124 >= new_y >= 0 == new_y % 1:
        return Ascan_y(initial_ascan_y, new_x, new_y)
    else:
        x0 = int(np.floor(new_x))
        y0 = int(np.floor(new_y))
        x1 = min(x0 + 1, 124)
        y1 = min(y0 + 1, 124)
        Ascan_vals = [
            np.array(Ascan_y(initial_ascan_y, x0, y0)),
            np.array(Ascan_y(initial_ascan_y, x1, y0)),
            np.array(Ascan_y(initial_ascan_y, x0, y1)),
            np.array(Ascan_y(initial_ascan_y, x1, y1))
        ]
        # 中心差分估计方向导数
        dx_val = [(Ascan_vals[1] - Ascan_y(initial_ascan_y, max(x0 - 1, 0), y0)) / 2,
                  (Ascan_vals[3] - Ascan_y(initial_ascan_y, max(x0 - 1, 0), y1)) / 2]
        dy_val = [(Ascan_vals[2] - Ascan_y(initial_ascan_y, x0, max(y0 - 1, 0))) / 2,
                  (Ascan_vals[3] - Ascan_y(initial_ascan_y, x1, max(y0 - 1, 0))) / 2]
        t_x = new_x - x0
        t_y = new_y - y0

        # 在 x 方向上使用三次厄米特样条插值
        interp_y0 = cubic_hermite_interpolation(Ascan_vals[0], Ascan_vals[1], dx_val[0], dx_val[1], t_x)
        interp_y1 = cubic_hermite_interpolation(Ascan_vals[2], Ascan_vals[3], dx_val[0], dx_val[1], t_x)

        # 在 y 方向上使用三次厄米特样条插值
        final_interp = cubic_hermite_interpolation(interp_y0, interp_y1, dy_val[0], dy_val[1], t_y)
        return final_interp


def all_resampling_y_fast_level3(initial_ascan_y, Disp):
    volume_2 = np.zeros(15625)
    for i in range(125):
        for j in range(125):
            volume_2[i * 125 + j] = resampling_at_time_0_y_level3(initial_ascan_y, j, i,
                                                                  Disp[125 ** 2 + 124 + 125 * j - i])

    return volume_2


def similarity_term_level3(initial_ascan_x, initial_ascan_y, Disp):
    v_1 = all_resampling_x_fast_level3(initial_ascan_x, Disp)
    v_2 = all_resampling_y_fast_level3(initial_ascan_y, Disp)
    count = np.sum((v_1 - v_2) ** 2)

    return count


def regularity_term_level3(Disp):
    t_1 = np.linspace(0, 15624, 15625)
    t_2 = np.linspace(15625, 31249, 15625)

    # 直接分割数组，不需要二维索引
    disp1 = Disp[0:15625]
    disp2 = Disp[15625:31250]

    # 计算第一部分的导数
    fx_derivative = np.gradient(disp1, t_1) ** 2

    # 计算第二部分的导数
    fy_derivative = np.gradient(disp2, t_2) ** 2

    # 计算积分
    f_integral = (integrate.cumulative_trapezoid(fx_derivative, t_1, initial=0)[-1] +
                  integrate.cumulative_trapezoid(fy_derivative, t_2, initial=0)[-1])

    return f_integral


def objective_function_level3(initial_ascan_x, initial_ascan_y, Disp, alpha):
    return similarity_term_level3(initial_ascan_x, initial_ascan_y, Disp) + alpha * regularity_term_level3(Disp)


def resampling_at_time_0_x_level2(initial_ascan_x, position_x, position_y, disp_x):
    new_x = position_x + disp_x
    new_y = position_y
    # 将new_x,new_y拉回合法坐标范围
    new_x = clip_coordinate(new_x, 0, 61)
    new_y = clip_coordinate(new_y, 0, 61)
    if 61 >= new_x >= 0 == new_x % 1 and 61 >= new_y >= 0 == new_y % 1:
        return Ascan_x(initial_ascan_x, new_x, new_y)
    else:
        x0 = int(np.floor(new_x))
        y0 = int(np.floor(new_y))
        x1 = min(x0 + 1, 61)
        y1 = min(y0 + 1, 61)
        Ascan_vals = [
            np.array(Ascan_x(initial_ascan_x, x0, y0)),
            np.array(Ascan_x(initial_ascan_x, x1, y0)),
            np.array(Ascan_x(initial_ascan_x, x0, y1)),
            np.array(Ascan_x(initial_ascan_x, x1, y1))
        ]
        # 中心差分估计方向导数
        dx_val = [(Ascan_vals[1] - Ascan_x(initial_ascan_x, max(x0 - 1, 0), y0)) / 2,
                  (Ascan_vals[3] - Ascan_x(initial_ascan_x, max(x0 - 1, 0), y1)) / 2]
        dy_val = [(Ascan_vals[2] - Ascan_x(initial_ascan_x, x0, max(y0 - 1, 0))) / 2,
                  (Ascan_vals[3] - Ascan_x(initial_ascan_x, x1, max(y0 - 1, 0))) / 2]
        t_x = new_x - x0
        t_y = new_y - y0

        # 在 x 方向上使用三次厄米特样条插值
        interp_y0 = cubic_hermite_interpolation(Ascan_vals[0], Ascan_vals[1], dx_val[0], dx_val[1], t_x)
        interp_y1 = cubic_hermite_interpolation(Ascan_vals[2], Ascan_vals[3], dx_val[0], dx_val[1], t_x)

        # 在 y 方向上使用三次厄米特样条插值
        final_interp = cubic_hermite_interpolation(interp_y0, interp_y1, dy_val[0], dy_val[1], t_y)
        return final_interp


def all_resampling_x_fast_level2(initial_ascan_x, Disp):
    volume_1 = np.zeros(3844)
    for i in range(62):
        for j in range(62):
            volume_1[i * 62 + j] = resampling_at_time_0_x_level2(initial_ascan_x, j, i, Disp[i * 62 + j])

    return volume_1


def resampling_at_time_0_y_level2(initial_ascan_y, position_x, position_y, disp_y):
    new_x = position_x
    new_y = position_y + disp_y
    # 将new_x,new_y拉回合法坐标范围
    new_x = clip_coordinate(new_x, 0, 61)
    new_y = clip_coordinate(new_y, 0, 61)
    if 61 >= new_x >= 0 == new_x % 1 and 61 >= new_y >= 0 == new_y % 1:
        return Ascan_y(initial_ascan_y, new_x, new_y)
    else:
        x0 = int(np.floor(new_x))
        y0 = int(np.floor(new_y))
        x1 = min(x0 + 1, 61)
        y1 = min(y0 + 1, 61)
        Ascan_vals = [
            np.array(Ascan_y(initial_ascan_y, x0, y0)),
            np.array(Ascan_y(initial_ascan_y, x1, y0)),
            np.array(Ascan_y(initial_ascan_y, x0, y1)),
            np.array(Ascan_y(initial_ascan_y, x1, y1))
        ]
        # 中心差分估计方向导数
        dx_val = [(Ascan_vals[1] - Ascan_y(initial_ascan_y, max(x0 - 1, 0), y0)) / 2,
                  (Ascan_vals[3] - Ascan_y(initial_ascan_y, max(x0 - 1, 0), y1)) / 2]
        dy_val = [(Ascan_vals[2] - Ascan_y(initial_ascan_y, x0, max(y0 - 1, 0))) / 2,
                  (Ascan_vals[3] - Ascan_y(initial_ascan_y, x1, max(y0 - 1, 0))) / 2]
        t_x = new_x - x0
        t_y = new_y - y0

        # 在 x 方向上使用三次厄米特样条插值
        interp_y0 = cubic_hermite_interpolation(Ascan_vals[0], Ascan_vals[1], dx_val[0], dx_val[1], t_x)
        interp_y1 = cubic_hermite_interpolation(Ascan_vals[2], Ascan_vals[3], dx_val[0], dx_val[1], t_x)

        # 在 y 方向上使用三次厄米特样条插值
        final_interp = cubic_hermite_interpolation(interp_y0, interp_y1, dy_val[0], dy_val[1], t_y)
        return final_interp


def all_resampling_y_fast_level2(initial_ascan_y, Disp):
    volume_2 = np.zeros(3844)
    for i in range(62):
        for j in range(62):
            volume_2[i * 62 + j] = resampling_at_time_0_y_level2(initial_ascan_y, j, i, Disp[62 ** 2 + 61 + 62 * j - i])

    return volume_2


def similarity_term_level2(initial_ascan_x, initial_ascan_y, Disp):
    v_1 = all_resampling_x_fast_level2(initial_ascan_x, Disp)
    v_2 = all_resampling_y_fast_level2(initial_ascan_y, Disp)
    count = np.sum((v_1 - v_2) ** 2)

    return count


def regularity_term_level2(Disp):
    t_1 = np.linspace(0, 3843, 3844)
    t_2 = np.linspace(3844, 7687, 3844)

    # 直接分割数组，不需要二维索引
    disp1 = Disp[0:3844]
    disp2 = Disp[3844:7688]

    # 计算第一部分的导数
    fx_derivative = np.gradient(disp1, t_1) ** 2

    # 计算第二部分的导数
    fy_derivative = np.gradient(disp2, t_2) ** 2

    # 计算积分
    f_integral = (integrate.cumulative_trapezoid(fx_derivative, t_1, initial=0)[-1] +
                  integrate.cumulative_trapezoid(fy_derivative, t_2, initial=0)[-1])

    return f_integral


def objective_function_level2(initial_ascan_x, initial_ascan_y, Disp, alpha):
    return similarity_term_level2(initial_ascan_x, initial_ascan_y, Disp) + alpha * regularity_term_level2(Disp)


def resampling_at_time_0_x_level1(initial_ascan_x, position_x, position_y, disp_x):
    new_x = position_x + disp_x
    new_y = position_y
    # 将new_x,new_y拉回合法坐标范围
    new_x = clip_coordinate(new_x, 0, 30)
    new_y = clip_coordinate(new_y, 0, 30)
    if 30 >= new_x >= 0 == new_x % 1 and 30 >= new_y >= 0 == new_y % 1:
        return Ascan_x(initial_ascan_x, new_x, new_y)
    else:
        x0 = int(np.floor(new_x))
        y0 = int(np.floor(new_y))
        x1 = min(x0 + 1, 30)
        y1 = min(y0 + 1, 30)
        Ascan_vals = [
            np.array(Ascan_x(initial_ascan_x, x0, y0)),
            np.array(Ascan_x(initial_ascan_x, x1, y0)),
            np.array(Ascan_x(initial_ascan_x, x0, y1)),
            np.array(Ascan_x(initial_ascan_x, x1, y1))
        ]
        # 中心差分估计方向导数
        dx_val = [(Ascan_vals[1] - Ascan_x(initial_ascan_x, max(x0 - 1, 0), y0)) / 2,
                  (Ascan_vals[3] - Ascan_x(initial_ascan_x, max(x0 - 1, 0), y1)) / 2]
        dy_val = [(Ascan_vals[2] - Ascan_x(initial_ascan_x, x0, max(y0 - 1, 0))) / 2,
                  (Ascan_vals[3] - Ascan_x(initial_ascan_x, x1, max(y0 - 1, 0))) / 2]
        t_x = new_x - x0
        t_y = new_y - y0

        # 在 x 方向上使用三次厄米特样条插值
        interp_y0 = cubic_hermite_interpolation(Ascan_vals[0], Ascan_vals[1], dx_val[0], dx_val[1], t_x)
        interp_y1 = cubic_hermite_interpolation(Ascan_vals[2], Ascan_vals[3], dx_val[0], dx_val[1], t_x)

        # 在 y 方向上使用三次厄米特样条插值
        final_interp = cubic_hermite_interpolation(interp_y0, interp_y1, dy_val[0], dy_val[1], t_y)
        return final_interp


def all_resampling_x_fast_level1(initial_ascan_x, Disp):
    volume_1 = np.zeros(961)
    for i in range(31):
        for j in range(31):
            volume_1[i * 31 + j] = resampling_at_time_0_x_level1(initial_ascan_x, j, i, Disp[i * 31 + j])

    return volume_1


def resampling_at_time_0_y_level1(initial_ascan_y, position_x, position_y, disp_y):
    new_x = position_x
    new_y = position_y + disp_y
    # 将new_x,new_y拉回合法坐标范围
    new_x = clip_coordinate(new_x, 0, 30)
    new_y = clip_coordinate(new_y, 0, 30)
    if 30 >= new_x >= 0 == new_x % 1 and 30 >= new_y >= 0 == new_y % 1:
        return Ascan_y(initial_ascan_y, new_x, new_y)
    else:
        x0 = int(np.floor(new_x))
        y0 = int(np.floor(new_y))
        x1 = min(x0 + 1, 30)
        y1 = min(y0 + 1, 30)
        Ascan_vals = [
            np.array(Ascan_y(initial_ascan_y, x0, y0)),
            np.array(Ascan_y(initial_ascan_y, x1, y0)),
            np.array(Ascan_y(initial_ascan_y, x0, y1)),
            np.array(Ascan_y(initial_ascan_y, x1, y1))
        ]
        # 中心差分估计方向导数
        dx_val = [(Ascan_vals[1] - Ascan_y(initial_ascan_y, max(x0 - 1, 0), y0)) / 2,
                  (Ascan_vals[3] - Ascan_y(initial_ascan_y, max(x0 - 1, 0), y1)) / 2]
        dy_val = [(Ascan_vals[2] - Ascan_y(initial_ascan_y, x0, max(y0 - 1, 0))) / 2,
                  (Ascan_vals[3] - Ascan_y(initial_ascan_y, x1, max(y0 - 1, 0))) / 2]
        t_x = new_x - x0
        t_y = new_y - y0

        # 在 x 方向上使用三次厄米特样条插值
        interp_y0 = cubic_hermite_interpolation(Ascan_vals[0], Ascan_vals[1], dx_val[0], dx_val[1], t_x)
        interp_y1 = cubic_hermite_interpolation(Ascan_vals[2], Ascan_vals[3], dx_val[0], dx_val[1], t_x)

        # 在 y 方向上使用三次厄米特样条插值
        final_interp = cubic_hermite_interpolation(interp_y0, interp_y1, dy_val[0], dy_val[1], t_y)
        return final_interp


def all_resampling_y_fast_level1(initial_ascan_y, Disp):
    volume_2 = np.zeros(961)
    for i in range(31):
        for j in range(31):
            volume_2[i * 31 + j] = resampling_at_time_0_y_level1(initial_ascan_y, j, i, Disp[31 ** 2 + 30 + 31 * j - i])

    return volume_2


def similarity_term_level1(initial_ascan_x, initial_ascan_y, Disp):
    v_1 = all_resampling_x_fast_level1(initial_ascan_x, Disp)
    v_2 = all_resampling_y_fast_level1(initial_ascan_y, Disp)
    count = np.sum((v_1 - v_2) ** 2)

    return count


def regularity_term_level1(Disp):
    t_1 = np.linspace(0, 960, 961)
    t_2 = np.linspace(961, 1921, 961)

    # 直接分割数组，不需要二维索引
    disp1 = Disp[0:961]
    disp2 = Disp[961:1922]

    # 计算第一部分的导数
    fx_derivative = np.gradient(disp1, t_1) ** 2

    # 计算第二部分的导数
    fy_derivative = np.gradient(disp2, t_2) ** 2

    # 计算积分
    f_integral = (integrate.cumulative_trapezoid(fx_derivative, t_1, initial=0)[-1] +
                  integrate.cumulative_trapezoid(fy_derivative, t_2, initial=0)[-1])

    return f_integral


def objective_function_level1(initial_ascan_x, initial_ascan_y, Disp, alpha):
    return similarity_term_level1(initial_ascan_x, initial_ascan_y, Disp) + alpha * regularity_term_level1(Disp)


class LBFGSOptimizer:
    def __init__(self, initial_vector, initial_ascan_x, initial_ascan_y, objective_function, alpha=0.1, level=1):
        self.initial_vector = initial_vector
        self.initial_ascan_x = initial_ascan_x
        self.initial_ascan_y = initial_ascan_y
        self.objective_function = objective_function
        self.alpha = alpha  # alpha 参数
        self.level = level  # 优化的层级
        self.optimization_history = []

    def objective_wrapper(self, x):
        """
        包装目标函数，记录优化历史
        """
        # 调用新的目标函数
        value = self.objective_function(self.initial_ascan_x, self.initial_ascan_y, x, self.alpha, self.level)
        self.optimization_history.append(value)
        return value

    def optimize(self, max_iter=1000, tolerance=1e-6, maxfun=100000):
        bounds = [(-self.level * 3, self.level * 3)] * len(self.initial_vector)
        result = minimize(
            fun=self.objective_wrapper,
            x0=self.initial_vector,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': tolerance * 10,
                'gtol': tolerance * 10,
                'maxcor': 180,
                'disp': True,
                'maxfun': maxfun  # 设置最大评估次数
            }
        )
        return result

    def plot_convergence(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.optimization_history)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Optimization Convergence History')
        plt.yscale('log')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    image_fol_1 = 'x.bmp'
    image_fol_2 = 'y.bmp'
    image_fol_1_level1 = 'x_level1.bmp'
    image_fol_2_level1 = 'y_level1.bmp'
    image_fol_1_level2 = 'x_level2.bmp'
    image_fol_2_level2 = 'y_level2.bmp'
    image_fol_1_level3 = 'x_level3.bmp'
    image_fol_2_level3 = 'y_level3.bmp'
    image_fol_1_level4 = 'x_level4.bmp'
    image_fol_2_level4 = 'y_level4.bmp'
    mean_x = np.mean(cv2.imread(image_fol_1, cv2.IMREAD_GRAYSCALE))
    std_x = np.std(cv2.imread(image_fol_1, cv2.IMREAD_GRAYSCALE))
    img1, img2 = read_and_normalize_images(image_fol_1, image_fol_2)
    img1_level1, img2_level1 = read_and_normalize_images(image_fol_1_level1, image_fol_2_level1)
    img1_level2, img2_level2 = read_and_normalize_images(image_fol_1_level2, image_fol_2_level2)
    img1_level3, img2_level3 = read_and_normalize_images(image_fol_1_level3, image_fol_2_level3)
    img1_level4, img2_level4 = read_and_normalize_images(image_fol_1_level4, image_fol_2_level4)
    data_x = read_ascan_x(img1)
    data_y = read_ascan_y(img2)
    data_x_level1 = read_ascan_x(img1_level1)
    data_y_level1 = read_ascan_y(img2_level1)
    data_x_level2 = read_ascan_x(img1_level2)
    data_y_level2 = read_ascan_y(img2_level2)
    data_x_level3 = read_ascan_x(img1_level3)
    data_y_level3 = read_ascan_y(img2_level3)
    data_x_level4 = read_ascan_x(img1_level4)
    data_y_level4 = read_ascan_y(img2_level4)


    def obj_computing(initial_ascan_x, initial_ascan_y, Disp, alpha, level):
        if level == 1:
            v_1_l1 = (all_resampling_x_fast_level1(initial_ascan_x, Disp)).reshape(31, 31)
            v_2_l1 = (all_resampling_y_fast_level1(initial_ascan_y, Disp)).reshape(31, 31)
            data_range = np.max([v_1_l1.max() - v_1_l1.min(), v_2_l1.max() - v_2_l1.min()])
            return (1 - ssim(v_1_l1, v_2_l1, data_range=data_range, gaussian_weights=True,
                             sigma=1.5)) + alpha * regularity_term_level1(Disp)

        elif level == 2:
            v_1_l2 = all_resampling_x_fast_level2(initial_ascan_x, Disp).reshape(62, 62)
            v_2_l2 = all_resampling_y_fast_level2(initial_ascan_y, Disp).reshape(62, 62)
            data_range = np.max([v_1_l2.max() - v_1_l2.min(), v_2_l2.max() - v_2_l2.min()])
            return (1 - ssim(v_1_l2, v_2_l2, data_range=data_range, gaussian_weights=True,
                             sigma=1.5)) + alpha * regularity_term_level2(Disp)
        elif level == 3:
            v_1_l3 = all_resampling_x_fast_level3(initial_ascan_x, Disp).reshape(125, 125)
            v_2_l3 = all_resampling_y_fast_level3(initial_ascan_y, Disp).reshape(125, 125)
            data_range = np.max([v_1_l3.max() - v_1_l3.min(), v_2_l3.max() - v_2_l3.min()])
            return (1 - ssim(v_1_l3, v_2_l3, data_range=data_range, gaussian_weights=True,
                             sigma=1.5)) + alpha * regularity_term_level3(Disp)
        elif level == 4:
            v_1_l4 = all_resampling_x_fast_level4(initial_ascan_x, Disp).reshape(250, 250)
            v_2_l4 = all_resampling_y_fast_level4(initial_ascan_y, Disp).reshape(250, 250)
            data_range = np.max([v_1_l4.max() - v_1_l4.min(), v_2_l4.max() - v_2_l4.min()])
            return (1 - ssim(v_1_l4, v_2_l4, data_range=data_range, gaussian_weights=True,
                             sigma=1.5)) + alpha * regularity_term_level4(Disp)
        elif level == 5:
            v_1_l5 = all_resampling_x_fast(initial_ascan_x, Disp).reshape(500, 500)
            v_2_l5 = all_resampling_y_fast(initial_ascan_y, Disp).reshape(500, 500)
            data_range = np.max([v_1_l5.max() - v_1_l5.min(), v_2_l5.max() - v_2_l5.min()])
            return (1 - ssim(v_1_l5, v_2_l5, data_range=data_range, gaussian_weights=True,
                             sigma=1.5)) + alpha * regularity_term(Disp)


    """
    optimizer_1 = LBFGSOptimizer(np.zeros(1922), data_x_level1, data_y_level1, obj_computing, 0.01, 1)
    start_time_1 = time.time()
    # 运行优化，设置最大评估次数 maxfun=10000
    optimal_disp1 = optimizer_1.optimize(maxfun=100000)
    # 记录优化结束时间
    end_time_1 = time.time()
    # 计算优化总时间
    optimization_time_1 = end_time_1 - start_time_1
    np.save('disp_level1.npy', optimal_disp1.x)
    print("\nOptimization Results:")
    print(f"Optimization terminated with message: {optimal_disp1.message}")
    print(f"Success: {optimal_disp1.success}")
    print(f"Final objective value: {optimal_disp1.fun}")
    print(f"Optimal vector: {optimal_disp1.x}")
    print(f"Number of iterations: {optimal_disp1.nit}")
    print(f"Optimization Time: {optimization_time_1:.4f} seconds")
    """
    """
    disp_level1 = np.load('disp_level1.npy')
    initial_disp_level2_part1 = cv2.resize((disp_level1[0: 961]).reshape(31, 31), (62, 62),
                                           interpolation=cv2.INTER_LINEAR)
    initial_disp_level2_part2 = cv2.resize((disp_level1[961: 1922]).reshape(31, 31), (62, 62),
                                           interpolation=cv2.INTER_LINEAR)
    initial_disp_level2 = np.concatenate((initial_disp_level2_part1.flatten(), initial_disp_level2_part2.flatten()))

    optimizer_2 = LBFGSOptimizer(initial_disp_level2, data_x_level2, data_y_level2, obj_computing, 0.01, 2)
    start_time_2 = time.time()
    # 运行优化，设置最大评估次数
    optimal_disp2 = optimizer_2.optimize(maxfun=30000)
    end_time_2 = time.time()
    optimization_time_2 = end_time_2 - start_time_2
    np.save('disp_level2.npy', optimal_disp2.x)
    print("\nOptimization Results:")
    print(f"Optimization terminated with message: {optimal_disp2.message}")
    print(f"Success: {optimal_disp2.success}")
    print(f"Final objective value: {optimal_disp2.fun}")
    print(f"Optimal vector: {optimal_disp2.x}")
    print(f"Number of iterations: {optimal_disp2.nit}")
    print(f"Optimization Time: {optimization_time_2:.4f} seconds")
    disp_level2 = np.load('disp_level2.npy')
    initial_disp_level3_part1 = cv2.resize((disp_level2.x[0: 3844]).reshape(62, 62), (125, 125),
                                           interpolation=cv2.INTER_LINEAR)
    initial_disp_level3_part2 = cv2.resize((disp_level2.x[3844: 7688]).reshape(62, 62), (125, 125),
                                           interpolation=cv2.INTER_LINEAR)
    initial_disp_level3 = np.concatenate((initial_disp_level3_part1.flatten(), initial_disp_level3_part2.flatten()))
    optimizer_3 = LBFGSOptimizer(initial_disp_level3, data_x_level3, data_y_level3, obj_computing, 0.01, 3)
    start_time_3 = time.time()
    # 运行优化，设置最大评估次数
    optimal_disp3 = optimizer_3.optimize(maxfun=3000)
    end_time_3 = time.time()
    optimization_time_3 = end_time_3 - start_time_3
    np.save('disp_level3.npy', optimal_disp3.x)
    print("\nOptimization Results:")
    print(f"Optimization terminated with message: {optimal_disp3.message}")
    print(f"Success: {optimal_disp3.success}")
    print(f"Final objective value: {optimal_disp3.fun}")
    print(f"Optimal vector: {optimal_disp3.x}")
    print(f"Number of iterations: {optimal_disp3.nit}")
    print(f"Optimization Time: {optimization_time_3:.4f} seconds")
    """
    disp_level3 = np.load('disp_level3.npy')
    initial_disp_level4_part1 = cv2.resize((disp_level3.x[0: 15625]).reshape(125, 125), (250, 250),
                                           interpolation=cv2.INTER_LINEAR)
    initial_disp_level4_part2 = cv2.resize((disp_level3.x[15625: 31250]).reshape(125, 125), (250, 250),
                                           interpolation=cv2.INTER_LINEAR)
    initial_disp_level4 = np.concatenate((initial_disp_level4_part1.flatten(), initial_disp_level4_part2.flatten()))
    optimizer_4 = LBFGSOptimizer(initial_disp_level4, data_x_level4, data_y_level4, obj_computing, 0.01, 4)
    start_time_4 = time.time()
    # 运行优化，设置最大评估次数
    optimal_disp4 = optimizer_4.optimize(maxfun=600)
    end_time_4 = time.time()
    optimization_time_4 = end_time_4 - start_time_4
    np.save('disp_level4.npy', optimal_disp4.x)
    print("\nOptimization Results:")
    print(f"Optimization terminated with message: {optimal_disp4.message}")
    print(f"Success: {optimal_disp4.success}")
    print(f"Final objective value: {optimal_disp4.fun}")
    print(f"Optimal vector: {optimal_disp4.x}")
    print(f"Number of iterations: {optimal_disp4.nit}")
    print(f"Optimization Time: {optimization_time_4:.4f} seconds")
    initial_disp_level5_part1 = cv2.resize((optimal_disp4.x[0: 62500]).reshape(250, 250), (500, 500),
                                           interpolation=cv2.INTER_LINEAR)
    initial_disp_level5_part2 = cv2.resize((optimal_disp4.x[62500: 125000]).reshape(250, 250), (500, 500),
                                           interpolation=cv2.INTER_LINEAR)
    initial_disp_level5 = np.concatenate((initial_disp_level5_part1.flatten(), initial_disp_level5_part2.flatten()))
    optimizer_5 = LBFGSOptimizer(initial_disp_level5, data_x, data_y, obj_computing, 0.01, 5)
    start_time_5 = time.time()
    # 运行优化，设置最大评估次数
    optimal_disp5 = optimizer_5.optimize(maxfun=200)
    end_time_5 = time.time()
    optimization_time_5 = end_time_5 - start_time_5
    np.save('disp_final.npy', optimal_disp5.x)
    print("\nOptimization Results:")
    print(f"Optimization terminated with message: {optimal_disp5.message}")
    print(f"Success: {optimal_disp5.success}")
    print(f"Final objective value: {optimal_disp5.fun}")
    print(f"Optimal vector: {optimal_disp5.x}")
    print(f"Number of iterations: {optimal_disp5.nit}")
    print(f"Optimization Time: {optimization_time_5:.4f} seconds")

    # 绘制收敛历史
    optimizer_4.plot_convergence()
    optimizer_5.plot_convergence()

    """
    result_x = np.clip((all_resampling_x_fast(data_x, array
                                                     )).reshape((500, 500)) * std_x + mean_x, 0, 255)
    gamma_image = np.uint8(np.clip(255 * (result_x / 255) ** 1.2, 0, 255))
    plt.imshow(gamma_image, cmap='gray')
    plt.show()
    """
    """
    optimal_disp1 = particle_swarm_optimization(
        initial_ascan_x=data_x_level1,
        initial_ascan_y=data_y_level1,
        alpha=1,
        n_particles=20,
        n_iterations=400,
        dim=1922,
        level=1,
        initial_disp=np.zeros(1922)
    )

    print(optimal_disp1.shape)
    initial_disp_level2_part1 = cv2.resize((optimal_disp1[0: 961]).reshape(31, 31), (62, 62),
                                           interpolation=cv2.INTER_LINEAR)
    initial_disp_level2_part2 = cv2.resize((optimal_disp1[961: 1922]).reshape(31, 31), (62, 62),
                                           interpolation=cv2.INTER_LINEAR)
    initial_disp_level2 = np.concatenate((initial_disp_level2_part1.flatten(), initial_disp_level2_part2.flatten()))
    print(objective_function_level2(data_x_level2, data_y_level2, np.zeros(7688), 1))
    optimal_disp2 = particle_swarm_optimization(
        initial_ascan_x=data_x_level2,
        initial_ascan_y=data_y_level2,
        alpha=1,
        n_particles=20,
        n_iterations=50,
        dim=7688,
        level=2,
        initial_disp=initial_disp_level2
    )
    print(optimal_disp2)
    print(optimal_disp2.shape)
    """
    """
    result_x = np.clip((all_resampling_x_fast_level1(data_x_level1, optimal_disp1
                                                     )).reshape((31, 31)) * std_x + mean_x, 0, 255)
    gamma_image = np.uint8(np.clip(255 * (result_x / 255) ** 1.2, 0, 255))
    plt.imshow(gamma_image, cmap='gray')
    plt.show()
    """
"""
def calculate_particle_fitness(particle_position, initial_ascan_x, initial_ascan_y, alpha, level):
    if level == 1:
        fitness_value = objective_function_level1(initial_ascan_x, initial_ascan_y, particle_position, alpha)
    elif level == 2:
        fitness_value = objective_function_level2(initial_ascan_x, initial_ascan_y, particle_position, alpha)
    elif level == 3:
        fitness_value = objective_function_level3(initial_ascan_x, initial_ascan_y, particle_position, alpha)
    elif level == 4:
        fitness_value = objective_function_level4(initial_ascan_x, initial_ascan_y, particle_position, alpha)
    elif level == 5:
        fitness_value = objective_function(initial_ascan_x, initial_ascan_y, particle_position, alpha)
    else:
        raise ValueError("Level must be between 1 and 5.")

    return fitness_value


def particle_swarm_optimization(initial_ascan_x, initial_ascan_y, alpha, n_particles, n_iterations, dim, n_processes,
                                initial_disp=None, level=1):
    # Initialize particle positions and velocities
    if initial_disp is None:
        particles_position = np.random.rand(n_particles, dim) * (6 * level) - (3 * level)
    else:
        particles_position = np.repeat(np.expand_dims(initial_disp, axis=0), n_particles, axis=0)
    particles_velocity = np.zeros_like(particles_position)
    pbest_position = np.copy(particles_position)

    pool = Pool(processes=n_processes)
    partial_fitness = partial(calculate_particle_fitness,
                              initial_ascan_x=initial_ascan_x,
                              initial_ascan_y=initial_ascan_y,
                              alpha=alpha,
                              level=level)

    # Parallel calculation of initial fitness values
    pbest_value = np.array(pool.map(partial_fitness, particles_position))

    # Initialize global best
    gbest_idx = np.argmin(pbest_value)
    gbest_position = np.copy(pbest_position[gbest_idx])
    gbest_value = pbest_value[gbest_idx]

    # PSO parameters
    w = 0.5  # Inertia weight
    c1 = 1.5  # Individual learning factor
    c2 = 1.5  # Social learning factor

    try:
        for t in range(n_iterations):
            # Update velocities and positions of all particles
            r1 = np.random.rand(n_particles, dim)
            r2 = np.random.rand(n_particles, dim)

            particles_velocity = (w * particles_velocity +
                                  c1 * r1 * (pbest_position - particles_position) +
                                  c2 * r2 * (gbest_position - particles_position))

            particles_position += particles_velocity

            # Limit particle positions within the dynamic range
            particles_position = np.clip(particles_position, -3 * level, 3 * level)

            # Parallel calculation of new fitness values
            fitness_values = np.array(pool.map(partial_fitness, particles_position))

            # Update personal bests
            improved_indices = fitness_values < pbest_value
            pbest_value[improved_indices] = fitness_values[improved_indices]
            pbest_position[improved_indices] = particles_position[improved_indices]

            # Update global best
            current_best_idx = np.argmin(fitness_values)
            if fitness_values[current_best_idx] < gbest_value:
                gbest_value = fitness_values[current_best_idx]
                gbest_position = particles_position[current_best_idx].copy()

            print(f"Iteration {t + 1}/{n_iterations}, Best Value: {gbest_value}")

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    finally:
        # Ensure the process pool is properly closed
        pool.close()
        pool.join()

    return gbest_position
"""
