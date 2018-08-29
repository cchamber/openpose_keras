
import numpy as np
import pandas as pd
import cv2
from time import sleep
from scipy.ndimage.filters import gaussian_filter, maximum_filter

# this is find peak function
from scipy.optimize import linear_sum_assignment


def find_peaks(layer, thre1=0.01):
    map_ori = cv2.resize(layer, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    map = gaussian_filter(map_ori, sigma=3)
    peaks_binary = (map == maximum_filter(map, 3)) & (map > thre1)

    if np.count_nonzero(peaks_binary) > 50:
        return []  #safety valve from N^2 in next stages

    peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
    peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]

    return peaks_with_score


def assign_peaks(layer_y, layer_gt):

    if len(layer_y) == 0 and len(layer_gt) == 0:
        return np.nan

    if len(layer_y) == 0 or len(layer_gt) == 0:
        return 400

    d = np.array(layer_y)
    t = np.array(layer_gt)

    dx = np.subtract.outer(d[:, 0], t[:, 0])
    dy = np.subtract.outer(d[:, 1], t[:, 1])
    distance = np.sqrt(dx ** 2 + dy ** 2)
    # print(distance)

    y, gt = linear_sum_assignment(distance)
    # print(np.array(list(zip(y,gt))))

    dist = [distance[foo] for foo in zip(y, gt)]  # TODO: use numpy
    # print(dist)

    dist += [400] * (len(layer_y) - len(y))
    dist += [400] * (len(layer_gt) - len(gt))

    dist = np.mean(dist)

    return dist


def calc_batch_metrics(batch_no, gt, Y, heatmap_layers):

    MAE = Y - gt
    MAE = np.abs(MAE)
    MAE = np.mean(MAE, axis=(1, 2))

    RMSE = (Y - gt) ** 2
    RMSE = np.mean(RMSE, axis=(1, 2))
    RMSE = np.sqrt(RMSE)


    gt_parts = np.full((gt.shape[0], gt.shape[3]), np.nan)
    y_parts = np.full((gt.shape[0], gt.shape[3]), np.nan)
    y_dist = np.full((gt.shape[0], gt.shape[3]), np.nan)


    for n in range(gt.shape[0]):
        for l in heatmap_layers:
            y_peaks = find_peaks(Y[n, :, :, l])
            y_parts[n, l] = len(y_peaks)
            gt_peaks = find_peaks(gt[n, :, :, l])
            gt_parts[n, l] = len(gt_peaks)
            y_dist[n, l] = assign_peaks(y_peaks, gt_peaks)

    batch_index = np.full(fill_value=batch_no, shape=MAE.shape)
    item_index, layer_index = np.mgrid[0:MAE.shape[0], 0:MAE.shape[1]]

    metrics = pd.DataFrame({'batch': batch_index.ravel(),
                            'item': item_index.ravel(),
                            'layer': layer_index.ravel(),
                            'MAE': MAE.ravel(),
                            'RMSE': RMSE.ravel(),
                            'GT_PARTS': gt_parts.ravel(),
                            'Y_PARTS': y_parts.ravel(),
                            'DIST': y_dist.ravel()
                            },
                           columns=('batch', 'item', 'layer', 'MAE', 'RMSE', 'GT_PARTS', 'Y_PARTS', 'DIST')
                           )

    return metrics

