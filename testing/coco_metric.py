import os
import json
import math

import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import tqdm
import cv2

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import util

from config_reader import config_reader

params, model_params = config_reader()
params['scale_search'] = list(params['scale_search'])


# the middle joints heatmap correpondence
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
          [55, 56], [37, 38], [45, 46]]

limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]


def predict(image, model, model_params):
    # print (image.shape)
    heatmap_avg = np.zeros((image.shape[0], image.shape[1], 19))
    paf_avg = np.zeros((image.shape[0], image.shape[1], 38))
    multiplier = [x * model_params['boxsize'] / image.shape[0] for x in params['scale_search']]
    # print(multiplier)
    for m in range(len(multiplier)):
        scale = multiplier[m]

        imageToTest = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                                 (3, 0, 1, 2))  # required shape (1, width, height, channels)

        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    return heatmap_avg, paf_avg


def find_peaks(heatmap_avg, threshold):
    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > threshold))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    return all_peaks


def find_connections(all_peaks, paf_avg, image_width, threshold):
    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * image_width / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > threshold)[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    return connection_all, special_k


def find_people(connection_all, special_k, all_peaks):
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    return subset, candidate


dt_gt_mapping = {
    0: 0,
    1: None,
    2: 6,
    3: 8,
    4: 10,
    5: 5,
    6: 7,
    7: 9,
    8: 12,
    9: 14,
    10: 16,
    11: 11,
    12: 13,
    13: 15,
    14: 2,
    15: 1,
    16: 4,
    17: 3,
}


def process(input_image, params, model, model_params):
    oriImg = cv2.imread(input_image)  # B,G,R order
    heatmap_avg, paf_avg = predict(oriImg, model, model_params)

    all_peaks = find_peaks(heatmap_avg, params['thre1'])
    connection_all, special_k = find_connections(all_peaks, paf_avg, oriImg.shape[0], params['thre2'])
    subset, candidate = find_people(connection_all, special_k, all_peaks)

    keypoints = []
    for s in subset:
        keypoint_indexes = s[:18]
        person_keypoint_coordinates = []
        for index in keypoint_indexes:
            if index == -1:
                # "No candidate for keypoint"
                X, Y = 0, 0
            else:
                X, Y = candidate[index.astype(int)][:2]
            person_keypoint_coordinates.append((X, Y))
        person_keypoint_coordinates_coco = [None] * 17

        for dt_index, gt_index in dt_gt_mapping.items():
            if gt_index is None:
                continue
            person_keypoint_coordinates_coco[gt_index] = person_keypoint_coordinates[dt_index]

        keypoints.append((person_keypoint_coordinates_coco, 1 - 1.0 / s[18]))
    return keypoints


def get_image_name(coco, image_id):
    return coco.imgs[image_id]['file_name']


def predict_many(coco, images_directory, validation_ids, params, model, model_params):
    assert (not set(validation_ids).difference(set(coco.getImgIds())))

    keypoints = {}
    for image_id in tqdm.tqdm(validation_ids):
        image_name = get_image_name(coco, image_id)
        image_name = os.path.join(images_directory, image_name)
        keypoints[image_id] = process(image_name, dict(params), model, dict(model_params))
    return keypoints


def format_results(keypoints, resFile):
    format_keypoints = []

    for image_id, people in keypoints.items():
        for keypoint_list, score in people:
            format_keypoint_list = []
            for x, y in keypoint_list:
                for v in [int(x), int(y), 1 if x > 0 or y > 0 else 0]:
                    format_keypoint_list.append(v)

            format_keypoints.append({
                "image_id": image_id,
                "category_id": 1,
                "keypoints": format_keypoint_list,
                "score": score,
            })

    json.dump(format_keypoints, open(resFile, 'w'))


def validation(model, dump_name, validation_ids=None, dataset='val2017'):
    annType = 'keypoints'
    prefix = 'person_keypoints'

    dataDir = 'dataset'
    annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataset)
    cocoGt = COCO(annFile)

    if validation_ids == None:
        validation_ids = cocoGt.getImgIds()

    resFile = '%s/results/%s_%s_%s100_results.json'
    resFile = resFile % (dataDir, prefix, dataset, dump_name)
    os.makedirs(os.path.dirname(resFile), exist_ok=True)

    keypoints = predict_many(cocoGt, os.path.join(dataDir, dataset), validation_ids, params, model, model_params)
    format_results(keypoints, resFile)

    cocoDt = cocoGt.loadRes(resFile)

    cocoEval = COCOeval(cocoGt, cocoDt, annType)

    cocoEval.params.imgIds = validation_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval


def per_image_scores(eval_result):
    def convert_match_to_score(match):
        matches = match['gtMatches'][:, np.array(match['gtIgnore']) == 0]
        scores = {
            'image_id': match['image_id'],
            'gt_person_count': matches.shape[1],
        }

        for i in range(matches.shape[0]):
            okp_threshold = eval_result.params.iouThrs[i]
            scores['matched_%.2f' % okp_threshold] = sum(matches[i, :] != 0)
        scores['average'] = np.mean(np.sum(matches != 0, axis=1)) / scores['gt_person_count']

        return scores

    evalImgs = eval_result.evalImgs
    scores = [convert_match_to_score(image_match) for image_match in evalImgs if image_match is not None]

    return pd.DataFrame(scores)
