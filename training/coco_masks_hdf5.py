#!/usr/bin/env python

from pycocotools.coco import COCO
from scipy.spatial.distance import cdist
import numpy as np
import cv2
import os
import os.path
import h5py
import json

dataset_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'dataset'))

tr_anno_path = os.path.join(dataset_dir, "annotations/person_keypoints_train2017.json")
tr_img_dir = os.path.join(dataset_dir, "train2017")

val_anno_path = os.path.join(dataset_dir, "annotations/person_keypoints_val2017.json")
val_img_dir = os.path.join(dataset_dir, "val2017")

datasets = [
    (val_anno_path, val_img_dir, "COCO_val"),  # it is important to have 'val' in validation dataset name, look for 'val' below
    (tr_anno_path, tr_img_dir, "COCO")
]


tr_hdf5_path = os.path.join(dataset_dir, "coco_train_dataset.h5")
val_hdf5_path = os.path.join(dataset_dir, "coco_val_dataset.h5")

val_size = 2645 # size of validation set



def make_mask(img_dir, img_id, img_anns, coco):

    img_path = os.path.join(img_dir, "%012d.jpg" % img_id)
    img = cv2.imread(img_path)
    h, w, c = img.shape

    mask_all = np.zeros((h, w), dtype=np.uint8)
    mask_miss = np.zeros((h, w), dtype=np.uint8)

    flag = 0
    for p in img_anns:
        p['segmentation'] =  [[p['bbox'][0],p['bbox'][1], p['bbox'][0],p['bbox'][1]+p['bbox'][3],p['bbox'][0]+p['bbox'][2],p['bbox'][1]+p['bbox'][3],p['bbox'][0]+p['bbox'][2],p['bbox'][1]]]
        p['area'] = p['bbox'][2]*p['bbox'][3]
        seg = p["segmentation"]
        # seg = p["segmentation"]

        if p["iscrowd"] == 1:
            mask_crowd = coco.annToMask(p)
            temp = np.bitwise_and(mask_all, mask_crowd)
            mask_crowd = mask_crowd - temp
            flag += 1
            continue
        else:
            mask = coco.annToMask(p)

        mask_all = np.bitwise_or(mask, mask_all)

        if p["num_keypoints"] <= 0:
            mask_miss = np.bitwise_or(mask, mask_miss)

    if flag<1:
        mask_miss = np.logical_not(mask_miss)
    elif flag == 1:
        mask_miss = np.logical_not(np.bitwise_or(mask_miss, mask_crowd))
        mask_all = np.bitwise_or(mask_all, mask_crowd)
    else:
        raise Exception("crowd segments > 1")

    mask_miss = mask_miss.astype(np.uint8)
    mask_miss *= 255

    return img, mask_miss

def process_image(image_rec, img_id, image_index, img_anns, dataset_type):

    print("Image ID: ", img_id)

    numPeople = len(img_anns)
    h, w = image_rec['height'], image_rec['width']

    all_persons = []

    for p in range(numPeople):

        pers = dict()

        person_center = [img_anns[p]["bbox"][0] + img_anns[p]["bbox"][2] / 2,
                         img_anns[p]["bbox"][1] + img_anns[p]["bbox"][3] / 2]

        pers["objpos"] = person_center
        pers["bbox"] = img_anns[p]["bbox"]
        pers["segment_area"] = img_anns[p]["area"]
        pers["num_keypoints"] = img_anns[p]["num_keypoints"]

        anno = img_anns[p]["keypoints"]

        pers["joint"] = np.zeros((17, 3))
        for part in range(17):
            pers["joint"][part, 0] = anno[part * 3]
            pers["joint"][part, 1] = anno[part * 3 + 1]

            # visible/invisible
            # COCO - Each keypoint has a 0-indexed location x,y and a visibility flag v defined as v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible.
            # OURS - # 3 never marked up in this dataset, 2 - not marked up in this person, 1 - marked and visible, 0 - marked but invisible
            if anno[part * 3 + 2] == 2:
                pers["joint"][part, 2] = 1
            elif anno[part * 3 + 2] == 1:
                pers["joint"][part, 2] = 0
            else:
                pers["joint"][part, 2] = 2

        pers["scale_provided"] = img_anns[p]["bbox"][3] / 368

        all_persons.append(pers)

    main_persons = []
    prev_center = []


    for pers in all_persons:

        # skip this person if parts number is too low or if
        # segmentation area is too small
        if pers["num_keypoints"] < 5 or pers["segment_area"] < 32 * 32:
            continue

        person_center = pers["objpos"]

        # skip this person if the distance to exiting person is too small
        flag = 0
        for pc in prev_center:
            a = np.expand_dims(pc[:2], axis=0)
            b = np.expand_dims(person_center, axis=0)
            dist = cdist(a, b)[0]
            if dist < pc[2] * 0.3:
                flag = 1
                continue

        if flag == 1:
            continue

        main_persons.append(pers)
        prev_center.append(np.append(person_center, max(img_anns[p]["bbox"][2], img_anns[p]["bbox"][3])))


    template = dict()
    template["dataset"] = dataset_type

    if image_index < val_size and 'val' in dataset_type:
        isValidation = 1
    else:
        isValidation = 0

    template["isValidation"] = isValidation
    template["img_width"] = w
    template["img_height"] = h
    template["image_id"] = img_id
    template["annolist_index"] = image_index
    template["img_path"] = '%012d.jpg' % img_id

    for p, person in enumerate(main_persons):

        instance = template.copy()

        instance["objpos"] = [ main_persons[p]["objpos"] ]
        instance["joints"] = [ main_persons[p]["joint"].tolist() ]
        instance["scale_provided"] = [ main_persons[p]["scale_provided"] ]

        lenOthers = 0

        for ot, operson in enumerate(all_persons):

            if person is operson:
                assert not "people_index" in instance, "several main persons? couldn't be"
                instance["people_index"] = ot
                continue

            if operson["num_keypoints"] == 0:
                continue

            instance["joints"].append(all_persons[ot]["joint"].tolist())
            instance["scale_provided"].append(all_persons[ot]["scale_provided"])
            instance["objpos"].append(all_persons[ot]["objpos"])

            lenOthers += 1

        assert "people_index" in instance, "No main person index"
        instance["numOtherPeople"] = lenOthers

        yield instance


def writeImage(grp, img_grp, data, img, mask_miss, count, image_id, mask_grp=None):

    serializable_meta = data
    serializable_meta['count'] = count

    nop = data['numOtherPeople']

    assert len(serializable_meta['joints']) == 1 + nop, [len(serializable_meta['joints']), 1 + nop]
    assert len(serializable_meta['scale_provided']) == 1 + nop, [len(serializable_meta['scale_provided']), 1 + nop]
    assert len(serializable_meta['objpos']) == 1 + nop, [len(serializable_meta['objpos']), 1 + nop]

    img_key = "%012d" % image_id
    if not img_key in img_grp:

        if mask_grp is None:
            img_and_mask = np.concatenate((img, mask_miss[..., None]), axis=2)
            img_ds = img_grp.create_dataset(img_key, data=img_and_mask, chunks=None)
        else:
            _, img_bin = cv2.imencode(".jpg", img)
            _, img_mask = cv2.imencode(".png", mask_miss)
            img_ds1 = img_grp.create_dataset(img_key, data=img_bin, chunks=None)
            img_ds2 = mask_grp.create_dataset(img_key, data=img_mask, chunks=None)


    key = '%07d' % count
    required = { 'image':img_key, 'joints': serializable_meta['joints'], 'objpos': serializable_meta['objpos'], 'scale_provided': serializable_meta['scale_provided'] }
    ds = grp.create_dataset(key, data=json.dumps(required), chunks=None)
    ds.attrs['meta'] = json.dumps(serializable_meta)

    print('Writing sample %d' % count)


def process():

    tr_h5 = h5py.File(tr_hdf5_path, 'w')
    tr_grp = tr_h5.create_group("dataset")
    tr_write_count = 0
    tr_grp_img = tr_h5.create_group("images")
    tr_grp_mask = tr_h5.create_group("masks")

    val_h5 = h5py.File(val_hdf5_path, 'w')
    val_grp = val_h5.create_group("dataset")
    val_write_count = 0
    val_grp_img = val_h5.create_group("images")
    val_grp_mask = val_h5.create_group("masks")

    for _, ds in enumerate(datasets):

        anno_path = ds[0]
        img_dir = ds[1]
        dataset_type = ds[2]

        coco = COCO(anno_path)
        ids = list(coco.imgs.keys())

        for image_index, img_id in enumerate(ids):
            ann_ids = coco.getAnnIds(imgIds=img_id)
            img_anns = coco.loadAnns(ann_ids)
            image_rec = coco.imgs[img_id]

            img = None
            mask_miss = None
            cached_img_id = None

            for data in process_image(image_rec, img_id, image_index, img_anns, dataset_type):

                if cached_img_id!=data['image_id']:
                    assert img_id == data['image_id']
                    cached_img_id = data['image_id']
                    img, mask_miss = make_mask(img_dir, cached_img_id, img_anns, coco)

                if data['isValidation']:
                    writeImage(val_grp, val_grp_img, data, img, mask_miss, val_write_count, cached_img_id, val_grp_mask)
                    val_write_count += 1
                else:
                    writeImage(tr_grp, tr_grp_img, data, img, mask_miss, tr_write_count, cached_img_id, tr_grp_mask)
                    tr_write_count += 1

    tr_h5.close()
    val_h5.close()

if __name__ == '__main__':
    process()
