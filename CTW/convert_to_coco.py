import os, os.path as osp
import glob
import numpy as np
import cv2

import json

from coco_annotation import CocoAnnotationClass

def get_polygons_from_annotation(anno, polygon_per_word=True):
    """
    polygon_per_word: If True, return a polygon for every word
    If False, return the convex polygon for the full text (containing 1 or more words)
    """
    polygons = []
    if polygon_per_word:
        for text_data in anno:
            for word_data in text_data:
                poly = np.array(word_data['polygon'], dtype=np.float32) # (4,2)
                polygons.append(poly)
    else:
        for text_data in anno:
            word_polygons = [np.array(w['polygon'], dtype=np.float32) for w in text_data]
            convex_polygon = cv2.convexHull(np.vstack(word_polygons)).squeeze()
            polygons.append(convex_polygon)
    return polygons


GREEN = (0,255,0)
RED = (0,0,255)
BLUE = (255,0,0)

VIS = True

root_img_dir = "/home/vincent/hd/datasets/OCR/CTW/Images"
root_annot_dir = "/home/vincent/hd/datasets/OCR/CTW/Annotations"

dataset = "train" # "val"

PER_WORD_POLYGON = False  # if True, get polygon for each word instead of word string (forming a text) in image

VIS = False

dataset_to_annot_map = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test_cls.jsonl" # NOT IMPLEMENTED
}
dataset_to_img_dir_map = {
    "train": "trainval",
    "val": "trainval",
    "test": "test"
}

img_dir = osp.join(root_img_dir, dataset_to_img_dir_map[dataset])
annot_file = osp.join(root_annot_dir, dataset_to_annot_map[dataset])

# coco init
coco_annot = CocoAnnotationClass(["text"], "ctw") # COCO IS 1-indexed, don't include BG CLASS
cls_idx = 1 # default 

ANNOT_ID = 0

with open(annot_file, 'r') as f: 
    data = f.readlines()

total = len(data)
for i in range(total):
    print("%d of %d"%(i+1, total))

    d = json.loads(data[i])

    img_file = d['file_name']

    IMG_ID = int(d['image_id'])
    img_width = d['width'] 
    img_height = d['height']

    anno = d['annotations']

    img_path = osp.join(img_dir, img_file)
    if not osp.exists(img_path):
        print("Could not find %s"%(img_path))
        continue

    if VIS:
        img_path = osp.join(img_dir, img_file)
        img = cv2.imread(img_path)  

        if img is None:
            print("Could not read %s"%(img_path))
            continue

        assert img.shape[:2] == (img_height, img_width)

    polygons = get_polygons_from_annotation(anno, PER_WORD_POLYGON)

    for poly in polygons:
        ANNOT_ID += 1

        coco_annot.add_annot(ANNOT_ID, IMG_ID, cls_idx, poly)

        if VIS:
            n = len(poly)
            for ix, px in enumerate(poly):
                px = tuple(px)
                cv2.line(img, px, tuple(poly[(ix+1)%n]), GREEN, 3)
                cv2.circle(img, px, 3, RED, -1)

    if VIS:
        img_resized = cv2.resize(img, (960, 960))
        cv2.imshow("img (resized)", img_resized)
        cv2.waitKey(0)

    coco_annot.add_image(IMG_ID, img_width, img_height, img_file)

# coco_annot.save("ctw_%s.json"%(dataset))
