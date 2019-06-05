import os, os.path as osp
import glob
import numpy as np
import cv2

from coco_annotation import CocoAnnotationClass

def get_polygons_from_annotation(annot_path):
    polygons = []
    with open(annot_path, 'r') as f:
        lines = f.readlines()
        # remove weird encoding in start of gt file (train set)
        lines[0] = lines[0].replace("\ufeff","")  
        for l in lines:
            poly = l.split(",")[:8]
            poly = np.array(poly, dtype=np.int32).reshape((4,2))
            polygons.append(poly)
    return polygons

GREEN = (0,255,0)
RED = (0,0,255)
BLUE = (255,0,0)

VIS = True

root_img_dir = "/media/bot/hd1/datasets/OCR/ICDAR/2015/Images"
root_annot_dir = "/media/bot/hd1/datasets/OCR/ICDAR/2015/Annotations"

# dataset = "Train"
dataset = "Test"

img_dir = osp.join(root_img_dir, dataset)
annot_dir = osp.join(root_annot_dir, dataset)

img_files = os.listdir(img_dir)

# coco init
coco_annot = CocoAnnotationClass(["text"], "icdar15_ist") # COCO IS 1-indexed, don't include BG CLASS
cls_idx = 1 # default 

ANNOT_ID = 0

total = len(img_files)
for img_id in range(1, total+1): 
    print("%d of %d"%(img_id, total))

    img_name = "img_%d.jpg"%(img_id) 
    ann_name = "gt_img_%d.txt"%(img_id)

    img_path = osp.join(img_dir, img_name)
    annot_path = osp.join(annot_dir, ann_name)

    img = cv2.imread(img_path)
    if img is None:
        print("Could not read %s"%(img_path))
        continue

    img_height, img_width = img.shape[:2]

    if not osp.exists(annot_path):
        print("Could not find %s"%(annot_path))
        continue

    polys = get_polygons_from_annotation(annot_path)

    IMG_ID = img_id
    for poly in polys:
        if len(poly) < 3:
            continue
        ANNOT_ID += 1

        coco_annot.add_annot(ANNOT_ID, IMG_ID, cls_idx, poly.astype(np.float32))

        if VIS:
            n = len(poly)
            for ix, px in enumerate(poly):
                px = tuple(px)
                cv2.line(img, px, tuple(poly[(ix+1)%n]), GREEN)
                cv2.circle(img, px, 2, RED, -1)

    if VIS:
        cv2.imshow("img", img)
        cv2.waitKey(0)

    coco_annot.add_image(IMG_ID, img_width, img_height, img_name)

coco_annot.save("icdar_2015_ist_%s.json"%(dataset))
