import os, os.path as osp
import glob
import numpy as np
import cv2

from coco_annotation import CocoAnnotationClass

def get_polygons_from_annotation(annot_path):
    polygons = []
    with open(annot_path, 'r') as f:
        line = f.readlines()
        for l in line:
            poly = l.split(",")[:8]
            poly = np.array(poly, dtype=np.int32).reshape((4,2))
            polygons.append(poly)
    return polygons

GREEN = (0,255,0)
RED = (0,0,255)
BLUE = (255,0,0)

VIS = False

IMG_EXTENSIONS = [".jpg",".png",".gif"]

root_img_dir = "/home/vincent/hd/datasets/OCR/ICDAR/2017/MLT/Images"
root_annot_dir = "/home/vincent/hd/datasets/OCR/ICDAR/2017/MLT/Annotations"

# dataset = "Train" # "Val"
dataset = "Train" # "Val"

img_dir = osp.join(root_img_dir, dataset)
annot_dir = osp.join(root_annot_dir, dataset)

img_files = os.listdir(img_dir)

# coco init
coco_annot = CocoAnnotationClass(["text"], "icdar17_mlt") # COCO IS 1-indexed, don't include BG CLASS
cls_idx = 1 # default 

ANNOT_ID = 0

total = len(img_files)
for img_id in range(1, total+1): 
    print("%d of %d"%(img_id, total))

    img_name = "img_%d.jpg"%(img_id) 
    ann_name = "gt_img_%d.txt"%(img_id)

    img_path = osp.join(img_dir, img_name)
    annot_path = osp.join(annot_dir, ann_name)

    if not osp.exists(img_path):
        for ext in IMG_EXTENSIONS[1:]:  # check other extensions
            img_name = img_name[:-4] + ext
            img_path = osp.join(img_dir, img_name)
            if osp.exists(img_path):
                break

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

coco_annot.save("icdar_2017_mlt_%s.json"%(dataset))
