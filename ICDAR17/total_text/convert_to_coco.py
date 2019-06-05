import os, os.path as osp
import glob
import numpy as np
import cv2
from scipy.io import loadmat

from natsort import natsorted

from coco_annotation import CocoAnnotationClass

GREEN = (0,255,0)
RED = (0,0,255)
BLUE = (255,0,0)

VIS = False

root_img_dir = "/media/bot/hd1/datasets/ICDAR/2017/total_text/Images"
root_annot_dir = "/media/bot/hd1/datasets/ICDAR/2017/total_text/Annotations"

# dataset = "Train" 
dataset = "Test" 

img_dir = osp.join(root_img_dir, dataset)
annot_dir = osp.join(root_annot_dir, dataset)

img_files = natsorted(os.listdir(img_dir))
# img_files = [f for f in img_files if "664" in f]

# coco init
coco_annot = CocoAnnotationClass(["text"], "icdar17_total_text") # COCO IS 1-indexed, don't include BG CLASS
cls_idx = 1 # default 

ANNOT_ID = 0

total = len(img_files)
for fx, f in enumerate(img_files, 1): 
    print("%d of %d) %s"%(fx, total, f))

    IMG_ID = int(f.lower().replace("img","").replace(".jpg",""))
    img_path = osp.join(img_dir, f)
    ann_f = f.lower().replace(".jpg",".mat").replace("img", "poly_gt_img")
    annot_path = osp.join(annot_dir, ann_f)

    img = cv2.imread(img_path)
    if img is None:
        print("Could not read %s"%(img_path))
        continue

    img_height, img_width = img.shape[:2]

    ann = loadmat(annot_path)['polygt']
    for a in ann:
        x_coords = a[1][0]
        y_coords = a[3][0]
        poly = np.vstack((x_coords, y_coords)).T

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

    coco_annot.add_image(IMG_ID, img_width, img_height, f)

coco_annot.save("icdar_2017_total_text_%s.json"%(dataset))
