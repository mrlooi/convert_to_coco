import os, os.path as osp
import glob
import numpy as np
import cv2
from scipy.io import loadmat

from natsort import natsorted

from coco_annotation import CocoAnnotationClass


def convert_rect_to_pts(roi):
    x_c, y_c, w, h, theta = roi
    rect = ((x_c, y_c), (w, h), theta)
    rect = cv2.boxPoints(rect)
    # rect = np.int0(np.round(rect))
    return rect

def get_polygons_from_annotation(annot_path):
    polygons = []
    with open(annot_path, 'r') as f:
        line = f.readlines()
        for l in line:
            values = l.strip().split(" ")
            index, is_difficult = values[:2]
            x, y, w, h, angle_rad = np.array(values[2:], dtype=np.float32)

            angle = np.rad2deg(angle_rad)

            xc = x + w / 2
            yc = y + h / 2
            roi = [xc, yc, w, h, angle]

            poly = convert_rect_to_pts(roi)
            polygons.append(poly)
    return polygons


GREEN = (0,255,0)
RED = (0,0,255)
BLUE = (255,0,0)

VIS = False

root_img_dir = "/home/vincent/hd/datasets/OCR/MSRA-TD500/Images"
root_annot_dir = "/home/vincent/hd/datasets/OCR/MSRA-TD500/Annotations"


# dataset = "train"
dataset = "test"

img_dir = osp.join(root_img_dir, dataset)
annot_dir = osp.join(root_annot_dir, dataset)

img_files = natsorted(os.listdir(img_dir))
# img_files = [f for f in img_files if "664" in f]

# coco init
coco_annot = CocoAnnotationClass(["text"], "msra_td500") # COCO IS 1-indexed, don't include BG CLASS
cls_idx = 1 # default 

ANNOT_ID = 0

total = len(img_files)
for fx, f in enumerate(img_files, 1): 
    print("%d of %d) %s"%(fx, total, f))

    img_path = osp.join(img_dir, f)
    IMG_ID = f.replace("IMG_","").replace(".JPG","")

    ann_f = f.replace(".JPG",".gt")
    annot_path = osp.join(annot_dir, ann_f)

    img = cv2.imread(img_path)
    if img is None:
        print("Could not read %s"%(img_path))
        continue
    if not osp.exists(annot_path):
        print("Could not find %s"%(annot_path))
        continue

    polys = get_polygons_from_annotation(annot_path)
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

    img_height, img_width = img.shape[:2]

    coco_annot.add_image(IMG_ID, img_width, img_height, f)

coco_annot.save("msra_td500_%s.json"%(dataset))
