import numpy as np
import cv2
import json
import os.path as osp
import copy
import datetime



def convert_datetime_to_string(dt=datetime.datetime.now(), formt="%Y-%m-%d %H:%M:%S"):
    return dt.strftime(formt)

class CocoAnnotationClass(object):
    def __init__(self, classes, supercategory=""):
        self.classes = classes
        self.data = self._get_default_data()
        self._init_var()
        for c,idx in self.map_classes_idx.items():
            self._add_category(c,idx,supercategory)

    def _init_var(self):
        self.map_classes_idx = {c: ix+1 for ix,c in enumerate(self.classes)}  # coco is 1-indexed
        self.map_idx_classes = {v:k for k,v in self.map_classes_idx.items()}

    def _get_default_data(self):
        default_d = {
            "info": {
                "year" : 2019, 
                "version" : "", 
                "description" : "", 
                "contributor" : "", 
                "url" : "", 
                "date_created" : convert_datetime_to_string()
            },
            "images": [],
            "annotations": [],
            "categories": [],
            "licenses": [
                {
                    "id" : 1, 
                    "name" : "", 
                    "url" : ""
                }
            ]
        }
        return default_d

    def set_classes(self, classes):
        self.classes = classes

    def clear(self):
        self.data = self._get_default_data()

    def _add_category(self, name, id=None, supercategory=""):
        cat_id = len(self.data["categories"]) + 1 if id is None else id
        cat_data = {
                    "id" : cat_id, 
                    "name" : name, 
                    "supercategory" : supercategory
                }
        self.data["categories"].append(cat_data)

    def add_annot(self, id, img_id, img_cls, seg_data, meta_data={}, is_crowd=0):
        """
        CAN NOW SUPPORT MULTIPLE SEG POLYGONS
        DEPRECATED: ONLY SUPPORTS seg polygons of len 1 i.e. cannot support multiple polygons that refer to the same id"""
        if isinstance(img_cls, str):
            if img_cls not in self.map_classes_idx:
                print("%s not in coco classes!"%(img_cls))
                return 
            cat_id = self.map_classes_idx[img_cls]
        else:
            assert img_cls in self.map_idx_classes
            cat_id = img_cls
        # seg_data_arr = np.array(seg_data)
        # if len(seg_data_arr.shape) == 2:
        #     seg_data_arr = seg_data_arr[None,:]
        # assert seg_data_arr.shape[-1] == 2 # x,y
        if len(seg_data) == 0:
            print("Polygon (seg_data) is empty!")
            return 
        seg_data_arr = seg_data if type(seg_data[0][0]) in [list, np.ndarray] else [seg_data]
        concat_arr = np.concatenate(seg_data_arr)
        bbox = np.array([np.amin(concat_arr, axis=0), np.amax(concat_arr, axis=0)]).reshape(4)
        bbox[2:] -= bbox[:2]
        bbox = bbox.tolist()
        area = sum([cv2.contourArea(arr) for arr in seg_data_arr])
        annot_data =    {
                    "id" : id,
                    "image_id" : img_id,
                    "category_id" : cat_id,
                    "segmentation" : [arr.flatten().tolist() for arr in seg_data_arr],
                    "area" : area,
                    "bbox" : bbox,
                    "iscrowd" : is_crowd,
                    "meta": meta_data # CUSTOM
                }
        self.data["annotations"].append(annot_data)

    def add_image(self, id, width, height, file_name, meta_data={}, date_captured=convert_datetime_to_string()):
        img_data =  {
                    "id" : id,
                    "width" : width,
                    "height" : height,
                    "file_name" : file_name,
                    "license" : 1,
                    "flickr_url" : "",
                    "coco_url" : "",
                    "date_captured" : date_captured,
                    "meta": meta_data
                }

        self.data["images"].append(img_data)

    def get_annot_json(self):
        return copy.deepcopy(self.data)

    def save(self, out_file):
        with open(out_file, "w") as f:
            json.dump(self.data, f)
            print("Saved to %s"%(out_file))

    def load(self, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)
            print("Loaded from %s"%(json_file))
        self.classes = [c['name'] for c in self.data["categories"]]
        self._init_var()
