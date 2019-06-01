import numpy as np
import cv2
from transforms3d.quaternions import mat2quat, quat2mat
import open3d

from coco_annotation import CocoAnnotationClass

def get_camera_settings_intrinsic_matrix(camera_settings):
    intrinsic_settings = camera_settings['intrinsic_settings']
    intrinsic_matrix = np.identity(3)
    intrinsic_matrix[0,0] = intrinsic_settings['fx']
    intrinsic_matrix[1,1] = intrinsic_settings['fy']
    intrinsic_matrix[0,2] = intrinsic_settings['cx']
    intrinsic_matrix[1,2] = intrinsic_settings['cy']
    return intrinsic_matrix

def create_cloud(points, normals=[], colors=[], T=None):
    cloud = open3d.PointCloud()
    cloud.points = open3d.Vector3dVector(points)
    if len(normals) > 0:
        assert len(normals) == len(points)
        cloud.normals = open3d.Vector3dVector(normals)
    if len(colors) > 0:
        assert len(colors) == len(points)
        cloud.colors = open3d.Vector3dVector(colors)

    if T is not None:
        cloud.transform(T)
    return cloud

def get_random_color():
    return (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))

def backproject_camera(im_depth, intrinsic_matrix, factor_depth):

    depth = im_depth.astype(np.float32, copy=True) / factor_depth

    # get intrinsic matrix
    K = np.matrix(intrinsic_matrix)
    K = np.reshape(K, (3,3))
    Kinv = np.linalg.inv(K)

    # compute the 3D points        
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

    # backprojection
    R = Kinv * x2d.transpose()

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)

    return np.array(X)

def get_4x4_transform(pose):
    object_pose_matrix4f = np.identity(4)
    object_pose = np.array(pose)
    if object_pose.shape == (4,4):
        object_pose_matrix4f = object_pose
    elif object_pose.shape == (3,4):
        object_pose_matrix4f[:3,:] = object_pose
    elif len(object_pose) == 7:
        object_pose_matrix4f[:3,:3] = quat2mat(object_pose[:4])
        object_pose_matrix4f[:3,-1] = object_pose[4:]
    else:
        print("[WARN]: Object pose for %s is not of shape (4,4) or (3,4) or 1-d quat (7)")
        return None
    return object_pose_matrix4f

def render_depth_pointcloud(im, depth, pose_data, points, intrinsic_matrix, factor_depth=1):

    rgb = im.copy()[:,:,::-1]
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255

    X = backproject_camera(depth, intrinsic_matrix, factor_depth)
    cloud_rgb = rgb # .astype(np.float32)[:,:,::-1] / 255
    cloud_rgb = cloud_rgb.reshape((cloud_rgb.shape[0]*cloud_rgb.shape[1],3))
    scene_cloud = create_cloud(X.T, colors=cloud_rgb)

    coord_frame = open3d.create_mesh_coordinate_frame(size = 0.6, origin = [0, 0, 0])

    if len(pose_data) == 0:
        open3d.draw_geometries([scene_cloud, coord_frame])
        return 

    all_objects_cloud = open3d.PointCloud()
    for pd in pose_data:
        object_cls = pd["cls"]
        object_pose = pd["pose"]
        # object_cloud_file = osp.join(object_model_dir,object_name,"points.xyz")
        object_pose_matrix4f = get_4x4_transform(object_pose)
        if object_pose_matrix4f is None:
            continue

        # object_pose_matrix4f[2,3]
        object_pts3d = points[object_cls] # read_xyz_file(object_cloud_file)
        object_cloud = create_cloud(object_pts3d)
        object_cloud.transform(object_pose_matrix4f)
        all_objects_cloud += object_cloud

        # print("Showing %s"%(object_name))
    open3d.draw_geometries([scene_cloud, all_objects_cloud, coord_frame])

def get_object_data(annotation, object_seg_ids):
    objects = annotation['objects']
    data = []
    for o in objects:
        d = {}
        q = o['quaternion_xyzw']
        t = np.array(o['location']) / 100  # cm to M
        pose = [q[-1],q[0],q[1],q[2],t[0],t[1],t[2]]
        d['pose'] = np.array(pose)
        d['transform'] = get_4x4_transform(pose)
        d['projected_cuboid'] = np.array(o['projected_cuboid'])
        d['visibility'] = o['visibility']
        cls = o['class']
        d['cls'] = cls
        d['seg_id'] = object_seg_ids[cls]
        data.append(d)
    return data


def load_points_from_obj_file(obj_file):
    """Loads a Wavefront OBJ file. """
    vertices = []
    print("Loading obj file %s..."%(obj_file))

    material = None
    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = values[1:4]
                # if swapyz:
                #     v = v[0], v[2], v[1]
                vertices.append(v)
    return np.array(vertices).astype(np.float32)

def download_models():
    """
    for i in "002_master_chef_can" "003_cracker_box" "004_sugar_box" "005_tomato_soup_can" "006_mustard_bottle" 
    "007_tuna_fish_can" "008_pudding_box" "009_gelatin_box" "010_potted_meat_can" "011_banana" "019_pitcher_base" "021_bleach_cleanser" 
    "024_bowl" "025_mug" "035_power_drill" "036_wood_block" "037_scissors" "040_large_marker" "051_large_clamp" 
    "052_extra_large_clamp" "061_foam_brick"; 
        do wget 'http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/'$i'_google_16k.tgz'; done
    """

def get_2d_projected_points(points, intrinsic_matrix):#, M):
    """
    points: (N,3) np array
    intrinsic matrix: 3x3 matrix
    """
    x3d = points.transpose()

    # projection
    # RT = M[:3,:]
    x2d = np.matmul(intrinsic_matrix, x3d)
    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
    x = np.transpose(x2d[:2,:], [1,0]).astype(np.int32)
    return x

def transform_points(points, M):
    """
    points: (N,3) np array
    M: 4x4 matrix
    """
    x3d = np.ones((4, len(points)), dtype=np.float32)
    x3d[0, :] = points[:,0]
    x3d[1, :] = points[:,1]
    x3d[2, :] = points[:,2]

    # projection
    RT = M[:3,:]
    pts_T = np.matmul(RT, x3d)
    pts_T = pts_T.transpose()  # (3,N) to (N,3)
    return pts_T

def get_valid_objects(im, seg_mask, object_data, points, intrinsic_matrix, min_visibility=0.0, min_proj_pct=0.4):
    h,w = im.shape[:2]

    valid_object_data = []

    close_kernel = np.ones((7,7),np.uint8)

    for pd in object_data:
        cls = pd["cls"]
        seg_id = pd['seg_id']
        vis = pd['visibility']

        if vis < min_visibility:
            continue 

        M = pd['transform']

        cls_pts = np.vstack((points[cls], [0.,0.,0.])) # the object points are centered at [0,0,0]. Get the 2D projection of it in pixels
        # projection

        pts_transformed = transform_points(cls_pts, M)
        x = get_2d_projected_points(pts_transformed, intrinsic_matrix)
        vertex_center = x[-1].astype(np.int32)
        x = x[:-1]

        if not (0 <= vertex_center[0] < w and 0 <= vertex_center[1] < h):
            continue

        mask = np.zeros((h,w), dtype=np.uint8)

        total_pts = len(x)
        proj_pts = []
        for px in x:
            if 0 <= px[0] < w and 0 <= px[1] < h:
                if seg_mask[px[1],px[0]] == seg_id:
                    # cv2.circle(mask, tuple(px), 1, 255, -1)
                    mask[px[1],px[0]] = 255
                    proj_pts.append(px)
        pct = float(len(proj_pts)) / total_pts
        if pct < min_proj_pct:
            continue

        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        _, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        polygons = convert_contours_to_polygons(contours)

        pd['center'] = vertex_center
        pd['polygons'] = polygons
        # add min max of points
        pd['bounds'] = np.array([np.min(pts_transformed, axis=0), np.max(pts_transformed, axis=0)])
        pd['centroid'] = pts_transformed[-1]
        valid_object_data.append(pd)

    return valid_object_data

def visualize_pose(im, seg_mask, object_data, points, intrinsic_matrix):
    im_copy = im.copy()
    im_copy2 = im.copy()
    h,w = im_copy.shape[:2]
    canvas = np.zeros((h,w,3), dtype=np.uint8)

    close_kernel = np.ones((5,5),np.uint8)

    for pd in object_data:
        cls = pd["cls"]
        pose = pd["pose"]
        seg_id = pd['seg_id']
        vis = pd['visibility']
        # if vis < 0.2:
        #     continue 

        color = get_random_color()
        cls_pts = np.vstack((points[cls], [0.,0.,0.])) # the object points are centered at [0,0,0]. Get the 2D projection of it in pixels

        # projection
        M = pd['transform']

        pts_transformed = transform_points(cls_pts, M)
        x = get_2d_projected_points(pts_transformed, intrinsic_matrix)
        vertex_center = x[-1].astype(np.int32)
        x = x[:-1]

        if not (0 <= vertex_center[0] < w and 0 <= vertex_center[1] < h):
            continue

        mask = np.zeros((h,w), dtype=np.uint8)

        total_pts = len(x)
        proj_pts = []
        for px in x:
            px = tuple(px)
            if 0 <= px[0] < w and 0 <= px[1] < h:
                cv2.circle(im_copy, px, 3, color, -1)
                if seg_mask[px[1],px[0]] == seg_id:
                    cv2.circle(canvas, px, 3, color, -1)
                    cv2.circle(mask, px, 2, 255, -1)
                    proj_pts.append(px)
        pct = float(len(proj_pts)) / total_pts
        if pct < 0.4:
            continue
        proj_pts = np.array(proj_pts)
        mean_pt = np.mean(proj_pts, axis=0).astype(np.int32)
        center_pt = (vertex_center[0],vertex_center[1])
        cv2.circle(im_copy2, center_pt, 3, (255,0,0), -1)
        try:
            im_copy2 = cv2.putText(im_copy2, "%.2f"%pct, tuple(mean_pt), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,255,255))
        except Exception:#, e:
            print(mean_pt)
            continue

        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        # cv2.imshow("mask", mask)
        # cv2.imshow("closing", closing)
        # cv2.waitKey(0)

        _, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        polygons = convert_contours_to_polygons(contours)
        approx = polygons[0]
        total = len(approx)
        for j,p in enumerate(approx):
            cv2.circle(im_copy2, tuple(p), 2, color, -1)
            cv2.line(im_copy2, tuple(p), tuple(approx[(j+1)%total]), color, 1)

        # draw 3 axis pose
        # from https://github.com/jerryhouuu/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV/blob/master/pose_estimation.py
        axis = np.float32([[0.1,0,0],  # blue axis
                              [0,0.1,0], # green axis
                              [0,0,0.1]])  # red axis
        R = M[:3,:3]
        T = M[:3,-1]
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        imgpts, jac = cv2.projectPoints(axis, R, T, intrinsic_matrix, dist_coeffs)
        _imgpts, jac = cv2.projectPoints(axis, np.identity(3), T, intrinsic_matrix, dist_coeffs)
        # print(imgpts)
        cv2.line(im_copy2, center_pt, tuple(imgpts[0].ravel()), (255,0,0), 3) #BLUE
        cv2.line(im_copy2, center_pt, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
        cv2.line(im_copy2, center_pt, tuple(imgpts[2].ravel()), (0,0,255), 3) #RED
        cv2.imshow("polygons", im_copy2)

    cv2.imshow("proj_poses", im_copy)
    cv2.imshow("proj_poses_refined", canvas)


def draw_cuboid_lines(img2, points, color):
    for ix in range(len(points)):
        cv2.putText(img2, "%d"%(ix), points[ix], cv2.FONT_HERSHEY_COMPLEX, 0.5, color)
    cv2.line(img2, points[0], points[1], color)
    cv2.line(img2, points[1], points[2], color)
    cv2.line(img2, points[3], points[2], color)
    cv2.line(img2, points[3], points[0], color)
    
    # draw back
    cv2.line(img2, points[4], points[5], color)
    cv2.line(img2, points[6], points[5], color)
    cv2.line(img2, points[6], points[7], color)
    cv2.line(img2, points[4], points[7], color)
    
    # draw sides
    cv2.line(img2, points[0], points[4], color)
    cv2.line(img2, points[7], points[3], color)
    cv2.line(img2, points[5], points[1], color)
    cv2.line(img2, points[2], points[6], color)

def get_cls_contours(label, cls):
    mask = np.zeros((label.shape), dtype=np.uint8)
    y,x = np.where(label==cls)
    mask[y,x] = 255

    _, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    return contours

def approx_contour(cnt, eps=0.005):
    if len(cnt) == 0:
        return []
    arclen = cv2.arcLength(cnt, True)
    epsilon = arclen * eps
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    approx = approx.squeeze()
    return approx

def convert_contours_to_polygons(contours, eps=0.005):
    polygons = [approx_contour(cnt, eps) for cnt in contours]
    polygons = [p for p in polygons if len(p) >= 3] # need at least 3 points to form a polygon
    return polygons

def visualize_proj_cuboid(im, object_data):
    im_copy = im.copy()
    h,w = im_copy.shape[:2]
    mask = np.zeros((h,w,3), dtype=np.uint8)
    for d in object_data:
        color = get_random_color()
        projc = d['projected_cuboid'].astype(np.int32)
        # cv2.fillConvexPoly(mask, projc[::2].copy(), color)

        projc = [tuple(pt) for pt in projc]
        for pt in projc:
            cv2.circle(im_copy, pt, 3, color, -1)
        draw_cuboid_lines(im_copy, projc, color)

    cv2.imshow("proj_cuboid", im_copy)
    # cv2.imshow("proj_cuboid_masks", mask)

coord_frame = open3d.create_mesh_coordinate_frame(size = 0.6, origin = [0, 0, 0])

if __name__ == '__main__':
    import glob
    import json 

    CLASSES = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                     '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                     '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                     '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
    CLASSES_INDEX = dict((c, ix) for ix,c in enumerate(CLASSES))

    ROOT_DIR = "/home/vincent/hd/datasets/FAT"
    MODEL_DIR = ROOT_DIR + "/models"
    SUPERCATEGORY = "FAT"

    points = {}

    coco_annot = CocoAnnotationClass(CLASSES[1:], SUPERCATEGORY) # COCO IS 1-indexed, don't include BG CLASS

    IMG_ID = 0
    ANNOT_ID = 0

    # LOAD DIR SETTINGS

    # sample_dir = ROOT_DIR + "/single/002_master_chef_can_16k/temple_0"
    # fat_dir = "/"
    sample_dir = ROOT_DIR + "/mixed/temple_0"
    camera_type = "left"
    camera_settings_json = sample_dir + "/_camera_settings.json"
    object_settings_json = sample_dir + "/_object_settings.json"

    with open(camera_settings_json, "r") as f:
        camera_settings = json.load(f)
    with open(object_settings_json, "r") as f:
        object_settings = json.load(f)
    camera_settings = camera_settings['camera_settings']
    intrinsics = dict((c['name'], get_camera_settings_intrinsic_matrix(c)) for c in camera_settings)
    OBJ_CLASSES = object_settings["exported_object_classes"]
    object_settings = object_settings["exported_objects"]
    object_seg_ids = dict((o['class'], o['segmentation_class_id']) for o in object_settings)
            
    # Convert transforms from column-major to row-major, and from cm to m
    object_transforms = dict((o['class'], np.array(o['fixed_model_transform']).transpose() / 100) for o in object_settings)
    
    for cls in OBJ_CLASSES:
        if cls not in points:
            ocls = cls.lower().replace("_16k","")
            obj_file = MODEL_DIR + "/%s/google_16k/textured.obj"%(ocls)
            obj_points = load_points_from_obj_file(obj_file)
            cloud = create_cloud(obj_points, T=object_transforms[cls])
            points[cls] = np.asarray(cloud.points)
            xyz_file = MODEL_DIR + "/%s/points.xyz"%(ocls)
            obj_points = np.savetxt(xyz_file, points[cls][::3])
            # open3d.draw_geometries([cloud, coord_frame])

    factor_depth = 10000

    # LOAD FILES IN DIR
    n_samples = 10 # 2
    dir_files = glob.glob("%s/*.jpg"%(sample_dir))[:n_samples]  #[1000:1000+n_samples]
    total_files = len(dir_files)
    print("%s folder: %d images"%(sample_dir, total_files))
    for fx,file in enumerate(dir_files):
        sample_file = file.replace(".jpg","")
        # camera_type = sample_file
        # sample_file = sample_dir + "/000000.%s"%camera_type
        camera_type = sample_file.split(".")[-1]  # left or right
        intrinsic_matrix = intrinsics[camera_type]

        annot_file = sample_file + ".json"
        img_file = sample_file + ".jpg"
        seg_file = sample_file + ".seg.png"
        depth_file = sample_file + ".depth.png"

        with open(annot_file, "r") as f:
            annotation = json.load(f)

        img = cv2.imread(img_file)
        if img is None:
            print("Could not find %s"%(img_file))
            continue

        label = cv2.imread(seg_file, cv2.IMREAD_UNCHANGED)
        if label is None:
            print("Could not find %s"%(seg_file))
            continue

        object_data = get_object_data(annotation, object_seg_ids)
        if len(object_data) == 0:
            continue
        IMG_ID += 1

        valid_object_data = get_valid_objects(img, label, object_data, points, intrinsic_matrix)
        if len(valid_object_data) == 0:
            continue

        for od in valid_object_data:
            ANNOT_ID += 1
            cls = od['cls'].lower().replace("_16k","")
            polygons = od['polygons']
            # print(polygons)
            # break
            cls_idx = CLASSES_INDEX[cls]
            meta_data = {
                'center': od['center'].tolist(),
                'centroid': od['centroid'].tolist(),
                'bounds': od['bounds'].tolist(),
                'pose': od['pose'].flatten().tolist(),
                'intrinsic_matrix': intrinsic_matrix.tolist()
             }
            coco_annot.add_annot(ANNOT_ID, IMG_ID, cls_idx, polygons[0], meta_data)

        img_height, img_width = img.shape[:2]
        img_name = img_file.replace(ROOT_DIR, "").strip("/")
        depth_name = depth_file.replace(ROOT_DIR, "").strip("/")
        meta_data = {
            "depth_file_name": depth_name,
            "factor_depth": factor_depth
        }
        coco_annot.add_image(IMG_ID, img_width, img_height, img_name, meta_data=meta_data)

        print("Loaded %d of %d (%s)"%(fx+1, total_files, sample_dir))

        # # """VISUALIZE"""
        visualize_pose(img, label, object_data, points, intrinsic_matrix)
        visualize_proj_cuboid(img, object_data)
        # cv2.imshow("img", img)
        # cv2.imshow("seg", label)
        cv2.waitKey(0)
        
        # depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        # render_depth_pointcloud(img, depth, object_data, points, intrinsic_matrix, factor_depth)
    
    # n_samples = "all"
    # coco_annot.save("coco_fat_debug_%s.json"%(n_samples))
