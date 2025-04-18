# inference with detector, gdrn, and refiner
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)
print(PROJ_ROOT)

# from predictor_yolo import YoloPredictor
from predictor_gdrn import GdrnPredictor
import os

import cv2
import torch

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
def get_image_list(rgb_images_path, depth_images_path=None):
    image_names = []

    rgb_file_names = os.listdir(rgb_images_path)
    rgb_file_names.sort()
    for filename in rgb_file_names:
        apath = os.path.join(rgb_images_path, filename)
        ext = os.path.splitext(apath)[1]
        if ext in IMAGE_EXT:
            image_names.append(apath)

    if depth_images_path is not None:
        depth_file_names = os.listdir(depth_images_path)
        depth_file_names.sort()
        for i, filename in enumerate(depth_file_names):
            apath = os.path.join(depth_images_path, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names[i] = (image_names[i], apath)
                # depth_names.append(apath)

    else:
        for i, filename in enumerate(rgb_file_names):
            image_names[i] = (image_names[i], None)

    return image_names


if __name__ == "__main__":
    image_paths = get_image_list(osp.join(PROJ_ROOT,"/home/robodev/Documents/BPC/gdrnpp_bop2022/datasets/BOP_DATASETS/test4/rgb"), 
                                 osp.join(PROJ_ROOT,"/home/robodev/Documents/BPC/gdrnpp_bop2022/datasets/BOP_DATASETS/test4/depth"))
    import matplotlib.pyplot as plt
    img = cv2.imread("/home/robodev/Documents/BPC/gdrnpp_bop2022/datasets/BOP_DATASETS/test4/rgb/000009_cam1_000000.png")
    plt.imshow(img)
    # yolo_predictor = YoloPredictor(
    #                    exp_name="yolox-x",
    #                    config_file_path=osp.join(PROJ_ROOT,"/home/robodev/Documents/BPC/gdrnpp_bop2022/configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_itodd_pbr_itodd_bop_test.py"),
    #                    ckpt_file_path=osp.join(PROJ_ROOT,"/home/robodev/Documents/BPC/gdrnpp_bop2022/pretrained_models/yolox/yolox_x.pth"),
    #                    fuse=True,
    #                    fp16=False
    #                  )
    gdrn_predictor = GdrnPredictor(
        config_file_path=osp.join(PROJ_ROOT,"configs/gdrn/ipdPbrSO/4.py"),
        ckpt_file_path=osp.join(PROJ_ROOT,"output/gdrn/ipdPbrSO/4/model_0044429.pth"),
        camera_json_path=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ipd/camera_cam2.json"),
        path_to_obj_models=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ipd/models")
    )
    gdrn_predictor.cls_names = ['obj_0', 'obj_0', 'obj_0', 'obj_0']

    for rgb_img, depth_img in image_paths:
        rgb_img = cv2.imread(rgb_img)
        if depth_img is not None:
            depth_img = cv2.imread(depth_img, 0)
        # outputs = yolo_predictor.inference(image=rgb_img)
        outputs = [427,472,250,221,1,1,0]
        # tensor_list = [torch.tensor([i],dtype=torch.float32) for i in outputs]  
        # tensor_list = [t.to('cuda') for t in tensor_list]
        # tensor_list = torch.stack(tensor_list)
        # tensor_list = torch.tensor([[[427., 472., 250., 221., 1., 1., 0.]]], device='cuda:0')

        tensor_list = torch.tensor([[2263.62, 762.01, 2444.67, 861.78, 1., 1., 0.]], device='cuda:0')
        # [1704.39, 668.54, 1888.49, 834.40, 1., 1., 0.] 
        # [2014.41, 306.19, 2199.78, 475.12, 1., 1., 0.], 
        # [2367.34, 351.64, 2531.15, 487.81, 1., 1., 0.], 
        # [2263.62, 762.01, 2444.67, 861.78, 1., 1., 0.]


        data_dict = gdrn_predictor.preprocessing(outputs=tensor_list, image=rgb_img, depth_img=depth_img)
        out_dict = gdrn_predictor.inference(data_dict)
        poses = gdrn_predictor.postprocessing(data_dict, out_dict)
        gdrn_predictor.gdrn_visualization(batch=data_dict, out_dict=out_dict, image=rgb_img)

