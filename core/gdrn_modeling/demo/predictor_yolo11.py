# the core predictor classes for gdrn
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)
import matplotlib.pyplot as plt

import torchvision
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from contextlib import ExitStack

class YoloPredictor():

    def __init__(self, 
                 model_path=osp.join(PROJ_ROOT, "output/yolo11/model_final.pt"),
                 conf=0.35,
                 iou=0.45):
        """
        Initialize YOLOv11 predictor
        
        Args:
            model_path (str): Path to the trained model weights
            conf (float): Confidence threshold
            iou (float): IoU threshold for non-maximum suppression
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def visual_yolo(self, results, rgb_image, class_names=None, cls_conf=0.35):
        """
        Visualize YOLO detection results
        
        Args:
            results (list): Detection results from YOLO
            rgb_image (np.ndarray): Input image
            class_names (list, optional): List of class names
            cls_conf (float, optional): Confidence threshold for visualization
        """
        if not results or len(results) == 0:
            return rgb_image

        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                continue

            for box in boxes:
                # Check confidence
                if box.conf[0] >= cls_conf:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = box.conf[0]
                    
                    # Draw rectangle
                    color = (0, 255, 0)  # Green color
                    cv2.rectangle(rgb_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{class_names[cls] if class_names else str(cls)}: {conf:.2f}"
                    cv2.putText(rgb_image, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # cv2.imshow('Detection', rgb_image)
        # cv2.waitKey(0)
        # plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image)
        plt.title("Detection Results")
        plt.axis('off')
        plt.show()
        return rgb_image

    def inference(self, image, verbose=False):
        """
        Run inference on an input image
        
        Args:
            image (np.ndarray): Input RGB image
            verbose (bool): Whether to print detection details
        
        Returns:
            list: Processed detection results
        """
        with ExitStack():
            # Run inference
            results = self.model.predict(
                image, 
                conf=self.conf, 
                iou=self.iou, 
                verbose=verbose,
            )
        
        return results

    def postprocess(self, results, num_classes, conf_thre=0.7, nms_thre=0.45, 
                class_agnostic=False, keep_single_instance=False):
        """
        Postprocess detection results to match YOLOX original format
        
        Args:
            results (list): Ultralytics detection results
            num_classes (int): Number of classes
            conf_thre (float): Confidence threshold
            nms_thre (float): NMS threshold
            class_agnostic (bool): Whether to perform class-agnostic NMS
            keep_single_instance (bool): Keep only highest confidence instance per class
        
        Returns:
            list: Postprocessed detections in torch tensor format
        """
        # Initialize output list with None for each image
        output = [None for _ in range(len(results))]
        
        for i, result in enumerate(results):
            # Skip if no detections
            if len(result.boxes) == 0:
                continue
            
            # Convert boxes to tensor
            boxes = result.boxes
            
            # Prepare detection tensor with YOLOX-like format
            # Format: [x_center, y_center, width, height, obj_conf, class_conf, class_pred]
            image_pred = torch.zeros((len(boxes), 5 + num_classes + 1))
            
            for j, box in enumerate(boxes):
                # XYWH format
                x_center, y_center, width, height = box.xywh[0]
                
                # Object confidence and class confidence
                obj_conf = box.conf[0]
                cls = int(box.cls[0])
                
                # Populate tensor
                image_pred[j, 0] = x_center  # x_center
                image_pred[j, 1] = y_center  # y_center
                image_pred[j, 2] = width     # width
                image_pred[j, 3] = height    # height
                image_pred[j, 4] = obj_conf  # object confidence
                
                # Set class confidence
                image_pred[j, 5 + cls] = obj_conf
                image_pred[j, -1] = cls  # class prediction
            
            # Convert to corner box format (x1, y1, x2, y2)
            box_corner = image_pred.new(image_pred.shape)
            box_corner[:, 0] = image_pred[:, 0] - image_pred[:, 2] / 2
            box_corner[:, 1] = image_pred[:, 1] - image_pred[:, 3] / 2
            box_corner[:, 2] = image_pred[:, 0] + image_pred[:, 2] / 2
            box_corner[:, 3] = image_pred[:, 1] + image_pred[:, 3] / 2
            image_pred[:, :4] = box_corner[:, :4]
            
            # Get class confidence and prediction
            class_conf, class_pred = torch.max(image_pred[:, 5:5+num_classes], 1, keepdim=True)
            
            # Filter by confidence threshold
            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            
            # Keep single instance per class if requested
            if keep_single_instance:
                instance_detections = torch.zeros(num_classes, 7)
                for class_num in range(num_classes):
                    max_conf = 0
                    for detection in detections[detections[:, 6] == class_num]:
                        if detection[4] * detection[5] > max_conf:
                            instance_detections[class_num] = detection
                            max_conf = detection[4] * detection[5]
                detections = instance_detections
            
            # Skip if no detections after filtering
            if detections.size(0) == 0:
                continue
            
            # Perform NMS
            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre
                )
            
            # Apply NMS and sort
            detections = detections[nms_out_index]
            detections = detections[detections[:, 6].argsort()]
            
            # Store in output
            output[i] = detections
        
        return output


if __name__ == "__main__":
    # Initialize the YoloPredictor
    predictor = YoloPredictor(
        model_path="/home/robodev/Documents/BPC/bpc_baseline/runs/detect/train13/weights/best.pt",
        conf=0.35,
        iou=0.45
    )
    img_path = "/home/robodev/Documents/BPC/gdrnpp_bop2022/datasets/BOP_DATASETS/test1/rgb/000011_cam1_000003.png"
    # img_path = osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/lmo/test/000001/rgb/000000.jpg")
    image_rgb = cv2.imread(img_path)
    # result = predictor.inference(img)
    # predictor.visual_yolo(result[0], img, ["cls_name_1", "cls_name_2"])
    # Run inference
    results = predictor.inference(image_rgb)
    class_names = ['0']
    # Postprocess the results (optional)
    processed_results = predictor.postprocess(
        results, 
        num_classes=len(class_names), 
        conf_thre=0.35, 
        nms_thre=0.45
    )
    
    # Visualize the results
    predictor.visual_yolo(
        results, 
        image_rgb, 
        class_names=class_names, 
        cls_conf=0.35
    )
    
    # Additional processing of results if needed
    for result in processed_results:
        if result is not None:
            for detection in result:
                # Print details of each detection
                print(f"Class: {detection.cls}, Confidence: {detection.conf}, Bbox: {detection.xyxy}")

