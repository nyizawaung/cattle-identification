'''
ReID in interval of 300 frames (1 minute)
'''

import timm
import torch, detectron2
import torch.nn as nn
import torch.optim as optim
from detectron2.utils.logger import setup_logger
import random
import torch.nn.functional as F
from typing import Iterable, List, NamedTuple

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.modeling import build_model
from detectron2.structures import Instances

from sklearn.metrics.pairwise import cosine_similarity

import detectron2.data.transforms as T
import cv2
import numpy as np
import os
from helpers.helper import Helper
from helpers.sleeper_divider import SleeperDivider
from batch_pipeline_patched_single_cam_v3 import PipelineMixin, STOP

from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms, models
from datetime import datetime
from collections import Counter
import csv
from PIL import Image
import math
from enum import Enum
import pandas as pd
import json
from datetime import datetime
import time
import queue
from dataclasses import dataclass
from typing import List, Any, Tuple, Optional


import heapq
import itertools
from torch.utils.data import DataLoader, Dataset
from numpy import ndarray
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

class _Stop: pass
STOP = _Stop()

torch.cuda.empty_cache()
class CheckSimilarity():
    model = timm.create_model('resnet101', pretrained=True, num_classes=0)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def extract_embedding(self,img: Image.Image) -> np.ndarray:
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)
        return embedding.squeeze().cpu().numpy()

    def cosine_similarity_search(self,last_image_paths,current_image: Image.Image): #current_image: Image.Image):
        query_embedding = self.extract_embedding(current_image)

        db_embeddings = []
        valid_paths = []

        for image in last_image_paths:
            try:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                emb = self.extract_embedding(img)
                db_embeddings.append(emb)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if not db_embeddings:
            return None, 0.0

        db_embeddings_np = np.array(db_embeddings)
        similarities = cosine_similarity([query_embedding], db_embeddings_np)[0]

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_match_path = valid_paths[best_idx]

        return best_match_path, best_score
class MASKLOCATION():
    def __init__(self, tracking_id, mask, cam_counter, current_point, is_touching_bottom=False, is_touching_top=False):
        self.tracking_id = tracking_id
        self.mask = mask
        self.is_touching_bottom = is_touching_bottom
        self.is_touching_top = is_touching_top
        self.cam_counter = cam_counter
        self.current_point = current_point #centroid
        self.last_mask_area = np.sum(mask)
        self.mask_area = np.sum(mask)

class CoreProcessResponseModel():
    def __init__(self, cam_counter, boxes,tracked_indexes, tracked_ids, predicted_ids, original_predicted_ids, colored_mask, recheck_periods):
        self.cam_counter = cam_counter
        self.boxes = boxes
        self.tracked_indexes = tracked_indexes
        self.tracked_ids = tracked_ids
        self.predicted_ids = predicted_ids
        self.original_predicted_ids = original_predicted_ids
        self.colored_mask = colored_mask
        self.recheck_periods = recheck_periods
    
    
class Prediction(NamedTuple):
    x: float
    y: float
    width: float
    height: float
    score: float
    class_name: str

images = []
class ImageDataset(Dataset):

    def __init__(self, imagery: List[Path]):
        self.imagery = imagery

    def __getitem__(self, index) -> ndarray:
        return cv2.imread(self.imagery[index])

    def __len__(self):
        return len(self.imagery)


class TRACKING_TYPE(Enum):
    BOX = 'BOX'
    CENTROID = 'CENTROID'
    IOU = 'IOU'

class ProtoInfer:
    def __init__(self, model, device, proto_ckpt_path, class_mapping_path=None):
        """
        class_mapping_path: path to class_mappings.json (optional)
        Supports:
          - dict index->name  (keys int or numeric strings)
          - list              (index->name)
          - dict protoLabel->name (keys are strings that match checkpoint proto_labels)
        """
        self.model = model.eval().to(device)
        self.device = device
        ckpt = torch.load(proto_ckpt_path, map_location=device)

        self.prototypes  = ckpt["prototypes"].to(device)   # [K,D] L2-normalized
        self.proto_labels = ckpt["proto_labels"]           # list[str]
        self.tau = 4.0#float(ckpt["tau"])
        emb_dim = ckpt.get("embedding_dim", None)

        if emb_dim is not None and self.prototypes.shape[1] != emb_dim:
            raise ValueError(
                f"Embedding dim mismatch: ckpt={emb_dim}, prototypes D={self.prototypes.shape[1]}"
            )

        self.idx_to_name = None         # list or dict[int]->str
        self.label_to_name = None       # dict[str]->str

        if class_mapping_path is not None:
            with open(class_mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)

            if isinstance(mapping, list):
                self.idx_to_name = list(mapping)

            elif isinstance(mapping, dict):
                def _is_int_like(k):
                    if isinstance(k, int): return True
                    if isinstance(k, str) and k.isdigit(): return True
                    return False

                if all(_is_int_like(k) for k in mapping.keys()):
                    self.idx_to_name = {int(k): v for k, v in mapping.items()}
                else:
                    self.label_to_name = dict(mapping)

        self._have_idx_map = self.idx_to_name is not None
        self._have_lbl_map = self.label_to_name is not None

    @torch.no_grad()
    def predict_stack(self, images_tensors, tau=None):
        """
        images_tensors: list[Tensor[C,H,W]] transformed like val_transform
        returns: list[[class_name, count], ...] (includes 'unknown')
        """
        if tau is None:
            tau = self.tau

        x = torch.stack(images_tensors).to(self.device)  # [B,C,H,W]
        feats = get_embeddings(self.model, x)            # [B,D]
        sims  = feats @ self.prototypes.T                # [B,K]
        max_sims, idxs = sims.max(dim=1)

        labels = []
        for s, i in zip(max_sims.tolist(), idxs.tolist()):
            if s >= tau:
                if self._have_idx_map and i in self.idx_to_name:
                    name = self.idx_to_name[i]
                else:
                    proto_lbl = self.proto_labels[i]
                    if self._have_lbl_map and proto_lbl in self.label_to_name:
                        name = self.label_to_name[proto_lbl]
                    else:
                        name = proto_lbl
                labels.append(name)
            else:
                labels.append("unknown")

        counts = Counter(labels)
        return [[k, v] for k, v in counts.items()]

@torch.no_grad()
def get_embeddings(model, x):
    """
    Get L2-normalized pre-logits embeddings from ConvNeXtV2.
    timm convnextv2: forward_features -> forward_head(pre_logits=True).
    """
    feats = model.forward_features(x)                  # pre-classifier features
    emb  = model.forward_head(feats, pre_logits=True)  # pooled pre-logits
    return F.normalize(emb, dim=1)  


class BatchPredictor():
    def __init__(self, cfg: CfgNode, batch_size: int, workers: int):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.batch_size = batch_size
        self.workers = workers
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __collate(self, batch):
        data = []
        for image in batch:
            if self.input_format == "RGB":
                image = image[:, :, ::-1]
            height, width = image.shape[:2]

            image = image.astype("float32").transpose(2, 0, 1)
            image = torch.as_tensor(image)
            data.append({"image": image, "height": height, "width": width})
        return data

    def __call__(self, imagery) -> Iterable[List[Prediction]]:
        """[summary]

        :param imagery: [description]
        :type imagery: List[Path]
        :yield: Predictions for each image
        :rtype: [type]
        """
        dataset = imagery
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.__collate,
            pin_memory=True
        )

        with torch.no_grad():
            for batch in loader:
                results: List[Instances] = self.model(batch)
                yield from results

    def __map_predictions(self, instances: Instances):
        instance_predictions = zip(
            instances.get('pred_boxes'),
            instances.get('pred_masks')
        )

        predictions = []
        for box, score, class_index in instance_predictions:
            x1 = box[0].item()
            y1 = box[1].item()
            x2 = box[2].item()
            y2 = box[3].item()
            width = x2 - x1
            height = y2 - y1
            prediction = Prediction(
                x1, y1, width, height, score.item())
            predictions.append(prediction)
        return predictions

class CorrectDistortedImage:
    def createSetting(self, k1,k2,k3,p1,p2,focal_length_x,focal_length_y,center_x,center_y,scale,aspect):

            self.image = None
            self.original_image = None 


            setting = {
                'k1': k1,
                'k2': k2,
                'k3': k3,
                'p1': p1,
                'p2': p2,
                'focal_length_x': focal_length_x, 
                'focal_length_y': focal_length_y,
                'center_x': center_x, 
                'center_y': center_y,
                'scale': scale, 
                'aspect': aspect
            }
            return setting

    def apply_correction(self,image,settings):
            if image is None:
                return None
            height, width = image.shape[:2]
            
            center_x = width * settings['center_x']
            center_y = height * settings['center_y']
            
            focal_x = settings['focal_length_x'] * width
            focal_y = settings['focal_length_y'] * width * settings['aspect']
            
            camera_matrix = np.array([
                [focal_x, 0, center_x],
                [0, focal_y, center_y],
                [0, 0, 1]
            ], dtype=np.float32)
            
            distortion_coeffs = np.array([settings['k1'], settings['k2'], settings['p1'], settings['p2'], settings['k3']], dtype=np.float32)
            
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                camera_matrix, distortion_coeffs, (width, height), 1, (width, height))
            
            new_camera_matrix[0, 0] *= settings['scale']  # Scale fx
            new_camera_matrix[1, 1] *= settings['scale']  # Scale fy
            
            undistorted_image = cv2.undistort(
                image, camera_matrix, distortion_coeffs, None, new_camera_matrix)
            
            undistorted_image = undistorted_image[15:height, 0:width]  # Crop the image to remove black borders
            return undistorted_image
    
    def show_image(self, image,window_name="CorrectedImage"):
            cv2.imshow(window_name, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def save_CorretedImage(self, image, filename):
            if image is not None:
                cv2.imwrite(filename, image)
                print(f"Corrected image saved as {filename}")
            else:
                print("No image to save.")



class CATTLE_IDENTIFICATION(PipelineMixin):
    def load_detector():
        
        detection_model = 'models\\KNP Night December 2025\Base_rtx8000_03_December_2025_20000_v2/model_best.pth'
        model_config = 'models\\KNP Night December 2025\Base_rtx8000_03_December_2025_20000_v2/config.yml'
        
        cfg = get_cfg()
        cfg.merge_from_file(cfg_filename=model_config)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6 #0.6
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3 #0.3
        
        cfg.MODEL.WEIGHTS = detection_model
        
        print("Initializing model")
        return BatchPredictor(cfg, batch_size=1, workers=0)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    def predict_stack_cattle_id(self, images, threshold=0.7):
        """
        Predict cattle IDs from a stack of images.
        
        Args:
            images: List of PIL Images or numpy arrays
            threshold: Confidence threshold for predictions
        
        Returns:
            List of [class_name, count] pairs for predictions above threshold
        """
        self.model.eval()

        
        image_tensor = torch.stack(images).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probabilities, 1)

            filtered_classes = [
                self.class_names[str(pred.item())] 
                for pred, prob in zip(preds, max_probs) 
                if prob.item() >= threshold
            ]

            class_counts = Counter(filtered_classes)
            filtered_counts = [[key, count] for key, count in class_counts.items()]

        return filtered_counts

    def predict_single_image(self, image, return_probability=False):
        """
        Predict cattle ID for a single image.
        
        Args:
            image: PIL Image or numpy array
            return_probability: Whether to return the confidence score
        
        Returns:
            predicted class name (and probability if return_probability=True)
        """
        self.model.eval()
        
        processed_image = self.preprocess_image(image)
        image_tensor = processed_image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            max_prob, pred = torch.max(probabilities, 1)

        predicted_class = self.class_mapping[pred.item()]
        
        if return_probability:
            return predicted_class, max_prob.item()
        return predicted_class
    
    def _load_model(classLen, model_path,device):
        
        model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=False, num_classes=classLen)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        
        return model.to(device)

    def _load_model_resnet(classLen, model_path,device):
        model = timm.create_model('resnet101', pretrained=False)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, classLen)
        )
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model.to(device)
    
    def get_files_from_folder(self,path, limitX=20):
        files = os.listdir(path)
        
        largest_files = heapq.nlargest(limitX, files, key=lambda x: int(x.split('.')[0]))  # Assumes numeric filenames
        
        return np.asarray(largest_files)
    
    def get_files_from_folder_scan_sort(self,path, limitX=20):
        with os.scandir(path) as entries:
            files = [entry.name for entry in entries if entry.is_file()]
            
            files.sort(key=lambda x: int(x.split('_')[0]), reverse=True)  # Adjust as needed for your filenames
            
            return files[:limitX]

    def batch_identification(self,save_dir,tracking_id,FORCE_MISSED_DETECTION = False,path = None):
        counter = 20
        save_path = path
        if path is None:
            save_path = f"{save_dir}//{tracking_id}"
        batch_size = self.identification_batch_size
        images = self.get_last_X_crops(tracking_id,batch_size)
        stacked_images = []
        if images is None:
            return None
        for image in images:
            
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            input_tensor = self.data_transforms['validation'](frame).unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(self.device)

            stacked_images.append(input_tensor[0])  # Append the first element to keep it as (C, H, W)
        
        return self.infer.predict_stack(stacked_images,0.3)
    def calculate_center_of_box(self,box):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        cx = int(x1) + int((x2-x1)/2)
        cy = int(y1) + int((y2-y1)/2)
        print(f"cx : {cx}, cy : {cy}")


    def draw_circle(self,cv2, center_coordinate, image, color):
        cv2.circle(image, center_coordinate, radius=3, color=color, thickness=-1)
    
    def get_color(self,tracking_id):
        try:
            idx = self.STORED_IDS.index(int(tracking_id))
            if idx<=4:
                color = self.colors[idx-1]
            else:
                idx = idx * 3
                color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
            return color
        except:
            return (0, 255, 0)  # Default color if tracking_id not found
    def draw_bounding_box(self,image, box, label,is_recheck_periods,tracking_id,font_scale = 2,color =(0, 255, 0), draw_tracking=True):
        x1, y1, x2, y2 = box

        if label == 'Identifying' or label == -1:
            label = f'Tracking'
        else:
            label = f'{label}'
        text = label
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        thickness = 2

        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        center_x = x1 + (x2 - x1) // 2
        center_y = y1 + (y2 - y1) // 2

        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        color = color if not is_recheck_periods else (255, 0,0 )  # Orange for recheck periods
        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        if draw_tracking:
            text = f'{tracking_id}'
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x2 - text_width #right side
            text_y = y1 if y1 >= 20 else y1 + 10 + text_height #same height 
            
            cv2.rectangle(image, (text_x, text_y - text_height - 10), (text_x + text_width, text_y), color, -1)

            cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    def overlay(self,image, mask, color, alpha, resize=None):
        """Combines image and its segmentation mask into a single image.
        
        Params:
            image: Training image. np.ndarray,
            mask: Segmentation mask. np.ndarray,
            color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
            alpha: Segmentation mask's transparency. float = 0.5,
            resize: If provided, both image and its mask are resized before blending them together.
            tuple[int, int] = (1024, 1024))

        Returns:
            image_combined: The combined image. np.ndarray

        """
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        if resize is not None:
            image = cv2.resize(image.transpose(1, 2, 0), resize)
            image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        return image_combined

    def SUMIYOSHI_ROI(self,x1,y1,x2,y2,h,w):
        print (x2-x1,' is cow width', x2, ' - ', x1 )
        if y1 < 250 or y2 > 3600 : 
            print('I am outside of the ROI')
            return False
        if(y2 - y1>2400 or y2-y1<1000): #1400 to 700 Beforee
            print(y2-y1, ' I am too big')
            return False
        
        return True 

    def HOKKAIDO_ROI(self,x1,y1,x2,y2,h,w):
        
        X1=200 #same as NEW_BLACK_X1
        X2=400 #same as NEW_BLACK_X2 # incase of x2 out of bound
        
        Y1_NEW=50#125  #decrease here to extend, increase to shrink 
        Y2_NEW=630  #5
        default = 640
        X1
        if(x1<int(X1*(w/default)) or x2>int(X2*(w/default)) or y1<int(Y1_NEW*(h/default)) or y2>int(Y2_NEW*(h/default)) or x1>=int(X2*(w/default))):
            return False
        return True  

    def FilterSize(self,y1,y2,h25,area,h75,max_freq_pos):
        print('Filter size y2 and y1 :->',y2,' - ' ,y1 )
        if y1 >= h25*1.5 and y2 <= h75 and area < 80000 : #within specific range  # new camera
            print(y2-y1,' size skipped due to',y2,' - ' ,y1 )
            return False
        return True

    def get_filename_from_file(self,file_path,ext):
        head,filename = os.path.split(file_path)
        filename = filename.split(ext)[0]
        return filename



    def current_datetime(self):
        current_datetime = datetime.now()
        now = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        return str(now)

    def CalculateBoxArea(self,xyxy):
        w = xyxy[2] - xyxy[0] + 1
        h = xyxy[3] - xyxy[1] + 1 
        return w*h

    def CalculateIOU(self,prev,current):
        x1 = max(prev[0],current[0])
        y1 = max(prev[1],current[1])
        x2 = min(prev[2],current[2])
        y2 = min(prev[3],current[3])
        w = x2 - x1
        h = y2 - y1
        if w<=0 or h<=0:
            return 0
        area = max(0,x2-x1 +1) * max(0,y2-y1+1)
        prev_area = self.CalculateBoxArea(prev)
        current_area = self.CalculateBoxArea(current)
        difference = abs(prev_area - current_area)
        prev_area = current_area if prev_area > current_area else prev_area
        
        if prev_area / 4 >= difference : # if size is not too different than previous
            return area/prev_area   
        return 0    
        
        
    
    def IOU_Tracking(self,x1,y1,x2,y2):
        
        is_new = True
        current_id = -1
        global LAST_Y
        is_Moving = True
        if len(self.STORED_IDS)>0:
            LAST_Y = self.STORED_XYXY[-1][1] #x1y1x2y2
        
        dynamic_IOU = self.IOU_TH
        if y1 < self.middle and False:
            if y1<(self.middle/2):
                dynamic_IOU += (1-self.IOU_TH) * ((self.middle-y1) / (self.middle-30))  #dynamic IOU threshold
            else : 
                dynamic_IOU += (1-self.IOU_TH) * ((self.middle-y1) / (self.middle))  #dynamic IOU threshold
            print('dynamic_threshold is :',dynamic_IOU)
        closet = -1
        closet_iou = 0
        for i in range(len(self.STORED_IDS)):
            iou =self.CalculateIOU(self.STORED_XYXY[i],[x1,y1,x2,y2])
            if is_new and iou > dynamic_IOU :    
                is_new = False
                closet = i
                break
                if iou > closet_iou:
                    closet_iou = iou
                    closet = i
        
        if closet>-1 :
            self.STORED_XYXY[closet] = [x1,y1,x2,y2]
            current_id = str(self.STORED_IDS[closet])
            self.TOTAL_MISSED_COUNT += self.STORED_MISS[closet] - 1
            self.STORED_MISS[closet] = 1
            is_Moving = closet_iou < 90
                    
        if is_new:
            self.CATTLE_LOCAL_ID+=1
            self.STORED_IDS.append(self.CATTLE_LOCAL_ID)
            self.STORED_MISS.append(1)
            self.STORED_XYXY.append([x1,y1,x2,y2])
            current_id = str(self.CATTLE_LOCAL_ID)
        
        
        
        return [current_id],is_Moving



    def CALCULATE_EUCLIDEAN_DISTANCE(self,prev_centroid, current_centroid):
        x_sq = (prev_centroid[0] - current_centroid[0]) ** 2
        y_sq = (prev_centroid[1] - current_centroid[1]) ** 2
        distance = ( x_sq + y_sq ) ** 0.5
        return distance
    def Centroid_Tracking(self,mask,w,h,cam_counter,y1,y2):
        
        tracking_missed_frame = 1
        is_new = True
        current_id = -1
        closet = -1
        closet_iou = 0
        threshold = (w/2) if w<=h else (h/2)
        threshold = max(threshold, 55)
        current_point = self.find_mask_center(mask,cam_counter)
        least_missed_frame = 100
        closed_distance_new = 1000
        closed_distance_i = -1
        for i, stored_point in enumerate(self.STORED_XYXY):
            distance = self.CALCULATE_EUCLIDEAN_DISTANCE(stored_point, current_point)    
            if distance <= threshold :    
                if distance < threshold/2: #distance is very close
                    is_new = False
                    closet = i    
                    
                    break
                elif distance > closet_iou:
                    closet_iou = distance
                    closet = i
                    
        
        if closet>-1 :
            
            current_id = str(self.STORED_IDS[closet])
            self.TOTAL_MISSED_COUNT += self.STORED_MISS[closet] - 1
            tracking_missed_frame = self.STORED_MISS[closet]-1
            self.STORED_XYXY[closet] = current_point
            self.STORED_MISS[closet] = 1
            is_new = False

        return [current_id,current_point,tracking_missed_frame,is_new]

    def Centroid_Tracking_PerCamera(self,mask,w,h,cam_counter,y1,y2,frame):
        
        tracking_missed_frame = 1
        is_new = True
        current_id = -1
        
        closet = -1
        closet_iou = 0
        threshold = (w/2) if w<=h else (h/2)
        threshold = threshold  # +30 is original
        current_point = self.find_mask_center(mask,cam_counter)
        least_missed_frame = 100
        if cam_counter in self.CAM_TRACKER:

            for tid,data in self.CAM_TRACKER[cam_counter].items():
                distance = self.CALCULATE_EUCLIDEAN_DISTANCE(data["tracking_xyxy"],current_point)
                
                if is_new and distance <= threshold :    
            
                    if distance < threshold/2: #distance is very close
                        is_new = False
                        closet = tid
                        
                        break
                    elif distance > closet_iou:
                        closet_iou = distance
                        closet = tid

            if closet!=-1 :
                
                current_id = str(closet)
                self.CAM_TRACKER[cam_counter][closet]['tracking_xyxy'] = current_point
                self.TOTAL_MISSED_COUNT = self.CAM_TRACKER[cam_counter][closet]['tracking_missed_frame'] - 1
                self.CAM_TRACKER[cam_counter][closet]['tracking_missed_frame'] = 1
                is_new = False

                is_touching_bottom = self.is_touching_bottom(y2,cam_counter)
                is_touching_top = self.is_touching_top(y1,frame)
                self.update_tracking_mask_location(current_id,mask,cam_counter,current_point,is_touching_top,is_touching_bottom)
            
        else:
            self.CAM_TRACKER[cam_counter] = {}     
        
        return [current_id,current_point,tracking_missed_frame,is_new]
        
    
    def Centroid_Tracking_v2(self,mask,w,h,cam_counter):
        
        tracking_missed_frame = 1
        is_new = True
        current_id = -1
        if len(self.STORED_IDS)>0:
            LAST_Y = self.STORED_CENTROID[-1][1] #x1y1x2y2
        closet = -1
        closet_iou = 0
        threshold = (w/2) if w<=h else (h/2)
        current_point = self.find_mask_center(mask,cam_counter)
        least_missed_frame = 100
        for i in range(len(self.STORED_IDS)):
            distance = self.CALCULATE_EUCLIDEAN_DISTANCE(self.STORED_CENTROID[i],current_point)
            
            if is_new and distance <= threshold :    
                if distance < threshold/2: #distance is very close
                    is_new = False
                    closet = i    
                    
                    break
                elif distance > closet_iou:
                    closet_iou = distance
                    closet = i

        
        if closet>-1 :
            self.STORED_CENTROID[closet] = current_point
            current_id = str(self.STORED_IDS[closet])
            self.TOTAL_MISSED_COUNT += self.STORED_MISS[closet] - 1
            tracking_missed_frame = self.STORED_MISS[closet]-1
            self.STORED_MISS[closet] = 1
            is_new = False
        

   
            
        return [current_id,current_point,tracking_missed_frame,is_new]
    def addNewTracking_v2(self,mask,current_point,x1,y1,x2,y2,cam_counter,w,h):
            


        
        
        self.CATTLE_LOCAL_ID+=1
        self.STORED_IDS.append(self.CATTLE_LOCAL_ID)
        self.STORED_MISS.append(1)
        self.STORED_XYXY.append(current_point)
        current_id = str(self.CATTLE_LOCAL_ID)
        self.STORED_SIZE.append(w*h)

        return current_id

    def addNewTracking_PerCamera(self,mask,current_point,x1,y1,x2,y2,cam_counter,w,h):
        self.CATTLE_LOCAL_ID+=1
        track_info = {
               "tracking_xyxy" : current_point,
               "tracking_missed_frame" : 1}
        self.CAM_TRACKER[cam_counter][self.CATTLE_LOCAL_ID] = track_info
        current_id = str(self.CATTLE_LOCAL_ID)
        is_touching_bottom = self.is_touching_bottom(y2,cam_counter)
        is_touching_top = self.is_touching_top(y1)
        self.update_tracking_mask_location(current_id,mask,cam_counter,current_point,is_touching_top,is_touching_bottom)
        return current_id
    def switchTrackingId(self,old_id,new_id):
        for m in self.TRACKING_MASK_LOCATION:
            if m.tracking_id == old_id:
                m.tracking_id = new_id
                break

    def addNewTracking(self,current_point,w,h):
        self.CATTLE_LOCAL_ID+=1
        self.STORED_IDS.append(self.CATTLE_LOCAL_ID)
        self.STORED_MISS.append(1)
        self.STORED_XYXY.append(current_point)
        current_id = str(self.CATTLE_LOCAL_ID)
        self.STORED_SIZE.append(w*h)
        return current_id

    def get_lowest_y(self,mask):
        Ys, Xs = np.where(mask)
        if Ys.size == 0:
            return None
        max_y = np.max(Ys)-5
        xs_at_max_y = Xs[Ys == max_y ]
        return {
            "y": max_y,                # lowest y (bottom edge)
            "x1": np.min(xs_at_max_y),  # leftmost x at max_y
            "x2": np.max(xs_at_max_y)   # rightmost x at max_y
        }
        
    def get_highest_y(self,mask):
        Ys, Xs = np.where(mask)
        if Ys.size == 0:
            return None
        min_y = np.min(Ys) + 5
        xs_at_min_y = Xs[Ys == min_y]
        return {
            "y": min_y,                # highest y (top edge)
            "x1": np.min(xs_at_min_y),  # leftmost x at min_y
            "x2": np.max(xs_at_min_y)   # rightmost x at min_y
        }
        
    def is_touching_bottom(self,y2,cam_counter, threshold=10):

        image_height = self.camera_heights[cam_counter]
        return abs(y2 - (image_height - 1)) <= threshold

    def is_touching_top(self,y1,frame, threshold=10):
        return y1 <= threshold
    

    def update_tracking_mask_location(self, tracking_id, mask, cam_counter,current_point, is_touching_top, is_touching_bottom):
        
        for m in self.TRACKING_MASK_LOCATION:
            if m.tracking_id == tracking_id:
                m.is_touching_bottom = is_touching_bottom

                m.cam_counter = cam_counter
                m.mask = mask
                m.is_touching_top = is_touching_top
                m.current_point = current_point
                m.last_mask_area = np.sum(mask)
                if (not is_touching_top and not is_touching_bottom and m.mask_area < m.last_mask_area) or m.mask_area < m.last_mask_area:# or m.mask_area is null :
                    m.mask_area = m.last_mask_area

            
    
    def isOverlappedXs_original(self,first, second, threshold = 0.1):
        first_x1 = first[0]
        first_x2 = first[1]
        second_x1 = second[0]
        second_x2 = second[1]
        min_length = min((first_x2 - first_x1), (second_x2 - second_x1))
        percentage = 0
        threshold = 0
        if first_x1 < second_x1 and first_x2 > second_x2: #second is inside first
            return True,1
        elif first_x1 > second_x1 and first_x2 < second_x2 : #first is inside second
            return True,1
        elif first_x1 < second_x1 and first_x2 > second_x1: 
            percentage = self.overLappedPercentage(min_length, abs(first_x2 - second_x1))
            if percentage > threshold:
                return True,percentage
            return False, percentage
        elif second_x1 < first_x1 and second_x2 > first_x1 :
            percentage = self.overLappedPercentage(min_length, abs(second_x2 - first_x1))
            if percentage > threshold:
                return True, percentage
            return False,percentage
        return False,percentage
    def isOverlappedXs(self, first, second, threshold=0.1):
        first_x1, first_x2 = first
        second_x1, second_x2 = second

        overlap_start = max(first_x1, second_x1)
        overlap_end = min(first_x2, second_x2)
        overlap_length = max(0, overlap_end - overlap_start)

        if overlap_length <= 0:
            return False, 0

        min_length = min(first_x2 - first_x1, second_x2 - second_x1)
        percentage = self.overLappedPercentage(min_length, overlap_length)

        if percentage > threshold:
            return True, percentage
        return False, percentage

    def overLappedPercentage(self, length1, length2):
        return length2 / length1 if length1 > 0 else 0

    def overLappedPercentage_original(self,length1, length2):
        
        percentage = 0
        if length1 > length2:
            percentage = length2 / length1
        else:
            percentage = length1 / length2
        return percentage

    def Is_Similar(self,mask,tracking_id,x1,y1,x2,y2,frame):
        rgb_mask = np.zeros_like(frame)
        rgb_mask[mask] = frame[mask]
        box_h = max(y2-y1,x2-x1)
        crop = rgb_mask[y1:y2, x1:x2] # take crop
        crop = self.resize_image(crop,(box_h,box_h)) #then cattle image to square box
        crop = cv2.resize(crop, (self.SIZE_224,self.SIZE_224)) #resize to
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        
        images = self.get_last_X_crops(tracking_id,20)
        
        if images is None:
            return None
        images = images[5:15] # 10 images from 5th to 15th
        match_path, score = self.similarity_checker.cosine_similarity_search(images, pil_img)
        if score > 0.7:
            print(f"✅ Match with score {score:.2f}")
            return True
        else:
            print(f"❌ Not similar with score {score:.2f}") #print(f"Deleting tracking {self.STORED_IDS[i-removed]} due to missed count {missed}")
            return False
    
    def align_centroid_per_camera(self,current_point,cam_counter):
        print(current_point)
        x,y = current_point
        y += (self.image_height * cam_counter) #adjust for multi cam
        
        return (x,y)

    def find_cattle_location(self,x1,x2):
        if x1 < 150:
            return "LEFT"
        elif x2 > 900:
            return "RIGHT"
        else:
            return "MIDDLE"

    def find_matching_mask(self, current_mask,current_point, cam_counter,tracking_id,x1,y1,x2,y2,frame, threshold=0.1):
        closed_tracking = tracking_id
        is_touching_bottom = self.is_touching_bottom(y2, cam_counter)
        is_touching_top = self.is_touching_top(y1, frame)
        return tracking_id,is_touching_bottom,is_touching_top #single cam no need to0 check
        distance_threshold = 100
        check_camera = cam_counter + (0 if is_touching_bottom else - 1)
        check_camera = max(check_camera,0)
        is_overlapped = False
        cattle_position = self.find_cattle_location(x1,x2)
        mask_area_threshold = 35000
        if is_touching_bottom:
            
            target_counter = cam_counter + 1
            line_info = self.get_lowest_y(current_mask)
            if line_info is None:
                return closed_tracking,is_touching_bottom,is_touching_top

            for m in self.TRACKING_MASK_LOCATION:
                if m.cam_counter == target_counter and m.is_touching_top:
                    other_line = self.get_highest_y(m.mask)
                    if other_line is None:
                        continue
                    isOverlapped, percentage = self.isOverlappedXs([line_info["x1"],line_info["x2"]], [other_line["x1"],other_line["x2"]],threshold)
                    distance = 1000
                    if other_line is not None and isOverlapped:
                        centroid = m.current_point
                        if percentage > 0.3:
                            mask_area = np.sum(current_mask)
                            
                            total_area = mask_area + m.last_mask_area
                            
                            if total_area > max( mask_area_threshold, 1.3 * (m.mask_area)):
                                print(f"{m.tracking_id} : last_mask : {m.last_mask_area} , m.mask_area : {m.mask_area}")
                                continue
                            else:
                                return m.tracking_id,is_touching_bottom,is_touching_top

        elif is_touching_top:
            target_counter = cam_counter - 1
            if target_counter < 0:
                return closed_tracking,is_touching_bottom,is_touching_top

            line_info = self.get_highest_y(current_mask)
            if line_info is None:
                print("No line found in current mask")
                return closed_tracking,is_touching_bottom,is_touching_top
            for m in self.TRACKING_MASK_LOCATION:
                if m.cam_counter == target_counter and m.is_touching_bottom:
                    other_line = self.get_lowest_y(m.mask)
                    if other_line is None:
                        continue
                    isOverlapped, percentage = self.isOverlappedXs([line_info["x1"],line_info["x2"]], [other_line["x1"],other_line["x2"]],threshold)
                    distance = 1000
                    if other_line is not None and isOverlapped:
                        if percentage > 0.3:
                            mask_area = np.sum(current_mask)
                            
                            total_area = mask_area + m.last_mask_area
                            
                            if total_area > max( mask_area_threshold, 1.3 * (m.mask_area)):
                                print(f"{m.tracking_id} : last_mask : {m.last_mask_area} , m.mask_area : {m.mask_area}")
                                continue
                            else:
                                return m.tracking_id,is_touching_bottom,is_touching_top
        return closed_tracking,is_touching_bottom,is_touching_top
    
    def delete_mask_by_tracking_id(self, ids):
        deleted_ids = []
        for id in ids:
            self.delete_last_20_path(id)
        for m in self.TRACKING_MASK_LOCATION:
            if m.tracking_id in ids:
                deleted_ids.append(m.tracking_id)
                print(f"Deleting mask with tracking ID: {m.tracking_id}")
                self.TRACKING_MASK_LOCATION.remove(m)


    def IncreaseMissedCount(self,tracking_ids):
        ids = []
        if(len(self.STORED_IDS)>0): 
            total_length = len(self.STORED_IDS)
        
            removed = 0
            threshold = 30
            for i in range(total_length):
                if self.STORED_IDS[i-removed] not in tracking_ids: #didn't detected in this frame
                    self.STORED_MISS[i-removed]+=1
                    missed = self.STORED_MISS[i-removed]
                    
                    if missed>threshold: #if missed 3 frames
                        ids.append(self.STORED_IDS[i-removed])
                        del self.STORED_MISS[i-removed]  
                        del self.STORED_XYXY[i-removed]
                        del self.STORED_SIZE[i-removed]
                        del self.STORED_IDS[i-removed]
                        removed+=1
        if len(ids)>0:
            self.delete_mask_by_tracking_id(ids)

                        
    def GetNewTrackingIDForTrackingConflict(self,tracking_id,current_point,w,h):
        
        try:
            index = self.STORED_IDS.index(tracking_id)
            del self.STORED_MISS[index]  
            del self.STORED_XYXY[index]
            del self.STORED_SIZE[index]
            del self.STORED_IDS[index]
            
        
        except:
            print(f"Error: Tracking ID {tracking_id} not found in stored IDs.")
        new_tracking_id =  self.addNewTracking(current_point,w,h)
        self.switchTrackingId(tracking_id,new_tracking_id)
        return new_tracking_id

    def resize_image(self,input_array,new_size):
        original_image = Image.fromarray(input_array)

        new_image = Image.new("RGB", new_size, (0, 0, 0))

        x_offset = (new_size[0] - original_image.width) // 2
        y_offset = (new_size[1] - original_image.height) // 2

        new_image.paste(original_image, (x_offset, y_offset))
        return np.array(new_image)

        
        


        

    def save_crop(self, frame, mask, x1, y1, x2, y2, save_dir, prev_id,
              is_touching, touching_pixels, 
              frame_count,
              is_save=True, is_small=False,
              predicted_id=-1, area=1000, IS_RECHECK_PERIOD=False, is_outside_gap=False):

        rgb_mask = np.zeros_like(frame)
        rgb_mask[mask] = frame[mask]
        box_h = max(y2 - y1, x2 - x1)
        crop = rgb_mask[y1:y2, x1:x2]  # take crop
        crop = self.resize_image(crop, (box_h, box_h))  # square box
        crop = cv2.resize(crop, (self.SIZE_224, self.SIZE_224))  # resize to final size
        img_file_name = None
        if touching_pixels:
            self.print_border_pixels(mask, (x1, y1, x2, y2))

        if is_save and not is_small and not is_touching:
            base_path = str(Path(f'{save_dir}/{prev_id[0]}'))
            recheck_txt = 'RC_TRUE' if IS_RECHECK_PERIOD else 'RC_FALSE'
            img_file_name = f'{self.image_count}_f_{frame_count}_{predicted_id}_GAP_{"T" if is_outside_gap else "F"}_T_{touching_pixels}_A_{area}_{recheck_txt}.jpg'
            demo_annotated_img_save_path = Path(
                base_path + '/' + img_file_name
            )

            os.makedirs(base_path, exist_ok=True)
            cv2.imwrite(str(demo_annotated_img_save_path), crop)


            self.add_last_20_crops(prev_id[0], crop)
            self.image_count += 1
        return img_file_name
    def add_last_20_paths(self,tracking_id,save_path):

        if tracking_id in self.LAST_20_PATH:
            self.LAST_20_PATH[tracking_id].append(save_path)
            if len(self.LAST_20_PATH[tracking_id])>20:
                self.LAST_20_PATH[tracking_id].pop(0)  

        else:
            self.LAST_20_PATH[tracking_id]= [save_path]
    def get_last_X_paths(self,tracking_id,take = 20):
        if type(tracking_id) == int :
            
            tracking_id = str(tracking_id)
        if tracking_id not in self.LAST_20_PATH:
            print('No images ??? really???')
            return None
        length = len(self.LAST_20_PATH[tracking_id])
        start_from = length - take if length > take else 0
        response = self.LAST_20_PATH[tracking_id][start_from:]
        return response
    def add_last_20_crops(self,tracking_id,crop):

        if tracking_id in self.LAST_20_PATH:
            self.LAST_20_PATH[tracking_id].append(crop)
            if len(self.LAST_20_PATH[tracking_id])>20:
                self.LAST_20_PATH[tracking_id].pop(0)  

        else:
            self.LAST_20_PATH[tracking_id]= [crop]
    
    def get_last_X_crops(self,tracking_id,take = 20):
        if type(tracking_id) == int :
            
            tracking_id = str(tracking_id)
        if tracking_id not in self.LAST_20_PATH:
            print('No images ??? really???')
            return None
        length = len(self.LAST_20_PATH[tracking_id])
        start_from = length - take if length > take else 0
        response = self.LAST_20_PATH[tracking_id][start_from:]
        return response
    
    def delete_last_20_path(self,tracking_id):
        if tracking_id in self.LAST_20_PATH:
            del self.LAST_20_PATH[tracking_id]
    def draw_mask_multiple_masks(self,colored_mask,masks,indexes):
        for index in indexes:
            colored_mask[masks[index].astype(bool)] = (0, 255, 255)  # red color mask
        return colored_mask

    def diff_Time(self,start,end,process):
        return
        
    def roi_no_crop(self,scale):
        w, h, c_x, c_y = 1096, 1096, 548, 548
        x1, y1 = int(c_x - (scale * w / 2.0)), int(c_y - (0.75 * h / 2.0))
        x2, y2 = int(x1 + (scale * w)), int(y1 + (0.75 * h))

        mask_roi = cv2.cvtColor(np.zeros((1096, 1096, 1), np.uint8) * 255, cv2.COLOR_GRAY2BGR)

        mask_roi = cv2.rectangle(mask_roi, (x1, y1), (x2, y2), (255, 255, 255), -1)

        return mask_roi

    def draw_mask(self,colored_mask,mask):
        boolean_mask = mask.astype(bool)
        colored_mask[boolean_mask] = (0, 255, 255)  # red color mask
        return colored_mask
    
    def reset_duplicate_tracking_identification(self,tracking_ids):
        default = self.Get_TrackingBatch_Default(isReset=True)
        for tracking_id in tracking_ids:
            track_index = -1
            if tracking_id in self.TRACKER: #already have tracking record
                track_index = self.TRACKER.index(tracking_id)
            
            
            
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] = 0
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = 'Identifying'
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['STANDING_IDENTIFICATION'] = False 
                self.CLASSIFICATION_TRACKER[track_index] = default
            
    def UpdateTrackingIDAndBatchInfo(self,old_tracking_id,location,predicted_id,w,h,IS_RECHECK_PERIOD):
        tracking_id = self.GetNewTrackingIDForTrackingConflict(old_tracking_id,location,w,h)
        if type(tracking_id) == 'str':
            tracking_id = int(tracking_id)
        default = {'GT': [predicted_id], 'COUNT':[self.batch_size+1], 'IS_STABLE':False,  'HAS_MISSED_FRAME' : False, 'REIDENTIFY_MISSED_COUNT' : self.REIDENTIFY_MISSED_COUNT, 'RECHECK_COUNT':0 }
        self.TRACKER.append(tracking_id) #add new tracking record
        track_index = len(self.TRACKER) - 1
        batch_default = self.Get_BatchDefault_Dict(location, self.batch_size+1, predicted_id)
        self.BATCH_CLASSIFICATION_TRACKER.append(batch_default)
        self.CLASSIFICATION_TRACKER.append(default)
        self.ALL_TIME_CLASSIFICATION_TRACKER.append(default) #to keep the whole result]
        self.TRACKING_RESULT[tracking_id] = [] #default
        self.TRACKING_RESULT[tracking_id].append(predicted_id)
        return (predicted_id,tracking_id,IS_RECHECK_PERIOD)
    
    def getTwoMaxIndexes(self,array):
        if(len(array) < 1):
            return -1,-1
        if len(array) < 2:
            return 0,-1
        max1 = max(array)
        max1Index = array.index(max1)
        array[max1Index] = 0
        max2 = max(array)
        max2Index = array.index(max2)
        array[max1Index] = max1
        if max1-max2<10:
            return max1Index,max2Index
        return max1Index, -1
    
 
    def Get_TrackingBatch_Default(self,isReset=False):
        
        GT = 'Reidentifying' if isReset else 'Identifying'
        RECHECK_COUNT = 5 if isReset else 0
        default = {'GT': [GT], 'COUNT': [0], 'IS_STABLE': False, 'HAS_MISSED_FRAME': False,
                   'REIDENTIFY_MISSED_COUNT': 0, 'RECHECK_COUNT': RECHECK_COUNT }
        return default
    def Get_BatchDefault_Dict(self,location, total_count = 0, predicted_id = 'Identifying'):
        return {'TOTAL_COUNT': 0, 'HOLDING': predicted_id, 'LOCATION': location, 'TOTAL_MISSED_FRAME': 0,
                         'TOTAL_DETECTION': 0, 'PREVIOUS_MAX_PREDICTED_ID': None, 'STANDING_IDENTIFICATION': False}
    
    def need_identification_due_to_standing(self, track_index, is_outside_gap, tracking_id):
        if is_outside_gap and not self.BATCH_CLASSIFICATION_TRACKER[track_index]['STANDING_IDENTIFICATION'] and \
           not self.BATCH_CLASSIFICATION_TRACKER[track_index]['PREVIOUS_MAX_PREDICTED_ID'] in [None,'Identifying','Reidentifying','unknown']: #if never of standing then do it
            self.BATCH_CLASSIFICATION_TRACKER[track_index]['STANDING_IDENTIFICATION'] = True
            print(f"Tracking ID {tracking_id} is outside gap area, marking as standing identification.")
            return True
        return False
    
    def GetBatchPredictedId(self, tracking_id, is_small, save_dir, missed_frame_count, location, w, h, is_outside_gap):
        batch_default = self.Get_BatchDefault_Dict(location)
        
        track_index = -1
        has_missed_frame = False
        if missed_frame_count > 3:
            has_missed_frame = True
        FORCE_IDENTIFICATION = False
        IS_RECHECK_PERIOD = False
        FORCE_MISSED_IDENTIFICATION = False
        HAS_TRACKING_CONFLICT = False

        IS_STABLE = False
        

        if tracking_id in self.TRACKER:  # already have tracking record

            track_index = self.TRACKER.index(tracking_id)

        else:
            self.TRACKER.append(tracking_id)  # add new tracking record
            track_index = len(self.TRACKER) - 1
            tracking_default = self.Get_TrackingBatch_Default()
            self.BATCH_CLASSIFICATION_TRACKER.append(batch_default)
            self.CLASSIFICATION_TRACKER.append(tracking_default)
            self.ALL_TIME_CLASSIFICATION_TRACKER.append(tracking_default)  # to keep the whole result

            self.TRACKING_RESULT[tracking_id] = []  # 

        
        self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_MISSED_FRAME'] += missed_frame_count
        self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_DETECTION'] += 1

        need_standing_identification = self.need_identification_due_to_standing(track_index, is_outside_gap, tracking_id)

#        total_predictions = 0
        total_predictions = self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] 
        if not is_small:
            self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] += 1
            
            if self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME']  or self.CLASSIFICATION_TRACKER[track_index]['RECHECK_COUNT'] > 0:  # keep searching if small
                if self.CLASSIFICATION_TRACKER[track_index]['RECHECK_COUNT'] > 0:
                    self.CLASSIFICATION_TRACKER[track_index]['RECHECK_COUNT'] -= 1  # reduce by one
                    IS_RECHECK_PERIOD = True
                

                if self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME']:
                    self.CLASSIFICATION_TRACKER[track_index]['REIDENTIFY_MISSED_COUNT'] -= 1  # reduce by one
                
                FORCE_MISSED_IDENTIFICATION = self.CLASSIFICATION_TRACKER[track_index]['REIDENTIFY_MISSED_COUNT'] < 1 or self.CLASSIFICATION_TRACKER[track_index]['RECHECK_COUNT'] < 1
                if FORCE_MISSED_IDENTIFICATION:
                    self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME'] = False  # identify in 10
                    self.CLASSIFICATION_TRACKER[track_index]['RECHECK_COUNT'] = 0
        if IS_RECHECK_PERIOD and self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME']:
            self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME'] = False  #remove one if having both
            
        if has_missed_frame and total_predictions > self.batch_size and self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] != 'Identifying':

            self.CLASSIFICATION_TRACKER[track_index][
                'REIDENTIFY_MISSED_COUNT'] = self.REIDENTIFY_MISSED_COUNT  # re identify in 10
            self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME'] = True  # do force missed detection
            self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = 'Identifying'
            self.CLASSIFICATION_TRACKER[track_index]['IS_STABLE'] = False  # if missed then not stable


        #IS_STABLE = self.CLASSIFICATION_TRACKER[track_index]['IS_STABLE']
        if FORCE_MISSED_IDENTIFICATION:
            print("Force missed detection now")
        # distance = self.CALCULATE_EUCLIDEAN_DISTANCE(location, self.BATCH_CLASSIFICATION_TRACKER[track_index][
        #             'LOCATION'])  # compare two distance
        is_moving = need_standing_identification #distance > 100 


        
        if (total_predictions > self.total_predictions
            and not self.CLASSIFICATION_TRACKER[track_index][
            'HAS_MISSED_FRAME'] and not FORCE_MISSED_IDENTIFICATION and not IS_RECHECK_PERIOD) or is_moving:
            if  total_predictions % 50 == 0:  # only re calculate every 100th frame
                
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['LOCATION'] = location

            if is_moving :  # NOT RESETTING #total_predictions % 300 == 0 or removed every 300 frames
                IS_RECHECK_PERIOD = True
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['LOCATION'] = location
                self.CLASSIFICATION_TRACKER[track_index]['RECHECK_COUNT'] = self.RECHECK_COUNT
                print("this is recheck period!")
            else:
                if FORCE_MISSED_IDENTIFICATION:
                    print("Damn,, Escaped from force missed identification")
                return self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'], tracking_id, IS_RECHECK_PERIOD


        maxPredictedId = -1
        isDoingIdentification = False
        if total_predictions == self.batch_size or (
                total_predictions > 0 and total_predictions % self.batch_size == 0 and total_predictions <= self.total_predictions) or FORCE_IDENTIFICATION or FORCE_MISSED_IDENTIFICATION:

            if not FORCE_MISSED_IDENTIFICATION and (self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME'] or IS_RECHECK_PERIOD):
                return self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'], tracking_id, IS_RECHECK_PERIOD

            maxCounters = self.batch_identification(save_dir, tracking_id, FORCE_MISSED_IDENTIFICATION)
            previous_max_predicted_id = self.BATCH_CLASSIFICATION_TRACKER[track_index][
                'PREVIOUS_MAX_PREDICTED_ID']  # update previous max predicted ID

            if FORCE_MISSED_IDENTIFICATION:  # force missed identification
                print("Begin force missed identification process, tracking id :",tracking_id)
                if maxCounters is None or len(maxCounters) == 0:
                    maxPredictedId = None
                else:
                    maxPredictedId = max(maxCounters,key=lambda x:x[1])[0]
                if maxPredictedId is None:  # if none keep searching
                    self.BATCH_CLASSIFICATION_TRACKER[track_index][
                        'REIDENTIFY_MISSED_COUNT'] = self.REIDENTIFY_MISSED_COUNT  # re identify in 10
                    self.BATCH_CLASSIFICATION_TRACKER[track_index][
                        'HAS_MISSED_FRAME'] = True  # do force missed detection
                
                elif previous_max_predicted_id != maxPredictedId:  # different cow

                    self.CLASSIFICATION_TRACKER[track_index] = self.Get_TrackingBatch_Default()
                    self.CLASSIFICATION_TRACKER[track_index]['GT'] = [maxPredictedId]
                    self.CLASSIFICATION_TRACKER[track_index]['COUNT'] = [self.identification_batch_size]
                    self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = maxPredictedId
                    self.BATCH_CLASSIFICATION_TRACKER[track_index]['PREVIOUS_MAX_PREDICTED_ID'] = maxPredictedId
                    print("Set up new ID for tracking id :",tracking_id, " to ", maxPredictedId)
                elif previous_max_predicted_id == maxPredictedId:
                    self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = maxPredictedId
                    self.BATCH_CLASSIFICATION_TRACKER[track_index]['PREVIOUS_MAX_PREDICTED_ID'] = maxPredictedId
                    return (maxPredictedId, tracking_id,IS_RECHECK_PERIOD)

            else:
                if maxCounters is not None:
                    for maxCounter in maxCounters:
                        predicted_id, predicted_count = maxCounter
                        if predicted_id not in self.CLASSIFICATION_TRACKER[track_index]['GT']:

                            self.CLASSIFICATION_TRACKER[track_index]['GT'].append(predicted_id)
                            self.CLASSIFICATION_TRACKER[track_index]['COUNT'].append(predicted_count)
                        else:
                            predicted_index = self.CLASSIFICATION_TRACKER[track_index]['GT'].index(predicted_id)
                            self.CLASSIFICATION_TRACKER[track_index]['COUNT'][predicted_index] += predicted_count

                        if predicted_id not in self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['GT']:

                            self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['GT'].append(predicted_id)
                            self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['COUNT'].append(predicted_count)
                        else:
                            predicted_index = self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['GT'].index(predicted_id)
                            self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['COUNT'][predicted_index] += predicted_count

                maxCount = max(self.CLASSIFICATION_TRACKER[track_index]['COUNT'])
                maxCountIndex = self.CLASSIFICATION_TRACKER[track_index]['COUNT'].index(maxCount)
                maxPredictedId = self.CLASSIFICATION_TRACKER[track_index]['GT'][maxCountIndex]
                self.TRACKING_RESULT[tracking_id].append(maxPredictedId)  # add to log
                if FORCE_IDENTIFICATION:  # due to moving
                    self.CLASSIFICATION_TRACKER[track_index]['IS_STABLE'] = previous_max_predicted_id == maxPredictedId

                self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = maxPredictedId
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['PREVIOUS_MAX_PREDICTED_ID'] = maxPredictedId

        
        maxPredictedId = self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING']
        if total_predictions >= 100 and (maxPredictedId == None or maxPredictedId == 'None' or
        maxPredictedId == 'Reidentifying' or maxPredictedId == 'Identifying' or maxPredictedId == 'unknown'): #added unknown
            self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] = 0
            self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = 'Identifying'

        if tracking_id not in self.tracking_by_predicted_id:
            self.tracking_by_predicted_id[maxPredictedId] = tracking_id
        
        if maxPredictedId is None:
            maxPredictedId = 'Reidentifying'
        return (maxPredictedId, tracking_id,IS_RECHECK_PERIOD)
    
    def GetBatchPredictedId_TwoSimilar(self,tracking_id,is_small,save_dir,missed_frame_count,location,w,h):
        batch_default = {'TOTAL_COUNT': 0, 'HOLDING': 'Identifying', 'LOCATION': location, 'TOTAL_MISSED_FRAME': 0,
                         'TOTAL_DETECTION': 0, 'PREVIOUS_MAX_PREDICTED_ID': None}
        track_index = -1
        has_missed_frame = False
        if missed_frame_count > 5:
            has_missed_frame = True
        default = {'GT': [None], 'COUNT': [0], 'IS_STABLE': False, 'HAS_MISSED_FRAME': has_missed_frame,
                   'REIDENTIFY_MISSED_COUNT': self.REIDENTIFY_MISSED_COUNT}
        FORCE_IDENTIFICATION = False
        FORCE_MISSED_IDENTIFICATION = False
        HAS_TRACKING_CONFLICT = False

        IS_STABLE = False

        if tracking_id in self.TRACKER:  # already have tracking record

            track_index = self.TRACKER.index(tracking_id)

        else:
            self.TRACKER.append(tracking_id)  # add new tracking record
            track_index = len(self.TRACKER) - 1

            self.BATCH_CLASSIFICATION_TRACKER.append(batch_default)
            self.CLASSIFICATION_TRACKER.append(default)
            self.ALL_TIME_CLASSIFICATION_TRACKER.append(default)  # to keep the whole result

            self.TRACKING_RESULT[tracking_id] = []  # default

        
        self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_MISSED_FRAME'] += missed_frame_count
        self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_DETECTION'] += 1
        total_predictions = 0
        if not is_small:
            self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] += 1
            total_predictions = self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] 
            if self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME'] == True:  # keep searching if small

                self.CLASSIFICATION_TRACKER[track_index]['REIDENTIFY_MISSED_COUNT'] -= 1  # reduce by one
                FORCE_MISSED_IDENTIFICATION = self.CLASSIFICATION_TRACKER[track_index]['REIDENTIFY_MISSED_COUNT'] < 1
                if FORCE_MISSED_IDENTIFICATION:
                    print(f"Reach limit and will do identification for missed frame of {tracking_id}")
                    self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME'] = False  # identify in 10
        
        if has_missed_frame and total_predictions > self.batch_size and self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] != 'Identifying':
            print(f"FORCE MISSED DETECTION FOR TRACKING {tracking_id}")

            self.CLASSIFICATION_TRACKER[track_index][
                'REIDENTIFY_MISSED_COUNT'] = self.REIDENTIFY_MISSED_COUNT  # re identify in 10
            self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME'] = True  # do force missed detection
            self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = 'Identifying'
            self.CLASSIFICATION_TRACKER[track_index]['IS_STABLE'] = False  # if missed then not stable


        IS_STABLE = self.CLASSIFICATION_TRACKER[track_index]['IS_STABLE']
        if IS_STABLE:
            maxPredictedId = self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING']
            return maxPredictedId, tracking_id

        if total_predictions > 300 and not IS_STABLE and not self.CLASSIFICATION_TRACKER[track_index][
            'HAS_MISSED_FRAME'] and not FORCE_MISSED_IDENTIFICATION:  # and if not stable
            is_moving = False
            if total_predictions % 100 == 0:  # only re calculate every 100th frame
                distance = self.CALCULATE_EUCLIDEAN_DISTANCE(location, self.BATCH_CLASSIFICATION_TRACKER[track_index][
                    'LOCATION'])  # compare two distance

                is_moving = distance > 100
                if is_moving:
                    print(self.BATCH_CLASSIFICATION_TRACKER[track_index]['LOCATION'])
                    print(location)
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['LOCATION'] = location

            if is_moving:  # or total_predictions %1000 == 0: #every 1000th frame
                print(f're identifying tracking {tracking_id} due to moving')
                FORCE_IDENTIFICATION = True
            elif total_predictions == 1000 and False:  # NOT RESETTING
                print(f're identifying tracking {tracking_id} due to 1000th images')
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] = 0
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = 'Identifying'
                self.CLASSIFICATION_TRACKER[track_index] = default
                FORCE_IDENTIFICATION = True
            else:
                return self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'], tracking_id


        maxPredictedId = -1
        if total_predictions == self.batch_size or (
                total_predictions > 0 and total_predictions % self.batch_size == 0) or FORCE_IDENTIFICATION or FORCE_MISSED_IDENTIFICATION:

            if self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME']:
                return self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'], tracking_id

            maxCounters = self.batch_identification(save_dir, tracking_id, FORCE_MISSED_IDENTIFICATION)
            previous_max_predicted_id = self.BATCH_CLASSIFICATION_TRACKER[track_index][
                'PREVIOUS_MAX_PREDICTED_ID']  # update previous max predicted ID

            if FORCE_MISSED_IDENTIFICATION:  # force missed identification
                print(f"counter {maxCounters}, previous_max_predicted_id : {previous_max_predicted_id}")

                if len(maxCounters) == 0:
                    maxPredictedId = None
                else:
                    maxPredictedId = maxCounters[0][0]
                print(
                    f"FORCE_MISSED_IDENTIFICATION RESULT: {maxPredictedId}, ' PREVIOUS_MAX_PREDICTED_ID : {previous_max_predicted_id}")
                if maxPredictedId is None:  # if none keep searching
                    print(f"FORCE MISSED DETECTION FOR TRACKING {tracking_id}")
                    self.BATCH_CLASSIFICATION_TRACKER[track_index][
                        'REIDENTIFY_MISSED_COUNT'] = self.REIDENTIFY_MISSED_COUNT  # re identify in 10
                    self.BATCH_CLASSIFICATION_TRACKER[track_index][
                        'HAS_MISSED_FRAME'] = True  # do force missed detection
                elif previous_max_predicted_id != maxPredictedId:  # different cow

                    self.TRACKING_RESULT[tracking_id].append(maxPredictedId)
                    return self.UpdateTrackingIDAndBatchInfo(tracking_id, location, maxPredictedId, w, h)
                elif previous_max_predicted_id == maxPredictedId:
                    self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = maxPredictedId
                    self.BATCH_CLASSIFICATION_TRACKER[track_index]['PREVIOUS_MAX_PREDICTED_ID'] = maxPredictedId
                    return (maxPredictedId, tracking_id)

            else:
                for maxCounter in maxCounters:
                    predicted_id, predicted_count = maxCounter
                    if predicted_id not in self.CLASSIFICATION_TRACKER[track_index]['GT']:

                        self.CLASSIFICATION_TRACKER[track_index]['GT'].append(predicted_id)
                        self.CLASSIFICATION_TRACKER[track_index]['COUNT'].append(predicted_count)
                    else:
                        predicted_index = self.CLASSIFICATION_TRACKER[track_index]['GT'].index(predicted_id)
                        self.CLASSIFICATION_TRACKER[track_index]['COUNT'][predicted_index] += predicted_count

                    if predicted_id not in self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['GT']:

                        self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['GT'].append(predicted_id)
                        self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['COUNT'].append(predicted_count)
                    else:
                        predicted_index = self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['GT'].index(predicted_id)
                        self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['COUNT'][predicted_index] += predicted_count

                maxCount = max(self.CLASSIFICATION_TRACKER[track_index]['COUNT'])
                maxCountIndex = self.CLASSIFICATION_TRACKER[track_index]['COUNT'].index(maxCount)
                maxPredictedId = self.CLASSIFICATION_TRACKER[track_index]['GT'][maxCountIndex]
                self.TRACKING_RESULT[tracking_id].append(maxPredictedId)  # add to log
                

                self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = maxPredictedId
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['PREVIOUS_MAX_PREDICTED_ID'] = maxPredictedId

        maxPredictedId = self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING']
        
        if maxPredictedId is None:
            maxPredictedId = 'Reidentifying'
        return (maxPredictedId, tracking_id)

    def GetPredictedIDFromTracking(self,tracking_id,predicted_id,is_small):
       
        track_index = -1
        if tracking_id in self.TRACKER: #already have tracking record
            track_index = self.TRACKER.index(tracking_id)
        else:
            self.TRACKER.append(tracking_id) #add new tracking record
            track_index = len(self.TRACKER) - 1
            default = {'GT': [predicted_id], 'COUNT':[0] , 'HOLDING' : predicted_id,'IS_PREDICTED' : False}
            self.CLASSIFICATION_TRACKER.append(default)
        total_predictions = sum(self.CLASSIFICATION_TRACKER[track_index]['COUNT']) 
        
        if not is_small:
            
            if predicted_id not in self.CLASSIFICATION_TRACKER[track_index]['GT']:
                self.CLASSIFICATION_TRACKER[track_index]['GT'].append(predicted_id)
                self.CLASSIFICATION_TRACKER[track_index]['COUNT'].append(1)
            else:
                predicted_index = self.CLASSIFICATION_TRACKER[track_index]['GT'].index(predicted_id)
                self.CLASSIFICATION_TRACKER[track_index]['COUNT'][predicted_index] +=1
            
        maxPredictedId = -1
        if total_predictions < 10:
            return -1
        if total_predictions == 10 or total_predictions % 10 ==0:
            maxCount = max(self.CLASSIFICATION_TRACKER[track_index]['COUNT'])
            maxCountIndex = self.CLASSIFICATION_TRACKER[track_index]['COUNT'].index(maxCount)
            maxPredictedId = self.CLASSIFICATION_TRACKER[track_index]['GT'][maxCountIndex]
            self.CLASSIFICATION_TRACKER[track_index]['HOLDING'] = maxPredictedId
            self.CLASSIFICATION_TRACKER[track_index]['IS_PREDICTED'] = True
            
            
        
        return f'{maxPredictedId}'
            
    def sortBoxAndMask(
    self,
    boxes,
    masks,
    is_last_cam: bool = False,
    y2_threshold=None,           # None = no filter. If 0<val<=1 → ratio of image height; else absolute pixels
    ):
        """
        boxes: np.ndarray of shape [N, 4] in (x1,y1,x2,y2)
        masks: sequence/array of N binary masks [H,W] (or bool np arrays)
        Returns:
            sorted_boxes_with_area: np.ndarray [N_kept, 5] (x1,y1,x2,y2,area) sorted by area desc
            sorted_masks: list of N_kept masks (largest connected component kept)
        """

        if boxes is None or len(boxes) == 0:
            return np.empty((0, 5), dtype=float), []
        boxes = np.asarray(boxes)

        widths  = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas   = widths * heights
        boxes_with_areas = np.hstack([boxes, areas.reshape(-1, 1)])

        if is_last_cam and y2_threshold is not None:
            img_h = None
            if hasattr(masks, "shape") and len(getattr(masks, "shape", [])) >= 3:
                img_h = masks.shape[-2]
            elif isinstance(masks, (list, tuple)) and len(masks) > 0 and hasattr(masks[0], "shape"):
                img_h = masks[0].shape[-2]

            if 0 < y2_threshold <= 1 and img_h is not None:
                thr = y2_threshold * img_h
            else:
                thr = float(y2_threshold)

            keep_idx = boxes_with_areas[:, 3] <= thr
            boxes_with_areas = boxes_with_areas[keep_idx]
            if isinstance(masks, (list, tuple)):
                masks = [m for k, m in zip(keep_idx, masks) if k]
            else:
                masks = masks[keep_idx]

        if boxes_with_areas.shape[0] == 0:
            return np.empty((0, 5), dtype=float), []

        sorted_indices = np.argsort(boxes_with_areas[:, -1])[::-1]
        sorted_boxes = boxes_with_areas[sorted_indices]

        sorted_masks = []
        if isinstance(masks, (list, tuple)):
            for i in sorted_indices:
                m = masks[int(i)]
                try:
                    sorted_masks.append(self.keep_largest_connected_mask(m))
                except Exception:
                    sorted_masks.append(m)
        else:
            for i in sorted_indices:
                m = masks[int(i)]
                try:
                    sorted_masks.append(self.keep_largest_connected_mask(m))
                except Exception:
                    sorted_masks.append(m)

        if self.MERGE_ROI is not None:
            return self.shouldMerge(sorted_boxes,sorted_masks)
        return sorted_boxes, sorted_masks


    def compute_iou(self,boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    def is_in_roi(self, box ):
        x1, y1, x2, y2 = box
        rx1, ry1, rx2, ry2 = self.MERGE_ROI
        
        return x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2
    def is_mask_centroid_in_roi(self, centroid):
        rx1, ry1, rx2, ry2 = self.MERGE_ROI
        cx, cy = centroid
        return rx1 <= cx <= rx2 and ry1 + 10 <= cy <= ry2 - 10
        

    def shouldMerge(self, boxes, masks, distance_thresh=230, max_size=185):
        merged_boxes = []
        merged_masks = []
        merged_flags = []
        used = set()

        for i in range(len(boxes)):
            if i in used:
                continue
            box1 = boxes[i][:4].astype(int)
            box_area = boxes[i][4]
            mask1 = masks[i]

            if not self.is_in_roi(box1):
                box1 = np.append(box1, box_area)
                merged_boxes.append(box1)
                merged_masks.append(mask1)
                merged_flags.append(False)
                continue
            

            center1 = self.find_mask_center(mask1,cam_counter=0)
            merged = False

            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                box2 = boxes[j][:4].astype(int)
                mask2 = masks[j]


                if not self.is_in_roi(box2):
                    continue
            
                iou = self.compute_iou(box1, box2)
                center2 = self.find_mask_center(mask2,cam_counter=0)
                if center1 is None or center2 is None:
                    continue
                center_distance = self.CALCULATE_EUCLIDEAN_DISTANCE(center1, center2)
                if center_distance > distance_thresh and iou < 0.1:
                    continue
                
                mx1 = min(box1[0], box2[0])
                my1 = min(box1[1], box2[1])
                mx2 = max(box1[2], box2[2])
                my2 = max(box1[3], box2[3])
                merged_box = (mx1, my1, mx2, my2)
                merged_h = my2 - my1
                merged_w = mx2 - mx1

                merged_mask = np.logical_or(mask1, mask2)
                area = np.sum(merged_mask)
                if area > 43000:
                    print("too big to merge")
                    continue
                
                merged_boxes.append(np.array(merged_box + (area,)))
                merged_masks.append(merged_mask)

                used.add(i)
                used.add(j)
                merged = True
                break

            if not merged:
                box1 = np.append(box1, box_area)
                merged_boxes.append(box1)
                merged_masks.append(mask1)

        return np.array(merged_boxes), merged_masks #, merged_flags




    def isDuplicate_box(self,prev,current, cam_counter = 0):
        return False
        if(int(prev[0])==int(current[0]) and int(prev[1])==int(current[1])  and int(prev[2])==int(current[2]) and int(prev[3])==int(current[3]) ):
            return False
        x1 = max(prev[0],current[0])
        y1 = max(prev[1],current[1])
        x2 = min(prev[2],current[2])
        y2 = min(prev[3],current[3])
        w = x2 - x1
        h = y2 - y1
        if w<=0 or h<=0:
            return 0
        area = max(0,x2-x1 +1) * max(0,y2-y1+1)
        prev_area = self.CalculateBoxArea(prev)
        current_area = self.CalculateBoxArea(current)
        prev_area = current_area if prev_area > current_area else prev_area
        if cam_counter ==1:
            print(area / prev_area, " is ratio of two box")
        if area / prev_area < 0.93: #check with 95% originally
            return False #not duplicate
        return True #duplicate
        
    def IsInsideAnotherBox(self,boxes,current_box , index, cam_counter = 0):
        if(index < 1): #first 
            return False
        contained_in_any_box = False
        until = min(index+1,len(boxes))
        for i in range(until):
            if self.isDuplicate_box(current_box, boxes[i],cam_counter):
                contained_in_any_box = True
                break
        return contained_in_any_box
    
    def isHuman(self,x1,y1,x2,y2, area):
        return area<self.SMALL_SIZE
    def isHumanRatio(self,w,h,area):
        if area > self.SMALL_SIZE + 500:
            return False
        
        if w>h:
            ratio = h/w
        else :
            ratio = w/h
            
        return ratio > 0.6
    
    def IsInMiddle(self,x1,y1,x2,y2):
        return x1 > 10 and y1 > 10 and x2 <self.boundary[0] + 5 and y2 < self.boundary[1]
    
    def IsTouchingBorder(self,box, threshold = 3):
        x1,y1,x2,y2 = box
        if  x1<threshold or y1<threshold or \
            y2>=self.boundary[1]-threshold \
            or x2>=self.boundary[0] -threshold: #or x2>=self.boundary[0]: #area < self.average_size 
            return False 
        
        return True

    def keep_largest_connected_mask(self,mask):
        """
        Retains only the largest connected component in the mask and removes all other disconnected regions.

        Parameters:
        - mask (numpy.ndarray): Boolean mask array.

        Returns:
        - numpy.ndarray: Mask with only the largest connected component.
        """
        mask = mask.astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels <= 1:
            return np.zeros_like(mask, dtype=bool)

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip background (index 0)

        largest_mask = (labels == largest_label)

        return largest_mask.astype(bool)


    def append_pair(self,tracking_id, predicted_id):
        global csv_file
        file_exists = os.path.exists(csv_file)
        
        updated_rows = []
        tracking_found = False

        if file_exists:
            with open(csv_file, mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row and int(row[0]) == tracking_id:
                        row.append(predicted_id)
                        tracking_found = True
                    updated_rows.append(row)

        if not tracking_found:
            updated_rows.append([tracking_id, predicted_id])

        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(updated_rows)


    def isInsideCorrectRoi(self,center):
        return cv2.pointPolygonTest(self.CORRECT_ROI, center, False) >= 0
    def isOutsideofGap(self,center,isTouching,bbox,area = 20000):
        area_th = 22000
        if self.GAP_ROI is None:
            return  area > area_th and self.isInsideCorrectRoi(center) # update with variable
        if cv2.pointPolygonTest(self.GAP_ROI, center, False)  >= 0 or area < area_th:
            return False
        
        if area > area_th and self.isInsideCorrectRoi(center): #not close to border
            return True

        return False


    def CoreProcess(self,boxes,masks_np,h,w,frame,cam_counter,frame_count):
        mask_count = -1
        tracked_ids = []
        colored_mask = np.zeros_like(frame)
        tracked_indexes = []
        predicted_ids = []
        recheck_periods = []
        original_predicted_ids = []
        img_file_names = []
        counter = -1
        duplicate_ids = []
        height_threshold = 80
        if not self.isEatingArea:
            height_threshold = 60
        
        
        for box in boxes:
            conver_time = datetime.now()
            x1, y1, x2, y2, _ = map(int, box)
            
            counter += 1
            mask_count+=1   
            isSmall = False

            if(self.IsInsideAnotherBox(boxes,box,counter, cam_counter)):
                continue #skip for this
            area = np.sum(masks_np[mask_count])

           
            touching_pixels = 0
            is_touching = False
            middle_height = 200
            is_touching_right = False
            if x1 <= 4 or y1 <= 4 or y2 >= self.boundary[1]: #check for 3 direction 
                touching_pixels,is_touching_right = self.count_border_pixels(masks_np[mask_count],(x1,y1,x2,y2))
                is_touching = touching_pixels >= self.border_threshold
                if is_touching: 
                    is_touching = not self.is_showing_most_body(masks_np[mask_count], (x1,y1,x2,y2))
                isSmall = is_touching or isSmall #middle_height < height_threshold # either touching or small height
            isSmall = self.isHuman(x1,y1,x2,y2,area) or isSmall
            
            
            if True :#HOKKAIDO_ROI(x1,y1,x2,y2,h,w):  #only cow class
                prev_id = self.Centroid_Tracking(masks_np[mask_count],x2-x1,y2-y1,cam_counter,y1,y2)                
                if(prev_id==-1): #skip cattle when prev_id // filter id is -1
                    continue
                
                
                middle_height = 200
               
                isInMiddle = self.IsInMiddle(x1,y1,x2,y2)
                if isInMiddle and area < 6500 and not is_touching_right:
                    print("I am middle : and small with area", area )
                    continue
                is_new = prev_id[3] # check is_new
                #don't remove this self.find_matching_mask line , it is placeholder for future improvement
                tracking_id,is_touching_bottom,is_touching_top = prev_id[0],False,False #self.find_matching_mask(masks_np[mask_count],prev_id[1],cam_counter,prev_id[0],x1,y1,x2,y2,frame)
                if tracking_id != -1:
                    if int(tracking_id) in self.STORED_IDS:
                        idx = self.STORED_IDS.index(int(tracking_id))
                        
                        self.STORED_XYXY[idx] = prev_id[1]

                        self.STORED_MISS[idx] = 1
                        prev_id[0] = tracking_id
                        is_new = False
                    else:
                        self.delete_mask_by_tracking_id([str(tracking_id)])

                if is_new:
                   
                    if isSmall and self.isHumanRatio(x2-x1,y2-y1,area) : #prioritize the size first
                        print(" I am too small haha")
                        continue
                    if isInMiddle and area < 8000 and not is_touching_right:
                        print("I am middle : and small with area", area )
                        continue
                       
                    else:
                        
                        prev_id[0] = self.addNewTracking_v2(masks_np[mask_count],prev_id[1],x1,y1,x2,y2,cam_counter,w,h)
                        #print("new tracking id : ",prev_id[0], " is touching  top ", is_touching_top, " is touching bottom ", is_touching_bottom, " camera ", cam_counter)
                        info = MASKLOCATION(prev_id[0],masks_np[mask_count],cam_counter,prev_id[1],is_touching_top,is_touching_bottom)
                        self.TRACKING_MASK_LOCATION.append(info)
                       
                
                if not isSmall and area < 11000 and isInMiddle and not is_touching_right:
                    isSmall = True


                
                tracked_ids.append(int(prev_id[0]))
                tracked_indexes.append(mask_count)
                
                missed_frame_count = int(prev_id[2])
                
               
                is_outside_gap = self.isOutsideofGap(prev_id[1], isTouching=isSmall,bbox=(x1,y1,x2,y2),area=area)
                predicted_id,tracking_id, IS_RECHECK_PERIOD = self.GetBatchPredictedId(int(prev_id[0]),isSmall,self.save_dir,missed_frame_count,prev_id[1],w,h, is_outside_gap)
                
                img_file_name = self.save_crop(frame,masks_np[mask_count],x1,y1,x2,y2,
                               self.save_dir,prev_id,
                               is_touching,touching_pixels,
                               frame_count,is_save=True,
                               is_small=isSmall,predicted_id= predicted_id,area=area,
                               IS_RECHECK_PERIOD=IS_RECHECK_PERIOD,
                                is_outside_gap = is_outside_gap 
                               )
                if predicted_id != '' and predicted_id!='Identifying' and predicted_id in predicted_ids :
                    duplicate_ids.append(predicted_id)
                recheck_periods.append(IS_RECHECK_PERIOD)
                img_file_names.append(img_file_name)
                predicted_ids.append(predicted_id)
                
        colored_mask = self.draw_mask_multiple_masks(colored_mask,masks_np,tracked_indexes)               
        
        response_model = CoreProcessResponseModel(cam_counter, boxes ,tracked_indexes, tracked_ids, predicted_ids, original_predicted_ids, colored_mask,recheck_periods)
        return response_model,img_file_names
        
  
    def dispose_model(self,predictor):
    
        del predictor  # Remove reference
        torch.cuda.empty_cache()  # Clear the CUDA memory
        print("Model disposed and GPU memory released.")

    def get_duplicate_indexes(self,array):
        index_map = defaultdict(list)
        for i, val in enumerate(array):
            index_map[val].append(i)

        duplicates = {val: idxs for val, idxs in index_map.items() if len(idxs) > 1}
        return duplicates
    def get_max_start_time_per_video(self,video_list):
        max_start_time = 0
        start_times = []
        print(video_list)

        for video in video_list:
            start_time = int(video.split('\\')[-1].split("-")[1])
            start_times.append(start_time)
            if start_time > max_start_time:
                max_start_time = start_time
        
        start_times = [max_start_time - x for x in start_times]

        return start_times

    
    def vertical_stitch_cut_both_4_6(self,img1, img2, img3, 
                                cut_top1=0.0, cut_bot1=0.0,
                                cut_top2=0.0, cut_bot2=0.0,
                                cut_top3=0.0, cut_bot3=0.0,
                                ):
        
        def crop(img, top_pct, bot_pct):
            h = img.shape[0]
            top = int(h * top_pct)
            bottom = int(h * (1 - bot_pct))
            return img[top:bottom]

        cropped1 = crop(img1, cut_top1, cut_bot1)
        cropped2 = crop(img2, cut_top2, cut_bot2)
        cropped3 = crop(img3, cut_top3, cut_bot3)

        overlap = 25
        positions_1_1 = [(0,2000)]  #for cam 4,5,6 x,y 
        block_size_1_1 = (280,overlap)
        positions_1_2 = [(670,2000)] #cam 4,5,6
        block_size_1_2 = (1152-670, overlap)  



            

        positions_2_1 = [(280,0)]
        block_size_2_1 = (400, overlap) 
        cropped2 = self.add_black_blocks(cropped2, positions_2_1, block_size_2_1)





        tetris1_2 = self.stack_tetris_crops(cropped1, cropped2, overlap=overlap)
        return [tetris1_2, cropped3]


    def vertical_stitch_cut_both_7_10(self,img1, img2, img3, img4,
                                cut_top1=0.0, cut_bot1=0.0,
                                cut_top2=0.0, cut_bot2=0.0,
                                cut_top3=0.0, cut_bot3=0.0,
                                cut_top4=0.0, cut_bot4=0.0):
        def crop(img, top_pct, bot_pct):
            h = img.shape[0]
            top = int(h * top_pct)
            bottom = int(h * (1 - bot_pct))
            return img[top:bottom]
        img3 = self.rotate_image(img3, 2) 
        img2 = self.rotate_image(img2, 1) 
        cropped1 = crop(img1, cut_top1, cut_bot1)
        cropped2 = crop(img2, cut_top2, cut_bot2)
        cropped3 = crop(img3, cut_top3, cut_bot3)
        cropped4 = crop(img4, cut_top4, cut_bot4)


        return [cropped1, cropped2,cropped3, cropped4] #stitched 

    def vertical_stitch_cut_both_11_14(self,img1, img2, img3, img4,
                                cut_top1=0.0, cut_bot1=0.0,
                                cut_top2=0.0, cut_bot2=0.0,
                                cut_top3=0.0, cut_bot3=0.0,
                                cut_top4=0.0, cut_bot4=0.0
                                ):
        
        def crop(img, top_pct, bot_pct):
            h = img.shape[0]
            top = int(h * top_pct)
            bottom = int(h * (1 - bot_pct))
            return img[top:bottom]
        img3 = self.rotate_image(img3, -1)  # Rotate img3 by 45 degrees clockwise
        cropped1 = crop(img1, cut_top1, cut_bot1)
        cropped2 = crop(img2, cut_top2, cut_bot2)
        cropped3 = crop(img3, cut_top3, cut_bot3)
        cropped4 = crop(img4, cut_top4, cut_bot4)
        overlap = 30
      
        positions_2 = [(0,2000)]
        block_size_2 = (400, overlap) 


        positions_2_1 = [(380,2000)]
        block_size_2_1 = (644,overlap)

        positions_3 = [(0,0)]
        block_size_3 = (500, overlap)


        return [cropped1, cropped2,cropped3, cropped4] #stitched 
        
    def stack_tetris_crops(self,img1,img2, overlap = 20):
        h1, w = img1.shape[:2]
        h2 = img2.shape[0]

        final_height = h1 + h2 - overlap

        result = np.zeros((final_height, w, 3), dtype=np.uint8)

        result[:h1 - overlap] = img1[:h1 - overlap]

        overlap_img1 = img1[h1 - overlap:]
        overlap_img2 = img2[:overlap]

        mask = np.any(overlap_img2 != [0, 0, 0], axis=2)

        blended_overlap = overlap_img1.copy()

        blended_overlap[mask] = overlap_img2[mask]

        result[h1 - overlap:h1] = blended_overlap

        result[h1:] = img2[overlap:]

        return result
    def add_black_blocks(self,image, xy_positions, block_size=(510, 20)):
        """
        Draws black blocks at specified (x, y) positions on the image.

        Args:
            image (np.ndarray): Input image (BGR).
            xy_positions (list of tuples): List of (x, y) top-left corners.
            block_size (tuple): Width and height of the block (w, h).

        Returns:
            np.ndarray: Modified image with black blocks drawn.
        """
        
        h_img, w_img = image.shape[:2]
        w_blk, h_blk = block_size
        for x, y in xy_positions:
            x_end = min(x + w_blk, w_img)
            y_end = min(y + h_blk, h_img)
            x_start = max(x, 0)
            y_start = min(max(y, 0),h_img-h_blk)
            image[y_start:y_end, x_start:x_end] = (0, 0, 0)  # Black in BGR

        return image

    def rotate_image(self,image, angle):
        (w, h) = self.resolution
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)  # Negative for clockwise
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def vertical_stitch_cut_both(self,img1, img2, img3, img4,
                                cut_top1=0.0, cut_bot1=0.0,
                                cut_top2=0.0, cut_bot2=0.0,
                                cut_top3=0.0, cut_bot3=0.0,
                                cut_top4=0.0, cut_bot4=0.0
                                ):
        
        def crop(img, top_pct, bot_pct):
            h = img.shape[0]
            top = int(h * top_pct)
            bottom = int(h * (1 - bot_pct))
            return img[top:bottom]
        img3 = self.rotate_image(img3, -2)  # Rotate cam 3 by 2 degrees

        cropped1 = crop(img1, cut_top1, cut_bot1)
        cropped2 = crop(img2, cut_top2, cut_bot2)
        cropped3 = crop(img3, cut_top3, cut_bot3)
        cropped4 = crop(img4, cut_top4, cut_bot4)
        overlap = 30





        positions_2_1 = [(380,2000)]
        block_size_2_1 = (644,overlap)
        cropped2 = self.add_black_blocks(cropped2, positions_2_1, block_size_2_1)

        positions_3 = [(0,0)]
        block_size_3 = (380, overlap)
        cropped3 = self.add_black_blocks(cropped3, positions_3, block_size_3)


        tetris2_3 = self.stack_tetris_crops(cropped2, cropped3, overlap=overlap)
        return [cropped1, tetris2_3, cropped4]

    def get_vertical_tiles(self, image, tile_height=800, overlap=200):
        h, w = image.shape[:2]
        tiles = []
        y_starts = list(range(0, h, tile_height - overlap))

        for y in y_starts:
            y_end = min(y + tile_height, h)
            tile = image[y:y_end]
            tiles.append((tile, y))
        return tiles

    def detect_tiles(self, stitched_img):
        all_boxes = []
        all_scores = []
        all_classes = []
        all_masks = []

        for tile, y_offset in self.get_vertical_tiles(stitched_img, tile_height=800, overlap=200):
            outputs = self.predictor(tile)
            instances = outputs["instances"].to("cpu")

            if len(instances) == 0:
                continue

            boxes = instances.pred_boxes.tensor.numpy()
            boxes[:, [1, 3]] += y_offset  # shift Y

            masks = instances.pred_masks.numpy()
            shifted_masks = np.zeros((masks.shape[0], stitched_img.shape[0], stitched_img.shape[1]), dtype=bool)
            for i in range(masks.shape[0]):
                mask = masks[i]
                h_tile = mask.shape[0]
                shifted_masks[i, y_offset:y_offset + h_tile, :] = mask  # shift vertically

            all_boxes.append(boxes)
            all_scores.append(instances.scores.numpy())
            all_classes.append(instances.pred_classes.numpy())
            all_masks.append(shifted_masks)

        return all_boxes, all_scores, all_classes, all_masks




    def get_Tracking_To_Merge(self,data):
        value_indices = defaultdict(list)
        for idx, val in enumerate(data):
            value_indices[val].append(idx)

        duplicates = {val: idxs for val, idxs in value_indices.items() if len(idxs) > 1}
        return duplicates
    def combine_boxes(self,box1, box2):
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        y2 = max(box1[3], box2[3])
        return [x1, y1, x2, y2]

    def save_data_to_skip(self, images, outputs):
        for i, img in enumerate(images):
            base_path = Path(self.save_dir) / f"frame_saver_{i}"
            base_path.mkdir(parents=True, exist_ok=True)

            img_path = base_path / f"{self.frame_number}.jpg"
            cv2.imwrite(str(img_path), img)

        output_dir = Path(self.save_dir) / "detections"
        output_dir.mkdir(parents=True, exist_ok=True)

        boxes_list, masks_list = [], []
        for out in outputs:
            boxes = out["instances"].pred_boxes.tensor.cpu().numpy()
            masks = out["instances"].pred_masks.cpu().numpy()

            boxes_list.append(boxes)
            masks_list.append(masks)

        with open(output_dir / f"{self.frame_number}_boxes.json", "w") as f:
            json.dump([b.tolist() for b in boxes_list], f)

        np.savez_compressed(output_dir / f"{self.frame_number}_masks.npz", *masks_list)

    def load_data_to_skip(self, frame_number):
        images = []
        i = 0
        while True:
            img_path = Path(self.path) / f"frame_saver_{i}" / f"{frame_number}.jpg"
            if not img_path.exists():
                break
            img = cv2.imread(str(img_path))
            images.append(img)
            i += 1

        boxes_file = Path(self.save_dir) / "detections" / f"{frame_number}_boxes.json"
        with open(boxes_file, "r") as f:
            boxes_list = json.load(f)

        masks_file = Path(self.save_dir) / "detections" / f"{frame_number}_masks.npz"
        masks_npz = np.load(masks_file, allow_pickle=True)
        masks_list = [masks_npz[key] for key in masks_npz.files]
        if frame_number <= 1 :
            for i in range(len(images)):
                self.camera_heights.append(images[i].shape[0])
        return images, boxes_list, masks_list
        

    def batch_frames(self, data, file_name, is_show_image, cap= None, tracking_cap = None, isFile = False, isCam = False):
        video_file = data[0]
        print(video_file, " I am video file")
        start_timer_adjustment = self.get_max_start_time_per_video(data)
        global images
        
        self.boundary = self.save_width-7,self.save_height-7
        video_readers = []
        apply_correction_on = {}
        video_counter = 0
        for vdo in data:
            
            video_reader = cv2.VideoCapture(vdo)
            

            video_readers.append(video_reader)
            video_counter += 1
        
        is_save_video = cap is not None
        is_save_tracking_video = tracking_cap is not None
        print('are we saving video? ',is_save_video)
        print('are we saving tracking video? ',is_save_tracking_video)
        print(f'current values small size {self.SMALL_SIZE}')

        middle = self.save_height * 0.25
        h25= self.save_height*0.25
        h75 = self.save_height*0.75

               
      
        skip_frame = self.frame_skip_count # math.ceil(CAP_PROP_FPS / 5)-1   
        frame_count = skip_frame+1
        
        ret = True
        skip_minute,skip_second = 4,45
        manual_skip = 0 #(skip_minute * 60 + skip_second)  * 20
        
        while ret:
            start_time = datetime.now()
            total_time_start = start_time
            images = []
            vid_counter = 0
            for video in video_readers:
                ret, frame = video.read()
                if manual_skip > 0:
                    continue
                if not ret :#or manual_skip > 0:
                    break
                images.append(cv2.resize(frame,self.resolution))
                vid_counter += 1
            if manual_skip > 0:
                print(manual_skip," skipping frames")
                manual_skip -= 1
                continue
        
            frame_count+=1
            if frame_count<=skip_frame: #read every 4th frame
                continue
            if(len(images) != len(video_readers)):
                print("All images len not match, len is ", len(images), " , videos len is ", len(video_readers))
                return
            if len(images) > 1:
                images = self.vertical_stitch_cut_both_4_6(
                            images[0], images[1], images[2],


                            


                            cut_top1=0.202,  cut_bot1=1-0.833,  # Keep top of Cam1, cut 18% bottom  // 20 pixels is 0.026 %  #10
                            cut_top2=0.182, cut_bot2=1-0.775,  # Cut 18% top and 21% bottom of Cam2  #9
                            cut_top3=0.202, cut_bot3=1-0.846,    # Cut 21% top of Cam3, keep full botto  #8
                        )
                if len(self.camera_heights) == 0:
                    print ("Adding images shape first time")
                    for i in range(len(images)):
                        self.camera_heights.append(images[i].shape[0])
                
            
            
            frame_count = 0
            batch_outputs = list(self.stupid_detector(images))
            self.frame_number +=1
            all_frame_infos = []
            cam_counter = 0
            for outputs in batch_outputs:  #per frame
                
                frame = images[cam_counter]
                start_time = datetime.now()
                instances = outputs["instances"].to("cpu")
                boxes,masks_np = self.sortBoxAndMask(instances.pred_boxes.tensor.numpy(),instances.pred_masks.numpy())
                mask_count = 0
                h, w, _ = frame.shape
                HAS_COW = False
                
                
                
                start_time = datetime.now()
                
                all_frame_infos.append(self.CoreProcess(boxes,masks_np,h,w,frame, cam_counter))
                
                
                
                self.diff_Time(start_time,datetime.now(),' tracking and validation process')
                
                
                images[cam_counter] = frame
                cam_counter+=1

            all_tracking_ids = [item for sublist in all_frame_infos for item in sublist.tracked_ids]
            all_duplicate_idexes = self.get_duplicate_indexes([item for sublist in all_frame_infos for item in sublist.predicted_ids])
            
            tracking_to_merge = self.get_Tracking_To_Merge(all_tracking_ids)
            
            all_duplicate_tracking_ids = []
            
            boxes_to_merge = defaultdict(list)
            for duplicate in all_duplicate_idexes:
                if duplicate in ("Identifying","Reidentifying","unknown",None):
                    continue
                duplicate_tracking_ids = [all_tracking_ids[i] for i in all_duplicate_idexes[duplicate]]
                is_same = all(val == duplicate_tracking_ids[0] for val in duplicate_tracking_ids)
                if not is_same:
                    for id in duplicate_tracking_ids:
                    
                        all_duplicate_tracking_ids.append(id)

            if(len(all_duplicate_tracking_ids) > 0):
                print("Found duplicate tracking ids across cameras: ",all_duplicate_tracking_ids)
                self.reset_duplicate_tracking_identification(all_duplicate_tracking_ids)
            
            for frame_info in all_frame_infos:
                mask_count = 0

                images[frame_info.cam_counter] = cv2.addWeighted(frame_info.colored_mask, 0.3,images[frame_info.cam_counter], 1 - 0.3, 0)
                for index in frame_info.tracked_indexes:
                    x1, y1, x2, y2, area = map(int, frame_info.boxes[index])
                    
                    tracking_id = frame_info.tracked_ids[mask_count]
                    label = 'Identifying' if frame_info.tracked_ids[mask_count] in all_duplicate_tracking_ids else f'{str(frame_info.predicted_ids[mask_count])}'
                    if tracking_id in tracking_to_merge.keys():
                        sum_y = 0
                        if frame_info.cam_counter > 0:
                            sum_y = sum(self.camera_heights[:frame_info.cam_counter])
                        boxes_to_merge[tracking_id] += [[x1,y1+sum_y,x2,y2+sum_y],label]
                    else:
                        self.draw_bounding_box(images[frame_info.cam_counter],(x1,y1,x2,y2),label,frame_info.recheck_periods[mask_count],str(frame_info.tracked_ids[mask_count]),font_scale=1) # draw with predicted id
                    mask_count+=1
            
            
            self.IncreaseMissedCount(all_tracking_ids)
            start_time = datetime.now()
            stacked_image = self.stack_image_from_bottom_to_top(images)
            for key,box_to_merge in boxes_to_merge.items():
                x1,y1,x2,y2 = self.combine_boxes(box_to_merge[0],box_to_merge[2])
                label = box_to_merge[1]
                try:
                    if y2-y1 > 650:
                        print(" too big to merge")
                        x1,y1,x2,y2 = box_to_merge[0]
                        self.draw_bounding_box(stacked_image,(x1,y1,x2,y2),box_to_merge[1],str(key),font_scale=1) # draw with predicted id
                        x1,y1,x2,y2 = box_to_merge[2]
                        self.draw_bounding_box(stacked_image,(x1,y1,x2,y2),box_to_merge[3],str(key),font_scale=1) # draw with predicted id
                    else:
                        self.draw_bounding_box(stacked_image,(x1,y1,x2,y2),label,str(key),font_scale=1) # draw with predicted id
                except:
                    print("Error in merging boxes for key:", key, "with box_to_merge:", box_to_merge)
            if stacked_image is None:
                print("Stacked image is None")
                break
            if is_save_video:
                cap.write(stacked_image)


            if is_save_tracking_video:
                tracking_cap.write(original_frame)
                
            self.diff_Time(start_time,datetime.now(),' file writing process')    # resizedImage = cv2.resize(v.get_image(), (1000, 1000)) 
            
            start_time = datetime.now()

            if is_show_image and True:
                imshow_size = 576 , int((self.image_height/2)*len(data))
                
                cv2.imshow("NASA doing centroid tracking on the cattle (detectron 2)", cv2.resize(stacked_image, (imshow_size)))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    pass
                    self.dispose_model(self.stupid_detector)
                    return -1
            self.diff_Time(start_time,datetime.now(),' image showing process')

            self.diff_Time(total_time_start,datetime.now(),' the whole process')
        return    

           
    

    def find_mask_center(self,mask,cam_counter):
        if not np.issubdtype(mask.dtype, np.bool_):
            mask = mask > 0
        
        y_coords, x_coords = np.where(mask)
        
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        center_y = center_y +  (0 if cam_counter == 0 else sum(self.camera_heights[:cam_counter]))
        return (center_x, center_y)



    def drawArrayOverImage(self,img):
        
        spacing = 0  # spacing between rectangles
        rect_height = 70   # height of each rectangle
        rect_width =  100 # width of each rectangle

        x_offset = 50
        y_offset = 2500

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 255)
        font_thickness = 2
        labels = ['ID','MISS']
        

        for i in range(2):
            array = self.STORED_IDS if i == 0 else self.STORED_MISS
        
            
            label_size = cv2.getTextSize(labels[i], font, font_scale, font_thickness)[0]
            label_x = x_offset - label_size[0] - 10
            label_y = y_offset + i * (rect_height + spacing) + rect_height // 2 + label_size[1] // 2
            
            cv2.putText(img, labels[i], (label_x, label_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            for j in range (26):
                
                value = '-'
                if j< len(array):
                    value = array[j]
                
                top_left = (x_offset + j * (rect_width + spacing), y_offset + i * (rect_height + spacing))
                bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)
                
                cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), 5)
                
                text_position = (top_left[0] + rect_width // 2 - 20, top_left[1] + rect_height // 2 + 10)
                cv2.putText(img, str(value), text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        return img

    def clearData(self):
        print("Clearing data")
        self.TRACKER.clear()  #To store tracking index to use in classifier
        self.CLASSIFICATION_TRACKER.clear()#[{'GT': [1, 2, 3], 'COUNT': [10, 2, 5]}] retrieve Index , GT is ground truth, COUNT is number of apperance
        self.BATCH_CLASSIFICATION_TRACKER.clear()
        self.ALL_TIME_CLASSIFICATION_TRACKER.clear()
        self.TRACKING_RESULT.clear()
        self.STORED_IDS.clear()   # move it up for each folder
        self.STORED_MISS.clear()
        self.STORED_XYXY.clear()
        self.CATTLE_LOCAL_ID = 0  
        self.cow_filter_size.clear()
        self.TRACKING_MASK_LOCATION.clear()
        self.LAST_20_PATH.clear()
        self.PASSING_CATTLE_BY_CAMERA.clear()
        

    def save_final_pdf(self,save_dir,file_name):
        print("Saving finalize pdf :",save_dir)
        default_data = ["ID"]
        final_result = []
        missed_frame_data = []
        default_missed_frame_data = ['Tracking','Total','Missed']
        missed_frame_data.append(default_missed_frame_data)
        for i in (self.class_names):
            default_data.append(self.class_names[i])
        
        final_result.append(default_data)
        for index in range(len(self.TRACKER)):
            try:

                tracking_id = self.TRACKER[index]
                if(tracking_id is None):
                    continue
                track_index = len(self.TRACKER) - 1
                dataset = self.ALL_TIME_CLASSIFICATION_TRACKER[index]
                stx = [int(tracking_id)]
                
                missed_data = []
                missed_data.append(tracking_id)
                missed_data.append(self.BATCH_CLASSIFICATION_TRACKER[index]['TOTAL_DETECTION'])
                missed_data.append(self.BATCH_CLASSIFICATION_TRACKER[index]['TOTAL_MISSED_FRAME'])
                missed_frame_data.append(missed_data)
                
                for idx in (self.class_names):
                    id = str(self.class_names[idx])
                    if id not in self.ALL_TIME_CLASSIFICATION_TRACKER[index]['GT']:
                        stx.append(0)    
                    else:
                        predicted_index = self.ALL_TIME_CLASSIFICATION_TRACKER[index]['GT'].index(id)
                        predicted_count = self.ALL_TIME_CLASSIFICATION_TRACKER[index]['COUNT'][predicted_index]
                        stx.append(predicted_count)
                    
                final_result.append(stx)   
            except Exception as e:
                print(f"Error processing index {index}: {e}")
        df = pd.DataFrame(final_result)
        df.to_csv(f"{save_dir}\\{file_name}.csv", index=False,header=False)
        missed_df = pd.DataFrame(missed_frame_data)
        missed_df.to_csv(f"{save_dir}\\missed_count.csv", index=False,header=False)
        self.save_accuracy_like_program(save_dir)
        
    def save_accuracy_like_program(self,save_dir,file_name='program_accuracy'):
        final_result = []
        for key in (self.TRACKING_RESULT):
            finalData = []
            finalData.append(key)
            
            for value in self.TRACKING_RESULT[key]:
                finalData.append(value)
            final_result.append(finalData)
        df = pd.DataFrame(final_result)
        df.to_csv(f"{save_dir}\\{file_name}.csv", index=False,header=False)
        
    
        

    def has_long_straight_lines(self,mask, min_length=100, frame_height=None):
        height = frame_height
        has_horizontal = False
        horizontal_y2 = None
        for y, row in enumerate(mask):
            line_starts = np.where(np.diff(np.concatenate(([False], row))))[0]
            line_ends = np.where(np.diff(np.concatenate((row, [False]))))[0]
            
            min_len = min(len(line_starts), len(line_ends))
            line_lengths = line_ends[:min_len] - line_starts[:min_len]
            
            if np.any(line_lengths >= min_length):
                has_horizontal = True
                horizontal_y2 = y
                break

        has_vertical = False
        vertical_y2 = None
        for x, col in enumerate(mask.T):
            line_starts = np.where(np.diff(np.concatenate(([False], col))))[0]
            line_ends = np.where(np.diff(np.concatenate((col, [False]))))[0]
            
            min_len = min(len(line_starts), len(line_ends))
            line_lengths = line_ends[:min_len] - line_starts[:min_len]
            
            if np.any(line_lengths >= min_length):
                has_vertical = True
                if line_ends[-1] == height:  # Check if the line extends to the bottom of the mask
                    vertical_y2 = frame_height - 1
                else:
                    vertical_y2 = line_ends[-1] - 1
                break

        return has_horizontal, has_vertical#, horizontal_y2, vertical_y2
    
    def check_lines_in_bbox(self,mask, bbox, min_length=50, threshold=50):
        x1, y1, x2, y2 = bbox
        height, width = mask.shape

        top_hits = 0
        left_hits = 0
        bottom_hits = 0

        for y in range(y1, y2 + 1):
            if y >= height:
                break
            row = mask[y, x1:x2+1]
            line_starts = np.where(np.diff(np.concatenate(([False], row))))[0]
            line_ends = np.where(np.diff(np.concatenate((row, [False]))))[0]
            
            min_len = min(len(line_starts), len(line_ends))
            line_lengths = line_ends[:min_len] - line_starts[:min_len]
            
            long_lines = line_lengths >= min_length
            if np.any(long_lines):
                if y == y1:
                    top_hits += np.sum(line_lengths[long_lines])
                elif y == y2:
                    bottom_hits += np.sum(line_lengths[long_lines])

        for x in range(x1, x2 + 1):
            if x >= width:
                break
            col = mask[y1:y2+1, x]
            line_starts = np.where(np.diff(np.concatenate(([False], col))))[0]
            line_ends = np.where(np.diff(np.concatenate((col, [False]))))[0]
            
            min_len = min(len(line_starts), len(line_ends))
            line_lengths = line_ends[:min_len] - line_starts[:min_len]
            
            long_lines = line_lengths >= min_length
            if np.any(long_lines):
                if x == x1:
                    left_hits += np.sum(line_lengths[long_lines])

        return top_hits >= threshold, left_hits >= threshold, bottom_hits >= threshold

    def check_orientation(self,box,ratio_threshold=1.5):
        
        width = box[2] - box[0]
        height = box[3] - box[1]

        if height > width * ratio_threshold:
            return 50 #"vertical" #100 , reducing to half due to re-id approach
        elif width > height * ratio_threshold:
            return 75#"horizontal" #150 
        else:
            return 40# "diagonal" $80
    def is_showing_most_body(self,mask, bbox):
        centroid_threshold = self.check_orientation(bbox)
        
        (x,y) = self.find_mask_center(mask,0)
        if x > centroid_threshold and x < self.image_width-centroid_threshold and y > centroid_threshold and y < self.image_height-centroid_threshold:
            return True
        return False
            
    
    def count_border_pixels(self,mask, bbox, border_margin=5):
        
        threshold = self.border_threshold
        """
        Count the number of pixels touching each border of the image and determine if it exceeds the threshold.
        
        :param mask: 2D numpy array (boolean mask of the object)
        :param bbox: tuple of (x1, y1, x2, y2) representing the bounding box
        :param threshold: int, the minimum number of pixels to consider the object as touching the border
        :param border_margin: int, margin to consider when checking if object touches the border
        :return: tuple of (is_partial, counts), where is_partial is a boolean and counts is a dict of pixel counts for each border
        """
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise ValueError("Input mask must be a 2D numpy array")

        height, width = mask.shape
        x1, y1, x2, y2 = bbox

       
        left_count = np.sum(mask[y1:y2+1, max(0, x1-border_margin):min(width, x1+border_margin)])
       
        right_count = np.sum(mask[y1:y2+1, max(0, x2-border_margin):min(width, x2+border_margin)])
        top_count = np.sum(mask[max(0, y1-border_margin):min(height, y1+border_margin), x1:x2+1])
        bottom_count = np.sum(mask[max(0, y2-border_margin):min(height, y2+border_margin), x1:x2+1])


        
        counts = [
             int(left_count),
             0, # 0 for right count
             int(top_count),
              int(bottom_count)
         ]
        maxTouching =  max(counts) #, counts
        return maxTouching, int(right_count)>2
    
    def print_border_pixels(self,mask, bbox, border_margin=3):
        
        threshold = self.border_threshold
        """
        Count the number of pixels touching each border of the image and determine if it exceeds the threshold.
        
        :param mask: 2D numpy array (boolean mask of the object)
        :param bbox: tuple of (x1, y1, x2, y2) representing the bounding box
        :param threshold: int, the minimum number of pixels to consider the object as touching the border
        :param border_margin: int, margin to consider when checking if object touches the border
        :return: tuple of (is_partial, counts), where is_partial is a boolean and counts is a dict of pixel counts for each border
        """
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise ValueError("Input mask must be a 2D numpy array")

        height, width = mask.shape
        x1, y1, x2, y2 = bbox


        left_count = np.sum(mask[y1:y2+1, max(0, x1-border_margin):min(width, x1+border_margin)])
        if self.isEatingArea:
            right_count = 0
        else:
            right_count = np.sum(mask[y1:y2+1, max(0, x2-border_margin):min(width, x2+border_margin)])
        top_count = np.sum(mask[max(0, y1-border_margin):min(height, y1+border_margin), x1:x2+1])
        bottom_count = np.sum(mask[max(0, y2-border_margin):min(height, y2+border_margin), x1:x2+1])

        counts = [
             int(left_count),
             int(right_count),
             int(top_count),
              int(bottom_count)
         ]
      
        
    
    def calculate_mask_height(self, mask, bbox):
        """
        Calculate the height of the mask along the longer dimension of the bounding box.
        
        :param mask: 2D numpy array (boolean mask)
        :param bbox: tuple of (x1, y1, x2, y2) representing the bounding box
        :return: tuple of (height, plane), where height is the calculated height and plane is either 'X' or 'Y'
        """
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise ValueError("Input must be a 2D numpy array")

        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        if bbox_width < bbox_height:
            plane = 'X'
            center = (y1 + y2) // 2
            row = mask[center, x1:x2+1]
            true_indices = np.where(row)[0]
            height = len(true_indices)
        else:
            plane = 'Y'
            center = (x1 + x2) // 2
            column = mask[y1:y2+1, center]
            true_indices = np.where(column)[0]
            height = len(true_indices)

        return height #height, plane
    
    def mask_height_from_center(self,mask):
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise ValueError("Input must be a 2D numpy array")

        height, width = mask.shape
        center_y = height // 2

        true_rows = np.where(mask.any(axis=1))[0]
        if len(true_rows) == 0:
            return 0  # No True values in the mask

        top = true_rows[0]
        bottom = true_rows[-1]

        height_above = max(center_y - top, 0)
        height_below = max(bottom - center_y, 0)

        return max(height_above, height_below)
    
    def get_all_videos(self,path: str, extensions=None):
        if extensions is None:
            extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        
        path = Path(path)
        video_files = [str(p) for p in path.rglob("*") if p.suffix.lower() in extensions]
        return video_files
    
    def unsharp_mask(self, bgr, radius=1.4, amount=1.0):
        blurred = cv2.GaussianBlur(bgr, (0,0), radius)
        sharp = cv2.addWeighted(bgr, 1+amount, blurred, -amount, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    def apply_clahe_ycrcb(self, bgr, clip=2.5, tile=8):
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        y2 = clahe.apply(y)
        merged = cv2.merge([y2, cr, cb])
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    
    def get_cattle_posture(self,mask: np.ndarray):
        """
        Determine cattle posture and facing direction from top-view mask.

        Args:
            mask (np.ndarray): Boolean or 0-255 binary mask from Detectron2.
            bbox (tuple): (x1, y1, x2, y2) bounding box (not used much, but may help cropping later).

        Returns:
            int: posture code (0–8)
                0 = lying
                1–8 = standing (↑ ↗ → ↘ ↓ ↙ ← ↖)
        """
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return -1  # no contour
        
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 5:
            return -1  # too small for ellipse fitting
        
        ellipse = cv2.fitEllipse(cnt)
        (center, axes, angle) = ellipse
        major, minor = max(axes), min(axes)
        ratio = major / minor if minor > 0 else 0
        
        if ratio < 1.5:
            return 0  # lying
        
        angle = angle % 360
        
        directions = {
            (337.5, 360): 1,   # top
            (0, 22.5): 1,      # top
            (22.5, 67.5): 2,   # top-right
            (67.5, 112.5): 3,  # right
            (112.5, 157.5): 4, # bottom-right
            (157.5, 202.5): 5, # bottom
            (202.5, 247.5): 6, # bottom-left
            (247.5, 292.5): 7, # left
            (292.5, 337.5): 8  # top-left
        }

        posture_code = None
        for (low, high), code in directions.items():
            if low <= angle < high:
                posture_code = code
                break
        
        return posture_code if posture_code is not None else -1


    def begin_process(self,cam_datas,isCam,camera_number=None,lane = "Left"):
        cap = None
        tracking_cap = None
        self.FRAME_COUNTER = 0
        
        number_of_cams = len(cam_datas)
        number_of_videos = len(cam_datas[0])
        
        self.frame_number = 0
        program_start_time = datetime.now()
        for idx in range(number_of_videos): 
            
            
            video_paths = []
            for i in range(number_of_cams):
                cam_videos = self.get_all_videos(cam_datas[i][idx])
                video_paths.append(cam_videos)
            print(len(video_paths), ' is video len and len video_paths[0] ', len(video_paths[0]))
            video_dir = cam_datas[0][0]
            csv_file = "tracking_prediction.csv"
            DATE = video_dir.split("\\")[-2]+'_'+video_dir.split("\\")[-1]
            channel = video_dir.split("\\")[-3] #hourly
            model_name = self.model_path.split("\\")[-1]
            project = f'H:/Output/runs/{self.FARMNAME}_identification/ConvNext_ArcFace 22 Jan 2026/KNP/{model_name}/{channel}' 
            #project = f'H:/Output/runs/{self.FARMNAME}_identification/ConvNext_ArcFace Day DataCollection Jan 2026/KNP/{model_name}/{channel}' 
            name = f'{DATE}_part ' #weight path and iou threshold
            
            self.save_dir = Helper.increment_path(Path(project) / name,mkdir=True)
            csv_file = f'{self.save_dir}\\{csv_file}'
            save_vid_name = video_dir.split("\\")[-1]+"_identification" # open this when running multiple videos
            tracking_vid_name = video_dir.split("\\")[-1]+"_tracking" # open this when running multiple videos

            isFile = False
            if video_dir.endswith(".mp4") or video_dir.endswith(".mkv"):
                isFile = True
                save_vid_name=  video_dir.split('\\')[-1].replace('.mkv','_track').replace('.mp4','_track')  #open this when running single video
                tracking_vid_name=  video_dir.split("\\")[-1].replace('.mkv','_cow_only_track').replace('.mp4','_cow_only_track')  #open this when running single video


            save_vid_path = str(Path(os.path.join(self.save_dir, save_vid_name)).with_suffix('.mp4'))
            self.JSON_SAVE_PATH = str(Path(os.path.join(self.save_dir, self.JSON_FRAME_FILE_NAME)))
            tracking_vid_path = str(Path(os.path.join(self.save_dir, tracking_vid_name)).with_suffix('.mp4'))
            is_quit = 1
            self.boundary = self.save_width-10,self.save_height-10
            
            
            self.resolution = (1024,768) # with number of camera because of stacking
            self.isEatingArea = False
            self.save_width,self.save_height = self.resolution  #4, 5, 6 with tetris



            self.boundary = self.save_width,self.save_height
            self.average_size = 30000
            self.SMALL_SIZE = 6000 # need to use it in clear data
            self.border_threshold = 70
            self.BIG_SIZE = 700000
                
                
            
            tracking_cap = None #cv2.VideoWriter(tracking_vid_path,cv2.VideoWriter_fourcc(*'mp4v'), 5, (self.save_width,self.save_height))
            cap = cv2.VideoWriter(save_vid_path,cv2.VideoWriter_fourcc(*'mp4v'), 5, (self.save_width,self.save_height))
            
            csv_main_file_path = str(self.save_dir) + "\\main_csv.csv"
            
            print(save_vid_path)
            skip_vid = 0
            for i in range(len(video_paths[0])):
                videos = []
                for cam_no in range(number_of_cams):
                    
                    print(video_paths[cam_no][i], ' is video file')
                    videos.append(video_paths[cam_no][i])
                
                if(skip_vid > 0):
                    skip_vid -= 1
                else:
                    try:
                        is_quit = self.batch_frames_queue(
                            data=videos,
                            file_name=save_vid_path,
                            is_show_image=True,
                            cap=cap,                # your cv2.VideoWriter or None
                            tracking_cap=None,         # optional second writer
                            isFile=True,
                            isCam=False
                            )
                        print("Finished one batch is_quit:", is_quit )
                    except KeyboardInterrupt:
                        is_quit = -1
                        print("Keyboard interrupt: quitting now")
                if is_quit == -1:
                    break 
            self.clearData()
            if is_quit == -1:
                print("quitting")
                break
        
        total_duration = datetime.now() - program_start_time
        print(f"Total duration: {total_duration.total_seconds()}")
        self.save_final_pdf(self.save_dir,'finalized_pdf')
        if cap is not None:
            cap.release()
        if tracking_cap is not None:
            tracking_cap.release()
        self.clearData()
        print('done saving everything')
        if cap is not None:
            cap.release()
        if tracking_cap is not None:
            tracking_cap.release()
        
        cv2.destroyAllWindows()
        return is_quit


    def setup_distortion_settings():
        setting5_52 = CorrectDistortedImage().createSetting(k1=0.0,k2=0.0,k3=0.0,p1=0.0,p2=-0.005,focal_length_x=1.0,focal_length_y=1.0,center_x=0.5,center_y=1.0,scale=1.01,aspect=1.0)    
        setting5_53 = CorrectDistortedImage().createSetting(k1=0.8,k2=-1.0,k3=-0.98,p1=-0.036,p2=-0.029,focal_length_x=1.47,focal_length_y=1.47,center_x=0.36,center_y=0.7,scale=1.03,aspect=1.0)    
        setting5_54 = CorrectDistortedImage().createSetting(k1=0.64,k2=0.86,k3=1.0,p1=-0.039,p2=-0.012,focal_length_x=1.1,focal_length_y=1.59,center_x=0.51,center_y=0.82,scale=1.01,aspect=1.0)
        setting6_03 = CorrectDistortedImage().createSetting(k1=-0.12,k2=0.17,k3=-0.05,p1=0.007,p2=0.008,focal_length_x=1.39,focal_length_y=0.64,center_x=0.53,center_y=0.53,scale=1.01,aspect=1.0)
        setting003 = CorrectDistortedImage().createSetting(k1=1.0,k2=0.0,k3=0.0,p1=-0.042,p2=-0.069,focal_length_x=4.71,focal_length_y=1.72,center_x=0.72,center_y=0.73,scale=1.02,aspect=1.64)
        setting004 = CorrectDistortedImage().createSetting(k1=-0.03,k2=0.29,k3=-0.14,p1=-0.041,p2=-0.0,focal_length_x=0.85,focal_length_y=0.81,center_x=0.51,center_y=0.54,scale=1.01,aspect=1.0)
        settings = {}
        keys = ["5-52","5-53","5-54","-6-03"," 003"," 004"]
        settings['5-52'] = setting5_52
        settings['5-53'] = setting5_53
        settings['5-54'] = setting5_54
        settings['-6-03'] = setting6_03
        settings[' 003'] = setting003
        settings[' 004'] = setting004
        return keys,settings

    def setup_overlapping_settings():
        overlapping_settings = {}
        overlapping_settings['5-52'] = True
        overlapping_settings['5-53'] = True
        overlapping_settings['5-54'] = False
        return overlapping_settings
    STORED_IDS =[]
    STORED_MISS = []
    STORED_XYXY = []
    STORED_CENTROID = []
    STORED_SIZE = []
    CATTLE_LOCAL_ID = 0
    IOU_TH = 0.5

    HOKKAIDO = "HOKKAIDO"
    HONKAWA = "HONKAWA"
    SUMIYOSHI = "SUMIYOSHI"
    YOSHII = "YOSHII"
    KUNNEPPU = "KUNNEPPU"
    FARMNAME = KUNNEPPU
    LAST_Y=0
    TOTAL_MISSED_COUNT = 0
    middle = 0
    
    DATE = '2024-08-07'
    
    removeFilterAfter = 0
    cow_filter_size = []
    average_size = 30000
    SMALL_SIZE = 6000 # need to use it in clear data
    border_threshold = 80
    BIG_SIZE = 700000
    RESET_AVERAGE_SIZE_AFTER = 50 # first is 50, rest is is
    batch_size = 5 #identification batch
    total_predictions = 50
    identification_batch_size = batch_size 
    frame_skip_count = 3 if FARMNAME == SUMIYOSHI else 5  #skip 3 frame read 1 frame, 5fps in 20 fps video

    image_width = 1152
    image_height = 648

    camera_heights = []
    isEatingArea = False

    TRACKER = []  #To store tracking index to use in classifier
    CLASSIFICATION_TRACKER = [] #[{'GT': [1, 2, 3], 'COUNT': [10, 2, 5]}] retrieve Index , GT is ground truth, COUNT is number of apperance
    ALL_TIME_CLASSIFICATION_TRACKER = []
    BATCH_CLASSIFICATION_TRACKER = []
    TRACKING_RESULT = {} # [TrackingID, [values from each batch]]
    REIDENTIFY_MISSED_COUNT = 5 # re identify if there are missed frame
    RECHECK_COUNT = 5
    manual_cow_count = 0
    image_count = 1
    cow_count = 1
    isWholeDay = False
    is_quit = 1
    resolution = (1024, 768)
    save_width,save_height = resolution
    boundary = save_width,save_height
    save_dir = ""
    SIZE_224 = 224
    image_count = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame_number = 0
    TRACKING_MASK_LOCATION = []
   
    model_path ="models\\Side Lane August 2025\\Base_rtx8000_10_August_2025_20000_v1" #Sumiyoshi_dec16_v3_human_2024_10000_iters_v6
    class_names = []
    predictor_path = Path("identification_models_KNP") / "KNP_CONVNEXT_ARCFACE_09_Feb_Day" #KNP_CONVNEXT_Nighttime_Jan_2026_demopc
    #predictor_path = Path("identification_models_KNP") / "KNP_CONVNEXT_ARCFACE_22_Jan_2026_fixed_02" #KNP_CONVNEXT_Nighttime_Jan_2026_demopc

    NUM_CLASSES = None
    model = None
    infer = None
    val_transform = None
    stupid_detector = None
    class_names = []
    _models_ready = False

    def __init__(self):
        super().__init__()
        self._ensure_models_loaded()
        self.model = self.__class__.model
        self.infer = self.__class__.infer
        self.val_transform = self.__class__.val_transform
        self.stupid_detector = self.__class__.stupid_detector
        self.class_names = self.__class__.class_names
        self.NUM_CLASSES = self.__class__.NUM_CLASSES
        self.predictor_path = self.__class__.predictor_path

    @classmethod
    def _ensure_models_loaded(cls):
        if cls._models_ready:
            return

        device = cls.device
        print("using ", device)

        class_mapping_path = cls.predictor_path / "class_mappings.json"
        with open(class_mapping_path, "r", encoding="utf-8") as f:
            cls.class_names = json.load(f)
        cls.NUM_CLASSES = len(cls.class_names)

        model = timm.create_model(
            "convnextv2_base.fcmae_ft_in22k_in1k",
            pretrained=False,
            num_classes=0
        ).to(device)

        state = torch.load(cls.predictor_path / "best_backbone.pth", map_location=device)
        filtered_state = {
            k: v for k, v in state.items()
            if not k.startswith("head") and "classifier" not in k
        }

        model.load_state_dict(filtered_state, strict=False)
        model.eval()
        cls.model = model

        cls.infer = ProtoInfer(
            model,
            device,
            proto_ckpt_path=cls.predictor_path / "best_prototypes.pt", #"prototypes_single_masked_convnext_only.pt",
            class_mapping_path=class_mapping_path
        )

        cls.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
        ])

        cls.stupid_detector = cls.load_detector()
        cls._models_ready = True
    
    LAST_20_PATH = {}
     
    PASSING_CATTLE_BY_CAMERA = {} ## camera counter [0]=2 cattle 
    barrel_distortion_corrector = CorrectDistortedImage()

    TRACKER_clone = []
    ALL_TIME_CLASSIFICATION_TRACKER_clone = []
    
    camera_lists = {}

    tracking_by_predicted_id = {}
    colors = [(255,0,0),(0,255,0),(0,0,255),(0,0,0)]

    CAM_TRACKER = {} 

    full_body_image = {}
    JSON_FRAME_FILE_NAME = "frame_data.json"
    JSON_SAVE_PATH = ""
    FRAME_COUNTER = 0
    TRACKING_BINARY_MASKS = {} # TRACKING_BINARY_MASKS[tracking_id]: binary_mask}
    GAP_ROI = None
    CORRECT_ROI = None
    MERGE_ROI = None #[140 , 290 , 675, 610]
    NEED_PREPROCESS = True #unsharp and clache 
    #SKIP_PREPROCESS_FOR_CAM = ["002_E2","003_E2","005_E2","006_E2"] #"002_E2","003_E2","005_E2","006_E2"
    SKIP_PREPROCESS_FOR_CAM = ["001_E2","002_E2","003_E2","004_E2","005_E2","006_E2","007_E2"] #"002_E2","003_E2","005_E2","006_E2"

def draw_bounding_boxes_out(image, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return image

def draw_masks_out(image, masks):
    try:
        for mask in masks:
            color = (0, 255, 0)  # Green color for the mask
            print(image.shape, ' is image')
            print(mask.shape, ' is mask')
            image[mask] = color
    except Exception as ex:
        cv2.imshow("Error in mask", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image
if __name__ == '__main__':
    
    sleeper_divider = SleeperDivider()
    begin_identification = CATTLE_IDENTIFICATION()
    inputs = [
     
           
            [
                 ["H:\\Nyi Zaw Aung\\815_CowDataChecking\\KNP\\Standard_Format\\ALL_Channel_One_Day\\Ch001_Camera 001_E2\\2025-12-24\\0621"]
            ],
            
            [
                 ["H:\\Nyi Zaw Aung\\815_CowDataChecking\\KNP\\Standard_Format\\ALL_Channel_One_Day\\Ch002_Camera 002_E2\\2025-12-24\\0621"]
            ],
            # [
            #      ["H:\\Nyi Zaw Aung\\815_CowDataChecking\\KNP\\Standard_Format\\ALL_Channel_One_Day\\Ch003_Camera 003_E2\\2025-12-26\\0621"]
            # ],
            # [
            #      ["H:\\Nyi Zaw Aung\\815_CowDataChecking\\KNP\\Standard_Format\\ALL_Channel_One_Day\\Ch004_Camera 004_E2\\2025-12-26\\0621"]
            # ], 
            #  [
            #      ["H:\\Nyi Zaw Aung\\815_CowDataChecking\\KNP\\Standard_Format\\ALL_Channel_One_Day\\Ch005_Camera 005_E2\\2025-12-26\\0621"]
            # ],
            
            # [
            #      ["H:\\Nyi Zaw Aung\\815_CowDataChecking\\KNP\\Standard_Format\\ALL_Channel_One_Day\\Ch006_Camera 006_E2\\2025-12-26\\0621"]
            # ],
            # [
            #      ["H:\\Nyi Zaw Aung\\815_CowDataChecking\\KNP\\Standard_Format\\ALL_Channel_One_Day\\Ch007_Camera 007_E2\\2025-12-26\\0621"]
            # ],
            
            
           
            
    ]
    
 
    for camPath in inputs:
        
        first_frame = None
        #camPath = allCams[0]    
        # for camPath in allCams:
        #     print(camPath, ' is cam path')
        #     video_files = begin_identification.get_all_videos(camPath[0])
        #     if not video_files:
        #         print(f"No video files found in {camPath}")
        #         continue
        #     cam = video_files[0]
        #     cap = cv2.VideoCapture(cam)
        #     ret, first_frame = cap.read()
        #     first_frame = cv2.resize(first_frame, begin_identification.resolution)
        #     cap.release()
        #     if ret:
        #         break
        
        # first_frame = begin_identification.apply_clahe_ycrcb(first_frame, clip=2, tile=8)  #B
        # first_frame = begin_identification.unsharp_mask(first_frame, radius=1.2, amount=1.0)  #B
        
        begin_identification.GAP_ROI = sleeper_divider.get_gap_roi_by_camera(camPath[0])
        print("GAP ROI is ", begin_identification.GAP_ROI)

        w, h = begin_identification.resolution
        x10 = int(w*0.1)
        y10 = int(0.1 * h)
        x20 = int(0.9 * w)
        y20 = int(0.9 * h)
        begin_identification.CORRECT_ROI = np.array([
            [x10, y10],
            [x20, y10],
            [x20, y20],
            [x10, y20]
        ])

        if any(skip_cam in camPath[0] for skip_cam in begin_identification.SKIP_PREPROCESS_FOR_CAM):
            begin_identification.NEED_PREPROCESS = False
        else:
            begin_identification.NEED_PREPROCESS = True

        if "001_E2" in camPath[0]:
            y10 = int(0.2 * h) #bar area (under brush)
            y20 = int(0.83 * h) #occluded one
            begin_identification.CORRECT_ROI = np.array([
                [x10, y10],
                [x20, y10],
                [x20, y20],
                [x10, y20]
            ])

        elif "002_E2" in camPath[0]:
            
            begin_identification.MERGE_ROI = [int(w*0.14), int(0.38 * h), int(0.64 * w), int(0.75 * h)] #x1,y1,x2,y2
            x1, y1 = int(w*0.14), int(0.38 * h)
            x2, y2 = int(0.64 * w), int(0.75 * h)
            begin_identification.GAP_ROI = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.int32)
            y20 = int(0.85 * h)
            begin_identification.CORRECT_ROI = np.array([
                [x10, y10],
                [x20, y10],
                [x20, y20],
                [x10, y20]
            ])
        
        elif "003_E2" in camPath[0]:
            y20 = int(0.83 * h) # occluded one
            begin_identification.CORRECT_ROI = np.array([
            [x10, y10],
            [x20, y10],
            [x20, y20],
            [x10, y20]
        ])
        elif "005_E2" in camPath[0]:
            y20 = int(0.83 * h) # occluded one
            begin_identification.CORRECT_ROI = np.array([
            [x10, y10],
            [x20, y10],
            [x20, y20],
            [x10, y20]
        ])
        elif "006_E2" in camPath[0]:
            y10 = int(0.15 * h) #bar area (under brush)
            begin_identification.CORRECT_ROI = np.array([
                [x10, y10],
                [x20, y10],
                [x20, y20],
                [x10, y20]
            ])
        else:
            begin_identification.MERGE_ROI = None
        print("GAP ROI is ", begin_identification.GAP_ROI)
        print("MERGE ROI is ", begin_identification.MERGE_ROI)
        
        is_quit = begin_identification.begin_process(camPath,isCam= False,camera_number=None)
        if is_quit ==-1:
            break
        try:
            print("Beginning processing now")
        
        except Exception as ex:
            print("We got a problem in running sir, ",ex)
    print("Finished")
    
    
