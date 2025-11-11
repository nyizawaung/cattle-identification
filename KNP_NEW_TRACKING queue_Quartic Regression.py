'''
In centroid tracking, each cattle from top camera will be used to find the possible match
in the next camera by projecting the red circle with Quartic regression.
'''

import math
import timm
import torch, detectron2
import torch.nn as nn
import torch.optim as optim

from detectron2.utils.logger import setup_logger


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
#from torchvision.models import EfficientNet_V2_M_Weights
import cv2
import numpy as np
import os
from helpers.helper import Helper
#from batch_pipeline import PipelineMixin
from batch_pipeline_patched_new_tracking import PipelineMixin, STOP
#from batch_pipeline_patched_v3 import PipelineMixin, STOP
#from batch_pipeline_patched_single_cam_v2 import PipelineMixin, STOP

from pathlib import Path
#import time
#from ANNModel import ANN
from PIL import Image
from torchvision import datasets, transforms, models
#import torch.nn.functional as F
#from efficientNet import Efficient_Net as EN
from datetime import datetime
from collections import Counter
import csv
from PIL import Image
from math import sin, asin
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
    #def __init__(self):
    #EfficientNetV2-S (pretrained, without classification head)
    #effnet_model = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0)
    model = timm.create_model('resnet101', pretrained=True, num_classes=0)
    model.eval()

    #Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #Preprocessing
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
        # Extract query embedding
        query_embedding = self.extract_embedding(current_image)

        db_embeddings = []
        valid_paths = []

        for image in last_image_paths:
            try:
                #img = Image.open(path).convert("RGB")
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                emb = self.extract_embedding(img)
                db_embeddings.append(emb)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if not db_embeddings:
            return None, 0.0

        # Calculate cosine similarity
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
    def __init__(self, cam_counter, boxes,tracked_indexes, tracked_ids, predicted_ids, original_predicted_ids, colored_mask, TRACKING_PROJECTILE = None):
        self.cam_counter = cam_counter
        self.boxes = boxes
        self.tracked_indexes = tracked_indexes
        self.tracked_ids = tracked_ids
        self.predicted_ids = predicted_ids
        self.original_predicted_ids = original_predicted_ids
        self.colored_mask = colored_mask
        self.TRACKING_PROJECTILE = TRACKING_PROJECTILE
    
        # response_model = {
        #     "cam_counter":cam_counter,
        #     "tracked_indexes": tracked_indexes,
        #     "tracked_ids": tracked_ids,
        #     "predicted_ids": predicted_ids,
        #     "original_predicted_ids": original_predicted_ids,
        #     "colored_mask": colored_mask
        # }
    
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
        self.tau = float(ckpt["tau"])
        emb_dim = ckpt.get("embedding_dim", None)

        if emb_dim is not None and self.prototypes.shape[1] != emb_dim:
            raise ValueError(
                f"Embedding dim mismatch: ckpt={emb_dim}, prototypes D={self.prototypes.shape[1]}"
            )

        # ---- Load optional class mapping
        self.idx_to_name = None         # list or dict[int]->str
        self.label_to_name = None       # dict[str]->str

        if class_mapping_path is not None:
            with open(class_mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)

            # list -> index mapping
            if isinstance(mapping, list):
                self.idx_to_name = list(mapping)

            # dict -> either index->name or protoLabel->name
            elif isinstance(mapping, dict):
                # detect if keys are index-like (ints or numeric strings)
                def _is_int_like(k):
                    if isinstance(k, int): return True
                    if isinstance(k, str) and k.isdigit(): return True
                    return False

                if all(_is_int_like(k) for k in mapping.keys()):
                    # build int-keyed dict (normalize str keys -> int)
                    self.idx_to_name = {int(k): v for k, v in mapping.items()}
                else:
                    # assume keys are proto label strings
                    self.label_to_name = dict(mapping)

        # Precompute a set for quick membership
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
                # try index mapping first
                if self._have_idx_map and i in self.idx_to_name:
                    name = self.idx_to_name[i]
                else:
                    # try proto-label mapping
                    proto_lbl = self.proto_labels[i]
                    if self._have_lbl_map and proto_lbl in self.label_to_name:
                        name = self.label_to_name[proto_lbl]
                    else:
                        # fallback to original proto label
                        name = proto_lbl
                labels.append(name)
            else:
                labels.append("unknown")

        counts = Counter(labels)
        return [[k, v] for k, v in counts.items()]

# =========================
# EMBEDDINGS / PROTOTYPES
# =========================
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
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, ::-1]
            height, width = image.shape[:2]

            #image = self.aug.get_transform(image).apply_image(image)
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
                #yield from [self.__map_predictions(result['instances']) for result in results]
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
            # self.root = root
            # self.root.title("Enhanced Barrel Distortion Correction")
            # self.root.geometry("1200x800")  # Larger window for both images and controls

            # Initialize parameters
            self.image = None
            self.original_image = None 

            #5-52 template
            # self.k1 = -0.08  # primary radial distortion coefficient
            # self.k2 = 0.0  # secondary radial distortion coefficient
            # self.k3 = 0.0  # tertiary radial distortion coefficient
            # self.p1 = 0.0  # tangential distortion coefficient
            # self.p2 = 0.0  # tangential distortion coefficient
            # self.focal_length_x = 1.0  # focal length x-axis
            # self.focal_length_y = 1.0  # focal length y-axis
            # self.center_x = 0.43  # center x (relative to width)
            # self.center_y = 0.02  # center y (relative to height)
            # self.scale = 1.01  # scale factor for the corrected image
            # self.aspect = 1.0  # aspect ratio adjustment

            # self.k1 = 0.03  # primary radial distortion coefficient
            # self.k2 = 0.0  # secondary radial distortion coefficient
            # self.k3 = 0.0  # tertiary radial distortion coefficient
            # self.p1 = 0.0  # tangential distortion coefficient
            # self.p2 = 0.0  # tangential distortion coefficient
            # self.focal_length_x = 1.0  # focal length x-axis
            # self.focal_length_y = 1.0  # focal length y-axis
            # self.center_x = 0.65  # center x (relative to width)
            # self.center_y = 0.0  # center y (relative to height)
            # self.scale = 1.01  # scale factor for the corrected image
            # self.aspect = 1.0  # aspect ratio adjustment
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
            #print(settings)
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Calculate center point in pixels
            center_x = width * settings['center_x']
            center_y = height * settings['center_y']
            
            # Adjust focal length based on image dimensions
            focal_x = settings['focal_length_x'] * width
            focal_y = settings['focal_length_y'] * width * settings['aspect']
            
            # Create camera matrix
            camera_matrix = np.array([
                [focal_x, 0, center_x],
                [0, focal_y, center_y],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Create distortion coefficients array [k1, k2, p1, p2, k3]
            distortion_coeffs = np.array([settings['k1'], settings['k2'], settings['p1'], settings['p2'], settings['k3']], dtype=np.float32)
            
            # Get optimal new camera matrix
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                camera_matrix, distortion_coeffs, (width, height), 1, (width, height))
            
            # Scale the new camera matrix
            new_camera_matrix[0, 0] *= settings['scale']  # Scale fx
            new_camera_matrix[1, 1] *= settings['scale']  # Scale fy
            
            # Undistort the image
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



#torch.cuda.empty_cache()
class CATTLE_IDENTIFICATION(PipelineMixin):
    def load_detector():
        # Other initialization code...
        
        # Pre-initialize the predictor once
        detection_model = 'models\\Side Lane August 2025\Base_rtx8000_10_August_2025_20000_v1/model_best.pth'
        model_config = 'models\\Side Lane August 2025\Base_rtx8000_10_August_2025_20000_v1/config.yml'
        
        cfg = get_cfg()
        cfg.merge_from_file(cfg_filename=model_config)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6 #0.6
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3 #0.3
        
        cfg.MODEL.WEIGHTS = detection_model
        
        print("Initializing model")
        # Set device explicitly
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return BatchPredictor(cfg, batch_size=8, workers=0)
        #print(self.predictor, ' is predictor')

    # Step 1: Set up data transformations
    data_transforms = {
        'train': transforms.Compose([
        #    transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
        #   transforms.Resize(256),
        #   transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    # Step 8: Prediction function
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

        # Preprocess all images
        #processed_images = [self.preprocess_image(img) for img in images]
        
        # Stack images into a single tensor
        image_tensor = torch.stack(images).to(self.device)

        with torch.no_grad():
            # Get model predictions
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probabilities, 1)

            # Filter predictions based on threshold
            filtered_classes = [
                self.class_names[str(pred.item())] 
                for pred, prob in zip(preds, max_probs) 
                if prob.item() >= threshold
            ]

            # Count occurrences of filtered classes
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
        
        # Preprocess image
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
        # Create model architecture
        # model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=classLen)
        # model.load_state_dict(torch.load(model_path, map_location=device))
        # model.eval()
        
        model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=False, num_classes=classLen)
         # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        
        return model.to(device)

    def _load_model_resnet(classLen, model_path,device):
        # Create model architecture
        model = timm.create_model('resnet101', pretrained=False)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, classLen)
        )
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model.to(device)
    
    def get_files_from_folder(self,path, limitX=20):
    # Get the list of files in the directory
        files = os.listdir(path)
        
        # Use heapq.nlargest to get the last 'limitX' largest numeric files without sorting everything
        largest_files = heapq.nlargest(limitX, files, key=lambda x: int(x.split('.')[0]))  # Assumes numeric filenames
        
        return np.asarray(largest_files)
    
    def get_files_from_folder_scan_sort(self,path, limitX=20):
    # Use os.scandir() for faster directory scanning
        with os.scandir(path) as entries:
        # Collect all filenames into a list
            files = [entry.name for entry in entries if entry.is_file()]
            
            # Sort the files based on the numeric part of the filenames (assuming filenames contain numbers)
            files.sort(key=lambda x: int(x.split('_')[0]), reverse=True)  # Adjust as needed for your filenames
            
            # Return the top 'limitX' files
            return files[:limitX]

    def batch_identification(self,save_dir,tracking_id,FORCE_MISSED_DETECTION = False,path = None):
        #start_time = datetime.now()
        counter = 20
        save_path = path
        #global batch_size 
        if path is None:
            save_path = f"{save_dir}//{tracking_id}"
        batch_size = self.identification_batch_size
        #images = self.get_files_from_folder_scan_sort(save_path,batch_size)
        images = self.get_last_X_crops(tracking_id,batch_size)
        #print('images: for tracking_id ',tracking_id , " are ", images)
        stacked_images = []
        if images is None:
            return None
        for image in images:
            #if counter < 1:
                #break
            #img_path = os.path.join(save_path, image)
            #frame = cv2.imread(image)
            #predict_cattle_id(frame)
            # Ensure the image is in RGB format
            
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Preprocess the image using the defined transforms
            input_tensor = self.data_transforms['validation'](frame).unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(self.device)

            # Append the processed image tensor to the list
            stacked_images.append(input_tensor[0])  # Append the first element to keep it as (C, H, W)
            #counter -= 1
        
        # Call predict_stack_cattle_id with the stacked_images
        #return self.predict_stack_cattle_id(stacked_images)
        #return self.predict_stack_cattle_id(stacked_images)
        return self.infer.predict_stack(stacked_images,0.25)
        #print("print total duration :", datetime.now()-start_time ) 
        #return 
    # Replace the path with your actual path
    #batch_size = 50
    #batch_identification("a",1,r"E:\Nyi Zaw Aung\Python\detectron2\Identification_dataset\Sumiyoshi-37\validation\M105")
    #batch_identification("a",1,r"E:\Nyi Zaw Aung\Python\detectron2\Identification_dataset\Random")
    def calculate_center_of_box(self,box):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        cx = int(x1) + int((x2-x1)/2)
        cy = int(y1) + int((y2-y1)/2)
        print(f"cx : {cx}, cy : {cy}")
        # match corner:
        #     case BoxCorner.CENTER: return (cx, cy)
        #     case BoxCorner.TOP_LEFT: return (x1, y1)
        #     case BoxCorner.TOP_RIGHT: return (x2, y1)
        #     case BoxCorner.BOTTOM_LEFT: return (x1, y2)
        #     case BoxCorner.BOTTOM_RIGHT: return (x2, y2) 

    #yellow = (0, 255, 255)
    #cam_10 = [(300,100), (550,100), (800,100), (1050,100), (1300,100), (1550,100),]

    def draw_circle(self,cv2, center_coordinate, image, color):
        cv2.circle(image, center_coordinate, radius=3, color=color, thickness=-1)
    
    def get_color(self,tracking_id):
        #get index of tracking_id
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
    # Function to draw a bounding box and annotate the image
    def draw_bounding_box(self,image, box, label,tracking_id,font_scale = 2,color =(0, 255, 0)):
        # Extract the coordinates from the box
        x1, y1, x2, y2 = box
        #print(x1,' ',y1,x2,y2)

        # Draw the bounding box rectangle on the image
        #get_color(tracking_id)
        #cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        #cv2.rectangle(image, (x1, y1), (x2, y2), self.get_color(tracking_id), 2)
        #text = f'{tracking_id}'
        # Define the text properties
        # if label == -1:
        #     label = '-1'
        if label == 'Identifying' or label == -1:
            label = f'Tracking'
        else:
            label = f'{label}'
        text = label
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        thickness = 2

        # Calculate the size of the text
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate center of the box
        center_x = x1 + (x2 - x1) // 2
        center_y = y1 + (y2 - y1) // 2

        # Adjust text position so it's centered
        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        
        # Put the label text on the image (no background)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # tracking label
        #size down
        #diagonal = self.CALCULATE_EUCLIDEAN_DISTANCE((x1,y1),(x2,y2))
        
        #text = f'd_{diagonal:.2f}'
        # if label == 'Identifying' or True:
        #     text = f'{tracking_id}'
        #     #font_scale = font_scale 
        #     (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        #     text_x = x2 - text_width #right side
        #     text_y = y1 if y1 >= 20 else y1 + 10 + text_height #same height 
            
        #     cv2.rectangle(image, (text_x, text_y - text_height - 10), (text_x + text_width, text_y), color, -1)

        #     # Put the label text on the image
        #     cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Example usage
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
        # color = color[::-1]
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
        #print(x1, '  ',y1, '  ',x2, '  ',y2, '  ',h, '  ',w)
        #if(x1<int(X1*(w/default)) or x2>int(X2*(w/default)) or y1<int(Y1_NEW*(h/default)) or y2>int(Y2_NEW*(h/default)) or x1>=int(X2*(w/default))):
        #    return False
        print (x2-x1,' is cow width', x2, ' - ', x1 )
        if y1 < 250 or y2 > 3600 : 
            print('I am outside of the ROI')
            return False
        #print(y1, " y1 and y2 ",y2) 
        if(y2 - y1>2400 or y2-y1<1000): #1400 to 700 Beforee
            print(y2-y1, ' I am too big')
            return False
        #if ( x1 < 45 * (w/429) or x2 > 500 * (w/429)): #measured on 429 x 724 size
        #    print(x1 ,'< ',50 * (w/429),' or ',x2 ,'>', 500 * (w/429))
        #    return False
        
        #if(x2-x1 > 500)
        return True 

    def HOKKAIDO_ROI(self,x1,y1,x2,y2,h,w):
        #print(x1, '  ',y1, '  ',x2, '  ',y2, '  ',h, '  ',w)
        
        X1=200 #same as NEW_BLACK_X1
        X2=400 #same as NEW_BLACK_X2 # incase of x2 out of bound
        
        Y1_NEW=50#125  #decrease here to extend, increase to shrink 
        Y2_NEW=630  #5
        default = 640
        X1
        if(x1<int(X1*(w/default)) or x2>int(X2*(w/default)) or y1<int(Y1_NEW*(h/default)) or y2>int(Y2_NEW*(h/default)) or x1>=int(X2*(w/default))):
            return False
        #if(y2 - y1>1500 or y2-y1<700): #1400 to 700 Before
        #    return False
        return True  

    def FilterSize(self,y1,y2,h25,area,h75,max_freq_pos):
        print('Filter size y2 and y1 :->',y2,' - ' ,y1 )
        #if y1 >= h25*1.5 and y2 <= h75 and (y2-y1<700 or area < 80000) : #within specific range # old camera
        if y1 >= h25*1.5 and y2 <= h75 and area < 80000 : #within specific range  # new camera
            print(y2-y1,' size skipped due to',y2,' - ' ,y1 )
            return False
        #elif y2>h75 and max_freq_pos>=50 and max_freq_pos<=100:
        #    return False
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
        #print('width ',w,' height ',h)
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
        #area = w * h
        #print('total area ',area)
        prev_area = self.CalculateBoxArea(prev)
        current_area = self.CalculateBoxArea(current)
        difference = abs(prev_area - current_area)
        #print('position 1',prev, 'position 2',current)
        #print('prev position, current position',prev_area,current_area)
        #return area / CalculateBoxArea(prev)
        #print('two box area ',prev_area,' ',current_area)
        prev_area = current_area if prev_area > current_area else prev_area
        
        if prev_area / 4 >= difference : # if size is not too different than previous
            return area/prev_area   
        return 0    
    #    iou = area / float(prev_area + current_area - area)
        
        #return io
        
    
    def IOU_Tracking(self,x1,y1,x2,y2):
        
        is_new = True
        current_id = -1
        global LAST_Y
        is_Moving = True
        if len(self.STORED_IDS)>0:
            LAST_Y = self.STORED_XYXY[-1][1] #x1y1x2y2
        
        #region dynamnic threshold
        dynamic_IOU = self.IOU_TH
        #middle = h*0.25
        if y1 < self.middle and False:
            if y1<(self.middle/2):
                dynamic_IOU += (1-self.IOU_TH) * ((self.middle-y1) / (self.middle-30))  #dynamic IOU threshold
            else : 
                dynamic_IOU += (1-self.IOU_TH) * ((self.middle-y1) / (self.middle))  #dynamic IOU threshold
            print('dynamic_threshold is :',dynamic_IOU)
        #endregion dynamic threshold
        closet = -1
        closet_iou = 0
        for i in range(len(self.STORED_IDS)):
            iou =self.CalculateIOU(self.STORED_XYXY[i],[x1,y1,x2,y2])
            #print(iou,' is iou percentage')
            #print(y1,' is current y1 position')
            if is_new and iou > dynamic_IOU :    
                is_new = False
                closet = i
                #continue
                break
                if iou > closet_iou:
                    closet_iou = iou
                    closet = i
        
        if closet>-1 :
            #print(closet, ' I am the chosen one!!!')
            self.STORED_XYXY[closet] = [x1,y1,x2,y2]
            current_id = str(self.STORED_IDS[closet])
            self.TOTAL_MISSED_COUNT += self.STORED_MISS[closet] - 1
            self.STORED_MISS[closet] = 1
            is_Moving = closet_iou < 90
                    
        if is_new:
            #if y1 < LAST_Y or y1 < middle+300: #new cattle is above the last cattle, means Id 5 is in front of 4. 
            #    print(y1 ,' is current y1 and last y1 is: ',LAST_Y,' middle is ',middle-300)
            #    return -1
            #print('Adding new cattle')
            self.CATTLE_LOCAL_ID+=1
            self.STORED_IDS.append(self.CATTLE_LOCAL_ID)
            self.STORED_MISS.append(1)
            self.STORED_XYXY.append([x1,y1,x2,y2])
            current_id = str(self.CATTLE_LOCAL_ID)
        
        
        
        return [current_id],is_Moving



    def CALCULATE_EUCLIDEAN_DISTANCE(self,prev_centroid, current_centroid):
        #print(prev_centroid)
        #print(current_centroid)
        x_sq = (prev_centroid[0] - current_centroid[0]) ** 2
        y_sq = (prev_centroid[1] - current_centroid[1]) ** 2
        distance = ( x_sq + y_sq ) ** 0.5
        #print(distance)
        return distance
    def Centroid_Tracking(self,mask,w,h,cam_counter,y1,y2):
        #PROJECTILE_TRACKING['1']=[{'id':1, 'x':100,'y':100},{'id':2, 'x':200,'y':200}] // this format
        tracking_missed_frame = 1
        is_new = True
        current_id = -1
        #if len(self.STORED_IDS)>0:
        #    LAST_Y = self.STORED_XYXY[-1][1] #x1y1x2y2
        closet = -1
        closet_iou = 0
        threshold = (w/2) if w<=h else (h/2)
        threshold = max(threshold, 55)
        #threshold = threshold * 0.8
        #threshold = threshold + 10 # 30 is original
        current_point = self.find_mask_center(mask,cam_counter)
        least_missed_frame = 100
        closed_distance_new = 1000
        closed_distance_i = -1
        # for i in range(len(self.STORED_IDS)):
        #     distance = self.CALCULATE_EUCLIDEAN_DISTANCE(self.STORED_XYXY[i],current_point)
        for i, stored_point in enumerate(self.STORED_XYXY):
            #print(f"stored_point: {stored_point}, current_point: {current_point}")
            distance = self.CALCULATE_EUCLIDEAN_DISTANCE(stored_point, current_point)    
            if distance <= threshold :    
                #continue
        #       is_new = False
        #       closet = i
        #       break
                if distance < threshold/2: #distance is very close
                    is_new = False
                    closet = i    
                    #is_moving = distance > threshold / 3
                    
                    break
                elif distance > closet_iou:
                    closet_iou = distance
                    closet = i
                
            # elif distance < closed_distance_new:
            #   closed_distance_new = distance
            #   closed_distance_i = i   
        
        updated_point = None
        
        if cam_counter == 0:
            target_cam = 1
            if current_point[1] > 227:
                #updated_point = current_point[0], self.Calculate_Projectile_Up(current_point[1])
                updated_point = current_point[0], self.Calculate_Projectile_From_Top(current_point[1])
        else:
            target_cam = 0
            updated_point = None
            #updated_point = current_point[0], self.Calculate_Projectile_Down(current_point[1])

        # if updated_point is not None and updated_point[1] is not None:
        #     if target_cam in self.PROJECTILE_TRACKING:
        #         self.PROJECTILE_TRACKING[target_cam]=[{updated_point}]
        #     else:
        #         print(self.PROJECTILE_TRACKING)
        #         self.PROJECTILE_TRACKING[target_cam].append({updated_point})

        if closet>-1 :
            #print(closet, ' I am the chosen one!!!')
            
            current_id = str(self.STORED_IDS[closet])
            self.TOTAL_MISSED_COUNT += self.STORED_MISS[closet] - 1
            tracking_missed_frame = self.STORED_MISS[closet]-1
            #if tracking_missed_frame > 2:
            #    print(f"{current_id} has {tracking_missed_frame} missed frames")
            self.STORED_XYXY[closet] = current_point
            self.STORED_MISS[closet] = 1
            is_new = False

        #elif closed_distance_i != -1: 
        #    print(" I am closed distance with ", str(self.STORED_IDS[closed_distance_i]), " ad distance is ", closed_distance_new) 
        return [current_id,current_point,tracking_missed_frame,is_new,target_cam,updated_point]
    def Calculate_Projectile_Down(self,y):
        #print(y, " from up")
        y_up = 60 * sin(0.012 * y - 1.38) + 337
        return y_up
    
    def Calculate_Projectile_From_Top(self,x, y_min=0.0, y_max=768.0): # find height in bottom image
        COEFFS = np.array([
            -1.97452e-7,   # a4
            4.38101e-4,   # a3
            -3.54014e-1,   # a2
            1.2405296e2,  # a1
            -1.5834903e4   # a0
        ], dtype=float)
        """
        Predict bottom-image y from top-image x using the quartic model.
        Supports scalar or array-like x. Returns float or np.ndarray.
        """
        x = np.asarray(x, dtype=float)
        # Horner's method for stability
        y = (((COEFFS[0]*x + COEFFS[1])*x + COEFFS[2])*x + COEFFS[3])*x + COEFFS[4]
        #if clip_to_image:
        y = np.clip(y, y_min, y_max)
        return float(y) if y.ndim == 0 else y

    def Calculate_Projectile_Up(self,y):
        #print(y, " from down")
        try:
            y_down = (1.38 + math.asin((y - 337) / 60)) / 0.012
            
            return y_down
        except:
            return None

    def Centroid_Tracking_PerCamera(self,mask,w,h,cam_counter,y1,y2,frame):
        
        tracking_missed_frame = 1
        is_new = True
        current_id = -1
        
        closet = -1
        closet_iou = 0
        threshold = (w/2) if w<=h else (h/2)
        #threshold = threshold * 0.8
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
                        #is_moving = distance > threshold / 3
                        
                        break
                    elif distance > closet_iou:
                        closet_iou = distance
                        closet = tid

            if closet!=-1 :
                #print(closet, ' I am the chosen one!!!')
                
                current_id = str(closet)
                self.CAM_TRACKER[cam_counter][closet]['tracking_xyxy'] = current_point
                self.TOTAL_MISSED_COUNT = self.CAM_TRACKER[cam_counter][closet]['tracking_missed_frame'] - 1
                #if tracking_missed_frame > 2:
                #    print(f"{current_id} has {tracking_missed_frame} missed frames")
                self.CAM_TRACKER[cam_counter][closet]['tracking_missed_frame'] = 1
                is_new = False

                #region update tracking
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
        #threshold = threshold * 0.8
        current_point = self.find_mask_center(mask,cam_counter)
        least_missed_frame = 100
        for i in range(len(self.STORED_IDS)):
            distance = self.CALCULATE_EUCLIDEAN_DISTANCE(self.STORED_CENTROID[i],current_point)
            
            if is_new and distance <= threshold :    
                #continue
        #       is_new = False
        #       closet = i
        #       break
                if distance < threshold/2: #distance is very close
                    is_new = False
                    closet = i    
                    #is_moving = distance > threshold / 3
                    
                    break
                elif distance > closet_iou:
                    closet_iou = distance
                    closet = i

        
        if closet>-1 :
            #print(closet, ' I am the chosen one!!!')
            self.STORED_CENTROID[closet] = current_point
            current_id = str(self.STORED_IDS[closet])
            self.TOTAL_MISSED_COUNT += self.STORED_MISS[closet] - 1
            tracking_missed_frame = self.STORED_MISS[closet]-1
            #if tracking_missed_frame > 2:
            #    print(f"{current_id} has {tracking_missed_frame} missed frames")
            self.STORED_MISS[closet] = 1
            is_new = False
        
        #else: #is it old from previous tracking

   
            
        return [current_id,current_point,tracking_missed_frame,is_new]
    def addNewTracking_v2(self,mask,current_point,x1,y1,x2,y2,cam_counter,w,h):
        # tracking_id,is_touching_bottom,is_touching_top = self.find_matching_mask(mask,current_point,cam_counter,y1,y2)
        # if tracking_id != "-1":
            
        #     self.update_tracking_mask_location(tracking_id,mask,cam_counter,current_point,is_touching_top,is_touching_bottom)
        #     idx = self.STORED_IDS.index(int(tracking_id))
        #     self.STORED_XYXY[idx] = current_point

        #     self.STORED_MISS[idx] = 1
        #     print("found closed id :",tracking_id)
        #     return tracking_id

        
        
        self.CATTLE_LOCAL_ID+=1
        self.STORED_IDS.append(self.CATTLE_LOCAL_ID)
        self.STORED_MISS.append(1)
        self.STORED_XYXY.append(current_point)
        current_id = str(self.CATTLE_LOCAL_ID)
        self.STORED_SIZE.append(w*h)
        #print(f"current new id {current_id}")

        # info = MASKLOCATION(current_id,mask,cam_counter,current_point,is_touching_top,is_touching_bottom)
        # self.TRACKING_MASK_LOCATION.append(info)
        return current_id

    def addNewTracking_PerCamera(self,mask,current_point,x1,y1,x2,y2,cam_counter,w,h):
        self.CATTLE_LOCAL_ID+=1
        track_info = {
               "tracking_xyxy" : current_point,
               "tracking_missed_frame" : 1}
        self.CAM_TRACKER[cam_counter][self.CATTLE_LOCAL_ID] = track_info
        current_id = str(self.CATTLE_LOCAL_ID)
        #region update tracking
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
        #print(f"current new id {current_id}")
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
        #print(image_height, " is image height")
        return abs(y2 - (image_height - 1)) <= threshold

    def is_touching_top(self,y1,frame, threshold=10):
        return y1 <= threshold

    def update_tracking_mask_location(self, tracking_id, mask, cam_counter,current_point, is_touching_top, is_touching_bottom):
        
        for m in self.TRACKING_MASK_LOCATION:
            if m.tracking_id == tracking_id:
                m.is_touching_bottom = is_touching_bottom
                #print("Updating tracking id:", tracking_id, "in cam ", m.cam_counter, ' to cam ', cam_counter)

                m.cam_counter = cam_counter
                m.mask = mask
                m.is_touching_top = is_touching_top
                m.current_point = current_point
                m.last_mask_area = np.sum(mask)
                if (not is_touching_top and not is_touching_bottom and m.mask_area < m.last_mask_area) or m.mask_area < m.last_mask_area:# or m.mask_area is null :
                    m.mask_area = m.last_mask_area

                
     
    def remove_tracking_in_other_camera(self,tracking_id, cam_counter):
        for cam_Data in self.CAM_TRACKER.values():
            if tracking_id in cam_Data:
                if cam_Data[tracking_id]['cam_counter'] != cam_counter:
                    del cam_Data[tracking_id]
                    print(f"Removed tracking {tracking_id} from camera {cam_Data[tracking_id]['cam_counter']}")
        
            

    def set_touching_cattle_count_per_cam(self):
        if m.is_touching_bottom:
            self.has_passing_cattle_camera(m.cam_counter)
            self.PASSING_CATTLE_BY_CAMERA[m.cam_counter] += 1
        if m.is_touching_top and m.cam_counter > 1:
            self.has_passing_cattle_camera(m.cam_counter-1)
            self.PASSING_CATTLE_BY_CAMERA[m.cam_counter-1] -= 1
    def has_passing_cattle_camera(self, cam_counter):
        if cam_counter not in self.PASSING_CATTLE_BY_CAMERA:
            self.PASSING_CATTLE_BY_CAMERA[cam_counter] = 0
    def get_touching_cattle_count_per_cam(self, cam_counter):
        if cam_counter in self.PASSING_CATTLE_BY_CAMERA:
            return self.PASSING_CATTLE_BY_CAMERA[cam_counter]
        return 0
    
    #compare how many pixels are overlapped from stroedX1x2 and currentX1x2
    def isOverlappedXs_original(self,first, second, threshold = 0.1):
        first_x1 = first[0]
        first_x2 = first[1]
        second_x1 = second[0]
        second_x2 = second[1]
        #print('first_x1 ',first_x1, ' first_x2 ',first_x2, ' second_x1 ',second_x1, ' second_x2 ',second_x2)
        min_length = min((first_x2 - first_x1), (second_x2 - second_x1))
        percentage = 0
        threshold = 0
        #print(first_x1, ' ',first_x2, ' ',second_x1, ' ',second_x2)
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

    #divide the longer length by the shorter length
    def overLappedPercentage_original(self,length1, length2):
        
        percentage = 0
        if length1 > length2:
            percentage = length2 / length1
        else:
            percentage = length1 / length2
        #print("checking overlapped percentage length1 ",length1, ' length2 ',length2, " percentage : ",percentage)
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

        #save_path = f"{self.save_dir}//{tracking_id}"
        
        #images = self.get_files_from_folder_scan_sort(save_path,10)
        images = self.get_last_X_crops(tracking_id,20)
        
        if images is None:
            return None
        images = images[5:15] # 10 images from 5th to 15th
        match_path, score = self.similarity_checker.cosine_similarity_search(images, pil_img)
        #print(f"Match: {match_path} with score {score:.2f}")
        if score > 0.7:
            print(f"✅ Match with score {score:.2f}")
            return True
        else:
            print(f"❌ Not similar with score {score:.2f}") #print(f"Deleting tracking {self.STORED_IDS[i-removed]} due to missed count {missed}")
            return False
    
    def align_centroid_per_camera(self,current_point,cam_counter):
        # Ensure mask is binary
        print(current_point)
        x,y = current_point
        y += (self.image_height * cam_counter) #adjust for multi cam
        
        return (x,y)

    #for curly version. 220 and 875 for straighten version
    def find_cattle_location(self,x1,x2):
        if x1 < 150:
            return "LEFT"
        elif x2 > 900:
            return "RIGHT"
        else:
            return "MIDDLE"

    def find_matching_mask(self, current_mask,current_point, cam_counter,tracking_id,x1,y1,x2,y2,frame, threshold=0.1):
        closed_tracking = tracking_id
        #return closed_tracking,False,False
        #if not is_mask_touching:
        #    print(f" skipping {tracking_id} due to not touching")
        #    return closed_tracking,False,False #if not touching, return tracking id
        is_touching_bottom = self.is_touching_bottom(y2, cam_counter)
        is_touching_top = self.is_touching_top(y1, frame)
        #if tracking_id != "-1":
            #print("tracking_id {}")
        return closed_tracking,is_touching_bottom,is_touching_top
        distance_threshold = 100
        check_camera = cam_counter + (0 if is_touching_bottom else - 1)
        check_camera = max(check_camera,0)
        is_overlapped = False
        cattle_position = self.find_cattle_location(x1,x2)
        mask_area_threshold = 35000
        #if x1 > 
        if is_touching_bottom:
            
            target_counter = cam_counter + 1
            line_info = self.get_lowest_y(current_mask)
            #print('current lowest line is :',line_info)
            if line_info is None:
                return closed_tracking,is_touching_bottom,is_touching_top

            #touching_cattle_count = self.get_touching_cattle_count_per_cam(cam_counter) #if touching bottom, take current
            #is_cam_overlapped = 
            #print("touching bottom, finding in camera :", target_counter)
            for m in self.TRACKING_MASK_LOCATION:
                #if m.cam_counter == cam_counter:
                #    continue
                if m.cam_counter == target_counter and m.is_touching_top:
                    other_line = self.get_highest_y(m.mask)
                    if other_line is None:
                        #print(f"No highest line found for tracking ID {m.tracking_id} in camera {m.cam_counter}")
                        continue
                    isOverlapped, percentage = self.isOverlappedXs([line_info["x1"],line_info["x2"]], [other_line["x1"],other_line["x2"]],threshold)
                    distance = 1000
                    if other_line is not None and isOverlapped:
                        centroid = m.current_point
                        #print(f"{tracking_id} overlapp percentage with _tracking Id touching bot : {m.tracking_id} is {percentage} in cam {m.cam_counter}")
                        if percentage > 0.3:
                            mask_area = np.sum(current_mask)
                            
                            total_area = mask_area + m.last_mask_area
                            
                            if total_area > max( mask_area_threshold, 1.3 * (m.mask_area)):
                                print(f"{m.tracking_id} : last_mask : {m.last_mask_area} , m.mask_area : {m.mask_area}")
                                continue
                            else:
                                return m.tracking_id,is_touching_bottom,is_touching_top
                    #else:
                        #print(f"{tracking_id} overlapp percentage with _tracking Id touching top : {m.tracking_id} is {percentage} in cam {m.cam_counter}")

        elif is_touching_top:
            target_counter = cam_counter - 1
            if target_counter < 0:
                return closed_tracking,is_touching_bottom,is_touching_top

            line_info = self.get_highest_y(current_mask)
            #print('current highest line is :',line_info)
            if line_info is None:
                print("No line found in current mask")
                return closed_tracking,is_touching_bottom,is_touching_top
            #touching_cattle_count = self.get_touching_cattle_count_per_cam(target_counter) #if touching top, take previous
            #print("touching top, finding in camera :", target_counter)
            for m in self.TRACKING_MASK_LOCATION:
                #if m.cam_counter == cam_counter:
                #    continue
                if m.cam_counter == target_counter and m.is_touching_bottom:
                    other_line = self.get_lowest_y(m.mask)
                    if other_line is None:
                        #print(f"No lowest line found for tracking ID {m.tracking_id} in camera {m.cam_counter}")
                        continue
                    isOverlapped, percentage = self.isOverlappedXs([line_info["x1"],line_info["x2"]], [other_line["x1"],other_line["x2"]],threshold)
                    distance = 1000
                    if other_line is not None and isOverlapped:
                        #print(f"{tracking_id} overlapp percentage with _tracking Id touching top : {m.tracking_id} is {percentage} in cam {m.cam_counter}")
                        if percentage > 0.3:
                            mask_area = np.sum(current_mask)
                            
                            total_area = mask_area + m.last_mask_area
                            
                            if total_area > max( mask_area_threshold, 1.3 * (m.mask_area)):
                                print(f"{m.tracking_id} : last_mask : {m.last_mask_area} , m.mask_area : {m.mask_area}")
                                #print(f"Overlapped {m.tracking_id} : {is_overlapped}, current mask area: {mask_area}, other mask area: {other_mask_area}, total area: {total_area}")
                                continue
                            else:
                                return m.tracking_id,is_touching_bottom,is_touching_top
                    #else:
                        #print(f"{tracking_id} overlapp percentage with _tracking Id touching top : {m.tracking_id} is {percentage} in cam {m.cam_counter}")
                    #if touching_cattle_count <= 1 and distance <distance_threshold + 100:
                        #print("Returning from touching top with cam_counter ",target_counter)
                    #    return m.tracking_id, is_touching_bottom,is_touching_top
        return closed_tracking,is_touching_bottom,is_touching_top
    
    def delete_mask_by_tracking_id(self, ids):
        #print(f"Deleting masks with tracking IDs: {ids}")
        deleted_ids = []
        for id in ids:
            self.delete_last_20_path(id)
        for m in self.TRACKING_MASK_LOCATION:
            if m.tracking_id in ids:
                deleted_ids.append(m.tracking_id)
                print(f"Deleting mask with tracking ID: {m.tracking_id}")
                self.TRACKING_MASK_LOCATION.remove(m)
        #if len(deleted_ids) != len(ids):
        #    print(f"Warning: Not all masks were deleted. Deleted IDs: {deleted_ids}, Requested IDs: {ids}")
            #for m in self.TRACKING_MASK_LOCATION:
                #print("Remaining mask with tracking ID:", m.tracking_id)


    def IncreaseMissedCount(self,tracking_ids):
        ids = []
        if(len(self.STORED_IDS)>0): 
            total_length = len(self.STORED_IDS)
        
            removed = 0
            threshold = 30
            for i in range(total_length):
                #print(i, ' ',len(STORED_MISS))
                if self.STORED_IDS[i-removed] not in tracking_ids: #didn't detected in this frame
                    self.STORED_MISS[i-removed]+=1
                    missed = self.STORED_MISS[i-removed]
                    #print(f"Increasing missed count for tracking {self.STORED_IDS[i-removed]}, missed count is now {missed} in frame {current_frame}")
                    #threshold = 5 if STORED_XYXY[i-removed][1]<= middle else 30 # if y1<= middle then above so miss count is less
                    
                    if missed>threshold: #if missed 3 frames
                        #deleted_id = self.STORED_IDS[i-removed]
                        #track_index = TRACKER.index(str(deleted_id))
                        #del TRACKER[track_index]
                        #del CLASSIFICATION_TRACKER[track_index]
                        ids.append(self.STORED_IDS[i-removed])
                        #print(f"deleting {self.STORED_IDS[i-removed]}, missed count is now {missed} in frame {current_frame}")
                        del self.STORED_MISS[i-removed]  
                        del self.STORED_XYXY[i-removed]
                        del self.STORED_SIZE[i-removed]
                        del self.STORED_IDS[i-removed]
                        removed+=1
                    #removed+=1
        if len(ids)>0:
            self.delete_mask_by_tracking_id(ids)
            #print(f"deleted ids : {ids}")

                        
    # use it when tracking conflict occurs, need to remove the previous tracking from ID not to confuse location and give incoming one as new                        
    def GetNewTrackingIDForTrackingConflict(self,tracking_id,current_point,w,h):
        
        try:
            index = self.STORED_IDS.index(tracking_id)
            del self.STORED_MISS[index]  
            del self.STORED_XYXY[index]
            del self.STORED_SIZE[index]
            del self.STORED_IDS[index]
            
        
        except:
            print(f"Error: Tracking ID {tracking_id} not found in stored IDs.")
            #new_tracking_id = self.addNewTracking(current_point,w,h)
        new_tracking_id =  self.addNewTracking(current_point,w,h)
        self.switchTrackingId(tracking_id,new_tracking_id)
        #return self.addNewTracking(current_point,w,h)
        return new_tracking_id

    def resize_image(self,input_array,new_size):
        # Open the image
        #new_size = (224,224)
        original_image = Image.fromarray(input_array)

        # Create a new image with the desired size and fill it with black pixels
        new_image = Image.new("RGB", new_size, (0, 0, 0))

        # Calculate the position to paste the original image in the center
        x_offset = (new_size[0] - original_image.width) // 2
        y_offset = (new_size[1] - original_image.height) // 2

        # Paste the original image onto the new image
        new_image.paste(original_image, (x_offset, y_offset))
        #return cv2.cvtColor(np.array(new_image),cv2.COLOR_RGB2BGR)
        return np.array(new_image)
        #bgr_array = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
        # Save the result_image
        #new_image.save(output_path)

    # def calculate_histogram(self,image, mask):
    #     # Convert image to grayscale if needed
    #     gray_image = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     #print(f"image shape : {len(image.shape)}")
        
    #     # Ensure mask is a numpy array of type uint8
    #     mask = np.asarray(mask, dtype=np.uint8)
        
    #     # Calculate histogram using mask
    #     hist = cv2.calcHist([gray_image], [0], mask, [256], [0, 256])
    #     # print(f"hist : {hist}")

    #     # Measure histogram characteristics
    #     max_value = np.max(hist)
    #     min_value = np.min(hist)
    #     mean_value = np.mean(hist)
    #     std_dev = np.std(hist)

    #     #print(f"Max Frequency: {max_value}")
    #     #print(f"Min Frequency: {min_value}")
    #     #print(f"Mean Frequency: {round(mean_value, 2)}")
    #     #print(f"Standard Deviation: {round(std_dev, 2)}")
    #     #print("--------------------------------------------")
        
    #     return hist.flatten()

    def save_crop(self, frame, mask, x1, y1, x2, y2, save_dir, prev_id, colored_mask,
              is_touching, touching_pixels, is_save=True, is_small=False,
              predicted_id=-1, area=1000):

        rgb_mask = np.zeros_like(frame)
        rgb_mask[mask] = frame[mask]
        box_h = max(y2 - y1, x2 - x1)
        crop = rgb_mask[y1:y2, x1:x2]  # take crop
        crop = self.resize_image(crop, (box_h, box_h))  # square box
        crop = cv2.resize(crop, (self.SIZE_224, self.SIZE_224))  # resize to final size

        if touching_pixels:
            self.print_border_pixels(mask, (x1, y1, x2, y2))

        if is_save and not is_small and not is_touching:
            base_path = str(Path(f'{save_dir}/{prev_id[0]}'))

            demo_annotated_img_save_path = Path(
                base_path + '/' + f'{self.image_count}_{predicted_id}_Touching_{touching_pixels}_area{area}.jpg'
            )

            os.makedirs(base_path, exist_ok=True)
            cv2.imwrite(str(demo_annotated_img_save_path), crop)

            # # === Random mid-grey background replacement ===
            # lower_black = np.array([0, 0, 0], dtype=np.uint8)
            # upper_black = np.array([5, 5, 5], dtype=np.uint8)
            # grey_val = random.randint(110, 160)  # random shade
            # random_grey = (grey_val, grey_val, grey_val)  # BGR
            # mask_black = cv2.inRange(crop, lower_black, upper_black)
            # crop[mask_black > 0] = random_grey
            # # ==============================================

            self.add_last_20_crops(prev_id[0], crop)
            self.image_count += 1

        #return crop
    def add_last_20_paths(self,tracking_id,save_path):
        #print("saving for tracking id :",tracking_id)

        if tracking_id in self.LAST_20_PATH:
            self.LAST_20_PATH[tracking_id].append(save_path)
            if len(self.LAST_20_PATH[tracking_id])>20:
                self.LAST_20_PATH[tracking_id].pop(0)  
            #print(self.LAST_20_PATH[tracking_id])

        else:
            self.LAST_20_PATH[tracking_id]= [save_path]
    def get_last_X_paths(self,tracking_id,take = 20):
        if type(tracking_id) == int :
            
            tracking_id = str(tracking_id)
        #print(type(tracking_id) ,' is tracking id type')
        #print("Fetching lax x paths for tracking id :",tracking_id)
        if tracking_id not in self.LAST_20_PATH:
            #print(self.LAST_20_PATH)
            print('No images ??? really???')
            return None
        length = len(self.LAST_20_PATH[tracking_id])
        start_from = length - take if length > take else 0
        response = self.LAST_20_PATH[tracking_id][start_from:]
        #print(len(response), ' is stored path for tracking_id :', tracking_id)
        return response
    #region last 20 crops
    def add_last_20_crops(self,tracking_id,crop):
        #print("saving for tracking id :",tracking_id)

        if tracking_id in self.LAST_20_PATH:
            self.LAST_20_PATH[tracking_id].append(crop)
            if len(self.LAST_20_PATH[tracking_id])>20:
                self.LAST_20_PATH[tracking_id].pop(0)  
            #print(self.LAST_20_PATH[tracking_id])

        else:
            self.LAST_20_PATH[tracking_id]= [crop]
    
    def get_last_X_crops(self,tracking_id,take = 20):
        if type(tracking_id) == int :
            
            tracking_id = str(tracking_id)
        #print(type(tracking_id) ,' is tracking id type')
        #print("Fetching lax x paths for tracking id :",tracking_id)
        if tracking_id not in self.LAST_20_PATH:
            #print(self.LAST_20_PATH)
            print('No images ??? really???')
            return None
        length = len(self.LAST_20_PATH[tracking_id])
        start_from = length - take if length > take else 0
        response = self.LAST_20_PATH[tracking_id][start_from:]
        #print(len(response), ' is stored path for tracking_id :', tracking_id)
        return response
    #endregion last 20 crops
    
    def delete_last_20_path(self,tracking_id):
        if tracking_id in self.LAST_20_PATH:
            del self.LAST_20_PATH[tracking_id]
    def draw_mask_multiple_masks(self,colored_mask,masks,indexes):
        #boolean_mask = mask.astype(bool)
        for index in indexes:
            colored_mask[masks[index].astype(bool)] = (0, 255, 255)  # red color mask
            center_of_mask = self.find_mask_center(masks[index],0)
            center_of_mask = tuple(map(int, center_of_mask))
            cv2.circle(colored_mask, center_of_mask, 5, (255, 0, 0), -1)
        return colored_mask

    def diff_Time(self,start,end,process):
        return
        #print(f'{end-start} duration for {process}')
        
    def roi_no_crop(self,scale):
        # create function for image roi
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
        for tracking_id in tracking_ids:
            track_index = -1
            if tracking_id in self.TRACKER: #already have tracking record
                track_index = self.TRACKER.index(tracking_id)
            
            #batch_default = {'TOTAL_COUNT':0 , 'HOLDING' : 'Identifying','LOCATION' : self.BATCH_CLASSIFICATION_TRACKER[track_index]['LOCATION']}
            #batch_default = {'TOTAL_COUNT':0 , 'HOLDING' : 'Identifying','LOCATION' : location,'TOTAL_MISSED_FRAME':0,'TOTAL_DETECTION':0}
            #batch_default = {'TOTAL_COUNT':0 , 'HOLDING' : 'Identifying','LOCATION' : self.BATCH_CLASSIFICATION_TRACKER[track_index]['LOCATION'] , }
            default = {'GT': ['Reidentifying'], 'COUNT':[0], 'IS_STABLE':False,  'HAS_MISSED_FRAME' : False, 'REIDENTIFY_MISSED_COUNT' : self.REIDENTIFY_MISSED_COUNT }
            
            self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] = 0
            self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = 'Identifying'
            self.CLASSIFICATION_TRACKER[track_index] = default
            
    def UpdateTrackingIDAndBatchInfo(self,old_tracking_id,location,predicted_id,w,h):
        tracking_id = self.GetNewTrackingIDForTrackingConflict(old_tracking_id,location,w,h)
        if type(tracking_id) == 'str':
            tracking_id = int(tracking_id)
        default = {'GT': [predicted_id], 'COUNT':[self.batch_size+1], 'IS_STABLE':False,  'HAS_MISSED_FRAME' : False, 'REIDENTIFY_MISSED_COUNT' : self.REIDENTIFY_MISSED_COUNT }
        self.TRACKER.append(tracking_id) #add new tracking record
        track_index = len(self.TRACKER) - 1
        #batch_default = {'TOTAL_COUNT':0 , 'HOLDING' : 'Identifying','LOCATION' : location,'TOTAL_MISSED_FRAME':0,'TOTAL_DETECTION':0}
        batch_default = {'TOTAL_COUNT':self.batch_size+1 , 'HOLDING' : predicted_id,'LOCATION' : location,'TOTAL_MISSED_FRAME':0,'TOTAL_DETECTION':0,'PREVIOUS_MAX_PREDICTED_ID':None}
        self.BATCH_CLASSIFICATION_TRACKER.append(batch_default)
        self.CLASSIFICATION_TRACKER.append(default)
        self.ALL_TIME_CLASSIFICATION_TRACKER.append(default) #to keep the whole result]
        self.TRACKING_RESULT[tracking_id] = [] #default
        self.TRACKING_RESULT[tracking_id].append(predicted_id)
        #print('Remove and insert new tracking ID from tracking conflict',self.TRACKER)
        return (predicted_id,tracking_id)
    
    def getTwoMaxIndexes(self,array):
        if(len(array) < 1):
            return -1,-1
        if len(array) < 2:
            return 0,-1
        max1 = max(array)
        max1Index = array.index(max1)
        #array.pop(max1Index)
        array[max1Index] = 0
        max2 = max(array)
        max2Index = array.index(max2)
        array[max1Index] = max1
        if max1-max2<10:
            return max1Index,max2Index
        return max1Index, -1
    
    def GetBatchPredictedId_detection_only(self, tracking_id, is_small, save_dir, missed_frame_count, location, w, h):
        batch_default = {'TOTAL_COUNT': 0, 'HOLDING': 'Identifying', 'LOCATION': location, 'TOTAL_MISSED_FRAME': 0,
                         'TOTAL_DETECTION': 0, 'PREVIOUS_MAX_PREDICTED_ID': None}
        track_index = -1
        has_missed_frame = False
        # if (tracking_id == '2' or tracking_id == 2) and missed_frame_count > 1:
        # print(f"missed frame count of tracking {tracking_id}: {missed_frame_count} ")
        if missed_frame_count > 3:
            has_missed_frame = True
            # print(f'{tracking_id} missed {missed_frame_count} frame and need to reidentify')# missed more than 5 frames then re-identify for 10 frames
        default = {'GT': [None], 'COUNT': [0], 'IS_STABLE': False, 'HAS_MISSED_FRAME': has_missed_frame,
                   'REIDENTIFY_MISSED_COUNT': self.REIDENTIFY_MISSED_COUNT}
        FORCE_IDENTIFICATION = False
        FORCE_MISSED_IDENTIFICATION = False
        HAS_TRACKING_CONFLICT = False

        IS_STABLE = False

        if tracking_id in self.TRACKER:  # already have tracking record

            track_index = self.TRACKER.index(tracking_id)

        else:
            #print(f'new tracking {tracking_id} to {self.TRACKER}')
            self.TRACKER.append(tracking_id)  # add new tracking record
            track_index = len(self.TRACKER) - 1

            self.BATCH_CLASSIFICATION_TRACKER.append(batch_default)
            self.CLASSIFICATION_TRACKER.append(default)
            self.ALL_TIME_CLASSIFICATION_TRACKER.append(default)  # to keep the whole result

            self.TRACKING_RESULT[tracking_id] = []  # default

            # print('added newq for tracking :',tracking_id)
        
        self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_MISSED_FRAME'] += missed_frame_count
        self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_DETECTION'] += 1
        # store missed_frame :
        
        return (tracking_id, tracking_id)
    def call_GetBatchPredictedId(self, prev_id, isSmall, missed_frame_count, w, h):
        return self.GetBatchPredictedId(int(prev_id[0]), isSmall, self.save_dir, missed_frame_count, prev_id[1], w, h)

    
    def GetBatchPredictedId(self, tracking_id, is_small, save_dir, missed_frame_count, location, w, h):
        batch_default = {'TOTAL_COUNT': 0, 'HOLDING': 'Identifying', 'LOCATION': location, 'TOTAL_MISSED_FRAME': 0,
                         'TOTAL_DETECTION': 0, 'PREVIOUS_MAX_PREDICTED_ID': None}
        track_index = -1
        has_missed_frame = False
        # if (tracking_id == '2' or tracking_id == 2) and missed_frame_count > 1:
        # print(f"missed frame count of tracking {tracking_id}: {missed_frame_count} ")
        if missed_frame_count > 5:
            has_missed_frame = True
            # print(f'{tracking_id} missed {missed_frame_count} frame and need to reidentify')# missed more than 5 frames then re-identify for 10 frames
        default = {'GT': [None], 'COUNT': [0], 'IS_STABLE': False, 'HAS_MISSED_FRAME': has_missed_frame,
                   'REIDENTIFY_MISSED_COUNT': self.REIDENTIFY_MISSED_COUNT}
        FORCE_IDENTIFICATION = False
        FORCE_MISSED_IDENTIFICATION = False
        HAS_TRACKING_CONFLICT = False

        IS_STABLE = False

        if tracking_id in self.TRACKER:  # already have tracking record

            track_index = self.TRACKER.index(tracking_id)

        else:
            #print(f'new tracking {tracking_id} to {self.TRACKER}')
            self.TRACKER.append(tracking_id)  # add new tracking record
            track_index = len(self.TRACKER) - 1

            self.BATCH_CLASSIFICATION_TRACKER.append(batch_default)
            self.CLASSIFICATION_TRACKER.append(default)
            self.ALL_TIME_CLASSIFICATION_TRACKER.append(default)  # to keep the whole result

            self.TRACKING_RESULT[tracking_id] = []  # default

            # print('added newq for tracking :',tracking_id)
        
        self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_MISSED_FRAME'] += missed_frame_count
        self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_DETECTION'] += 1
        # store missed_frame :
        total_predictions = 0
        if not is_small:
            # print(f" tracking_id {tracking_id} is small? : {is_small}")
            # predicted_index = CLASSIFICATION_TRACKER[track_index]['GT'].index(predicted_id)
            self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] += 1
            total_predictions = self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] 
            if self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME'] == True:  # keep searching if small

                self.CLASSIFICATION_TRACKER[track_index]['REIDENTIFY_MISSED_COUNT'] -= 1  # reduce by one
                FORCE_MISSED_IDENTIFICATION = self.CLASSIFICATION_TRACKER[track_index]['REIDENTIFY_MISSED_COUNT'] < 1
                if FORCE_MISSED_IDENTIFICATION:
                   # print(f"Reach limit and will do identification for missed frame of {tracking_id}")
                    # self.BATCH_CLASSIFICATION_TRACKER[track_index]['REIDENTIFY_MISSED_COUNT']  = 10 #re identify in 10
                    self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME'] = False  # identify in 10
        
        if has_missed_frame and total_predictions > self.batch_size and self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] != 'Identifying':
            #print(f"FORCE MISSED DETECTION FOR TRACKING {tracking_id}")

            self.CLASSIFICATION_TRACKER[track_index][
                'REIDENTIFY_MISSED_COUNT'] = self.REIDENTIFY_MISSED_COUNT  # re identify in 10
            self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME'] = True  # do force missed detection
            self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = 'Identifying'
            self.CLASSIFICATION_TRACKER[track_index]['IS_STABLE'] = False  # if missed then not stable

            # print(self.CLASSIFICATION_TRACKER[track_index])

        IS_STABLE = self.CLASSIFICATION_TRACKER[track_index]['IS_STABLE']

        #region check is moving
        distance = self.CALCULATE_EUCLIDEAN_DISTANCE(location, self.BATCH_CLASSIFICATION_TRACKER[track_index][
                    'LOCATION'])  # compare two distance
        is_moving = distance > 100

        if total_predictions > 0 and total_predictions % 100 == 0 and is_moving:
            FORCE_MISSED_IDENTIFICATION = True #simulate one time force missed identification on total_predictions 600 to ensure same or not

        if IS_STABLE:
            maxPredictedId = self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING']
            return maxPredictedId, tracking_id

        if total_predictions > self.total_predictions and not IS_STABLE and not self.CLASSIFICATION_TRACKER[track_index][
            'HAS_MISSED_FRAME'] and not FORCE_MISSED_IDENTIFICATION:  # and if not stable
            #is_moving = False
            if  total_predictions % 50 == 0:  # only re calculate every 100th frame
                
                #if is_moving:
                    #print(self.BATCH_CLASSIFICATION_TRACKER[track_index]['LOCATION'])
                    #print(location)
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['LOCATION'] = location

            if is_moving:  # or total_predictions %1000 == 0: #every 1000th frame
                # print("tracking id {tracking_id} is moving")
                # print("Before clearing ",self.ALL_TIME_CLASSIFICATION_TRACKER)
                #print(f're identifying tracking {tracking_id} due to moving')
                # self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] = 0
                # self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING']  = 'Identifying'
                # default = {'GT': [None], 'COUNT':[0], 'IS_STABLE' : False}
                # self.CLASSIFICATION_TRACKER[track_index]
                # self.CLASSIFICATION_TRACKER[track_index] = default
                FORCE_IDENTIFICATION = True
                # print("After clearing ",self.ALL_TIME_CLASSIFICATION_TRACKER)
            elif total_predictions == 1000 and False:  # NOT RESETTING
                #print(f're identifying tracking {tracking_id} due to 1000th images')
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] = 0
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = 'Identifying'
                # default = {'GT': [None], 'COUNT':[0], 'IS_STABLE' : False}
                # self.CLASSIFICATION_TRACKER[track_index]
                self.CLASSIFICATION_TRACKER[track_index] = default
                FORCE_IDENTIFICATION = True
            else:
                return self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'], tracking_id
        # total_predictions = self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT']

        # if missed_frame_count >
        # print('increase count for identification')
        # print("tracking ",track_index," have ", sum(CLASSIFICATION_TRACKER[track_index]['COUNT'])," counts")

        # if missed_frame_count >
        # print(f'total predictions for {tracking_id} is : {total_predictions}')
        maxPredictedId = -1
        isDoingIdentification = False
        if total_predictions == self.batch_size or (
                total_predictions > 0 and total_predictions % self.batch_size == 0 and total_predictions <= 100) or FORCE_IDENTIFICATION or FORCE_MISSED_IDENTIFICATION:

            # if has missed frame then show current holding -> could be identifying or id
            if self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME']:
                return self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'], tracking_id

            maxCounters = self.batch_identification(save_dir, tracking_id, FORCE_MISSED_IDENTIFICATION)
            previous_max_predicted_id = self.BATCH_CLASSIFICATION_TRACKER[track_index][
                'PREVIOUS_MAX_PREDICTED_ID']  # update previous max predicted ID
            # print(maxCounters)
            # maxPredictedId = maxCounters[0][0]

            if FORCE_MISSED_IDENTIFICATION:  # force missed identification
               # print(f"counter {maxCounters}, previous_max_predicted_id : {previous_max_predicted_id}")

                if maxCounters is None or len(maxCounters) == 0:
                    maxPredictedId = None
                else:
                    maxPredictedId = maxCounters[0][0]
                #print(
                 #   f"FORCE_MISSED_IDENTIFICATION RESULT: {maxPredictedId}, ' PREVIOUS_MAX_PREDICTED_ID : {previous_max_predicted_id}")
                if maxPredictedId is None:  # if none keep searching
                 #   print(f"FORCE MISSED DETECTION FOR TRACKING {tracking_id}")
                    self.BATCH_CLASSIFICATION_TRACKER[track_index][
                        'REIDENTIFY_MISSED_COUNT'] = self.REIDENTIFY_MISSED_COUNT  # re identify in 10
                    self.BATCH_CLASSIFICATION_TRACKER[track_index][
                        'HAS_MISSED_FRAME'] = True  # do force missed detection
                # elif previous_max_predicted_id != None and previous_max_predicted_id != 'Reidentifying' and previous_max_predicted_id != 'Identifying': #if not same then update  due to tracking conflict
                elif previous_max_predicted_id != maxPredictedId:  # different cow
                    # batch_default = {'TOTAL_COUNT':10 , 'HOLDING' : maxPredictedId,'LOCATION' : self.BATCH_CLASSIFICATION_TRACKER[track_index]['LOCATION'] , 'HAS_MISSED_FRAME' : False, 'REIDENTIFY_MISSED_COUNT' : self.REIDENTIFY_MISSED_COUNT}
                    # default = {'GT': [maxPredictedId], 'COUNT':[10], 'IS_STABLE':False }
                    # self.BATCH_CLASSIFICATION_TRACKER[track_index] = batch_default
                    # self.CLASSIFICATION_TRACKER[track_index] = default

                    self.TRACKING_RESULT[tracking_id].append(maxPredictedId)
                    return self.UpdateTrackingIDAndBatchInfo(tracking_id, location, maxPredictedId, w, h)
                elif previous_max_predicted_id == maxPredictedId:
                    self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = maxPredictedId
                    self.BATCH_CLASSIFICATION_TRACKER[track_index]['PREVIOUS_MAX_PREDICTED_ID'] = maxPredictedId
                    return (maxPredictedId, tracking_id)

            else:
                # if len(maxCounters) == 0:
                #     maxPredictedId = None
                # region update to csv to check accuracy
                if maxCounters is not None:
                    for maxCounter in maxCounters:
                        # print(maxCounter)
                        predicted_id, predicted_count = maxCounter
                        # print(CLASSIFICATION_TRACKER[track_index])
                        ## record limited indexes
                        if predicted_id not in self.CLASSIFICATION_TRACKER[track_index]['GT']:

                            self.CLASSIFICATION_TRACKER[track_index]['GT'].append(predicted_id)
                            self.CLASSIFICATION_TRACKER[track_index]['COUNT'].append(predicted_count)
                            # print('new')
                        else:
                            predicted_index = self.CLASSIFICATION_TRACKER[track_index]['GT'].index(predicted_id)
                            self.CLASSIFICATION_TRACKER[track_index]['COUNT'][predicted_index] += predicted_count

                        # print(self.CLASSIFICATION_TRACKER)
                        if predicted_id not in self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['GT']:

                            self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['GT'].append(predicted_id)
                            self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['COUNT'].append(predicted_count)
                            # print('new')
                        else:
                            predicted_index = self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['GT'].index(predicted_id)
                            self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['COUNT'][predicted_index] += predicted_count

                maxCount = max(self.CLASSIFICATION_TRACKER[track_index]['COUNT'])
                maxCountIndex = self.CLASSIFICATION_TRACKER[track_index]['COUNT'].index(maxCount)
                maxPredictedId = self.CLASSIFICATION_TRACKER[track_index]['GT'][maxCountIndex]
                self.TRACKING_RESULT[tracking_id].append(maxPredictedId)  # add to log
                # self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] = 0
                if FORCE_IDENTIFICATION:  # due to moving
                   # print(f"Setting final ID for tracking {tracking_id} : {maxPredictedId}")
                    self.CLASSIFICATION_TRACKER[track_index]['IS_STABLE'] = previous_max_predicted_id == maxPredictedId

                self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = maxPredictedId
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['PREVIOUS_MAX_PREDICTED_ID'] = maxPredictedId

        # else:
        #    maxPredictedId =
        
        maxPredictedId = self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING']
        if total_predictions >= 100 and (maxPredictedId == None or maxPredictedId == 'None' or
        maxPredictedId == 'Reidentifying' or maxPredictedId == 'Identifying'):
            self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] = 0
            self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = 'Identifying'

        if tracking_id not in self.tracking_by_predicted_id:
            self.tracking_by_predicted_id[maxPredictedId] = tracking_id
        #secondMaxPredictedId = None
        # first , secondMaxIndex = self.getTwoMaxIndexes(self.CLASSIFICATION_TRACKER[track_index]['COUNT'])
        # if secondMaxIndex >0:
        #     maxPredictedId += "_"+ self.CLASSIFICATION_TRACKER[track_index]['GT'][secondMaxIndex]
        
        if maxPredictedId is None:
            maxPredictedId = 'Reidentifying'
        # if is_small:
        #    return f'{tracking_id}_{maxPredictedId}_size'
        #    #return 
        # else:
        # return (str(tracking_id)+"_"+maxPredictedId)
        return (maxPredictedId, tracking_id)
    
    def GetBatchPredictedId_TwoSimilar(self,tracking_id,is_small,save_dir,missed_frame_count,location,w,h):
        #print("tracking_id :",tracking_id, " is small :, ", is_small)
        batch_default = {'TOTAL_COUNT': 0, 'HOLDING': 'Identifying', 'LOCATION': location, 'TOTAL_MISSED_FRAME': 0,
                         'TOTAL_DETECTION': 0, 'PREVIOUS_MAX_PREDICTED_ID': None}
        track_index = -1
        has_missed_frame = False
        # if (tracking_id == '2' or tracking_id == 2) and missed_frame_count > 1:
        # print(f"missed frame count of tracking {tracking_id}: {missed_frame_count} ")
        if missed_frame_count > 5:
            has_missed_frame = True
            # print(f'{tracking_id} missed {missed_frame_count} frame and need to reidentify')# missed more than 5 frames then re-identify for 10 frames
        default = {'GT': [None], 'COUNT': [0], 'IS_STABLE': False, 'HAS_MISSED_FRAME': has_missed_frame,
                   'REIDENTIFY_MISSED_COUNT': self.REIDENTIFY_MISSED_COUNT}
        FORCE_IDENTIFICATION = False
        FORCE_MISSED_IDENTIFICATION = False
        HAS_TRACKING_CONFLICT = False

        IS_STABLE = False

        if tracking_id in self.TRACKER:  # already have tracking record

            track_index = self.TRACKER.index(tracking_id)

        else:
            #print(f'new tracking {tracking_id} to {self.TRACKER}')
            self.TRACKER.append(tracking_id)  # add new tracking record
            track_index = len(self.TRACKER) - 1

            self.BATCH_CLASSIFICATION_TRACKER.append(batch_default)
            self.CLASSIFICATION_TRACKER.append(default)
            self.ALL_TIME_CLASSIFICATION_TRACKER.append(default)  # to keep the whole result

            self.TRACKING_RESULT[tracking_id] = []  # default

            # print('added newq for tracking :',tracking_id)
        
        self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_MISSED_FRAME'] += missed_frame_count
        self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_DETECTION'] += 1
        # store missed_frame :
        total_predictions = 0
        if not is_small:
            # print(f" tracking_id {tracking_id} is small? : {is_small}")
            # predicted_index = CLASSIFICATION_TRACKER[track_index]['GT'].index(predicted_id)
            self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] += 1
            total_predictions = self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] 
            if self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME'] == True:  # keep searching if small

                self.CLASSIFICATION_TRACKER[track_index]['REIDENTIFY_MISSED_COUNT'] -= 1  # reduce by one
                FORCE_MISSED_IDENTIFICATION = self.CLASSIFICATION_TRACKER[track_index]['REIDENTIFY_MISSED_COUNT'] < 1
                if FORCE_MISSED_IDENTIFICATION:
                    print(f"Reach limit and will do identification for missed frame of {tracking_id}")
                    # self.BATCH_CLASSIFICATION_TRACKER[track_index]['REIDENTIFY_MISSED_COUNT']  = 10 #re identify in 10
                    self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME'] = False  # identify in 10
        
        if has_missed_frame and total_predictions > self.batch_size and self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] != 'Identifying':
            print(f"FORCE MISSED DETECTION FOR TRACKING {tracking_id}")

            self.CLASSIFICATION_TRACKER[track_index][
                'REIDENTIFY_MISSED_COUNT'] = self.REIDENTIFY_MISSED_COUNT  # re identify in 10
            self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME'] = True  # do force missed detection
            self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = 'Identifying'
            self.CLASSIFICATION_TRACKER[track_index]['IS_STABLE'] = False  # if missed then not stable

            # print(self.CLASSIFICATION_TRACKER[track_index])

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
                # print("tracking id {tracking_id} is moving")
                # print("Before clearing ",self.ALL_TIME_CLASSIFICATION_TRACKER)
                print(f're identifying tracking {tracking_id} due to moving')
                # self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] = 0
                # self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING']  = 'Identifying'
                # default = {'GT': [None], 'COUNT':[0], 'IS_STABLE' : False}
                # self.CLASSIFICATION_TRACKER[track_index]
                # self.CLASSIFICATION_TRACKER[track_index] = default
                FORCE_IDENTIFICATION = True
                # print("After clearing ",self.ALL_TIME_CLASSIFICATION_TRACKER)
            elif total_predictions == 1000 and False:  # NOT RESETTING
                print(f're identifying tracking {tracking_id} due to 1000th images')
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] = 0
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = 'Identifying'
                # default = {'GT': [None], 'COUNT':[0], 'IS_STABLE' : False}
                # self.CLASSIFICATION_TRACKER[track_index]
                self.CLASSIFICATION_TRACKER[track_index] = default
                FORCE_IDENTIFICATION = True
            else:
                return self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'], tracking_id
        # total_predictions = self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT']

        # if missed_frame_count >
        # print('increase count for identification')
        # print("tracking ",track_index," have ", sum(CLASSIFICATION_TRACKER[track_index]['COUNT'])," counts")

        # if missed_frame_count >
        # print(f'total predictions for {tracking_id} is : {total_predictions}')
        maxPredictedId = -1
        if total_predictions == self.batch_size or (
                total_predictions > 0 and total_predictions % self.batch_size == 0) or FORCE_IDENTIFICATION or FORCE_MISSED_IDENTIFICATION:

            # if has missed frame then show current holding -> could be identifying or id
            if self.CLASSIFICATION_TRACKER[track_index]['HAS_MISSED_FRAME']:
                return self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'], tracking_id

            maxCounters = self.batch_identification(save_dir, tracking_id, FORCE_MISSED_IDENTIFICATION)
            previous_max_predicted_id = self.BATCH_CLASSIFICATION_TRACKER[track_index][
                'PREVIOUS_MAX_PREDICTED_ID']  # update previous max predicted ID
            # print(maxCounters)
            # maxPredictedId = maxCounters[0][0]

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
                # elif previous_max_predicted_id != None and previous_max_predicted_id != 'Reidentifying' and previous_max_predicted_id != 'Identifying': #if not same then update  due to tracking conflict
                elif previous_max_predicted_id != maxPredictedId:  # different cow
                    # batch_default = {'TOTAL_COUNT':10 , 'HOLDING' : maxPredictedId,'LOCATION' : self.BATCH_CLASSIFICATION_TRACKER[track_index]['LOCATION'] , 'HAS_MISSED_FRAME' : False, 'REIDENTIFY_MISSED_COUNT' : self.REIDENTIFY_MISSED_COUNT}
                    # default = {'GT': [maxPredictedId], 'COUNT':[10], 'IS_STABLE':False }
                    # self.BATCH_CLASSIFICATION_TRACKER[track_index] = batch_default
                    # self.CLASSIFICATION_TRACKER[track_index] = default

                    self.TRACKING_RESULT[tracking_id].append(maxPredictedId)
                    return self.UpdateTrackingIDAndBatchInfo(tracking_id, location, maxPredictedId, w, h)
                elif previous_max_predicted_id == maxPredictedId:
                    self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = maxPredictedId
                    self.BATCH_CLASSIFICATION_TRACKER[track_index]['PREVIOUS_MAX_PREDICTED_ID'] = maxPredictedId
                    return (maxPredictedId, tracking_id)

            else:
                # region update to csv to check accuracy
                for maxCounter in maxCounters:
                    # print(maxCounter)
                    predicted_id, predicted_count = maxCounter
                    # print(CLASSIFICATION_TRACKER[track_index])
                    ## record limited indexes
                    if predicted_id not in self.CLASSIFICATION_TRACKER[track_index]['GT']:

                        self.CLASSIFICATION_TRACKER[track_index]['GT'].append(predicted_id)
                        self.CLASSIFICATION_TRACKER[track_index]['COUNT'].append(predicted_count)
                        # print('new')
                    else:
                        predicted_index = self.CLASSIFICATION_TRACKER[track_index]['GT'].index(predicted_id)
                        self.CLASSIFICATION_TRACKER[track_index]['COUNT'][predicted_index] += predicted_count

                    # print(self.CLASSIFICATION_TRACKER)
                    if predicted_id not in self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['GT']:

                        self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['GT'].append(predicted_id)
                        self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['COUNT'].append(predicted_count)
                        # print('new')
                    else:
                        predicted_index = self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['GT'].index(predicted_id)
                        self.ALL_TIME_CLASSIFICATION_TRACKER[track_index]['COUNT'][predicted_index] += predicted_count

                maxCount = max(self.CLASSIFICATION_TRACKER[track_index]['COUNT'])
                maxCountIndex = self.CLASSIFICATION_TRACKER[track_index]['COUNT'].index(maxCount)
                maxPredictedId = self.CLASSIFICATION_TRACKER[track_index]['GT'][maxCountIndex]
                self.TRACKING_RESULT[tracking_id].append(maxPredictedId)  # add to log
                # self.BATCH_CLASSIFICATION_TRACKER[track_index]['TOTAL_COUNT'] = 0
                if FORCE_IDENTIFICATION:  # due to moving
                    print(f"Setting final ID for tracking {tracking_id} : {maxPredictedId}")
                    self.CLASSIFICATION_TRACKER[track_index]['IS_STABLE'] = previous_max_predicted_id == maxPredictedId

                self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING'] = maxPredictedId
                self.BATCH_CLASSIFICATION_TRACKER[track_index]['PREVIOUS_MAX_PREDICTED_ID'] = maxPredictedId

        # else:
        #    maxPredictedId =
        maxPredictedId = self.BATCH_CLASSIFICATION_TRACKER[track_index]['HOLDING']
        #secondMaxPredictedId = None
        # first , secondMaxIndex = self.getTwoMaxIndexes(self.CLASSIFICATION_TRACKER[track_index]['COUNT'])
        # if secondMaxIndex >0:
        #     maxPredictedId += "_"+ self.CLASSIFICATION_TRACKER[track_index]['GT'][secondMaxIndex]
        
        if maxPredictedId is None:
            maxPredictedId = 'Reidentifying'
        # if is_small:
        #    return f'{tracking_id}_{maxPredictedId}_size'
        #    #return 
        # else:
        # return (str(tracking_id)+"_"+maxPredictedId)
        return (maxPredictedId, tracking_id)

    def GetPredictedIDFromTracking(self,tracking_id,predicted_id,is_small):
        #print("tracking_id :",tracking_id, " is small :, ", is_small)
       
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
                #print('new')
            else:
                predicted_index = self.CLASSIFICATION_TRACKER[track_index]['GT'].index(predicted_id)
                self.CLASSIFICATION_TRACKER[track_index]['COUNT'][predicted_index] +=1
            
            #print('increase count for identification')
        #print("tracking ",track_index," have ", sum(CLASSIFICATION_TRACKER[track_index]['COUNT'])," counts")
        maxPredictedId = -1
        if total_predictions < 10:
            return -1
        if total_predictions == 10 or total_predictions % 10 ==0:
            maxCount = max(self.CLASSIFICATION_TRACKER[track_index]['COUNT'])
            maxCountIndex = self.CLASSIFICATION_TRACKER[track_index]['COUNT'].index(maxCount)
            maxPredictedId = self.CLASSIFICATION_TRACKER[track_index]['GT'][maxCountIndex]
            #CLASSIFICATION_TRACKER
            self.CLASSIFICATION_TRACKER[track_index]['HOLDING'] = maxPredictedId
            self.CLASSIFICATION_TRACKER[track_index]['IS_PREDICTED'] = True
            
            
        
        #print(maxPredictedI)
        #else:
        #    maxPredictedId =
        return f'{maxPredictedId}'
        #if is_small:
        #    return f'{tracking_id}_{maxPredictedId}_size'
        #    #return 
        #else:
        #    return f'{tracking_id}_{maxPredictedId}'
            
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

        # Normalize inputs
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 5), dtype=float), []
        boxes = np.asarray(boxes)

        # Compute area
        widths  = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas   = widths * heights
        boxes_with_areas = np.hstack([boxes, areas.reshape(-1, 1)])

        # Optional filter for the "last cam": drop boxes with y2 beyond a threshold
        if is_last_cam and y2_threshold is not None:
            # Try to infer image height from masks if ratio given
            img_h = None
            if hasattr(masks, "shape") and len(getattr(masks, "shape", [])) >= 3:
                # e.g., masks: [N, H, W]
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

        # If everything got filtered out
        if boxes_with_areas.shape[0] == 0:
            return np.empty((0, 5), dtype=float), []

        # Sort by area (desc)
        sorted_indices = np.argsort(boxes_with_areas[:, -1])[::-1]
        sorted_boxes = boxes_with_areas[sorted_indices]

        # Keep largest connected component per mask
        sorted_masks = []
        # Index into masks in the same order as sorted_indices
        if isinstance(masks, (list, tuple)):
            for i in sorted_indices:
                m = masks[int(i)]
                try:
                    sorted_masks.append(self.keep_largest_connected_mask(m))
                except Exception:
                    sorted_masks.append(m)
        else:
            # masks is likely a numpy array [N, H, W]
            for i in sorted_indices:
                m = masks[int(i)]
                try:
                    sorted_masks.append(self.keep_largest_connected_mask(m))
                except Exception:
                    sorted_masks.append(m)

        return sorted_boxes, sorted_masks
        #return sorted_boxes,sorted_masks, [False] * len(sorted_boxes)
        #return self.shouldMerge(sorted_boxes,sorted_masks,cam_counter=0)

    #def shouldMerge(self, boxes, masks, cam_counter = 0):

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
        #return not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)
        
        return x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2
    def is_mask_centroid_in_roi(self, centroid):
        rx1, ry1, rx2, ry2 = self.MERGE_ROI
        cx, cy = centroid
        return rx1 <= cx <= rx2 and ry1 + 10 <= cy <= ry2 - 10
        

    def shouldMerge(self, boxes, masks, cam_counter=0, distance_thresh=230, max_size=185):
        merged_boxes = []
        merged_masks = []
        merged_flags = []
        used = set()

        for i in range(len(boxes)):
            if i in used:
                continue
            box1 = boxes[i][:4].astype(int)
            mask1 = masks[i]
            # if not self.is_small_box(box1, 250):

            if not self.is_in_roi(box1) or True:
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

                #iou = self.compute_iou(box1, box2)

                if not self.is_in_roi(box2):
                    continue
            
                iou = self.compute_iou(box1, box2)
                # if iou > 0:
                #     print(iou,  " IOU VALUE")
                center2 = self.find_mask_center(mask2,cam_counter=0)
                if center1 is None or center2 is None:
                    continue
                center_distance = self.CALCULATE_EUCLIDEAN_DISTANCE(center1, center2)
                #print(center_distance ," CENTROID DISTANCE !! ")
                if center_distance > distance_thresh and iou < 0.1:
                    continue
                
                # Perform merging
                mx1 = min(box1[0], box2[0])
                my1 = min(box1[1], box2[1])
                mx2 = max(box1[2], box2[2])
                my2 = max(box1[3], box2[3])
                merged_box = (mx1, my1, mx2, my2)
                merged_h = my2 - my1
                merged_w = mx2 - mx1

                #merged_mask1 = self.place_mask_in_box(mask1, box1, merged_box, (merged_h, merged_w))
                #merged_mask2 = self.place_mask_in_box(mask2, box2, merged_box, (merged_h, merged_w))
                merged_mask = np.logical_or(mask1, mask2)
                if np.sum(merged_mask) > 40000:
                    continue
                merged_boxes.append(np.array(merged_box))
                merged_masks.append(merged_mask)

                used.add(i)
                used.add(j)
                merged = True
                #print("merged one box")
                break

            if not merged:
                merged_boxes.append(box1)
                merged_masks.append(mask1)
            merged_flags.append(merged)

        return np.array(merged_boxes), merged_masks, merged_flags



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
        #area = w * h
        #print('total area ',area)
        prev_area = self.CalculateBoxArea(prev)
        current_area = self.CalculateBoxArea(current)
        #difference = abs(prev_area - current_area)
        #print('position 1',prev, 'position 2',current)
        #print('prev position, current position',prev_area,current_area)
        #return area / CalculateBoxArea(prev)
        #print('two box area ',prev_area,' ',current_area)
        prev_area = current_area if prev_area > current_area else prev_area
        #ratio = area / prev_area
        if cam_counter ==1:
            print(area / prev_area, " is ratio of two box")
        if area / prev_area < 0.93: #check with 95% originally
            return False #not duplicate
        return True #duplicate
        #return 
    #    iou = area / float(prev_area + current_area - area)
        
        #return iou
    def IsInsideAnotherBox(self,boxes,current_box , index, cam_counter = 0):
        if(index < 1): #first 
            return False
        contained_in_any_box = False
        #printMe =False
        until = min(index+1,len(boxes))
        #if cam_counter ==1:
        #    print(f"Checking total boxes {until} is inside another box")
        for i in range(until):
            if self.isDuplicate_box(current_box, boxes[i],cam_counter):
                #print("I am duplicated box ")
                contained_in_any_box = True
        #        printMe = True
                break
        #if printMe: 
        #    print("duplicate but :",contained_in_any_box)  
        return contained_in_any_box
    
    def isHuman(self,x1,y1,x2,y2, area):
        #if (x1 > 5 and x2<self.boundary[0] and y1 >5 and y2< self.boundary[1] and
        #already check place in isValidSize
        #print(" I am too small ")
        return area<self.SMALL_SIZE
        #return True
    def isHumanRatio(self,w,h,area):
        #override human ration if the size is big enough
        if area > self.SMALL_SIZE + 500:
            return False
        
        if w>h:
            ratio = h/w
        else :
            ratio = w/h
            
        #print(f'{ratio} is body ratio for visible cow, {w} , {h}')
        return ratio > 0.6
    
    def IsInMiddle(self,x1,y1,x2,y2):
        return x1 > 10 and y1 > 10 and x2 <self.boundary[0] + 5 and y2 < self.boundary[1]
    
    def IsTouchingBorder(self,x1,y1,x2,y2, area):
        x = x2-x1
        y = y2-y1
        if  x1<3 or y1<3 or y2>=self.boundary[1]: #or x2>=self.boundary[0]: #area < self.average_size 
            return False 
        #if x<100 or y<100 or area < SMALL_SIZE or area > BIG_SIZE  :
        
        #if area < self.SMALL_SIZE or x < 100 or y<100 or area > self.BIG_SIZE  :
            #print("size is ",x, " : " ,y)
        #    return False
        #if x<(200 * multiplier) or y<(200 * multiplier) or 
        #if area < average_size:
        #    print("size is ",x, " : " ,y)
        #    return False
        return True

    def keep_largest_connected_mask(self,mask):
        """
        Retains only the largest connected component in the mask and removes all other disconnected regions.

        Parameters:
        - mask (numpy.ndarray): Boolean mask array.

        Returns:
        - numpy.ndarray: Mask with only the largest connected component.
        """
        # Ensure mask is binary
        mask = mask.astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels <= 1:
            # No connected components found (only background)
            return np.zeros_like(mask, dtype=bool)

        # Find the largest component (excluding the background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip background (index 0)

        # Create a mask for the largest component
        largest_mask = (labels == largest_label)

        return largest_mask.astype(bool)


    def append_pair(self,tracking_id, predicted_id):
        # Check if the file exists
        global csv_file
        file_exists = os.path.exists(csv_file)
        
        # Read existing data from the file if it exists
        updated_rows = []
        tracking_found = False

        if file_exists:
            with open(csv_file, mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    # If the tracking ID is found, append the new predicted ID
                    if row and int(row[0]) == tracking_id:
                        row.append(predicted_id)
                        tracking_found = True
                    updated_rows.append(row)

        # If tracking ID is not found, add a new row
        if not tracking_found:
            updated_rows.append([tracking_id, predicted_id])

        # Write the updated rows back to the file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(updated_rows)

    # Test: Add pairs dynamically
    #append_pair(1, 'AB')
    #append_pair(2, 'CD')
    #append_pair(1, 'AB')
    #append_pair(1, 'ED')
    #append_pair(2, 'ED')
    #append_pair(3, 'FF')
    #append_pair(3, 'GG')

    #print(f"Data appended to {csv_file}")

    def Draw_Projectile(self,cam_counter,TRACKING_PROJECTILE, colored_mask):
        #print(TRACKING_PROJECTILE, ' is tracking projectilte' )
        if cam_counter in TRACKING_PROJECTILE:
            print("Drawing projectile for camera ", cam_counter)
            print(TRACKING_PROJECTILE[cam_counter], ' is tracking projectilte result' )
            for item in TRACKING_PROJECTILE[cam_counter]:
                center_of_mask = tuple(map(int, item))
                cv2.circle(colored_mask, center_of_mask, 20, (0, 0, 255), -1)
        return colored_mask
    def CoreProcess(self,boxes,masks_np,h,w,frame,cam_counter):
        #global frame_counter
        mask_count = -1
        tracked_ids = []
        colored_mask = np.zeros_like(frame)
        tracked_indexes = []
        predicted_ids = []
        original_predicted_ids = []
        counter = -1
        duplicate_ids = []
        height_threshold = 80
        #print("xxxxxxxxx ---------- xxxxxxxxxxxxxx")
        if not self.isEatingArea:
            height_threshold = 60
        TRACKING_PROJECTILE = {}
        for box in boxes:
            conver_time = datetime.now()
            x1, y1, x2, y2, _ = map(int, box)
            #self.diff_Time(conver_time,datetime.now(),' box convertion processing')
            
            counter += 1
            mask_count+=1   
            isSmall = False

            if(self.IsInsideAnotherBox(boxes,box,counter, cam_counter)):
                #print("I am skipping")
                continue #skip for this
            area = np.sum(masks_np[mask_count])

           
            touching_pixels = 0
            is_touching = False
            middle_height = 200
            #is_mask_touching = False
            if x1 <= 4 or y1 <= 4 or y2 >= self.boundary[1]: #check for 3 direction 
                touching_pixels = self.count_border_pixels(masks_np[mask_count],(x1,y1,x2,y2))
                is_touching = touching_pixels >= self.border_threshold
                #is_mask_touching = touching_pixels >= 5 # touching to border . no need to check how much touching
                if is_touching: 
                    is_touching = not self.is_showing_most_body(masks_np[mask_count], (x1,y1,x2,y2))
                #middle_height = self.calculate_mask_height(masks_np[mask_count],(x1,y1,x2,y2))
                #if is_touching:
                    #isSmall = True
                isSmall = is_touching or isSmall #middle_height < height_threshold # either touching or small height
            isSmall = self.isHuman(x1,y1,x2,y2,area) or isSmall
            #if self.IsInMiddle(x1,y1,x2,y2) and isSmall and self.isHumanRatio(x2-x1,y2-y1,area):
            #    continue
            
            
            if True :#HOKKAIDO_ROI(x1,y1,x2,y2,h,w):  #only cow class
                #Centroid_Tracking_PerCamera
                #prev_id = self.Centroid_Tracking(masks_np[mask_count],x2-x1,y2-y1,cam_counter,y1,y2)
                prev_id = self.Centroid_Tracking(masks_np[mask_count],x2-x1,y2-y1,cam_counter,y1,y2)                
                #predicted_id = en.predict_by_efficientnet(crop)
                #(prev_id,is_moving) =IOU_Tracking(x1,y1,x2,y2)
                #is_moving = True
                if(prev_id==-1): #skip cattle when prev_id // filter id is -1
                    #print("tracking skipping")
                    continue
                
                
                is_touching = False
                middle_height = 200
               
                isInMiddle = self.IsInMiddle(x1,y1,x2,y2)
                if isInMiddle and area < 6000:
                    #print("I am middle : and small with area", area )
                    continue
                is_new = prev_id[3] # check is_new
                
                tracking_id,is_touching_bottom,is_touching_top = self.find_matching_mask(masks_np[mask_count],prev_id[1],cam_counter,prev_id[0],x1,y1,x2,y2,frame)
                if tracking_id != -1:
                    if int(tracking_id) in self.STORED_IDS:
                        #print(f" tracking id {tracking_id} is in camera {cam_counter}")
                        #if not is_touching_top and not is_touching_bottom: # not touching then tracking can only be in one place
                        #    self.remove_tracking_in_other_camera(tracking_id,cam_counter) #remove tracking in other camera
                        idx = self.STORED_IDS.index(int(tracking_id))
                        self.update_tracking_mask_location(tracking_id,masks_np[mask_count],cam_counter,prev_id[1],is_touching_top,is_touching_bottom)
                        
                        self.STORED_XYXY[idx] = prev_id[1]

                        self.STORED_MISS[idx] = 1
                        #print("found closed id :",tracking_id)
                        #prev_id[2] = 
                        prev_id[0] = tracking_id
                        is_new = False
                    else:
                        #print(f"Unable to find tracking id {tracking_id} in stored Ids")
                        self.delete_mask_by_tracking_id([str(tracking_id)])
                    #return tracking_id
                    #if int(tracking_id) == 9:
                    #    print(f"tracking id {tracking_id} is in camera {cam_counter} and is touching top {is_touching_top} and bottom {is_touching_bottom}")

                if is_new:
                    # tracking_id,is_touching_bottom,is_touching_top = self.find_matching_mask(masks_np[mask_count],prev_id[1],cam_counter,y1,y2)
                    # if tracking_id != "-1":
                        
                    #     self.update_tracking_mask_location(tracking_id,masks_np[mask_count],cam_counter,prev_id[1],is_touching_top,is_touching_bottom)
                    #     idx = self.STORED_IDS.index(int(tracking_id))
                    #     self.STORED_XYXY[idx] = prev_id[1]

                    #     self.STORED_MISS[idx] = 1
                    #     print("found closed id :",tracking_id)
                    #     #prev_id[2] = 
                    #     prev_id[0] = tracking_id
                    #     #return tracking_id
                    if isSmall and self.isHumanRatio(x2-x1,y2-y1,area) : #prioritize the size first
                        #print(" I am too small haha")
                        #frame = draw_bounding_box(frame, (x1, y1, x2, y2), f'{area}',4,color=(218,200,30))
                        continue
                    elif isInMiddle and isSmall:
                        #print(" I am too small Middle")
                        #frame = draw_bounding_box(frame, (x1, y1, x2, y2), f'{area}',4,color=(218,200,30))
                        continue
                        #if isSmallSize #or is_touching or self.isHumanRatio(x2-x1,y2-y1,area): #if new is touching then wait it
                        #    continue
                    #print(prev_id[0],' area :',area, 'SMALL SIZE :', self.SMALL_SIZE )
                    else:
                        
                    #prev_id[0] = self.addNewTracking(prev_id[1],w,h) #if not touching then insert itssssssssss
                        prev_id[0] = self.addNewTracking_v2(masks_np[mask_count],prev_id[1],x1,y1,x2,y2,cam_counter,w,h)
                        print("new tracking id : ",prev_id[0], " is touching  top ", is_touching_top, " is touching bottom ", is_touching_bottom, " camera ", cam_counter)
                        info = MASKLOCATION(prev_id[0],masks_np[mask_count],cam_counter,prev_id[1],is_touching_top,is_touching_bottom)
                        self.TRACKING_MASK_LOCATION.append(info)
                        #print("new tracking id :",prev_id[0])
                #if self.isValidSize(x1, y1, x2, y2,area):
                #    print(f'Tracking {prev_id[0]},  is small due to having {area} < {self.SMALL_SIZE}')
                #    isSmall = True  
                #else :
                    #check for missing frame count and detect again
                        
                #print(prev_id[0],'s mask area is ',area)
                # if len(self.cow_filter_size)<100 and not isSmall and not is_touching and self.isEatingArea:
                #     self.cow_filter_size.append(area)
                #     if len(self.cow_filter_size) == 100:
                #         self.average_size = (sum(self.cow_filter_size)/100)
                #         self.SMALL_SIZE = self.average_size * 0.7 #BACK TO 0.6 AND ONLY APPLY TO NEW
                #         if self.isEatingArea and self.SMALL_SIZE < 11000:
                #             self.SMALL_SIZE = 11000
                #         elif not self.isEatingArea and self.SMALL_SIZE < 7000:
                #             self.SMALL_SIZE = 7000
                #         self.BIG_SIZE = self.average_size * 1.6
                #         #self.average_size = self.average * 0.9 #some time only partial stuck at bottom
                
                #     if self.RESET_AVERAGE_SIZE_AFTER < 1:
                #         self.cow_filter_size.clear()
                #         self.RESET_AVERAGE_SIZE_AFTER = 1100
                    
                #     self.RESET_AVERAGE_SIZE_AFTER -=1
                
                #predicted_id =
                #max_freq_pos = 120
                #with lock:
                #colored_mask = self.draw_mask(colored_mask,masks_np[mask_count])
                
                #if False and not FilterSize(y1,y2,h25,poly_area,h75,max_freq_pos) :
                    #await draw_bounding_box(frame, (x1, y1, x2, y2), f's{poly_area}',4,color=(218,200,30))
                #    continue
                #print("I was skipped")
                
                if not isSmall and area < 9000:
                    isSmall = True


                
                tracked_ids.append(int(prev_id[0]))
                tracked_indexes.append(mask_count)
                
                missed_frame_count = int(prev_id[2])
                
                #print(f'tracking id {prev_id[0]} is  small ? :::::: {isSmall}')
                #asyncio.create_task(save_crop(frame,mask,x1,y1,x2,y2,save_dir,prev_id,colored_mask))
                #crop = 
                #region partial detection
                    
                #if not isSmall :
                #isSmall = False # to know actual result
                #partial detection
                predicted_id,tracking_id = self.GetBatchPredictedId_detection_only(int(prev_id[0]),isSmall,self.save_dir,missed_frame_count,prev_id[1],w,h)
                #predicted_id
                #predicted_id,tracking_id = self.GetBatchPredictedId(int(prev_id[0]),isSmall,self.save_dir,missed_frame_count,prev_id[1],w,h)
                
                #prev_id[0] = tracking_id
                self.save_crop(frame,masks_np[mask_count],x1,y1,x2,y2,self.save_dir,prev_id,colored_mask,is_touching,touching_pixels,is_save=True,is_small=isSmall,predicted_id= predicted_id,area=area)
                if predicted_id != '' and predicted_id!='Identifying' and predicted_id in predicted_ids :
                    duplicate_ids.append(predicted_id)
                
                predicted_ids.append(predicted_id)
                target_cam, updated_point = prev_id[4],prev_id[5]
                if updated_point is not None and updated_point[1] is not None:
                    if target_cam in TRACKING_PROJECTILE:
                        TRACKING_PROJECTILE[target_cam].append(updated_point)
                    else:
                        TRACKING_PROJECTILE[target_cam] = [updated_point]
        colored_mask = self.draw_mask_multiple_masks(colored_mask,masks_np,tracked_indexes)  
        print("TRACKING PROJECTILE AFTER PROCESSING ", TRACKING_PROJECTILE)
        #colored_mask =          
        #reset ideentification for tracking                    
        # if len(duplicate_ids) > 0:
            
        #     duplicate_indexes = []
        #     print(duplicate_ids,' duplicate ids')
        #     for duplicate_id in duplicate_ids:
        #         duplicate_indexes.append([index for index, value in enumerate(predicted_ids) if value == duplicate_id])
        #     duplicate_tracking_indexes = []
        #     for duplicaters in duplicate_indexes: 
        #         #print(duplicate_indexes, ' is duplicate ids and processing on ',duplicaters )
        #         for index in duplicaters:
        #             predicted_ids[index] = 'Identifying'
        #             duplicate_tracking_indexes.append(tracked_ids[index])
        #     #print("Predited ID duplicate on tracking: " , duplicate_tracking_indexes)
        #     self.reset_duplicate_tracking_identification(duplicate_tracking_indexes)
            
        #if HAS_COW:
            #predicted_ids = self.parallel_identification(data_to_pred) 
        #print("xxxxxxxxx ---------- xxxxxxxxxxxxxx")
        response_model = CoreProcessResponseModel(cam_counter, boxes ,tracked_indexes, tracked_ids, predicted_ids, original_predicted_ids, colored_mask, TRACKING_PROJECTILE)
        return response_model
        
    def CoreProcess_Parallel(self,boxes,masks_np,h,w,frame,cam_counter):
        #global frame_counter
        mask_count = -1
        tracked_ids = []
        colored_mask = np.zeros_like(frame)
        tracked_indexes = []
        predicted_ids = []
        original_predicted_ids = []
        counter = -1
        duplicate_ids = []
        height_threshold = 80
        #print("xxxxxxxxx ---------- xxxxxxxxxxxxxx")
        if not self.isEatingArea:
            height_threshold = 60
        futures = []
        #results = [None] * len(boxes)  # to store results in order
        track_info_list = []

        total_box = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            for i, box in enumerate(boxes):
                conver_time = datetime.now()
                x1, y1, x2, y2, _ = map(int, box)
                #self.diff_Time(conver_time,datetime.now(),' box convertion processing')
                
                counter += 1
                mask_count+=1   
                isSmall = False

                if(self.IsInsideAnotherBox(boxes,box,counter, cam_counter)):
                    #print("I am skipping")
                    continue #skip for this
                area = np.sum(masks_np[mask_count])

            
                touching_pixels = 0
                is_touching = False
                middle_height = 200
                #is_mask_touching = False
                if x1 <= 4 or y1 <= 4 or y2 >= self.boundary[1]: #check for 3 direction 
                    touching_pixels = self.count_border_pixels(masks_np[mask_count],(x1,y1,x2,y2))
                    is_touching = touching_pixels >= self.border_threshold
                    #is_mask_touching = touching_pixels >= 5 # touching to border . no need to check how much touching
                    if is_touching: 
                
                        isSmall = is_touching or isSmall #middle_height < height_threshold # either touching or small height
                isSmall = self.isHuman(x1,y1,x2,y2,area) or isSmall
                
                
                prev_id = self.Centroid_Tracking(masks_np[mask_count],x2-x1,y2-y1,cam_counter,y1,y2)                
            
                if(prev_id==-1): #skip cattle when prev_id // filter id is -1
                    #print("tracking skipping")
                    continue
                
                
                is_touching = False
                middle_height = 200
                
                isInMiddle = self.IsInMiddle(x1,y1,x2,y2)
                if isInMiddle and area < 6000:
                    #print("I am middle : and small with area", area )
                    continue
                is_new = prev_id[3] # check is_new
                
                tracking_id,is_touching_bottom,is_touching_top = self.find_matching_mask(masks_np[mask_count],prev_id[1],cam_counter,prev_id[0],x1,y1,x2,y2,frame)
                if tracking_id != -1:
                    if int(tracking_id) in self.STORED_IDS:
                        
                        idx = self.STORED_IDS.index(int(tracking_id))
                        self.update_tracking_mask_location(tracking_id,masks_np[mask_count],cam_counter,prev_id[1],is_touching_top,is_touching_bottom)
                        
                        self.STORED_XYXY[idx] = prev_id[1]

                        self.STORED_MISS[idx] = 1
                        
                        prev_id[0] = tracking_id
                        is_new = False
                    else:
                    
                        self.delete_mask_by_tracking_id([str(tracking_id)])
                
                if is_new:
                    
                    if isSmall and self.isHumanRatio(x2-x1,y2-y1,area) : #prioritize the size first
                        #print(" I am too small haha")
                        #frame = draw_bounding_box(frame, (x1, y1, x2, y2), f'{area}',4,color=(218,200,30))
                        continue
                    elif isInMiddle and isSmall:
                        #print(" I am too small Middle")
                        #frame = draw_bounding_box(frame, (x1, y1, x2, y2), f'{area}',4,color=(218,200,30))
                        continue
                    
                    else:  
                    #prev_id[0] = self.addNewTracking(prev_id[1],w,h) #if not touching then insert itssssssssss
                        prev_id[0] = self.addNewTracking_v2(masks_np[mask_count],prev_id[1],x1,y1,x2,y2,cam_counter,w,h)
                        print("new tracking id : ",prev_id[0], " is touching  top ", is_touching_top, " is touching bottom ", is_touching_bottom, " camera ", cam_counter)
                        info = MASKLOCATION(prev_id[0],masks_np[mask_count],cam_counter,prev_id[1],is_touching_top,is_touching_bottom)
                        self.TRACKING_MASK_LOCATION.append(info)
                #total_box +=1
                tracked_ids.append(int(prev_id[0]))
                tracked_indexes.append(mask_count)
                
                missed_frame_count = int(prev_id[2])
                track_info_list.append((i, prev_id, isSmall, missed_frame_count, x1, y1, x2, y2, mask_count, area))

                #predicted_id,tracking_id = self.GetBatchPredictedId(int(prev_id[0]),isSmall,self.save_dir,missed_frame_count,prev_id[1],w,h)
                args = (prev_id, isSmall, missed_frame_count, w, h)

                futures.append((i, executor.submit(self.call_GetBatchPredictedId, *args)))
                    
                    #prev_id[0] = tracking_id
                    #self.save_crop(frame,masks_np[mask_count],x1,y1,x2,y2,self.save_dir,prev_id,colored_mask,is_touching,touching_pixels,is_save=True,is_small=isSmall,predicted_id= "--",area=area)
                    # if predicted_id != '' and predicted_id!='Identifying' and predicted_id in predicted_ids and False:
                    #     duplicate_ids.append(predicted_id)
                    
                    # predicted_ids.append(predicted_id)
            # for i, future in futures:
            #     try:
            #         predicted_id, tracking_id = future.result()
            #         results[i] = (predicted_id, tracking_id)
            #         #self.save_crop(frame,masks_np[mask_count],x1,y1,x2,y2,self.save_dir,prev_id,colored_mask,is_touching,touching_pixels,is_save=True,is_small=isSmall,predicted_id= predicted_id,area=area)
            #         if predicted_id != '' and predicted_id!='Identifying' and predicted_id in predicted_ids and False:
            #             duplicate_ids.append(predicted_id)
                    
            #         predicted_ids.append(predicted_id)
            #     except Exception as e:
            #         print(f"Error in prediction for index {i}: {e}")
            #         results[i] = ('Error', -1)    
            
            # Phase 2: Get predicted IDs in parallel
            results = [None] * len(track_info_list)
            #print(f"futures len: {len(futures)} , track_info_list len: {len(track_info_list)}")
            future_counter = 0
            for idx, future in futures:
                try:

                    predicted_id, tracking_id = future.result()
                    results[future_counter] = (predicted_id, tracking_id)
                    #print(f"Predicted ID for index {idx}: {predicted_id}, Tracking ID: {tracking_id}")
                except Exception as e:
                    #print("future result in error ", idx, " idx ",future.result())
                    print(f"Error: {e}")
                    results[idx] = ('Identifying', tracking_id)
                
                future_counter += 1

                #predicted_ids.append(predicted_id)

            for i, (track_idx, prev_id, isSmall, missed_frame_count, x1, y1, x2, y2, mask_count, area) in enumerate(track_info_list):
                predicted_id, tracking_id = results[i]

                # Save mask or crop
                self.save_crop(frame, masks_np[mask_count], x1, y1, x2, y2, self.save_dir,
                            prev_id, colored_mask, is_touching, touching_pixels,
                            is_save=True, is_small=isSmall, predicted_id=predicted_id, area=area)

                
                predicted_ids.append(predicted_id)
        response_model = CoreProcessResponseModel(cam_counter, boxes ,tracked_indexes, tracked_ids, predicted_ids, original_predicted_ids, colored_mask)
        return response_model
    
    def stack_image_from_bottom_to_top(self,image_list):
        """
        Stack images from bottom to top with a maximum height for each image.
        """
        #604 603 602 601 format
        
        stacked_image = None
        for img in image_list:
            #img = cv2.imread(imgpath)
             
            if stacked_image is None:
                #stacked_image = cv2.resize(img, (self.image_width, self.image_height))
                stacked_image = img
            else:
                # Resize the new image to fit the width of the stacked image

                
                # resized_img = cv2.resize(img, (self.image_width, self.image_height))
                
                # Stack the images vertically
                stacked_image = np.vstack((stacked_image, img))
        #print(stacked_image.shape, ' image shape')
        return stacked_image

    def dispose_model(self,predictor):
    # If using GPU, free up CUDA memory
    
        del predictor  # Remove reference
        torch.cuda.empty_cache()  # Clear the CUDA memory
        print("Model disposed and GPU memory released.")

    def get_duplicate_indexes(self,array):
        index_map = defaultdict(list)
        for i, val in enumerate(array):
            index_map[val].append(i)

        # Only keep values with duplicates
        duplicates = {val: idxs for val, idxs in index_map.items() if len(idxs) > 1}
        return duplicates
    def get_max_start_time_per_video(self,video_list):
        max_start_time = 0
        start_times = []
        print(video_list)

        for video in video_list:
            start_time = int(video.split('\\')[-1].split("-")[1])
            #print(start_time)
            start_times.append(start_time)
            if start_time > max_start_time:
                max_start_time = start_time
        
        #print(start_times)
        #extrack max start time from every start time
        start_times = [max_start_time - x for x in start_times]

        return start_times

    
    # ---------- STITCH IMAGES ----------
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


        #cropped1 = add_black_blocks(cropped1, positions_1_1, block_size_1_1)
        #cropped1 = add_black_blocks(cropped1, positions_1_2, block_size_1_2)

            

        positions_2_1 = [(280,0)]
        block_size_2_1 = (400, overlap) 
        cropped2 = self.add_black_blocks(cropped2, positions_2_1, block_size_2_1)

        #positions_2_2 = [(670,0)]
        #block_size_2_2 = (2000, overlap) 
        #cropped2 = add_black_blocks(cropped2, positions_2_2, block_size_2_2)

        #positions_2_2 = [(0,2000)]
        #block_size_2_2 = (644,overlap)
        #cropped2 = add_black_blocks(cropped2, positions_2_2, block_size_2_2)

        #positions_3 = [(0,0)]
        #block_size_3 = (380, overlap)
        #cropped3 = self.add_black_blocks(cropped3, positions_3, block_size_3)

        #image = cv2.imwrite("D:/stitched_tetris_cropped1.png",cropped1)
        #image = cv2.imwrite("D:/stitched_tetris_cropped2.png",cropped2)

        tetris1_2 = self.stack_tetris_crops(cropped1, cropped2, overlap=overlap)
        #stitched2_3 = np.vstack([cropped2, cropped3])
        #stitched = np.vstack([cropped1, cropped2, cropped3])
        #stitched = np.vstack([tetris1_2, cropped3])
        #return [tetris1_2, cropped3]
        #return stitched
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

        #cropped3 = self.draw_quarter_circles(cropped3,130)

        #stitched = np.vstack([cropped3,cropped4])
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
        #img3 = rotate_image(img3, 1)  # Rotate cam 3 by 2 degrees
        img3 = self.rotate_image(img3, -1)  # Rotate img3 by 45 degrees clockwise
        cropped1 = crop(img1, cut_top1, cut_bot1)
        cropped2 = crop(img2, cut_top2, cut_bot2)
        cropped3 = crop(img3, cut_top3, cut_bot3)
        cropped4 = crop(img4, cut_top4, cut_bot4)
        #cropped2 = apply_tetris_crop(cropped1, x1=150, x2=340, length=20, shape_type='T')
        #cropped2 = apply_tetris_crop(cropped2, x1=160, x2=670, length=20, shape_type='U')
        overlap = 30
      
        positions_2 = [(0,2000)]
        block_size_2 = (400, overlap) 
        #cropped2 = add_black_blocks(cropped2, positions_2, block_size_2)


        positions_2_1 = [(380,2000)]
        block_size_2_1 = (644,overlap)
        #cropped2 = add_black_blocks(cropped2, positions_2_1, block_size_2_1)

        positions_3 = [(0,0)]
        block_size_3 = (500, overlap)
        #cropped3 = add_black_blocks(cropped3, positions_3, block_size_3)

        #image = cv2.imwrite("D:/stitched_tetris_cropped1.png",cropped1)
        #image = cv2.imwrite("D:/stitched_tetris_cropped2.png",cropped2)

        #tetris2_3 = stack_tetris_crops(cropped2, cropped3, overlap=overlap)
        #stitched2_3 = np.vstack([cropped2, cropped3])
        #stitched = np.vstack([cropped1, cropped2, cropped3])
        #stitched = np.vstack([cropped1, tetris2_3])
        #return [cropped1, tetris2_3, cropped4]
        return [cropped1, cropped2,cropped3, cropped4] #stitched 
        
    def stack_tetris_crops(self,img1,img2, overlap = 20):
        h1, w = img1.shape[:2]
        h2 = img2.shape[0]

        # Calculate the final height
        final_height = h1 + h2 - overlap

        # Initialize result image
        result = np.zeros((final_height, w, 3), dtype=np.uint8)

        # Copy top part of img1 (non-overlapping)
        result[:h1 - overlap] = img1[:h1 - overlap]

        # Blend the overlapping region
        overlap_img1 = img1[h1 - overlap:]
        overlap_img2 = img2[:overlap]

        # Create a mask where img2 is non-black
        mask = np.any(overlap_img2 != [0, 0, 0], axis=2)

        # Start with overlap from img1
        blended_overlap = overlap_img1.copy()

        # Overwrite black pixels in img1 with non-black from img2
        blended_overlap[mask] = overlap_img2[mask]

        # Place blended overlap
        result[h1 - overlap:h1] = blended_overlap

        # Copy the remaining part of img2
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
        #print(xy_positions, ' positions to draw black blocks')
        for x, y in xy_positions:
            x_end = min(x + w_blk, w_img)
            y_end = min(y + h_blk, h_img)
            x_start = max(x, 0)
            y_start = min(max(y, 0),h_img-h_blk)
            #print('Drawing from ({}, {}) to ({}, {})'.format(x_start, y_start, x_end, y_end))
            image[y_start:y_end, x_start:x_end] = (0, 0, 0)  # Black in BGR

        return image

    def rotate_image(self,image, angle):
        (w, h) = self.resolution
        #print(self.resolution, ' resolution')
        center = (w // 2, h // 2)

        # Rotate by 45 degrees clockwise
        M = cv2.getRotationMatrix2D(center, angle, 1.0)  # Negative for clockwise
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

        #camera 11-14
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
        #cropped2 = apply_tetris_crop(cropped1, x1=150, x2=340, length=20, shape_type='T')
        #cropped2 = apply_tetris_crop(cropped2, x1=160, x2=670, length=20, shape_type='U')
        overlap = 30
        #positions_1_1 = [(0,2000)]  #for cam 4,5,6 x,y 
        #block_size_1_1 = (145,overlap)
        #positions_1_2 = [(670,2000)]
        #block_size_1_2 = (1152-670, overlap)  


        #cropped1 = add_black_blocks(cropped1, positions_1_1, block_size_1_1)
        #cropped1 = self.add_black_blocks(cropped1, positions_1_2, block_size_1_2)

        #positions_2 = [(0,0)]
        #block_size_2 = (670, overlap) 
        #cropped2 = self.add_black_blocks(cropped2, positions_2, block_size_2)


        positions_2_1 = [(380,2000)]
        block_size_2_1 = (644,overlap)
        cropped2 = self.add_black_blocks(cropped2, positions_2_1, block_size_2_1)

        positions_3 = [(0,0)]
        block_size_3 = (380, overlap)
        cropped3 = self.add_black_blocks(cropped3, positions_3, block_size_3)

        #image = cv2.imwrite("D:/stitched_tetris_cropped1.png",cropped1)
        #image = cv2.imwrite("D:/stitched_tetris_cropped2.png",cropped2)

        tetris2_3 = self.stack_tetris_crops(cropped2, cropped3, overlap=overlap)
        #stitched2_3 = np.vstack([cropped2, cropped3])
        #stitched = np.vstack([cropped1, cropped2, cropped3])
        #stitched = np.vstack([cropped1, tetris2_3])
        return [cropped1, tetris2_3, cropped4]

    # ---------- TILE IMAGE VERTICALLY ----------
    def get_vertical_tiles(self, image, tile_height=800, overlap=200):
        h, w = image.shape[:2]
        tiles = []
        y_starts = list(range(0, h, tile_height - overlap))

        for y in y_starts:
            y_end = min(y + tile_height, h)
            tile = image[y:y_end]
            tiles.append((tile, y))
        return tiles

    # ---------- RUN DETECTION ON TILES ----------
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


    # ---------- MERGE DETECTIONS WITH NMS ----------
    def apply_nms(self, all_boxes, all_scores, all_classes, all_masks, iou_threshold=0.5):
        if not all_boxes:
            return [], [], [], []

        boxes = np.vstack(all_boxes)
        scores = np.hstack(all_scores)
        classes = np.hstack(all_classes)
        masks = np.vstack(all_masks)  # shape: [N, H, W]

        boxes_tensor = torch.tensor(boxes)
        scores_tensor = torch.tensor(scores)

        keep = nms(boxes_tensor, scores_tensor, iou_threshold)

        final_boxes = boxes_tensor[keep].numpy()
        final_scores = scores_tensor[keep].numpy()
        final_classes = classes[keep.numpy()]
        final_masks = masks[keep.numpy()]

        return final_boxes, final_scores, final_classes, final_masks


    # ---------- DRAW DETECTIONS ----------
    def draw_detections(self, image, boxes, scores, classes):
        image_copy = image.copy()
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{cls} {score:.2f}"
            cv2.putText(image_copy, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return image_copy

    def draw_quarter_circles(self,image, radius=100):
        h, w = image.shape[:2]

        # Top-left corner (from corner to center - upper left quarter)
        cv2.ellipse(
            image,
            center=(0, 0),
            axes=(int(radius/2), int(radius/2)),
            angle=0,
            startAngle=0,
            endAngle=90,
            color=(0, 0, 0),
            thickness=-1
        )

        # Top-right corner (upper right quarter)
        cv2.ellipse(
            image,
            center=(w, 0),
            axes=(radius, radius),
            angle=0,
            startAngle=90,
            endAngle=180,
            color=(0, 0, 0),
            thickness=-1
        )

        return image
    def get_Tracking_To_Merge(self,data):
        # Dictionary to hold indices of each value
        value_indices = defaultdict(list)
        #print("")
        # Populate the dictionary
        for idx, val in enumerate(data):
            value_indices[val].append(idx)

        # Extract duplicates and their indexes
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
        # === Save images ===
        for i, img in enumerate(images):
            base_path = Path(self.save_dir) / f"frame_saver_{i}"
            base_path.mkdir(parents=True, exist_ok=True)

            img_path = base_path / f"{self.frame_number}.jpg"
            cv2.imwrite(str(img_path), img)

        # === Save detection outputs (boxes + masks only) ===
        output_dir = Path(self.save_dir) / "detections"
        output_dir.mkdir(parents=True, exist_ok=True)

        boxes_list, masks_list = [], []
        for out in outputs:
            boxes = out["instances"].pred_boxes.tensor.cpu().numpy()
            masks = out["instances"].pred_masks.cpu().numpy()

            boxes_list.append(boxes)
            masks_list.append(masks)

        # Save boxes as JSON
        with open(output_dir / f"{self.frame_number}_boxes.json", "w") as f:
            json.dump([b.tolist() for b in boxes_list], f)

        # Save masks as compressed npz
        np.savez_compressed(output_dir / f"{self.frame_number}_masks.npz", *masks_list)

    def load_data_to_skip(self, frame_number):
        # === Load images ===
        images = []
        i = 0
        while True:
            img_path = Path(self.path) / f"frame_saver_{i}" / f"{frame_number}.jpg"
            if not img_path.exists():
                break
            img = cv2.imread(str(img_path))
            images.append(img)
            i += 1

        # === Load boxes ===
        boxes_file = Path(self.save_dir) / "detections" / f"{frame_number}_boxes.json"
        with open(boxes_file, "r") as f:
            boxes_list = json.load(f)

        # === Load masks ===
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
        #if isCam:
        #    video_file = video_dir
        #else:
        #    video_file = f"{video_dir}" if isFile else f"{video_dir}\\{file_name}"
        
        self.boundary = self.save_width-7,self.save_height-7
        #print(video_file,' I am video file')
        # video_readers = cv2.VideoCapture(video_file)
        video_readers = []
        apply_correction_on = {}
        #distortion_keys,distortion_settings
        video_counter = 0
        for vdo in data:
            
            video_reader = cv2.VideoCapture(vdo)
            

            video_readers.append(video_reader)
            video_counter += 1
        
        is_save_video = cap is not None
        is_save_tracking_video = tracking_cap is not None
        #is_save_image = True
        print('are we saving video? ',is_save_video)
        print('are we saving tracking video? ',is_save_tracking_video)
        print(f'current values small size {self.SMALL_SIZE}')

        #w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        #h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        middle = self.save_height * 0.25
        h25= self.save_height*0.25
        h75 = self.save_height*0.75
        #h90 = h*0.9

               
      
        skip_frame = self.frame_skip_count # math.ceil(CAP_PROP_FPS / 5)-1   
        frame_count = skip_frame+1
        
        ret = True
        skip_minute,skip_second = 4,45
        manual_skip = 0 #(skip_minute * 60 + skip_second)  * 20
        
        while ret:
            #print("are we reading?")
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
            #if not ret:
            #    return 
            if(len(images) != len(video_readers)):
                print("All images len not match, len is ", len(images), " , videos len is ", len(video_readers))
                return
            #self.camera_heights.clear()
            if len(images) > 1:
                images = self.vertical_stitch_cut_both_4_6(
                            images[0], images[1], images[2],
                            #cut_top1=0.0,  cut_bot1=1-0.8833,  # Keep top of Cam1, cut 18% bottom
                            #cut_top2=0.4208, cut_bot2=1-0.7916,  # Cut 18% top and 21% bottom of Cam2
                            #cut_top3=0.19, cut_bot3=0.0    # Cut 21% top of Cam3, keep full bottom
                            #cut_top1=0.0,  cut_bot1=1-0.9093,  # Keep top of Cam1, cut 18% bottom
                            #cut_top2=0.4208, cut_bot2=1-0.7916,  # Cut 18% top and 21% bottom of Cam2
                            #cut_top3=0.19, cut_bot3=0.0    # Cut 21% top of Cam3, keep full bottom

                             #camera 11-14
                            # cut_top1=0.0,  cut_bot1=1-0.755,  # Keep top of Cam1, cut 18% bottom  // 20 pixels is 0.026 %  #14
                            # cut_top2=0.15, cut_bot2=1-0.815,  # Cut 18% top and 21% bottom of Cam2  #13 with tetris
                            # cut_top3=0.225, cut_bot3=1-0.829,    # Cut 21% top of Cam3, keep full botto  #12
                            # cut_top4=0.143, cut_bot4 = 1-0.781   #11
                            #camera 11-14

                            #camera 7-10
                            
                            #cut_top1=0.202,  cut_bot1=1-0.833,  # Keep top of Cam1, cut 18% bottom  // 20 pixels is 0.026 %  #10
                            #cut_top2=0.182, cut_bot2=1-0.775,  # Cut 18% top and 21% bottom of Cam2  #9
                            #cut_top3=0.202, cut_bot3=1-0.846,    # Cut 21% top of Cam3, keep full botto  #8
                            #cut_top4=0.229, cut_bot4 = 1-0.862   #7
                            #camera 7-10

                            #camera 4-6
                            #cut_top1=0.0,  cut_bot1=1-0.9093,  # Keep top of Cam1, cut 18% bottom Camera 6
                            #cut_top2=0.4208, cut_bot2=1-0.7916,  # Cut 18% top and 21% bottom of Camera 2
                            #cut_top3=0.19, cut_bot3=0.0    # Cut 21% top of Cam3, keep full bottom Camera 1

                            cut_top1=0.202,  cut_bot1=1-0.833,  # Keep top of Cam1, cut 18% bottom  // 20 pixels is 0.026 %  #10
                            cut_top2=0.182, cut_bot2=1-0.775,  # Cut 18% top and 21% bottom of Cam2  #9
                            cut_top3=0.202, cut_bot3=1-0.846,    # Cut 21% top of Cam3, keep full botto  #8
                            #camera 4-6
                        )
                #images = stitched
                if len(self.camera_heights) == 0:
                    print ("Adding images shape first time")
                    for i in range(len(images)):
                        self.camera_heights.append(images[i].shape[0])
                    #print(" after stitching , all images len is ", len(images))
                
            
            
            frame_count = 0
            batch_outputs = list(self.stupid_detector(images))
            # self.save_data_to_skip(images, batch_outputs)
            # print('predicted outputs:',batch_outputs)
            self.frame_number +=1
            #print("Frame number: ", frame_number)
            all_frame_infos = []
            cam_counter = 0
            #print("xxxxxxxxx ---Core start------- xxxxxxxxxxxxxx")
            for outputs in batch_outputs:  #per frame
                
                #print(len(outputs), ' is the length of outputs')
                frame = images[cam_counter]
                start_time = datetime.now()
                #print('how come')
                instances = outputs["instances"].to("cpu")
                #boxes = instances.pred_boxes.tensor.numpy()
                #classes = instances.pred_classes.numpy()
                #masks_np = instances.pred_masks.numpy()
                boxes,masks_np = self.sortBoxAndMask(instances.pred_boxes.tensor.numpy(),instances.pred_masks.numpy())
                mask_count = 0
                h, w, _ = frame.shape
                #print(outputs)
                HAS_COW = False
                
                
                
                start_time = datetime.now()
                
                all_frame_infos.append(self.CoreProcess(boxes,masks_np,h,w,frame, cam_counter))
                #all_frame_infos.append(self.CoreProcess_Parallel(boxes,masks_np,h,w,frame, cam_counter))
                
                #all_frame_infos.append(coreProcessResponse)
                
                
                self.diff_Time(start_time,datetime.now(),' tracking and validation process')
                
                
                #frame = cv2.addWeighted(images[cam_counter], 1.0, coreProcessResponse.colored_mask, 0.5, 0)
                images[cam_counter] = frame
                cam_counter+=1

            #update touching cattle count per cam for each camera
            #self.set_touching_cattle_count_per_cam()
            #print("xxxxxxxxx ---Core end------- xxxxxxxxxxxxxx")
            all_tracking_ids = [item for sublist in all_frame_infos for item in sublist.tracked_ids]
            all_duplicate_idexes = self.get_duplicate_indexes([item for sublist in all_frame_infos for item in sublist.predicted_ids])
            
            tracking_to_merge = self.get_Tracking_To_Merge(all_tracking_ids)
            #if len(tracking_to_merge) > 0:
            #    print(f"Need to merge {tracking_to_merge}")
            
            all_duplicate_tracking_ids = []
            #print(f"{all_tracking_ids} is all tracking id")
            #duplicate across all images
            
            boxes_to_merge = defaultdict(list)
            for duplicate in all_duplicate_idexes:
                if duplicate == "Identifying" or duplicate== "Reidentifying" or "unknown":
                    continue
                #print(duplicate, " duplicate value" )
                duplicate_tracking_ids = [all_tracking_ids[i] for i in all_duplicate_idexes[duplicate]]
                #print(duplicate_tracking_ids, " duplicate tracking ids")
                #is same tracking across images?
                is_same = all(val == duplicate_tracking_ids[0] for val in duplicate_tracking_ids)
                #no same tracking id across images, reset all
                if not is_same:
                    for id in duplicate_tracking_ids:
                        all_duplicate_tracking_ids.append(id)
                #print(is_same," is same ", list(set(duplicate_tracking_ids)))

            #Duplilcate across all images
            if(len(all_duplicate_tracking_ids) > 0):
                #print("resetting duplilcate ids from tracking : ",all_duplicate_tracking_ids   )
                self.reset_duplicate_tracking_identification(all_duplicate_tracking_ids)
            #all_duplicate_tracking_ids.clear()
            
            for frame_info in all_frame_infos:
                #print(frame_info.tracked_ids, ' is the tracked ids')
                mask_count = 0

                images[frame_info.cam_counter] = cv2.addWeighted(frame_info.colored_mask, 0.3,images[frame_info.cam_counter], 1 - 0.3, 0)
                for index in frame_info.tracked_indexes:
                    #print("I am cow")
                    x1, y1, x2, y2, area = map(int, frame_info.boxes[index])
                    
                    #print(tracked_ids, '  are predicted ids')
                    #await(draw_bounding_box(original_frame,(x1,y1,x2,y2),original_predicted_ids[mask_count],font_scale=1)) # draw with predicted id
                    #label = f'{str(tracked_ids[mask_count])}_{str(predicted_ids[mask_count])}'
                    #if isCam:
                    tracking_id = frame_info.tracked_ids[mask_count]
                    label = 'Identifying' if frame_info.tracked_ids[mask_count] in all_duplicate_tracking_ids else f'{str(frame_info.predicted_ids[mask_count])}'
                    if tracking_id in tracking_to_merge.keys():
                        sum_y = 0
                        if frame_info.cam_counter > 0:
                            sum_y = sum(self.camera_heights[:frame_info.cam_counter])
                        boxes_to_merge[tracking_id] += [[x1,y1+sum_y,x2,y2+sum_y],label]
                    else:
                        self.draw_bounding_box(images[frame_info.cam_counter],(x1,y1,x2,y2),label,str(frame_info.tracked_ids[mask_count]),font_scale=1) # draw with predicted id
                    # if is_save_tracking_video:
                    #     self.draw_bounding_box(original_frame,(x1,y1,x2,y2),label,tracked_ids[mask_count],font_scale=1) # draw with predicted id
                    #self.draw_mask(images[frame_info.cam_counter],frame_info.masks_np[mask_count])
                    mask_count+=1
            
            
            self.IncreaseMissedCount(all_tracking_ids)
            start_time = datetime.now()
            #frame = cv2.resize(frame, (1088, 1088))
            stacked_image = self.stack_image_from_bottom_to_top(images)
            #stacked_image = images[0]
            #print("stacked image shape is ", stacked_image.shape)
            for key,box_to_merge in boxes_to_merge.items():
                #print(key, box_to_merge)
                x1,y1,x2,y2 = self.combine_boxes(box_to_merge[0],box_to_merge[2])
                label = box_to_merge[1]
                try:
                    if y2-y1 > 650:
                        print(" too big to merge")
                        x1,y1,x2,y2 = box_to_merge[0]
                        self.draw_bounding_box(stacked_image,(x1,y1,x2,y2),box_to_merge[1],str(key),font_scale=1) # draw with predicted id
                        x1,y1,x2,y2 = box_to_merge[2]
                        self.draw_bounding_box(stacked_image,(x1,y1,x2,y2),box_to_merge[3],str(key),font_scale=1) # draw with predicted id
                    #else:
                    else:
                        self.draw_bounding_box(stacked_image,(x1,y1,x2,y2),label,str(key),font_scale=1) # draw with predicted id
                except:
                    print("Error in merging boxes for key:", key, "with box_to_merge:", box_to_merge)
            if stacked_image is None:
                print("Stacked image is None")
                break
            if is_save_video:
                #print('I am saving the footage of moon')
                #showVideo = cv2.resize(frame, (1080, 1080), interpolation=cv2.INTER_AREA)
                #print("saving stacked_iamge in shape", stacked_image.shape)
                cap.write(stacked_image)


            if is_save_tracking_video:
                #print('I am saving the footage of venus')
                tracking_cap.write(original_frame)
                
            self.diff_Time(start_time,datetime.now(),' file writing process')    # resizedImage = cv2.resize(v.get_image(), (1000, 1000)) 
            
            start_time = datetime.now()

            if is_show_image and True:
                #cv2.imshow("kunneppu",cv2.resize(frame, (1080, 1080)) )
                imshow_size = 576 , int((self.image_height/2)*len(data))
                #print(imshow_size, " Imshow size")
                cv2.imshow("NASA doing centroid tracking on the cattle (detectron 2)", cv2.resize(stacked_image, (imshow_size)))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    pass
                    self.dispose_model(self.stupid_detector)
                    return -1
            self.diff_Time(start_time,datetime.now(),' image showing process')

            self.diff_Time(total_time_start,datetime.now(),' the whole process')
        #self.dispose_model(predictor)
        return    

           
    

    def find_mask_center(self,mask,cam_counter):
        # Ensure mask is binary
        if not np.issubdtype(mask.dtype, np.bool_):
            mask = mask > 0
        
        # Find the coordinates of non-zero pixels
        y_coords, x_coords = np.where(mask)
        
        # Compute the center of the mask
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
         #region Adjust height for multi cams
        #center_y = center_y +  (0 if cam_counter == 0 else self.camera_heights[cam_counter-1])
        center_y = center_y +  (0 if cam_counter == 0 else sum(self.camera_heights[:cam_counter]))
        #if cam_counter > 0:
            #print("Center Y is adjusted for camera ", cam_counter, " to ", center_y)
        return (center_x, center_y)



    def drawArrayOverImage(self,img):
        
        # Parameters for drawing
        spacing = 0  # spacing between rectangles
        rect_height = 70   # height of each rectangle
        rect_width =  100 # width of each rectangle

        # Offset to position the arrays correctly on the image
        x_offset = 50
        y_offset = 2500

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 255)
        font_thickness = 2
        labels = ['ID','MISS']
        

        # Iterate over arrays and draw them
        for i in range(2):
            array = self.STORED_IDS if i == 0 else self.STORED_MISS
        
        #print('damn I am too kind!!! ',(zip(STORED_IDS, STORED_MISS)))
        #for i, (array) in (zip(STORED_IDS, STORED_MISS)):
            # Draw the label
            #print("I am drawling array")
            
            label_size = cv2.getTextSize(labels[i], font, font_scale, font_thickness)[0]
            label_x = x_offset - label_size[0] - 10
            label_y = y_offset + i * (rect_height + spacing) + rect_height // 2 + label_size[1] // 2
            
            # Draw the label
            cv2.putText(img, labels[i], (label_x, label_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            #for k in range(2):
            for j in range (26):
                
                value = '-'
                if j< len(array):
                    value = array[j]
                
                #y_dy_offset = y_offset if j<13 else y_offset * 2    
                #print('I am drawing Array value')
                top_left = (x_offset + j * (rect_width + spacing), y_offset + i * (rect_height + spacing))
                bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)
                
                # Draw rectangle
                cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), 5)
                
                # Put the text inside the rectangle
                text_position = (top_left[0] + rect_width // 2 - 20, top_left[1] + rect_height // 2 + 10)
                cv2.putText(img, str(value), text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        return img

    def clearData(self):
        #global CATTLE_LOCAL_ID
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
        #global TRACKER_clone
        #global ALL_TIME_CLASSIFICATION_TRACKER_clone
        #global class_names_clone
        #TRACKER_clone = self.TRACKER
        #ALL_TIME_CLASSIFICATION_TRACKER_clone = self.ALL_TIME_CLASSIFICATION_TRACKER
        print("Saving finalize pdf :",save_dir)
        default_data = ["ID"]
        final_result = []
        missed_frame_data = []
        default_missed_frame_data = ['Tracking','Total','Missed']
        missed_frame_data.append(default_missed_frame_data)
        #class_names_clone = self.class_names
        for i in (self.class_names):
            default_data.append(self.class_names[i])
        #print(default_data)
        
        final_result.append(default_data)
        #print(f"===> Stored data  Tracker {self.TRACKER}")
        #print(f"===> Stored data  ALL_TIME_CLASSIFICATION_TRACKER {self.ALL_TIME_CLASSIFICATION_TRACKER}")
        for index in range(len(self.TRACKER)):
            try:

                tracking_id = self.TRACKER[index]
                if(tracking_id is None):
                    continue
                track_index = len(self.TRACKER) - 1
                #dataset = list(range(100,141))
                dataset = self.ALL_TIME_CLASSIFICATION_TRACKER[index]
                stx = [int(tracking_id)]
                
                missed_data = []
                missed_data.append(tracking_id)
                missed_data.append(self.BATCH_CLASSIFICATION_TRACKER[index]['TOTAL_DETECTION'])
                missed_data.append(self.BATCH_CLASSIFICATION_TRACKER[index]['TOTAL_MISSED_FRAME'])
                missed_frame_data.append(missed_data)
                
                #batch_default = {'TOTAL_COUNT':0 , 'HOLDING' : 'Identifying','LOCATION' : location,'TOTAL_MISSED_FRAME':0,'TOTAL_DETECTION':0}
                #self.BATCH_CLASSIFICATION_TRACKER.append(batch_default))
                for idx in (self.class_names):
                    id = str(self.class_names[idx])
            #        print(id)
                    if id not in self.ALL_TIME_CLASSIFICATION_TRACKER[index]['GT']:
                        stx.append(0)    
                        #print(id,0)
                    else:
                        predicted_index = self.ALL_TIME_CLASSIFICATION_TRACKER[index]['GT'].index(id)
                        predicted_count = self.ALL_TIME_CLASSIFICATION_TRACKER[index]['COUNT'][predicted_index]
                        #print(id,predicted_count)
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
        #final_result.append("ID")
        for key in (self.TRACKING_RESULT):
            finalData = []
            finalData.append(key)
            #finalData.append(result_list[key])
            
            for value in self.TRACKING_RESULT[key]:
                finalData.append(value)
            final_result.append(finalData)
        #print(final_result)   
        df = pd.DataFrame(final_result)
        df.to_csv(f"{save_dir}\\{file_name}.csv", index=False,header=False)
        
        #self.save_missed_frame_count(save_dir)
    
        

    def has_long_straight_lines(self,mask, min_length=100, frame_height=None):
        #print(mask, ' is mask')
        height = frame_height
        # Check for horizontal lines
        has_horizontal = False
        horizontal_y2 = None
        for y, row in enumerate(mask):
            line_starts = np.where(np.diff(np.concatenate(([False], row))))[0]
            line_ends = np.where(np.diff(np.concatenate((row, [False]))))[0]
            
            # Ensure line_starts and line_ends have the same shape
            min_len = min(len(line_starts), len(line_ends))
            line_lengths = line_ends[:min_len] - line_starts[:min_len]
            
            if np.any(line_lengths >= min_length):
                has_horizontal = True
                horizontal_y2 = y
                break

        # Check for vertical lines
        has_vertical = False
        vertical_y2 = None
        for x, col in enumerate(mask.T):
            line_starts = np.where(np.diff(np.concatenate(([False], col))))[0]
            line_ends = np.where(np.diff(np.concatenate((col, [False]))))[0]
            
            # Ensure line_starts and line_ends have the same shape
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

        # Check for horizontal lines
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

        # Check for vertical lines
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
            return 100 #"vertical"
        elif width > height * ratio_threshold:
            return 150#"horizontal"
        else:
            return 80# "diagonal"
    def is_showing_most_body(self,mask, bbox):
        centroid_threshold = self.check_orientation(bbox)
        #print(centroid_threshold, ' is centroid threshold')
        
        (x,y) = self.find_mask_center(mask,0)
        #print(x , ' : ',  y, ' is centroid')
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

        # Check if the bounding box is partially outside the mask
       
        # Count pixels touching each border
        left_count = np.sum(mask[y1:y2+1, max(0, x1-border_margin):min(width, x1+border_margin)])
        if self.isEatingArea:
            right_count = 0
        else:
            right_count = np.sum(mask[y1:y2+1, max(0, x2-border_margin):min(width, x2+border_margin)])
        top_count = np.sum(mask[max(0, y1-border_margin):min(height, y1+border_margin), x1:x2+1])
        bottom_count = np.sum(mask[max(0, y2-border_margin):min(height, y2+border_margin), x1:x2+1])

        # counts = {
        #     'left': int(left_count),
        #     'right': int(right_count),
        #     'top': int(top_count),
        #     'bottom': int(bottom_count)
        # }

        # # Check if any border count exceeds the threshold
        # is_partial = any(count >= threshold for count in counts.values())
        #if is_partial:
        #    exceeding_counts = {value for key, value in counts.items() if value >= threshold}
        #    print(exceeding_counts)
        
        counts = [
             int(left_count),
             0,
             int(top_count),
              int(bottom_count)
         ]
        maxTouching =  max(counts) #, counts
        #if maxTouching == True:
        #    print(counts)
        return maxTouching
    
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

        # Check if the bounding box is partially outside the mask
        #if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
        #    return True #, {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}

        # Count pixels touching each border
        left_count = np.sum(mask[y1:y2+1, max(0, x1-border_margin):min(width, x1+border_margin)])
        if self.isEatingArea:
            right_count = 0
        else:
            right_count = np.sum(mask[y1:y2+1, max(0, x2-border_margin):min(width, x2+border_margin)])
        top_count = np.sum(mask[max(0, y1-border_margin):min(height, y1+border_margin), x1:x2+1])
        bottom_count = np.sum(mask[max(0, y2-border_margin):min(height, y2+border_margin), x1:x2+1])

        # counts = {
        #     'left': int(left_count),
        #     'right': int(right_count),
        #     'top': int(top_count),
        #     'bottom': int(bottom_count)
        # }

        # # Check if any border count exceeds the threshold
        # is_partial = any(count >= threshold for count in counts.values())
        #if is_partial:
        #    exceeding_counts = {value for key, value in counts.items() if value >= threshold}
        #    print(exceeding_counts)
        
        counts = [
             int(left_count),
             int(right_count),
             int(top_count),
              int(bottom_count)
         ]
      
        #print("Printing border pixels for touching_pixels getting True : ", counts)
        
    
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

        # Determine whether to use X or Y plane based on the longer dimension of the bbox
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

        # Find the center of the mask
        height, width = mask.shape
        center_y = height // 2

        # Find the first True value from top and bottom
        true_rows = np.where(mask.any(axis=1))[0]
        if len(true_rows) == 0:
            return 0  # No True values in the mask

        top = true_rows[0]
        bottom = true_rows[-1]

        # Calculate the height from the center
        height_above = max(center_y - top, 0)
        height_below = max(bottom - center_y, 0)

        # Return the maximum height from center
        return max(height_above, height_below)
    
    def get_all_videos(self,path: str, extensions=None):
        if extensions is None:
            extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        
        path = Path(path)
        video_files = [str(p) for p in path.rglob("*") if p.suffix.lower() in extensions]
        return video_files
    def begin_process(self,cam_datas,isCam,camera_number=None,lane = "Left"):
        cap = None
        tracking_cap = None
        
        number_of_cams = len(cam_datas)
        number_of_videos = len(cam_datas[0])
        
        self.frame_number = 0
        program_start_time = datetime.now()
        for idx in range(number_of_videos): 
            
            
            video_paths = []
            for i in range(number_of_cams):
                #videos.append(cam_datas[i][idx])
                #print(cam_datas[i][idx], ' is video file')
                cam_videos = self.get_all_videos(cam_datas[i][idx])
                #for vid in cam_videos:
                #    print(vid)
                video_paths.append(cam_videos)
            print(len(video_paths), ' is video len and len video_paths[0] ', len(video_paths[0]))
            video_dir = cam_datas[0][0]
            #print(video_dir)
            csv_file = "tracking_prediction.csv"
            DATE = video_dir.split("\\")[-2]+'_'+video_dir.split("\\")[-1]
            channel = video_dir.split("\\")[-3] #hourly
            model_name = self.model_path.split("\\")[-1]
            project = f'E:/Output/runs/{self.FARMNAME}_identification/KNP_Merge_and_Batch_identification/26 Sept Report 2025/reuse tetris_Sumiyoshi-44/{model_name}/testris_{channel}'
            name = f'{DATE}_part ' #weight path and iou threshold
            
            self.save_dir = Helper.increment_path(Path(project) / name,mkdir=True)
            csv_file = f'{self.save_dir}\\{csv_file}'
            #print(save_dir)
            #region Video data
            save_vid_name = video_dir.split("\\")[-1]+"_identification" # open this when running multiple videos
            tracking_vid_name = video_dir.split("\\")[-1]+"_tracking" # open this when running multiple videos

            isFile = False
            if video_dir.endswith(".mp4") or video_dir.endswith(".mkv"):
                isFile = True
                save_vid_name=  video_dir.split('\\')[-1].replace('.mkv','_track').replace('.mp4','_track')  #open this when running single video
                tracking_vid_name=  video_dir.split("\\")[-1].replace('.mkv','_cow_only_track').replace('.mp4','_cow_only_track')  #open this when running single video


            #print(save_vid_name,' is file :'  ,isFile)
            save_vid_path = str(Path(os.path.join(self.save_dir, save_vid_name)).with_suffix('.mp4'))
            tracking_vid_path = str(Path(os.path.join(self.save_dir, tracking_vid_name)).with_suffix('.mp4'))
            #continue
            is_quit = 1
            #cap =  None #: cv2.VideoWriter(save_vid_path,cv2.VideoWriter_fourcc(*'mp4v'), 6, (1920,1080))
            self.boundary = self.save_width-10,self.save_height-10
            
            
            self.resolution = (1024, 768) # with number of camera because of stacking
            #self.resolution = (819, 576) # with number of camera because of stacking 80% of normal
            self.isEatingArea = False
            self.save_width,self.save_height = 1024, 768 * number_of_cams  #4, 5, 6 with tetris
            #self.save_width,self.save_height = (1024,1580)  #4, 5, 6 with tetris
            #self.save_width,self.save_height = (1024,1921) # 7, 8, 9, 10 no tetris
            #self.save_width,self.save_height = (1024,2043) # 11 , 12 13, 14 with tetris and shortser cropped 3


            #self.save_width,self.save_height = (819,1121)  #4, 5, 6 with tetris 80% of normal
            #self.save_width,self.save_height = (1024,1921) # 7, 8, 9, 10 no tetris
            #self.save_width,self.save_height = (1024,2043) # 11 , 12 13, 14 with tetris and shortser cropped 3

            self.boundary = self.save_width,self.save_height
            self.average_size = 30000
            self.SMALL_SIZE = 6000 # need to use it in clear data
            self.border_threshold = 70
            self.BIG_SIZE = 700000
                
                
            #tracking_cap = cv2.VideoWriter(tracking_vid_path,cv2.VideoWriter_fourcc(*'mp4v'), 5, (int(resolution[0] * multiplier), int(resolution[1] *multiplier)))
            #cap = cv2.VideoWriter(save_vid_path,cv2.VideoWriter_fourcc(*'mp4v'), 5, (int(resolution[0] * multiplier), int(resolution[1] *multiplier)))
            
            tracking_cap = None #cv2.VideoWriter(tracking_vid_path,cv2.VideoWriter_fourcc(*'mp4v'), 5, (self.save_width,self.save_height))
            #print("save_width and save_height", self.save_width,self.save_height)
            cap = cv2.VideoWriter(save_vid_path,cv2.VideoWriter_fourcc(*'mp4v'), 5, (self.save_width,self.save_height))
            
            #endregion Video data
            csv_main_file_path = str(self.save_dir) + "\\main_csv.csv"
            #await generate_frames(video_dir,entry,True,cap,cow_only_cap)
            
            print(save_vid_path)
            skip_vid = 12
            for i in range(len(video_paths[0])):
                videos = []
                for cam_no in range(number_of_cams):
                    
                    print(video_paths[cam_no][i], ' is video file')
                    videos.append(video_paths[cam_no][i])
                    #continue
                    #print(video_file, ' is video file')
                    #cap = cv2.VideoWriter(save_vid_path,cv2.VideoWriter_fourcc(*'mp4v'), 5, (self.save_width,self.save_height))
                    #tracking_cap = cv2.VideoWriter(tracking_vid_path,cv2.VideoWriter_fourcc(*'mp4v'), 5, (self.save_width,self.save_height))
                    #print(video_dir,' I am video dir')
                
                #if isCam:
                #    video_file = video_dir
                #else:
                #    video_file = f"{video_dir}" if isFile else f"{video_dir}\\{file_name}"
                if(skip_vid > 0):
                    skip_vid -= 1
                else:
                    # is_quit = self.batch_frames( videos, lane, True, cap, tracking_cap)                                  
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
            #return
            #self.clearData()
            #task.wait()
            if is_quit == -1:
                print("quitting")
                break
            #else: 
                #return
        
        total_duration = datetime.now() - program_start_time
        print(f"Total duration: {total_duration.total_seconds()}")
        self.save_final_pdf(self.save_dir,'finalized_pdf')
        if cap is not None:
            cap.release()
        if tracking_cap is not None:
            tracking_cap.release()
        self.clearData()
        #cow_only_cap.release()
        print('done saving everything')
        if cap is not None:
            cap.release()
        if tracking_cap is not None:
            tracking_cap.release()
        
        cv2.destroyAllWindows()


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
        #settings['-6-03'] = setting6_03
        #settings[' 003'] = setting003
        #settings[' 004'] = setting004
        return overlapping_settings
    #region parameters
    #region Tracker
    STORED_IDS =[]
    STORED_MISS = []
    STORED_XYXY = []
    STORED_CENTROID = []
    STORED_SIZE = []
    CATTLE_LOCAL_ID = 0
    IOU_TH = 0.5
    #endregion Tracker
    ############################ tracking parameters #########################################

    #region variables
    HOKKAIDO = "HOKKAIDO"
    HONKAWA = "HONKAWA"
    SUMIYOSHI = "SUMIYOSHI"
    YOSHII = "YOSHII"
    KUNNEPPU = "KUNNEPPU"
    FARMNAME = KUNNEPPU
    LAST_Y=0
    prevId_record = []
    #endregion variables
    TOTAL_MISSED_COUNT = 0
    middle = 0

    thread_pool = []

    # Predict with the model
    TRACKING = TRACKING_TYPE.CENTROID # IOU, Box, Centroid
    WEIGHT_PATH = '' # train for Hokkaido path, train 3 for Sumiyoshi
    
    DATE = '2024-08-07'
    alpha = 1.2  # Contrast control (1.0-3.0)
    beta = 20   # Brightness control (0-100)
    isShowBrightness = True #show brightness False show original
    multiplier = 1
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
    #dataset = f"E:/Nyi Zaw Aung/SuLarbMon/815_CowDataChecking/{DATE}/{HOUR}/20230317_061029_BB92.mkv" #2023-11-27 09 and 18 cows

    TRACKER = []  #To store tracking index to use in classifier
    CLASSIFICATION_TRACKER = [] #[{'GT': [1, 2, 3], 'COUNT': [10, 2, 5]}] retrieve Index , GT is ground truth, COUNT is number of apperance
    ALL_TIME_CLASSIFICATION_TRACKER = []
    BATCH_CLASSIFICATION_TRACKER = []
    TRACKING_RESULT = {} # [TrackingID, [values from each batch]]
    REIDENTIFY_MISSED_COUNT = 5 # re identify if there are missed frame
    manual_cow_count = 0
    image_count = 1
    cow_count = 1
    # base_path = "E:\\sumiyoshi\\ch006\\2024-06-10\\15\\"
    isWholeDay = False
    is_quit = 1
    #resolution = (1920,1080)
    resolution = (1152,648)
    save_width,save_height = resolution
    boundary = save_width,save_height
    save_dir = ""
    SIZE_224 = 224
    image_count = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using ",device)
    frame_number = 0
    #region added attributes
    TRACKING_MASK_LOCATION = []
    PROJECTILE_TRACKING = {}
    #PROJECTILE_TRACKING['1']=[{'id':1, 'x':100,'y':100},{'id':2, 'x':200,'y':200}] // this format
   
    #model_path ="models\\December 2024\\Sumiyoshi_dec16_v2_human_2024_10000_iters_v1" #Sumiyoshi_dec16_v3_human_2024_10000_iters_v6
    model_path ="models\\Side Lane August 2025\\Base_rtx8000_10_August_2025_20000_v1" #Sumiyoshi_dec16_v3_human_2024_10000_iters_v6
    #print("initializing under main")
    #similarity_checker = CheckSimilarity()
    class_names = []
    #FeaturesExtractor\\Sumiyoshi-40-24Oct-2024\\label-sumiyoshi-40-v2-50epochs-pretrained_27oct2024.json oct 24 #final    
    #with open2 is('FeaturesExtractor\\Sumiyoshi-38-ep50\\label-sumiyoshi-38-v1-50epochs-pretrained_27oct2024.json', 'r') as f:
    #predictor_path = "identification_models_KNP\\KNP_ConvNeXtV2_fcmae_ft_in22k_in1k_v_RTX8000_1"KNP_ConvNeXtV2_fcmae_ft_in22k_in1k_v_4
    #KNP_ConvNeXtV2_fcmae_ft_in22k_in1k_demo_light_grey_v1
    #KNP_ConvNeXtV2_fcmae_ft_in22k_in1k_demo_random_light_grey_v1
    #KNP_ConvNeXtV2_fcmae_ft_in22k_in1k_demo_v_4 last black
    #predictor_path = "identification_models_KNP\\KNP_ConvNeXtV2_fcmae_ft_in22k_in1k_v_RTX8000_v5" 
    predictor_path = Path("identification_models_KNP") / "KNP_ConvNeXtV2_fcmae_ft_in22k_in1k_v_RTX8000_Feature_distance_unknown_v_10"

    # Load model + weights
    NUM_CLASSES = len(json.load(open(predictor_path / "class_mappings.json")))  # dynamic
    # model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k',
    #                         pretrained=False, num_classes=NUM_CLASSES).to(device)
    # model.load_state_dict(torch.load(predictor_path / "last_model.pth", map_location=device))
    # model.eval()
    class_mapping_path = f'{predictor_path}\\class_mappings.json'
    with open(class_mapping_path, 'r') as f:
        class_names = json.load(f)

    # infer = ProtoInfer(model, device, proto_ckpt_path=predictor_path / "prototypes.pt",class_mapping_path=class_mapping_path )

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
    ])

    #print(class_names)
    #identification_model_path = f"{predictor_path}\\efficientnetv2s_cattle.pth" #99.70% weight_epoch_140_weights
    
    #identification_model_path = f"{predictor_path}\\last_model.pth" 
    #model = _load_model(len(class_names),identification_model_path,device)
    #model = _load_model(len(class_names), identification_model_path, device)
    stupid_detector = load_detector()
    #model = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    #num_ftrs = model.classifier[1].in_features
    #model.classifier[1] = nn.Linear(num_ftrs, len(class_names))  # Update to your number of classes
    #FeaturesExtractor\\Sumiyoshi-40-31Oct-2024\\efficientv2-m-sumiyoshi-40-v1-50epochs-pretrained_31oct2024.pth
    #model.load_state_dict(torch.load("FeaturesExtractor\\Sumiyoshi-38-ep50\\efficientv2-m-sumiyoshi-38-v1-50epochs-pretrained_27oct2024.pth")) # 23 oct update 4 browns
    #model.load_state_dict(torch.load("Identification models\\efficientv2-m-sumiyoshi-40-v1-50epochs-pretrained_31oct2024.pth")) # 31 oct update 4 browns
    LAST_20_PATH = {}
     
    PASSING_CATTLE_BY_CAMERA = {} ## camera counter [0]=2 cattle 
    #print("total class is : ", len(class_names))
    #endregion parameters
    barrel_distortion_corrector = CorrectDistortedImage()

    TRACKER_clone = []
    ALL_TIME_CLASSIFICATION_TRACKER_clone = []
    
    #distortion_keys,distortion_settings = setup_distortion_settings()
    # distortion_keys,distortion_settings = setup_distortion_settings_straight()
    # camera_overlapping_setting = setup_overlapping_settings()
    camera_lists = {}

    tracking_by_predicted_id = {}
    colors = [(255,0,0),(0,255,0),(0,0,255),(0,0,0)]

    CAM_TRACKER = {} 

    full_body_image = {}

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
    

    begin_identification = CATTLE_IDENTIFICATION()
    inputs = [
     
            # [
            #     ["G:\\Nyi Zaw Aung\\815_CowDataChecking\\KNP\\Standard_Format\\ALL_Channel_One_Day\\Ch005_KP-FHDD-4-05\\2025-09-06\\0514"],
            #     ["G:\\Nyi Zaw Aung\\815_CowDataChecking\\KNP\\Standard_Format\\ALL_Channel_One_Day\\Ch002_KP-FHDD-3-02\\2025-09-06\\0514"],
            #     ["G:\\Nyi Zaw Aung\\815_CowDataChecking\\KNP\\Standard_Format\\ALL_Channel_One_Day\\Ch001_KP-FHDD-3-01\\2025-09-06\\0514"],
                
            # ],
            [
                # ["G:\\Nyi Zaw Aung\\815_CowDataChecking\\KNP\\Standard_Format\\ALL_Channel_One_Day\\Ch005_KP-FHDD-4-05\\2025-09-17\\1317"],
                # ["G:\\Nyi Zaw Aung\\815_CowDataChecking\\KNP\\Standard_Format\\ALL_Channel_One_Day\\Ch002_KP-FHDD-3-02\\2025-09-17\\1317"],
                # ["G:\\Nyi Zaw Aung\\815_CowDataChecking\\KNP\\Standard_Format\\ALL_Channel_One_Day\\Ch001_KP-FHDD-3-01\\2025-09-17\\1317"],
                ["E:\\Ch005_KP-FHDD-4-05\\2025-09-20\\0514"],
                ["E:\\Ch002_KP-FHDD-3-02\\2025-09-20\\0514"],
                #["E:\\Ch001_KP-FHDD-3-01\\2025-09-20\\0514"],
                # ["G:\\Nyi Zaw Aung\\815_CowDataChecking\\KNP\\Standard_Format\\ALL_Channel_One_Day\\Ch001_Camera 001_E2\\2025-09-17\\0514"],
            ],
            #[
                # ["G:\\Nyi Zaw Aung\\815_CowDataChecking\\KNP\\Standard_Format\\ALL_Channel_One_Day\\Ch001_Camera 001_E2\\2025-09-17\\1521"],
            #    ["E:\\Ch005_KP-FHDD-4-05\\2025-09-20\\1521"],
            #    ["E:\\Ch002_KP-FHDD-3-02\\2025-09-20\\1521"],
                #["E:\\Ch001_KP-FHDD-3-01\\2025-09-20\\1521"],
            #]
          
          

          
            
    ]
    
 
    for allCams in inputs:
        begin_identification.begin_process(allCams,isCam= False,camera_number=None)
        try:
            print("Beginning processing now")
            #begin_identification.begin_process(input_data,isCam= False,camera_number=None)
        
        except Exception as ex:
            print("We got a problem in running sir, ",ex)
    print("Finished")
    
    