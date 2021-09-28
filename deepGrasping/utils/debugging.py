import os
import sys
import copy
import yaml

sys.path.append(os.getcwd())

from src.jacquardPreprocessing import RectangleDrawing
from graspnetAPI import GraspNet, RectGraspGroup

import numpy as np
from PIL import Image
import cv2
import random

class Debugging:

    def __init__(self, dataPreprocessing):
        
        self.loadConfigFile()   
        self.dataPreprocessing = dataPreprocessing
         
    def loadConfigFile(self):

        with open("config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.workingDir = cfg.get("dirConfig").get("workingDir")
        self.dataDir = cfg.get("dirConfig").get("dataDir")
        self.processedDataDir = cfg.get("dirConfig").get("processedDataDir")
        self.dataset = cfg.get("dirConfig").get("dataSet")
        self.colorModel = cfg.get("dirConfig").get("colorModel")

        if self.dataset == "jacquard":
            self.splitPerJawSize = cfg.get("jacquardConfig").get("splitPerJawSize")
            self.jawSize = cfg.get("jacquardConfig").get("jawSize")

        elif self.dataset == "cornell":
            self.backgroundMappingFile = cfg.get("cornellConfig").get("backgroundMappingFile")
            self.backgroundDir = cfg.get("cornellConfig").get("backgroundDir")

        elif self.dataset == "graspnet":
            self.camera = cfg.get("graspnetConfig").get("camera")
            self.split = cfg.get("graspnetConfig").get("split")
            self.fric_coef_thresh = cfg.get("graspnetConfig").get("fric_coef_thresh")

    def saveImageAfterSubBackgroundWithRect(self,  image_path=None, img_name=None, scene_id=None, ann_id=None, num_rects=3):
    
        if self.dataset == "cornell":
            img = self.openImage(os.path.join("processedData/imagesWithoutBackground", img_name))
            rec = self.dataPreprocessing.dataPreprocessing.graspingRectanglesMapping_dict[img_name]
            for _, rectangle in rec.items():
                rect = cv2.minAreaRect(np.array(rectangle))
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                img = cv2.drawContours(np.float32(img), [box], 0, (0, 0, 255), 2)
            self.saveImage(img, img_name)

        if self.dataset == "jacquard":
            img = self.openImage(os.path.join("processedData/imagesWithoutBackground", img_name))
            rec = self.dataPreprocessing.dataPreprocessing.getGraspingRectangles(img_name)
            
            i_num_rect = 0
            for _, rectangle in rec.items():
                i_num_rect += 1
                rectangleDrawing = RectangleDrawing(rectangle)
                img = rectangleDrawing.draw(np.float32(img))
                if i_num_rect == num_rects: break
                
            self.saveImage(img, img_name)
        
        if self.dataset == "graspnet":

            rectGraspGroup = self.dataPreprocessing.dataPreprocessing.g.loadGrasp(sceneId=scene_id, annId=ann_id, camera=self.camera, format="rect", fric_coef_thresh=self.fric_coef_thresh)
            object_id = random.choice(np.unique(rectGraspGroup.object_ids)) 
            img_name = f"{scene_id}_{ann_id}_{object_id}.png"
            img = self.openImage(os.path.join("processedData/imagesWithoutBackground", img_name))
            rectGraspObjectArray = rectGraspGroup.rect_grasp_group_array[rectGraspGroup.object_ids == object_id]
            rect_grasp_object = RectGraspGroup()
            rect_grasp_object.rect_grasp_group_array = copy.deepcopy(rectGraspObjectArray)
            rect_grasp = rect_grasp_object
            img = rect_grasp.to_opencv_image(np.float32(img), rectGraspObjectArray.shape[0])
            self.saveImage(img, img_name)

    def saveImagesAfterSubBackgroundWithRect(self):

        if self.dataset == "cornell":
            outputList = list(self.dataPreprocessing.dataPreprocessing.output_dict.items())
            imageName = random.choice(outputList)[0]
            self.saveImageAfterSubBackgroundWithRect(img_name=imageName)
            
        if self.dataset == "jacquard":
            outputList = list(self.dataPreprocessing.dataPreprocessing.output_dict.items())
            imageName =  random.choice(outputList)[0] 
            self.saveImageAfterSubBackgroundWithRect(img_name=imageName)

        if self.dataset == "graspnet":
            sceneId = 0*random.choice(self.dataPreprocessing.dataPreprocessing.g.sceneIds)
            annId = random.choice(range(256))
            self.saveImageAfterSubBackgroundWithRect(scene_id=sceneId, ann_id=annId)

    def openImage(self, image_path):

        assert isinstance(image_path, str), "Image path name must be a string!"
        try:
            # img = cv2.imread(image_path) ######
            img = Image.open(image_path)
        except:
            raise Exception(f"Image {image_path} not found!")
        else:
            return img
      
    def saveImage(self, image, imageName):
        directory = "figures/"
        imagePath = os.path.join(directory, imageName)       
        cv2.imwrite(imagePath, np.float32(image))