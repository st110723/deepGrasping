import yaml
import pprint
import numpy as np
import cv2
from graspnetAPI import GraspNet, RectGraspGroup
from PIL import Image, ImageChops
from skimage.color import rgb2yuv
import os
import sys
sys.path.append(os.getcwd())

class GraspnetPreprocessing:

    def __init__(self):

        print(f"[+] Loading configuaration file!")
        self.loadConfigFile()
        print(f"[+] Loading configuaration file done!")

    def loadConfigFile(self):

        with open("config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.workingDir = cfg.get("dirConfig").get("workingDir")
        self.dataDir = cfg.get("dirConfig").get("dataDir")
        self.processedDataDir = cfg.get("dirConfig").get("processedDataDir")
        self.dataSet = cfg.get("dirConfig").get("dataSet")
        assert self.dataSet == "graspnet", "Dataset not understood!"
        self.colorModel = cfg.get("dirConfig").get("colorModel")
        assert self.colorModel in ("RGB", "YUV", "HSV"), "Color model not understood!"

        self.cameraType = cfg.get("graspnetConfig").get("camera")
        self.split = cfg.get("graspnetConfig").get("split")
        self.fric_coef_thresh = cfg.get("graspnetConfig").get("fric_coef_thresh")

        self.g = GraspNet(self.dataDir, camera = self.cameraType, split = self.split)

    def openImage(self, sceneId, annId):
        img = self.g.loadBGR(sceneId = sceneId, camera = self.cameraType, annId = annId)
        return img

    def RGBtoYUV(self, image):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        return img_yuv

    def RGBtoHSV(self, image):
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return img_hsv

    def getGraspingRectangles(self, image_path):
        # graspingRectanglesMapping_dict[index] = rectangle
        # index += 1
        return graspingRectanglesMapping_dict

    def substractBackground(self, sceneId, annId):
        
        img = self.openImage(sceneId, annId)
        
        if self.colorModel == "YUV":
            img = self.RGBtoYUV(img)
        elif self.colorModel == "HSV":
            img = self.RGBtoHSV(img)
        
        mask = self.g.loadMask(sceneId = sceneId, camera = self.cameraType, annId = annId)
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        objIds = self.g.getObjIds(sceneIds=sceneId)
        
        imgList = dict()
        for objId in objIds:
            oneObject = (mask == (objId)+1)
            imgList[objId] = img * oneObject

        return imgList

    def mapGraspingRectanglesToImage(self):
        self.output_dict = {}

        # for sceneId in self.g.sceneIds:
        for sceneId in [0,1]:
            # for annId in range(256):
            for annId in range(10):
                rectGraspGroup = self.g.loadGrasp(sceneId=sceneId, annId=annId, camera=self.cameraType, format="rect", fric_coef_thresh=self.fric_coef_thresh)
                objectIds = np.unique(rectGraspGroup.object_ids)
                for objectId in objectIds:
                    rectGraspObjectArray = rectGraspGroup.rect_grasp_group_array[rectGraspGroup.object_ids == objectId]
                    self.output_dict[f"{sceneId}_{annId}_{objectId}.png"] = {index:self.graspnetToCornellLabel(rectGraspObjectArray[index,:]) for index in range(rectGraspObjectArray.shape[0])}

    def graspnetToCornellLabel(self, graspnetLabel):
        
        x = graspnetLabel[0]
        y = graspnetLabel[1]
        h = graspnetLabel[4]
        w = np.sqrt(np.power((graspnetLabel[2]-graspnetLabel[0]), 2) + np.power((graspnetLabel[3]-graspnetLabel[1]), 2))

        if graspnetLabel[0] > graspnetLabel[2]:
            theta = np.arctan2((graspnetLabel[3]-graspnetLabel[1]), (graspnetLabel[0]-graspnetLabel[2]))
        elif graspnetLabel[0] < graspnetLabel[2]:
            theta = np.arctan2((graspnetLabel[1]-graspnetLabel[3]), (graspnetLabel[2]-graspnetLabel[0]))
        else:
            theta = -np.pi/2
        
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)

        return x, y, h, w, sinTheta, cosTheta
        
        

 