import yaml
import math
import pprint
import numpy as np
import cv2
from PIL import Image, ImageChops
from skimage.color import rgb2yuv
import os
import sys
sys.path.append(os.getcwd())

class JacquardPreprocessing:

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
        assert self.dataSet == "jacquard", "Dataset not understood!"
        self.colorModel = cfg.get("dirConfig").get("colorModel")
        assert self.colorModel in ("RGB", "YUV", "HSV"), "Color model not understood!"
        self.jawSize = cfg.get("jacquardConfig").get("jawSize")

    def openImage(self, image_path):
        assert isinstance(image_path, str), "Image path name must be a string!"
        try:
            img = cv2.imread(image_path)
        except:
            raise Exception(f"Image {image_path} not found!")
        else:
            return img

    def RGBtoYUV(self, image):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        return img_yuv

    def RGBtoHSV(self, image):
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return img_hsv

    def mapImageToBackground(self, image_path):
        imageName = os.path.split(image_path)[1][:-8]
        backgroundName = imageName + "_mask.png"
        backgroundName = os.path.join(os.path.split(image_path)[0], backgroundName)
        return backgroundName

    def substractBackground(self, image_path):
        img = self.openImage(image_path)
        background = self.openImage(self.mapImageToBackground(image_path))

        if self.colorModel == "YUV":
            img = self.RGBtoYUV(img)
        elif self.colorModel == "HSV":
            img = self.RGBtoHSV(img)
        
        return cv2.bitwise_and(img, background)

    def mapGraspingRectanglesToImage(self):
        self.graspingRectanglesMapping_dict = {}
        self.output_dict = {}

        for obj_dir in os.listdir(self.dataDir):
            oneObjectDir = os.path.join(self.dataDir, obj_dir)
            for file in os.listdir(oneObjectDir):
                filePath = os.path.join(oneObjectDir, file)
                
                if filePath.endswith("_RGB.png"): 
                    self.output_dict[os.path.basename(filePath)] = self.getGraspingRectangles(filePath)

    def splitPerJawsSize(self):
        
        for obj_dir in os.listdir(self.dataDir):
            oneObjectDir = os.path.join(self.dataDir, obj_dir)
            for file in os.listdir(oneObjectDir):
                filePath = os.path.join(oneObjectDir, file)
                if filePath.endswith("_grasps.txt"): 
                    
                    oneCat = []
                    with open(filePath) as f:
                        lines = f.readlines()
                        current = lines[0] 
                        
                        while lines:
                            line = lines.pop(0)
                           
                            if line.split(";")[:2] == current.split(";")[:2]:
                                oneCat.append(line)
                            else:
                                current = line
                                self.store(oneCat, filePath)
                                oneCat.clear()
                                oneCat.append(current)
                        self.store(oneCat, filePath)

    def store(self, listToAffect, path):
            directory = os.path.join(self.processedDataDir, "splitPerJawSize")  
            if not(os.path.exists(directory)):
                os.mkdir(directory)

            for index, ele in enumerate(listToAffect):
                if index == 0:
                    filename = os.path.basename(path[:-4])+f"_{index+2}.txt"
                    f = open(os.path.join(directory, filename), "a")
                    f.write(ele)
                    f.close()
                else:
                    ratio = float(ele.split(";")[-1])/float(listToAffect[0].split(";")[-1])
                    filename = os.path.basename(path[:-4])+f"_{round(2*ratio)}.txt"
                    f = open(os.path.join(directory, filename), "a")
                    f.write(ele)
                    f.close()

    def getGraspingRectangles(self, filePath):
        graspingRectanglesPath = os.path.join(self.processedDataDir, "splitPerJawSize/") + os.path.basename(filePath)[:-7] + f"grasps_{self.jawSize}.txt" 
        return self.mapOutputToImage(graspingRectanglesPath)

    def mapOutputToImage(self, graspingRectanglesPath):

        outputMapping_dict = {}
        rectangleIndex = 0
        with open(graspingRectanglesPath) as f:
            while True:
                line = f.readline()
                
                if not(line):
                    break 

                line_splited = line.split(";")
                x = float(line_splited[0])
                y = float(line_splited[1])
                w = float(line_splited[3])
                h = float(line_splited[4])
                sinTheta = float(np.sin(float(line_splited[2]) * np.pi/180))
                cosTheta = float(np.cos(float(line_splited[2]) * np.pi/180))
                
                outputMapping_dict[str(rectangleIndex)] = (x, y, h, w, sinTheta, cosTheta)
                rectangleIndex += 1

        return outputMapping_dict

class RectangleDrawing:
    
    def __init__(self, rec):

        self.x = rec[0]
        self.y = rec[1]
        self.h = rec[2]
        self.w = rec[3]
        self.sinTheta = rec[4]
        self.cosTheta = rec[5]

        self.P0 = (int(self.cosTheta*(-self.w/2)-self.sinTheta*(-self.h/2)+self.x), int(-self.sinTheta*(self.w/2)+self.cosTheta*(-self.h/2)+self.y))
        self.P1 = (int(self.cosTheta*(self.w/2)-self.sinTheta*(-self.h/2)+self.x), int(self.sinTheta*(self.w/2)+self.cosTheta*(-self.h/2)+self.y))
        self.P2 = (int(self.cosTheta*(self.w/2)-self.sinTheta*(self.h/2)+self.x), int(self.sinTheta*(self.w/2)+self.cosTheta*(self.h/2)+self.y))
        self.P3 = (int(self.cosTheta*(-self.w/2)-self.sinTheta*(self.h/2)+self.x), int(-self.sinTheta*(self.w/2)+self.cosTheta*(self.h/2)+self.y))

        self.verts = [self.P0,self.P1,self.P2,self.P3]

    def draw(self, image):
        
        for i in range(len(self.verts)-1):
            cv2.line(image, (self.verts[i][0], self.verts[i][1]), (self.verts[i+1][0],self.verts[i+1][1]), (0,255,0), 2)
        
        cv2.line(image, (self.verts[3][0], self.verts[3][1]), (self.verts[0][0], self.verts[0][1]), (0,255,0), 2)
        
        return image
