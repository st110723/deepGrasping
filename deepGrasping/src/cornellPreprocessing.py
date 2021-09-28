import yaml
import pprint
import numpy as np
import cv2
from PIL import Image, ImageChops
from skimage.color import rgb2yuv
import os
import sys
sys.path.append(os.getcwd())

class CornellPreprocessing:

    def __init__(self):

        print(f"[+] Loading configuaration file!")
        self.loadConfigFile()
        print(f"[+] Loading configuaration file done!")

        print()

        print(f"[+] Loading background mapping file!")
        self.loadBackgroundMappingFile()
        print(f"[+] Loading background mapping file done!")

    def loadConfigFile(self):

        with open("config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.workingDir = cfg.get("dirConfig").get("workingDir")
        self.dataDir = cfg.get("dirConfig").get("dataDir")
        self.processedDataDir = cfg.get("dirConfig").get("processedDataDir")
        self.dataSet = cfg.get("dirConfig").get("dataSet")
        assert self.dataSet == "cornell", "Dataset not understood!"
        self.colorModel = cfg.get("dirConfig").get("colorModel")
        assert self.colorModel in ("RGB", "YUV", "HSV"), "Color model not understood!"

        self.backgroundDir = cfg.get("cornellConfig").get("backgroundDir")
        self.backgroundMappingFile = cfg.get("cornellConfig").get("backgroundMappingFile")

    def loadBackgroundMappingFile(self):
        backgroundMappingFile = os.path.join(
            self.dataDir, self.backgroundMappingFile)

        self.backgroundMapping_dict = {}
        try:
            with open(backgroundMappingFile) as f:
                for line in f:
                    self.backgroundMapping_dict[line.strip().split()[0]] = line.strip().split()[
                        1]
        except:
            raise Exception(
                f"Background mapping file not found in {backgroundMappingFile}!")

    def openImage(self, image_path):

        assert isinstance(image_path, str), "Image path name must be a string!"
        try:
            img = cv2.imread(image_path) 
            # img = Image.open(image_path)

        except:
            raise Exception(f"Image {image_path} not found!")
        else:
            return img

    def RGBtoYUV(self, image): 
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        return img_yuv

    def RGBtoHSV(self, image):
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # img_hsv = image.convert('HSV')
        return img_hsv

    def mapImageToBackground(self, image_path):
        imageName = os.path.split(image_path)[1]
        backgroundName = self.backgroundMapping_dict.get(imageName, None)
        if backgroundName == None:
            raise Exception(f"Background of the image {image_path} not found!")
        return os.path.join(self.backgroundDir, backgroundName.replace('_', ''))

    
    def getGraspingRectangles(self, image_path):
        from itertools import islice

        def processReadLines(readLines):
            import math
            sublist = []
            for coordinates in readLines:
                try:
                    x, y = int(float(coordinates.split()[0])), int(
                        float(coordinates.split()[1]))
                except:
                    return None
                else:
                    sublist.append((x, y))
            return sublist

        def next_n_lines(file_opened, N):
            return [x.strip() for x in islice(file_opened, N)]

        graspingRectanglesMapping_dict = {}

        basepath, imageName = os.path.split(
            image_path)[0], os.path.split(image_path)[1]
        graspingRectanglesFilename = imageName[:-5] + "cpos.txt"
        graspingRectanglesFilePath = os.path.join(
            basepath, graspingRectanglesFilename)

        index = 0
        with open(graspingRectanglesFilePath) as f:
            while True:
                readLines = next_n_lines(f, 4)
                if not readLines:
                    break
                rectangle = processReadLines(readLines)
                if rectangle == None:
                    pass
                else:
                    graspingRectanglesMapping_dict[index] = rectangle
                    index += 1

        return graspingRectanglesMapping_dict

    def substractBackground(self, image_path):
        img = self.openImage(image_path)
        background = self.openImage(self.mapImageToBackground(image_path))

        if self.colorModel == "YUV":
            img = self.RGBtoYUV(img)
            background = self.RGBtoYUV(background)
        elif self.colorModel == "HSV":
            img = self.RGBtoHSV(img)
            background = self.RGBtoHSV(background)
        # return ImageChops.difference(img, background)
        # return cv2.cvtColor(np.float32(ImageChops.difference(img, background)), cv2.COLOR_BGR2RGB)
        return cv2.subtract(background, img)
        # return cv2.cvtColor(cv2.subtract(background, img), cv2.COLOR_BGR2RGB)

    def mapGraspingRectanglesToImage(self):
        self.graspingRectanglesMapping_dict = {}
        self.output_dict = {}

        for directory in os.listdir(self.dataDir):
            if directory.isdecimal():
                oneObjectDir = os.path.join(self.dataDir, directory)
                for file in os.listdir(oneObjectDir):
                    filePath = os.path.join(oneObjectDir, file)
                    if filePath.endswith("png"):
                        self.graspingRectanglesMapping_dict[os.path.basename(filePath)] = self.getGraspingRectangles(
                            filePath)
                        self.output_dict[os.path.basename(filePath)] = self.mapOutputToImage(
                            self.getGraspingRectangles(filePath))
                        
    def mapOutputToImage(self, graspingRectangles):

        def bboxes_to_grasps(rectangle):
            from math import sqrt, pow

            x = 0
            y = 0
            for coordinates in rectangle:
                x += coordinates[0]
                y += coordinates[1]
            x = x / 4
            y = y / 4
            h = sqrt(pow(rectangle[1][0]-rectangle[0][0],
                         2) + pow(rectangle[1][1]-rectangle[0][1], 2))
            w = sqrt(pow(rectangle[2][0]-rectangle[1][0],
                         2) + pow(rectangle[2][1]-rectangle[1][1], 2))
            sinTheta = abs(rectangle[1][1]-rectangle[0][1])/h
            cosTheta = abs(rectangle[1][0]-rectangle[0][0])/h
            return x, y, h, w, sinTheta, cosTheta

        outputMapping_dict = {}
        for index, rectangle in graspingRectangles.items():
            outputMapping_dict[index] = bboxes_to_grasps(rectangle)

        return outputMapping_dict





