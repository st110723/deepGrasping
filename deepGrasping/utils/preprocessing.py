import os
import sys
import yaml
import csv  
import random
import cv2
import numpy as np
import pandas as pd
from shutil import copyfile

sys.path.append(os.getcwd())

import albumentations as DataAugmentation     

from src.cornellPreprocessing import CornellPreprocessing
from src.jacquardPreprocessing import JacquardPreprocessing
from src.jacquardPreprocessing import RectangleDrawing
from src.graspnetPreprocessing import GraspnetPreprocessing
from utils.debugging import Debugging


class Preprocessing:
    
    def __init__(self):
        
        self.loadConfigFile()
            
        if self.dataset == "cornell":
            self.dataPreprocessing = CornellPreprocessing()
        
        if self.dataset == "jacquard":
            self.dataPreprocessing = JacquardPreprocessing()
            if self.splitPerJawSize:
                self.dataPreprocessing.splitPerJawsSize()

        if self.dataset == "graspnet":
            self.dataPreprocessing = GraspnetPreprocessing()


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

        self.imagesFolderPath = cfg.get("dataAugmentation").get("images_folder_path")
        self.augmentedImagesFolderPath = cfg.get("dataAugmentation").get("augmented_images_folder_path")
        self.labelsFilePath = cfg.get("dataAugmentation").get("labels_file_path")
        self.augmentedLabelsFilePath = cfg.get("dataAugmentation").get("augmented_labels_file_path")
        self.augmentedOneLabelFilePath = cfg.get("dataAugmentation").get("augmented_one_label_file_path")

        self.augmented_images_set = None
        self.not_augmented_images_set = None

    def mapGraspingRectanglesToImages(self):
        self.dataPreprocessing.mapGraspingRectanglesToImage()

    def saveImageAfterSubBackground(self, image_path, name):
        img = self.dataPreprocessing.substractBackground(image_path)
        self.saveImage(img, name, "processedData/imagesWithoutBackground")

    def saveImagesAfterSubBackground(self):

        if self.dataset == "cornell":
            for imageName in self.dataPreprocessing.output_dict.keys():
                directory = imageName[3:5]
                imagePath = os.path.join(self.dataDir, directory, imageName)
                self.saveImageAfterSubBackground(imagePath, imageName)
        
        if self.dataset == "jacquard":
            for imageName in self.dataPreprocessing.output_dict.keys():
                obj = imageName[2:-8]
                imagePath = os.path.join(self.dataDir, obj, imageName)
                self.saveImageAfterSubBackground(imagePath, imageName)

        if self.dataset == "graspnet":
            # for sceneId in self.dataPreprocessing.g.sceneIds:
            #     for annId in range(256):
            for sceneId in [0, 1]:
                for annId in range(10):
                    imgList = self.dataPreprocessing.substractBackground(sceneId, annId)
                    for objId in imgList.keys():
                        self.saveImage(imgList[objId], f"{sceneId}_{annId}_{objId}.png", "processedData/imagesWithoutBackground")

    def saveImage(self, image, imageName, directory):
        
        if not(os.path.exists(directory)):
                print(f"[+] Directory {directory} will be created!")
                os.mkdir(directory)

        imagePath = os.path.join(directory, imageName)       
        cv2.imwrite(imagePath, np.float32(image))

    def saveGraspingRectanglesToFile(self):
    
        header = ['img', 'x', 'y', 'h', 'w', 'sin', 'cos']
        output_dict = self.dataPreprocessing.output_dict

        with open(self.labelsFilePath, 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write the data
            for img, rects in output_dict.items():
                for index, rec in rects.items():
                    x, y, h, w, sinTheta, cosTheta = rec
                    writer.writerow([img, x, y, h, w, sinTheta, cosTheta])

    def data_augmentation(self, per_augmentation=0.5, always_apply=True, probability=0.5):
            
        vertical_flip = DataAugmentation.VerticalFlip(always_apply=always_apply, p=probability)
        horizontal_flip = DataAugmentation.HorizontalFlip(always_apply=always_apply, p=probability)
        # transpose = DataAugmentation.Transpose(always_apply=always_apply, p=probability)
        # rotate_90 = DataAugmentation.RandomRotate90(always_apply=always_apply, p=probability)

        color_jitter = DataAugmentation.ColorJitter(always_apply=always_apply, p=probability)
        gauss_noise = DataAugmentation.GaussNoise(per_channel=True, always_apply=always_apply, p=probability)

        spatial_transforms_list = {
            "verticalFlip": vertical_flip, 
            "horizontalFlip": horizontal_flip, 
            # "transpose": transpose, 
            # "rotate": rotate_90
            }  

        color_transforms_list = {
            "colorJitter": color_jitter, 
            "gaussNoise": gauss_noise
            }

        assert (0 < per_augmentation < 1), "per_augmentation not in [0, 1]!"
        num_augmented_images = int(len(os.listdir(self.imagesFolderPath)) * per_augmentation)
        self.augmented_images_set = set(random.choices(os.listdir(self.imagesFolderPath), k=num_augmented_images))
        self.not_augmented_images_set = set(os.listdir(self.imagesFolderPath)) - self.augmented_images_set
        self.augmented_images_dict = dict()

        for img in self.augmented_images_set:
            
            img_path = os.path.join(self.imagesFolderPath, img)
            image = cv2.imread(img_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            spatial_transform = random.choice(list(spatial_transforms_list.keys()))
            color_transform = random.choice(list(color_transforms_list.keys()))
            transform = DataAugmentation.Compose([spatial_transforms_list[spatial_transform], color_transforms_list[color_transform]])
            
            augmented_image = transform(image=image)['image']
            
            self.saveImage(image, img, self.augmentedImagesFolderPath)
            self.saveImage(augmented_image, f"{os.path.splitext(img)[0]}_{spatial_transform}_{color_transform}.png", self.augmentedImagesFolderPath)
            
            self.augmented_images_dict[img] = (spatial_transform, color_transform)

        for img in self.not_augmented_images_set:
            
            img_path = os.path.join(self.imagesFolderPath, img)
            image = cv2.imread(img_path)
            
            self.saveImage(image, img, self.augmentedImagesFolderPath)

    def adapt_labels_after_augmentation(self):

        copyfile(self.labelsFilePath, self.augmentedLabelsFilePath)

        labels_dataframe = pd.read_csv(self.augmentedLabelsFilePath)

        with open(self.augmentedLabelsFilePath, mode='a') as augmented_labels_file:

            augmented_labels_file_writer = csv.writer(augmented_labels_file, delimiter=',')

            for img in self.augmented_images_set:

                img_path = os.path.join(self.imagesFolderPath, img)
                image = cv2.imread(img_path)
                image_height, image_width, _ = image.shape
                one_img_spatial_transform, one_img_color_transform = self.augmented_images_dict[img] 
                one_img_labels_dataframe = labels_dataframe[labels_dataframe["img"] == img]

                for one_img_one_label in one_img_labels_dataframe.iterrows():
                    _, x, y, h, w, sin_, cos_ = one_img_one_label[1]
                    output = (x, y, h, w, sin_, cos_)
                    corners = self.output_to_corners(output)
                    transformed_corners = [self.apply_transformation(P.reshape(2, -1), one_img_spatial_transform, image_width, image_height) for P in corners]
                    x_, y_, h_, w_, sin__, cos__ = self.corners_to_output(transformed_corners)

                    augmented_labels_file_writer.writerow([f"{os.path.splitext(img)[0]}_{one_img_spatial_transform}_{one_img_color_transform}.png", x_, y_, h_, w_, sin__, cos__])

    def keep_one_label_after_augmentation(self):
        
        labels_dataframe = pd.read_csv(self.augmentedLabelsFilePath)
        one_label_dataframe = labels_dataframe.drop_duplicates(subset=['img'], keep='last')
        one_label_dataframe.to_csv(self.augmentedOneLabelFilePath, sep=",", index=False)

    def split_data(self, per_training, per_validation, per_testing):

        training_data_path = "/home/alaa/maha/myDeepGrasping/trainingData/"
        validation_data_path = "/home/alaa/maha/myDeepGrasping/validationData/"
        testing_data_path = "/home/alaa/maha/myDeepGrasping/testingData/"

        assert round(per_training + per_validation + per_testing, 3) == 1, "sum should be equal to 1!"

        num_training_images = int(len(os.listdir(self.augmentedImagesFolderPath)) * per_training)
        num_validation_images = int(len(os.listdir(self.augmentedImagesFolderPath)) * per_validation)
        
        training_images = random.sample(os.listdir(self.augmentedImagesFolderPath), k=num_training_images)
        validation_testing_images = set(os.listdir(self.augmentedImagesFolderPath)) - set(training_images)
        validation_images = random.sample(validation_testing_images, k=num_validation_images)
        testing_images = set(validation_testing_images) - set(validation_images)

        for img in training_images:
            img_path = os.path.join(self.augmentedImagesFolderPath, img)
            img_dest_path = os.path.join(training_data_path, img)
            copyfile(img_path, img_dest_path)

        for img in validation_images:
            img_path = os.path.join(self.augmentedImagesFolderPath, img)
            img_dest_path = os.path.join(validation_data_path, img)
            copyfile(img_path, img_dest_path)

        for img in testing_images:
            img_path = os.path.join(self.augmentedImagesFolderPath, img)
            img_dest_path = os.path.join(testing_data_path, img)
            copyfile(img_path, img_dest_path)

    @staticmethod
    def apply_transformation(coordinates, augmentation_type, img_x_size, img_y_size):

        transformation_dict = {
            "verticalFlip": lambda coordinates: np.matmul(np.array([[1, 0], [0, -1]]), coordinates) + np.array([[0], [img_y_size]]),
            "horizontalFlip": lambda coordinates: np.matmul(np.array([[-1, 0], [0, 1]]), coordinates) + np.array([[img_x_size], [0]])
        }

        return transformation_dict[augmentation_type](coordinates)

    @staticmethod
    def output_to_corners(output):

        x, y, h, w, sin_, cos_ = output
        
        rot_matrix = np.array([ [cos_,-sin_], [sin_, cos_] ])
        
        xmin = -w/2
        xmax = w/2
        ymin = -h/2
        ymax = h/2

        P1 = np.array([xmin, ymin])
        P2 = np.array([xmax, ymin])
        P3 = np.array([xmax, ymax])
        P4 = np.array([xmin, ymax])
        
        P1 = np.matmul(rot_matrix, P1)
        P2 = np.matmul(rot_matrix, P2)
        P3 = np.matmul(rot_matrix, P3)
        P4 = np.matmul(rot_matrix, P4)
        
        P1 += np.array([x, y])
        P2 += np.array([x, y])
        P3 += np.array([x, y])
        P4 += np.array([x, y])

        return (P1, P2, P3, P4)

    @staticmethod
    def corners_to_output(corners):

        x = (1/4) * sum(corners)[0].item()
        y = (1/4) * sum(corners)[1].item()

        h = np.sqrt(pow(corners[1][0]-corners[0][0], 2) + pow(corners[1][1]-corners[0][1], 2)).item()
        w = np.sqrt(pow(corners[2][0]-corners[1][0], 2) + pow(corners[2][1]-corners[1][1], 2)).item()

        # sin_ = (abs(corners[1][1]-corners[0][1])/h).item()
        # cos_ = (abs(corners[1][0]-corners[0][0])/h).item()
        sin_ = ((corners[1][1]-corners[0][1])/h).item()
        cos_ = ((corners[1][0]-corners[0][0])/h).item()
        
        
        return (x, y, h, w, sin_, cos_)


if __name__ == "__main__":
    
    preprocessing = Preprocessing()
    preprocessing.mapGraspingRectanglesToImages()
    preprocessing.saveImagesAfterSubBackground()
    preprocessing.saveGraspingRectanglesToFile()

    # debugging = Debugging(preprocessing)
    # debugging.saveImagesAfterSubBackgroundWithRect()
    
    preprocessing.data_augmentation(per_augmentation=0.5)
    preprocessing.adapt_labels_after_augmentation()
    preprocessing.keep_one_label_after_augmentation()
    preprocessing.split_data(per_training=0.7, per_validation=0.2, per_testing=0.1)
    