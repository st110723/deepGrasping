from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from customDataset import CustomDataSet
from PIL import Image
import numpy as np

class TransformerDebugger:

    def __init__(self, train_data, val_data, test_data, labels_file, batch_size):

        self.batch_size = batch_size
        
        self.train_data = train_data 
        self.val_data = val_data 
        self.test_data = test_data 

        self.labels_file = labels_file 


    def preprocess(self, data, transformation, shuffle):
        
        if data is None:  
            return None

        transform = self.get_transformation(transformation)

        dataset = CustomDataSet(main_dir=data, labels_file=self.labels_file, transform=transform)
        
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        
        return loader


    def run(self, transformation, shuffle):

        train_loader = self.preprocess(self.train_data, transformation, shuffle)
        val_loader = self.preprocess(self.val_data, transformation, shuffle)
        test_loader = self.preprocess(self.test_data, transformation, shuffle)

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            save_image(inputs, f'debug/train_{i}_{transformation}.png')
            if i%100 == 0:
                print("train loader")
                print("inputs shape: ", inputs.shape)
                print("labels shape: ", labels.shape)

        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            save_image(inputs, f'debug/val_{i}_{transformation}.png')
            if i%100 == 0:
                print("val loader")
                print("inputs shape: ", inputs.shape)
                print("labels shape: ", labels.shape)

    @staticmethod
    def check_image(image_path):
        with Image.open(image_path) as image:
            image_as_array = np.asarray(image)
            print(f"image: {image_path}")
            print(f"image size: {image_as_array.shape[0]} x {image_as_array.shape[1]}")
            print(f"number of channels: {image_as_array.shape[2]}")
            print(f"ratio of 0 in channels 0: {(image_as_array[:,:,0]==0).sum()/np.prod(image_as_array[:,:,0].shape)} %")
            print(f"ratio of 0 in channels 1: {(image_as_array[:,:,1]==0).sum()/np.prod(image_as_array[:,:,1].shape)} %")
            print(f"ratio of 0 in channels 2: {(image_as_array[:,:,2]==0).sum()/np.prod(image_as_array[:,:,2].shape)} %")

            

    @staticmethod
    def get_transformation(transformation):

        resize_center_normalize = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        resize_center = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor()
        ])

        resize = transforms.Compose([
            transforms.Resize(256),  
            transforms.ToTensor()
        ])

        no_transformation = transforms.Compose([ 
            transforms.ToTensor()
        ])

        transformation_dict = {
            "resize_center_normalize": resize_center_normalize,
            "resize_center": resize_center,
            "resize": resize,
            "no_transformation": no_transformation
        }

        return transformation_dict[transformation]

            
train_data = "/home/alaa/maha/myDeepGrasping/processedData/imagesWithoutBackground"  
val_data = "/home/alaa/maha/myDeepGrasping/processedData/imagesWithoutBackground"    
test_data = None
labels_file = "/home/alaa/maha/myDeepGrasping/processedData/label.csv"
transformation = "no_transformation" # "resize_center_normalize" "resize_center" "resize" "no_transformation"

transformer_debugger = TransformerDebugger(train_data=train_data, val_data=val_data, test_data=test_data, labels_file=labels_file, batch_size=1)
# transformer_debugger.run(transformation=transformation, shuffle=False)
transformer_debugger.check_image("debug/train_16_resize_center.png")
