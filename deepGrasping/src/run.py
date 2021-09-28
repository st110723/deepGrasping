# Before running, install required packages:
# pip install numpy torch torchvision pytorch-ignite tensorboardX tensorboard

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, datasets, transforms
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from datetime import datetime
from tensorboardX import SummaryWriter
from customDataset import CustomDataSet
from customLossfunction import IOU
from functools import lru_cache

from model import Net 

class NeuralNet:

        
    def __init__(self, train_data, val_data, test_data, model_name, loss_func_name, optimizer_name, lr, weight_decay, batch_size, num_epochs, labels_file, num_classes=4):

        self.train_data = train_data 
        self.val_data = val_data 
        self.test_data = test_data

        self.model_name = model_name 
        self.model = self.get_model(model_name) 
        if model_name != "myNeuralNet":
            self.adapt_model(num_classes)

        self.labels_file = labels_file
        
        # Set up hyperparameters.
        self.lr = lr
        self.weight_decay = weight_decay 
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.loss_func_name = loss_func_name
        self.optimizer_name = optimizer_name
        self.loss_func = self.get_loss_func(loss_func_name)
        self.optimizer = self.get_optimizer(optimizer_name)

        # Set up logging.
        self.experiment_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.writer = SummaryWriter(logdir=f"/home/alaa/maha/myDeepGrasping/logs/{self.experiment_id}")
        self.print_every = 1  # batches

        # Set up device.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    # @lru_cache(maxsize=32)
    def get_model(self, model):

        from torchvision import models

        models = {
                "alexnet": models.alexnet(pretrained=True), 
                "resnet": models.resnet18(pretrained=True),
                "densenet": models.densenet169(pretrained=True),
                "vgg": models.vgg19_bn(pretrained=True),
                "myNeuralNet": Net()
                }

        return models[model]


    def get_loss_func(self, loss_func): 
        
        loss_funcs = {
                "crossEntropy": nn.CrossEntropyLoss(),
                "mse": nn.MSELoss(),
                "iou": IOU(),
                }

        return loss_funcs[loss_func]


    def get_optimizer(self, optimizer):

        optimizers = {
                "adam": optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay),
                "sgd": optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay),
                "rmsProp": optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                }

        return optimizers[optimizer]


    def adapt_model(self, num_classes):
        if self.model_name == "alexnet":
            self.model.classifier[-1] = torch.nn.Linear(in_features=self.model.classifier[-1].in_features, out_features=num_classes, bias=True)
            # self.model.classifier[-3] = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
            # self.model.classifier[-6] = torch.nn.Linear(in_features=9216, out_features=4096, bias=True)
            self.model = nn.Sequential(self.model, nn.Tanh())
        
        if self.model_name == "resnet":
            self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes, bias=True)
            self.model = nn.Sequential(self.model, torch.nn.Linear(in_features=num_classes, out_features=num_classes, bias=True))
            self.model = nn.Sequential(self.model, torch.nn.Linear(in_features=num_classes, out_features=num_classes, bias=True))
            self.model = nn.Sequential(self.model, nn.Tanh())
            print(self.model)

        if self.model_name == "densenet":
            self.model.classifier = torch.nn.Linear(in_features=self.model.classifier.in_features, out_features=num_classes, bias=True)
            
        if self.model_name == "vgg":
            self.model.classifier[-1] = torch.nn.Linear(in_features=self.model.classifier[-1].in_features, out_features=num_classes, bias=True)

    def preprocess(self, data, batch_size, shuffle, flag):
        
        if data is None:  
            return None

        # Read image files to pytorch dataset.
        transform = transforms.Compose([
            # transforms.Resize(256), 
            # transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = CustomDataSet(main_dir=data, labels_file=self.labels_file, transform=transform, flag=flag) #TODO: houni a3mel kfold

        # Wrap in data loader.
        if self.use_cuda:
            kwargs = {"pin_memory": True, "num_workers": 1}
        else:
            kwargs = {}
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        
        return loader

    def run(self):

        train_loader = self.preprocess(self.train_data, batch_size=self.batch_size, shuffle=False, flag="training")
        val_loader = self.preprocess(self.val_data, batch_size=10000000000, shuffle=False, flag="validation")
        test_loader = self.preprocess(self.test_data, batch_size=10000000000, shuffle=False, flag="testing")

        self.model = self.model.to(self.device)
        
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times

            epoch_loss = 0
            epoch_iou = 0
            
            for batch, data in enumerate(train_loader, start=1):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)

                loss, iou_mean, iou_max = self.loss_func(outputs, labels, "training")
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss
                epoch_iou += iou_mean
    
                print(f"epoch: {epoch}\tbatch: {batch}\tloss: {loss.item()}\tiou mean={iou_mean}\tiou max={iou_max}")

            epoch_loss /= batch
            epoch_iou /= batch
            
            print()
            print(f"epoch: {epoch}\ttraining mean loss: {epoch_loss.item()}\ttraining mean iou: {epoch_iou}")
            print()

            for batch, data in enumerate(val_loader, start=1):
                
                inputs, labels = data
                outputs = self.model(inputs)
                score, iou_max_sum = self.loss_func(outputs, labels, "validation")
                print(f"score: {score}\tiou max : {iou_max_sum}")

            print("-" * 80)
            print()

        print()
        print("-" * 80)
        print('Training done!')
        print("-" * 80)
        print()

        # for batch, data in enumerate(test_loader, start=1):
        #     inputs, labels = data
        #     outputs = self.model(inputs)
        #     loss, iou_mean, iou_max = self.loss_func(outputs, labels, "testing")
        #     print(f"testing loss: {loss.item()}\ttesting iou mean: {iou_mean}\ttesting iou max: {iou_max}")
        
        print()
        print("-" * 80)
        print('Testing done!')
        print("-" * 80)
        print()

    def __str__(self):

        msg = f"""
                [model name] \t {self.model_name} \n
                [model] \t {self.model} \n
                \n
                [optimizer] \t {self.optimizer_name} \n
                [loss function] \t {self.loss_func_name} \n
                [learning rate] \t {self.lr} \n
        """

        return msg

######################## config ############################################

train_data = "/home/alaa/maha/myDeepGrasping/trainingData/"  
val_data = "/home/alaa/maha/myDeepGrasping/validationData/"  
test_data = "/home/alaa/maha/myDeepGrasping/testingData/"

# labels_file = "/home/alaa/maha/myDeepGrasping/labels/augmented_one_label.csv"
labels_file = "/home/alaa/maha/myDeepGrasping/labels/augmented_labels.csv"

model_name = "myNeuralNet" # "alexnet" "myNeuralNet"

loss_func_name = "iou"
optimizer_name = "adam"
lr = 0.003 # 0.003 0.01
weight_decay = 0

num_epochs = 500
batch_size = 10

net = NeuralNet(train_data=train_data, val_data=val_data, test_data=test_data, model_name=model_name, loss_func_name=loss_func_name, optimizer_name=optimizer_name, lr=lr, weight_decay=weight_decay, batch_size=batch_size, num_epochs=num_epochs, labels_file=labels_file)
net.run()