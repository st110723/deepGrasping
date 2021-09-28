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


class NeuralNet:

    def __init__(self, model_name, loss_func_name, optimizer_name, lr, batch_size, num_epochs, num_classes=4):

        self.model_name = model_name 
        self.model = self.get_model(model_name)
        self.adapt_model(num_classes)
        
        # Set up hyperparameters.
        self.lr = lr
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

    def get_model(self, model):

        from torchvision import models
            
        models = {
                "alexnet": models.alexnet(pretrained=True), 
                "resnet": models.resnet18(pretrained=True),
                "densenet": models.densenet169(pretrained=True),
                "vgg": models.vgg19_bn(pretrained=True),
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
                "adam": optim.Adam(self.model.parameters(), lr=self.lr),
                "sgd": optim.SGD(self.model.parameters(), lr=self.lr),
                "sgd": optim.RMSprop(self.model.parameters(), lr=self.lr)
                }
        return optimizers[optimizer]

    def adapt_model(self, num_classes):
        if self.model_name == "alexnet":
            self.model.classifier[-1] = torch.nn.Linear(in_features=self.model.classifier[-1].in_features, out_features=num_classes, bias=True)
            self.model.classifier[-3] = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
            self.model.classifier[-6] = torch.nn.Linear(in_features=9216, out_features=4096, bias=True)
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

    def preprocess(self, data, shuffle):
        
        if data is None:  # val/test can be empty
            return None

        # Read image files to pytorch dataset.
        transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        labels_file = "/home/alaa/maha/myDeepGrasping/processedData/label.csv"
        dataset = CustomDataSet(main_dir=data, labels_file=labels_file, transform=transform)

        # Wrap in data loader.
        if self.use_cuda:
            kwargs = {"pin_memory": True, "num_workers": 1}
        else:
            kwargs = {}
        
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, **kwargs)
        
        return loader

    def run(self):

        # ----------------------------------- Setup -----------------------------------
        # INSERT YOUR DATA HERE
        # Expected format: [images, labels]
        # - images has array shape (num samples, color channels, height, width)
        # - labels has array shape (num samples, )
        
        train_data = "/home/alaa/maha/myDeepGrasping/processedData/imagesWithoutBackground"  # required
        val_data = "/home/alaa/maha/myDeepGrasping/processedData/imagesWithoutBackground"    # optional
        test_data = None

        train_loader = self.preprocess(train_data, shuffle=False)
        val_loader = self.preprocess(val_data, shuffle=False)
        test_loader = self.preprocess(test_data, shuffle=True)

        # ----------------------------------- Model -----------------------------------
        # Set up model in device
        self.model = self.model.to(self.device)
        
        # --------------------------------- Training ----------------------------------
        # Set up pytorch-ignite trainer and evaluator.
        trainer = create_supervised_trainer(
            self.model,
            self.optimizer,
            self.loss_func,
            device=self.device,
        )

        metrics = {
            "loss": Loss(self.loss_func),
        }

        evaluator = create_supervised_evaluator(
            self.model, metrics=metrics, device=self.device
        )

        @trainer.on(Events.ITERATION_COMPLETED(every=self.print_every))
        def log_batch(trainer):
            batch = (trainer.state.iteration - 1) % trainer.state.epoch_length + 1
            print(
                f"Epoch {trainer.state.epoch} / {self.num_epochs}, "
                f"batch {batch} / {trainer.state.epoch_length}: "
                f"loss: {trainer.state.output:.3f}"
            )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_epoch(trainer):

            print(f"Epoch {trainer.state.epoch} / {self.num_epochs} average results: ")

            def log_results(name, metrics, epoch):
                
                print(
                    f"{name + ':':6} loss: {metrics['loss']:.3f}"
                )
                self.writer.add_scalar(f"{name}_loss", metrics["loss"], epoch)
                

            # Train data.
            evaluator.run(train_loader)
            log_results("train", evaluator.state.metrics, trainer.state.epoch)
    
            # Val data.
            if val_loader:
                evaluator.run(val_loader)
                log_results("val", evaluator.state.metrics, trainer.state.epoch)

            # Test data.
            if test_loader:
                evaluator.run(test_loader)
                log_results("test", evaluator.state.metrics, trainer.state.epoch)

            print()
            print("-" * 80)
            print()


        # Start training.
        trainer.run(train_loader, max_epochs=self.num_epochs)

    def __str__(self):

        msg = f"""
                [model] \t {self.model_name} \n
                [optimizer] \t {self.optimizer_name} \n
                [loss function] \t {self.loss_func_name} \n
                [learning rate] \t {self.lr} \n
        """

        return msg
    
net = NeuralNet(model_name="alexnet", loss_func_name="iou", optimizer_name="adam", lr=1, batch_size=2000, num_epochs=100)
print(net)
net.run()
