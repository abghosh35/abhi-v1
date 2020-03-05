import sys
import numpy as np
import torch
import warnings
import time

class TrainIterator:
    """
    Run training job with validation (if validation data is provided)
    Inputs:
        model: (required) Model class
        train_loader: (required) Training data loader
        metric: (required) Dict : Provide the metric value as {metric_name: metric_function}
        loss_fn: (required) function: Pass the criterion function to compute the loss
        optimizer: function: Pass the optimizer function to run. If None then Adam optimizer is selected by default.
        valid_loader: None: Validation data loader
        lr: float: (default=0.001)
        epochs: Int: (default=1) Number of epochs to run the training job
        use_gpu: Boolean (default=True) Whether to use GPU or CPU for training

    Return: The final fitted model
    """
    def __init__(self, 
                 model, 
                 train_loader, 
                 valid_loader=None, 
                 loss_fn=None,
                 optimizer=None,
                 lr=0.001,
                 epochs=1, 
                 use_gpu=True,
                 metric=None
                ):
        
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fn = torch.nn.CrossEntropyLoss() if loss_fn is None else loss_fn
        self.optimizer = torch.optim.Adam
        self.lr = lr
        self.epochs = epochs
        self.use_gpu = use_gpu
        self.metric = metric[list(metric.keys())[0]]
        self.metric_name = list(metric.keys())[0]
        self.batch_size = len(self.train_loader)
        
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device('cpu')
            
        if use_gpu:
            if self.device.type == "cpu":
                warnings.warn("CUDA device is not available")
                
        self.device = torch.device(self.device)
        self.model.to(self.device)
        if self.device.type == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            
    
    def UpdateMetric(self, metric, new_value, count, mtype="avg"):
        if mtype == "avg":
            metric = (metric * count + new_value)/(count + 1)
        elif mtype == "sum":
            metric += new_value
            
        return metric, count + 1
            
    
    def PrintBatchLoss(self, epoch, batch_num, epoch_loss, epoch_accuracy, val_loss=None, val_accuracy=None):
        nbars = 50
        if self.valid_loader is None:
            batch_num += 1
        percent = (round((batch_num / self.batch_size) * 100, 2))
        nb_bar_fill = int(round((nbars * percent) / 100))
        bar_fill = '█' * nb_bar_fill
        bar_empty = ' ' * (nbars - nb_bar_fill)
        text = "Epoch#{}/{} |{}| {}% ({}/{})".format(epoch, self.epochs, str(bar_fill + bar_empty), percent, batch_num, self.batch_size)
        text += " Train (Loss: {:.4f}, {}: {:.4f})".format(epoch_loss, self.metric_name.capitalize(), epoch_accuracy)
        if val_loss is not None:
            text += " Valid (Loss: {:.4f}, {}: {:.4f})".format(val_loss, self.metric_name.capitalize(), val_accuracy)
        sys.stdout.write("\r" + str(text))
        if batch_num == self.batch_size:
            print()
        sys.stdout.flush()
            
    
    def RunTrainer(self):
        
        criterion = self.loss_fn
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)

        self.model.zero_grad()
        
        for epoch in range(1, (self.epochs + 1)):
            self.model.train()
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            i = 0
            self.PrintBatchLoss(epoch, i, epoch_loss, epoch_accuracy)
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                
                out = self.model(x)
                loss = criterion(out, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss, _ = self.UpdateMetric(epoch_loss, loss.item(), i)
                epoch_accuracy, _ = self.UpdateMetric(epoch_accuracy, self.metric(out, y), i)
                
                self.PrintBatchLoss(epoch, i, epoch_loss, epoch_accuracy)
            
            if self.valid_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_accuracy = 0.0
                for i, (x, y) in enumerate(self.valid_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    out = self.model(x)
                    loss = self.loss_fn(out, y)
                    
                    val_loss, _ = self.UpdateMetric(val_loss, loss.item(), i)
                    val_accuracy, _ = self.UpdateMetric(val_accuracy, self.metric(out, y), i)
                    
                self.PrintBatchLoss(epoch, self.batch_size, epoch_loss, epoch_accuracy, val_loss, val_accuracy)
                
        return self.model