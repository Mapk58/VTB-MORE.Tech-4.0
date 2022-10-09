import torch
from torch import optim
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
from classifier import Model


class Trainer:
    def __init__(self, params, arguments):
        self.params = params
        self.arguments = arguments
        self.device = self.get_device()

        # Model definition has to be implemented by the concrete model
        self.model = None
    
    def get_device(self):
        device = 'cpu'

        if torch.cuda.is_available():
            device = 'cuda'
        
        return torch.device(device)
    
    def fit(self, X, y, X_val=None, y_val=None):
        optimizer = optim.AdamW(self.model.parameters(), lr=0.01)

        X = torch.tensor(X)
        X_val = torch.tensor(X_val).float()
        y = torch.tensor(y)
        y_val = torch.tensor(y_val)

        loss_func = nn.CrossEntropyLoss()
        '''
        else:
            loss_func = nn.BCEWithLogitsLoss()
            y = y.float()
            y_val = y_val.float()
        '''
        
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.arguments.batch_size, shuffle=True, num_workers=4)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.arguments.val_batch_size, shuffle=True)

        min_val_loss, min_val_loss_idx = float('inf'), 0

        loss_history, val_loss_history = [], []

        for epoch in tqdm(range(1, self.arguments.epochs + 1), desc='Training'):
            for batch_x, batch_y in enumerate(train_loader):

                out = self.model(batch_x.to(self.device))
                
                loss = loss_func(out, batch_y.to(self.device))
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            val_loss, val_cnt = .0, 0

            for batch_val_x, batch_val_y in enumerate(val_loader):
                
                out = self.model(batch_val_x.to(self.device))

                val_loss += loss_func(out, batch_val_y.to(self.device))
                val_cnt += 1
            
            val_loss /= val_cnt
            val_loss_history.append(val_loss.item())

            if val_loss < min_val_loss:
                min_val_loss, min_val_loss_idx = val_loss, epoch

                self.save_model(extension='best', directory='tmp')
            
            if min_val_loss_idx + self.arguments.early_stopping_rounds < epoch:
                print("Early stopping applies.")
                break
        
        self.load_model(extension="best", directory="tmp")
        return loss_history, val_loss_history
    
    def predict(self, X):
        prediction_probabilities = self.predict_proba(X)
        self.predictions = np.argmax(prediction_probabilities, axis=1)
        return self.predictions
    
    def _predict(self, X):
        self.model.eval()

        X = torch.tensor(X).float()
        test_dataset = TensorDataset(X)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.arguments.val_batch_size, shuffle=False, num_workers=2)
        predictions = []

        with torch.no_grad():
            for batch_x in test_loader:
                preds = self.model(batch_x[0].to(self.device))

                if self.arguments.objective == 'binary':
                    torch.sigmoid_(preds)
                
                predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions)
    
    def predict_proba(self, X):
        probas = self._predict(X)

        if probas.shape[1] == 1:
            probas = np.concatenate((1 - probas, probas), 1)

        return probas
    
    def get_model_size(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def load_model(self, extension='', directory='models'):
        filename = self.get_output_path(filename='m', directory=directory, extension=extension, file_type='pt')
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict)
    
    def save_model(self, extension='', directory='models'):
        filename = self.get_output_path(filename='m', directory=directory, extension=extension, file_type='pt')
        torch.save(self.model.state_dict(), filename)

    def get_output_path(self, filename, file_type, directory=None, extension=None):
        output_dir = "output/"
        dir_path = output_dir + self.arguments.model_name + "/" + self.arguments.dataset

        if directory:
            # For example: .../models
            dir_path = dir_path + "/" + directory

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        file_path = dir_path + "/" + filename
        if extension is not None:
            file_path += "_" + str(extension)
        file_path += "." + file_type

        return file_path
    
    def clone(self):
        return self.__class__(self.params, self.arguments)