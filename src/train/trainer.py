from settings import *

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import xgboost as xgb

from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import uuid

from typing import Callable, Tuple
from copy import deepcopy


class EarlyStopping:
    def __init__(self, patience=100, min_delta=5e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.lr_counter = 0
        self.best_loss = None
        self.early_stop = False
        self.reduce_lr = False
        self.save_model = True

    def __call__(self, test_loss):
        self.save_model = False
        self.reduce_lr = False
        self.early_stop = False
        if self.best_loss is None:
            self.best_loss = test_loss
            self.save_model = True
        elif test_loss < self.best_loss - self.min_delta:
            self.best_loss = test_loss
            self.counter = 0
            self.lr_counter = 0
            self.save_model = True
        else:
            self.counter += 1
            self.lr_counter += 1
            if self.lr_counter >= self.patience // 2:
                self.reduce_lr = True
            if self.counter >= self.patience:
                self.early_stop = True


class Trainer:
    def __init__(
            self, 
            model: nn.Module, # TODO: Handdle XGBoost models
            loss: nn.Module = nn.MSELoss(reduction="sum"), 
            metric = None, 
            opt: torch.optim.Optimizer | None = None, 
            device: str | torch.device = DEVICE, 
            epochs: int = 500,
            lr: float = 5e-3,
            batch_size: int = 1024,
            name: str = f"{str(uuid.uuid4())}.pt",
            clip_grad: bool = True,
            verbose: bool = True,
            **kwargs
        ):
        """
        Description:

            Initialize the trainer.
        Args:
            model: The model to be trained (XGBoost, torch.nn.Module, etc.)
            loss: Loss function for neural network models
            metric: Evaluation metric function (custom or loss for now)
            optimizer: Optimizer for neural network models
            device: Device to train the model ('cpu', 'cuda')
            batch_size: Batch size for mini-batch training (default: 32)
        """
        self.model = model
        self.loss = loss
        self.metric = metric or loss
        self.optimizer = opt or torch.optim.AdamW(params=model.parameters(), lr=lr)
        self.device = device
        self.best_model = model
        self.best_score = None
        self.batch_size = batch_size  # Add batch size for mini-batch training
        self.model.to(device)
        self.name = name
        self.already_trained = False

        self.clip_grad = clip_grad
        self.epochs = epochs

        self.losses = []
        self.val_losses = []

        self.eval_step = 20
        
        self.verbose = verbose 
        for layer in model.main:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
        
        self.load_model()

    def fit(
            self, 
            X_train: torch.Tensor, 
            y_train: torch.Tensor, 
            X_val: torch.Tensor | None = None,
            y_val: torch.Tensor | None = None,
            epochs: int | None = None, 
            eval_on_test: bool = True,
            split_ratio: float = .9,
            force_train: bool = False,
            k_folds: int = 5
        ):
        """
        Description:

            Fit the model using k-fold cross-validation with mini-batch training.
        
        Args:
            X: Features (input data)
            y: Labels (target data)
            k_folds: Number of folds for cross-validation
            epochs: Number of epochs (for neural networks)
            eval_on_test: Whether to evaluate on the validation set during training
        """
        if self.already_trained and not force_train:
            print("Model already trained. Use force_train=True to retrain.")
            return
        
        epochs = epochs or self.epochs

        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        if X_val is not None and not isinstance(X_val, torch.Tensor):
            X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        
        if y_val is not None and not isinstance(y_val, torch.Tensor):
            y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        if k_folds > 1:
            if self.verbose and False:
                print("Cross-validation not supported yet.")
            return self.fit(X_train, y_train, X_val, y_val, epochs, eval_on_test, split_ratio, force_train, k_folds=1)
            # kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

            # fold_idx = 0
            # models = [ deepcopy(self.model)for _ in range(k_folds) ]
            

            # for train_idx, val_idx in kfold.split(X_train, y_train):
            #     print(f"Training fold {fold_idx + 1}/{k_folds}...")
            #     print("train_idx:", train_idx) # TODO: remove

            #     self.model = models[fold_idx]
            #     self.optimizer = torch.optim.AdamW(params=self.model.parameters())
            #     X_train, y_train = X[train_idx], y[train_idx]
            #     X_val, y_val = X[val_idx], y[val_idx]

            #     train_loader, val_loader = self._prepare_dataloaders(X_train, y_train, X_val, y_val, eval_on_test)
            #     self._train_nn(train_loader, val_loader, epochs, eval_on_test)

            #     fold_idx += 1

        else: 
            # TODO: Random split
            if eval_on_test and (not X_val or not y_val):
                idx = int(len(X_train) * split_ratio) if eval_on_test else len(X) - 1
                X_train, y_train = X_train[:idx], y_train[:idx]
                X_val, y_val = (X_train[idx:], y_train[idx:]) if eval_on_test else (None, None)
            
            train_loader, val_loader = self._prepare_dataloaders(X_train, y_train, X_val, y_val, eval_on_test)
            self._train_nn(train_loader, val_loader, epochs, eval_on_test)

    def _prepare_dataloaders(
            self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            X_val: torch.Tensor | None,
            y_val: torch.Tensor | None,
            eval_on_test: bool = True,
        ) -> Tuple[DataLoader, DataLoader]:

        train_dataset = TensorDataset(X_train, y_train)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if eval_on_test and X_val is not None and y_val is not None:
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None
        
        return train_loader, val_loader
    

    def _train_nn(
            self, 
            train_loader: DataLoader, 
            val_loader: DataLoader | None = None, 
            epochs: int = 10, 
            eval_on_test: bool = False
        ):
        """
        Description:

            Train the neural network model with mini-batch gradient descent.
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (if applicable)
            epochs: Number of epochs for training
            eval_on_test: Whether to evaluate during training
        """
        early_stopping = EarlyStopping()

        with tqdm(range(epochs), unit="epoch", colour="red", disable=not self.verbose) as tepoch:
            # tepoch.
            for epoch in range(epochs):
                self.model.train()
                running_loss = 0.
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss(outputs, targets)

                    self.optimizer.zero_grad()
                    loss.backward()

                    if self.clip_grad:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                    self.optimizer.step()
                    
                    running_loss += loss.item()

                avg_loss = running_loss / train_loader.dataset.__len__()
                self.losses.append(avg_loss)
                # print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

                if eval_on_test and val_loader and epoch % self.eval_step == 0:
                    val_loss = self._evaluate_nn(val_loader)
                    self._update_best_model(val_loss)
                
                tepoch.set_postfix(
                    loss = self.losses[-1] if self.losses else "?",
                    val_loss = self.val_losses[-1] if self.val_losses else "?",
                    best = self.best_score if self.best_score else "?",
                    early_stopping_step = early_stopping.counter,
                    lr_counting = early_stopping.lr_counter
                )
                tepoch.update(1)
                if eval_on_test:
                    early_stopping(self.val_losses[-1])

                if early_stopping.save_model:
                    self.save_model(best = True, verbose=False)

                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

                if early_stopping.early_stop:
                    for g in self.optimizer.param_groups:
                        g['lr'] *= .5

        print(f"Best model on val score: {self.best_score}")
        print(f"Model saved at {MODEL_FOLDER.joinpath(self.name)}")
        self.plot_losses()


    def _evaluate_nn(self, val_loader):
        """
        Description:

            Evaluate the neural network on validation data using mini-batches.
        
        Args:

        """
        self.model.eval()
        running_val_loss = 0.
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.metric(outputs, targets)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / val_loader.dataset.__len__()
        self.val_losses.append(avg_val_loss)
        # print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def _update_best_model(self, score):
        """
        Description:

            Update the best model based on validation score.
        """
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.best_model = deepcopy(self.model)

    def plot_losses(self, display: bool = False):
        folder = MODEL_FOLDER.joinpath(self.name + '.png')
        plt.ioff()

        fig = plt.figure()
        plt.plot(range(len(self.losses)), self.losses, label='Training Loss')
        plt.plot([i * self.eval_step for i in range(len(self.val_losses))], self.val_losses, label='Validation Loss')
        plt.legend()
        plt.savefig(folder)
        plt.close(fig)
        if display:
            plt.show()


    def find_hyperparameters(
            self, 
            X_train, 
            y_train, 
            param_grid, 
            search_method="grid", 
            k_folds=5
        ):
        """
        Description:

            Perform hyperparameter search on the model.
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Dictionary of parameters for GridSearchCV or RandomizedSearchCV
            search_method: 'grid' for GridSearch or 'random' for RandomizedSearch
            k_folds: Number of folds for cross-validation
        """
        if search_method == "grid":
            search = GridSearchCV(self.model, param_grid, cv=KFold(n_splits=k_folds), scoring=self.metric)
        elif search_method == "random":
            search = RandomizedSearchCV(self.model, param_grid, cv=KFold(n_splits=k_folds), scoring=self.metric)
        else:
            raise ValueError("Unsupported search method: choose 'grid' or 'random'.")

        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        print(f"Best hyperparameters: {search.best_params_}")
        self.save_model(f"best_model_{self.name}")

    def save_model(self, name: str = None, best: bool = False, verbose: bool = True):
        if not name:
            name = self.name
        if not name[:-3] == ".pt":
            name = name + ".pt"

        model = self.best_model if best else self.model

        torch.save(model.state_dict(), MODEL_FOLDER.joinpath(name))
        if verbose:
            print(f"Model saved at {MODEL_FOLDER.joinpath(name)}")

        # LOAD MODEL
        # model = torch.jit.load('model_scripted.pt')
        # model.eval()

    def load_model(self):
        if not MODEL_FOLDER.joinpath(self.name).exists():
            return
        self.model.load_state_dict(torch.load(MODEL_FOLDER.joinpath(self.name)))
        self.model.eval()
        self.already_trained = True

    def predict(self, X, pred_strat='n_in_1_out', seq_len=None):
        """
        Description:

            Make predictions using the model.
        Args:
            X: Input features
            initial_input: For sequential models, first input to start the prediction
            pred_strat: Prediction method ('n_in_1_out', 'n_in_m_out', 'n_in_n_out')
            seq_len: Sequence length for sequential models (if applicable)
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)

        if len(X.shape) > 2 and X.shape[0] > 1:
            dataset = TensorDataset(X)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            y_pred = []
            for x, in dataloader:
                y_pred.append(self._predict(x, pred_strat=pred_strat, seq_len=seq_len))
            y_pred = np.concatenate(y_pred)

        else: 
            y_pred = self._predict(X, pred_strat=pred_strat, seq_len=seq_len)

        return y_pred

    def _predict(
            self, 
            x: torch.Tensor, 
            pred_strat: str = 'n_in_1_out', 
            seq_len: bool = None
        ):
        """
        Description:

        Args:
            - 
        """
        self.model.eval()
        with torch.no_grad():
            # if pred_strat == 'n_in_1_out':
            #     return self._n_in_1_out(x)
            # elif pred_strat == 'n_in_m_out':
            #     return self._n_in_m_out(x, seq_len)
            # elif pred_strat == 'n_in_n_out':
            #     return self._n_in_n_out(x, seq_len)
            # else:
            #     raise ValueError(f"Unknown prediction strategy: {pred_strat}")

            return self.model(x).cpu().numpy()


    def _n_in_1_out(self, X):
        # Example implementation for 'n in, 1 out' strategy
        predictions = []
        for i in range(X.size(0)):
            output = self.model(X[i:i+1])
            predictions.append(output.cpu().numpy())
        return np.concatenate(predictions)

    def _n_in_m_out(self, X, seq_len):
        # Example implementation for 'n in, m out' strategy
        outputs = self.model(X[:, :seq_len])
        return outputs.cpu().numpy()

    def _n_in_n_out(self, X, seq_len):
        # Example implementation for 'n in, n out' strategy
        predictions = []
        for i in range(0, X.size(1) - seq_len + 1):
            output = self.model(X[:, i:i+seq_len])
            predictions.append(output.cpu().numpy())
        return np.concatenate(predictions, axis=1)

    def eval(self, X_test, y_test):
        """
        Description:

            Evaluate the model on test data.
        Args:
            X_test: Test samples
            y_test: Test targets
        """
        y_pred = self.predict(X_test)
        y_pred = torch.Tensor(y_pred).to(self.device)
        return self.metric(y_test, y_pred)
    


# TODO: Handle XGBoost
# if isinstance(self.model, xgb.XGBModel):
#     # Train XGBoost model using batch mode
#     self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)] if eval_on_test else None)
#     if eval_on_test:
#         y_pred = self.model.predict(X_val)
#         score = self.metric(y_val, y_pred)
#         self._update_best_model(score)
