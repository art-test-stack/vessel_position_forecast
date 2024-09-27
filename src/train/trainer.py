import torch
from torch import nn

import xgboost as xgb

from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from tqdm import tqdm

from typing import Callable, Tuple
from copy import deepcopy

class Trainer:
    def __init__(
            self, 
            model: nn.Module, # TODO: Handdle XGBoost models
            loss: nn.Module = nn.MSELoss(), 
            metric = None, 
            optimizer: torch.optim.Optimizer | None = None, 
            device: str | torch.device = 'cpu', 
            batch_size: int = 1024
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
        self.optimizer = optimizer or torch.optim.AdamW(params=model.parameters())
        self.device = device
        self.best_model = None
        self.best_score = None
        self.batch_size = batch_size  # Add batch size for mini-batch training
        self.model.to(device)

    def fit(
            self, 
            X: torch.Tensor | np.ndarray, 
            y: torch.Tensor | np.ndarray, 
            k_folds: int = 5, 
            epochs: int = 10, 
            eval_on_test: bool = False
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

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32).to(self.device)

        if k_folds > 1:

            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

            fold_idx = 0
            models = [ deepcopy(self.model)for _ in range(k_folds) ]
            

            for train_idx, val_idx in kfold.split(X, y):
                print(f"Training fold {fold_idx + 1}/{k_folds}...")
                print("train_idx:", train_idx) # TODO: remove

                self.model = models[fold_idx]
                self.optimizer = torch.optim.AdamW(params=self.model.parameters())
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                train_loader, val_loader = self._prepare_dataloaders(X_train, y_train, X_val, y_val, eval_on_test)
                self._train_nn(train_loader, val_loader, epochs, eval_on_test)

                fold_idx += 1

        else: 
            # TODO: Random split
            idx = int(len(X) * .9) if eval_on_test else len(X) - 1
            X_train, y_train = X[:idx], y[:idx]
            X_val, y_val = (X[idx:], y[idx:]) if eval_on_test else (None, None)
            
            train_loader, val_loader = self._prepare_dataloaders(X_train, y_train, X_val, y_val, eval_on_test)
            self._train_nn(train_loader, val_loader, epochs, eval_on_test)

    def _prepare_dataloaders(
            self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            X_val: torch.Tensor | None,
            y_val: torch.Tensor | None,
            eval_on_test: bool = False,
        ) -> Tuple[DataLoader, DataLoader]:

        train_dataset = TensorDataset(X_train, y_train)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if eval_on_test:
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
        for epoch in tqdm(range(epochs), colour="red"):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # avg_loss = running_loss / len(train_loader)
            # print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

            if eval_on_test and val_loader:
                val_loss = self._evaluate_nn(val_loader)
                self._update_best_model(val_loss)
            
        if eval_on_test:
            print(f"Best model on val score: {self.best_score}")
            

    def _evaluate_nn(self, val_loader):
        """
        Description:

            Evaluate the neural network on validation data using mini-batches.
        
        Args:

        """
        self.model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.metric(outputs, targets)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
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

    def find_hyperparameters(self, X_train, y_train, param_grid, search_method="grid", k_folds=5):
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
