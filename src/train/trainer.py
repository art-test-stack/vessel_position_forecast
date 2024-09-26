import torch
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from tqdm import tqdm

class Trainer:
    def __init__(
            self, 
            model, 
            loss=None, 
            metric=None, 
            optimizer=None, 
            device: str = 'cpu', 
            batch_size: int = 256
        ):
        """
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
        self.optimizer = optimizer
        self.device = device
        self.best_model = None
        self.best_score = None
        self.batch_size = batch_size  # Add batch size for mini-batch training
        self.model.to(device)

    def fit(self, X, y, k_folds=5, epochs=10, eval_on_test=False):
        """
        Fit the model using k-fold cross-validation with mini-batch training.
        Args:
            X: Features (input data)
            y: Labels (target data)
            k_folds: Number of folds for cross-validation
            epochs: Number of epochs (for neural networks)
            eval_on_test: Whether to evaluate on the validation set during training
        """
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        fold_idx = 0
        for train_idx, val_idx in kfold.split(X, y):
            print(f"Training fold {fold_idx + 1}/{k_folds}...")
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            if isinstance(self.model, xgb.XGBModel):
                # Train XGBoost model using batch mode
                self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)] if eval_on_test else None)
                if eval_on_test:
                    y_pred = self.model.predict(X_val)
                    score = self.metric(y_val, y_pred)
                    self._update_best_model(score)

            elif isinstance(self.model, torch.nn.Module):
                # Create DataLoader for mini-batch training
                train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                              torch.tensor(y_train, dtype=torch.float32))
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

                if eval_on_test:
                    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                                torch.tensor(y_val, dtype=torch.float32))
                    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                else:
                    val_loader = None

                self._train_nn(train_loader, val_loader, epochs, eval_on_test)

            fold_idx += 1

    def _train_nn(self, train_loader, val_loader=None, epochs=10, eval_on_test=False):
        """
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

            avg_loss = running_loss / len(train_loader)
            # print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

            if eval_on_test and val_loader:
                val_loss = self._evaluate_nn(val_loader)
                self._update_best_model(val_loss)

    def _evaluate_nn(self, val_loader):
        """Evaluate the neural network on validation data using mini-batches."""
        self.model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.metric(outputs, targets)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def _update_best_model(self, score):
        """Update the best model based on validation score."""
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.best_model = self.model

    def find_hyperparameters(self, X_train, y_train, param_grid, search_method="grid", k_folds=5):
        """
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

    def predict(self, X, initial_input=None, prediction_strategy='n_in_1_out', sequence_length=None):
        """
        Make predictions using the model.
        Args:
            X: Input features
            initial_input: For sequential models, first input to start the prediction
            prediction_strategy: Prediction method ('n_in_1_out', 'n_in_m_out', 'n_in_n_out')
            sequence_length: Sequence length for sequential models (if applicable)
        """
        if isinstance(self.model, xgb.XGBModel):
            return self.model.predict(X)
        elif isinstance(self.model, torch.nn.Module):
            X = torch.tensor(X).to(self.device)
            self.model.eval()
            with torch.no_grad():
                if prediction_strategy == 'n_in_1_out':
                    return self._n_in_1_out(X)
                elif prediction_strategy == 'n_in_m_out':
                    return self._n_in_m_out(X, sequence_length)
                elif prediction_strategy == 'n_in_n_out':
                    return self._n_in_n_out(X, sequence_length)
                else:
                    raise ValueError(f"Unknown prediction strategy: {prediction_strategy}")
        else:
            raise NotImplementedError("Unsupported model type for prediction.")

    def _n_in_1_out(self, X):
        # Example implementation for 'n in, 1 out' strategy
        predictions = []
        for i in range(X.size(0)):
            output = self.model(X[i:i+1])
            predictions.append(output.cpu().numpy())
        return np.concatenate(predictions)

    def _n_in_m_out(self, X, sequence_length):
        # Example implementation for 'n in, m out' strategy
        outputs = self.model(X[:, :sequence_length])
        return outputs.cpu().numpy()

    def _n_in_n_out(self, X, sequence_length):
        # Example implementation for 'n in, n out' strategy
        predictions = []
        for i in range(0, X.size(1) - sequence_length + 1):
            output = self.model(X[:, i:i+sequence_length])
            predictions.append(output.cpu().numpy())
        return np.concatenate(predictions, axis=1)

    def eval(self, X_test, y_test):
        """
        Evaluate the model on test data.
        Args:
            X_test: Test features
            y_test: Test labels
        """
        y_pred = self.predict(X_test)
        return self.metric(y_test, y_pred)
