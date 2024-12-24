# Imports
import numpy as np  # For numerical computations
import pandas as pd  # For data analysis
import torch  # For building and training neural networks
import torch.nn as nn  # For defining neural network structures
import torch.optim as optim  # For optimization algorithms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt  # For plotting graphs

# Global variables for paths
TRAIN_PATH = "~/Desktop/PythonProject/playground-series-s4e6/train.csv"
TEST_PATH = "~/Desktop/PythonProject/playground-series-s4e6/test.csv"

# Define reusable utility functions
def load_data(file_path):
    """
    Load dataset from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    return pd.read_csv(file_path)

# Data Preprocessing: Cleaning, handling missing values, and encoding features
def preprocess_data(data):
    """
    Perform data cleaning, handle missing values, and normalize numerical features,
    while ensuring the 'Target' column remains as one column for binary classification.

    Args:
        data (DataFrame): The dataset to preprocess.
    Returns:
        DataFrame: Preprocessed dataset.
    """
    # Exclude columns that should not be preprocessed
    columns_to_exclude = [
        "id",
        "Previous qualification (grade)",
        "Mother's occupation",
        "Admission grade",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (evaluations)",
    ]

    # Keep excluded columns separate
    excluded_data = data[columns_to_exclude]
    data = data.drop(columns=columns_to_exclude)

    # Handle missing values
    data.fillna(data.median(numeric_only=True), inplace=True)  # Fill numeric NaNs with median
    data.fillna("Unknown", inplace=True)  # Fill categorical NaNs with "Unknown"

    # Ensure 'Target' is excluded from encoding and remains as a single binary column
    if 'Target' in data.columns:
        data['Target'] = data['Target'].apply(lambda x: 1 if x == 'Graduate' else 0)

    # Encode categorical features (excluding 'Target')
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns
    categorical_cols = categorical_cols.drop("Target", errors="ignore")  # Exclude 'Target'
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Identify binary columns (0/1) and exclude them from normalization
    binary_cols = [
        col for col in data.columns if set(data[col].unique()).issubset({0, 1})
    ]
    binary_data = data[binary_cols]
    data = data.drop(columns=binary_cols)

    # Normalize numerical features
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()

    # Reattach excluded columns and binary columns
    data = pd.concat([excluded_data, binary_data, data], axis=1)

    return data

# Load and preprocess data
def preprocess_and_get_input_size(train_path, test_path):
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)
    
    # 1) feature_cols are all columns except "id" and "Target"
    feature_cols = train_data.drop(columns=["id", "Target"]).columns
    
    # 2) input_size is just how many columns remain in feature_cols
    input_size = len(feature_cols)
    
    return train_data, test_data, input_size

# Define models
def define_models(input_size):
    """
    Define the logistic regression and neural network models.
    Returns:
        tuple: Logistic regression model, neural network model.
    """
    logistic_model = nn.Sequential(
        nn.Linear(input_size, 1),  # Input to single output
        nn.Sigmoid()  # Sigmoid activation for binary classification
    )

    neural_net = nn.Sequential(
        nn.Linear(input_size, 32),  # Input to hidden layer
        nn.ReLU(),  # Activation function
        nn.Linear(32, 2)  # Hidden layer to output layer
    )

    return logistic_model, neural_net

# Define loss functions
def define_loss_functions():
    """
    Define loss functions for logistic regression and neural network models.
    Returns:
        tuple: Binary cross-entropy loss for logistic regression,
               Cross-entropy loss for neural network.
    """
    logistic_loss = nn.BCELoss()  # Binary Cross-Entropy Loss for Logistic Regression
    neural_net_loss = nn.CrossEntropyLoss()  # Cross-Entropy Loss for Neural Network

    return logistic_loss, neural_net_loss

# Function to define optimization algorithms
def define_optimizers(logistic_model, neural_net, learning_rate=0.0005):
    """
    Define optimizers for logistic regression and neural network models.
    Returns:
        tuple: Optimizer for logistic regression, optimizer for neural network.
    """
    # Optimizer for logistic regression
    logistic_optimizer = optim.SGD(logistic_model.parameters(), lr=learning_rate)

    # Optimizer for neural network
    neural_net_optimizer = optim.Adam(neural_net.parameters(), lr=learning_rate)

    return logistic_optimizer, neural_net_optimizer

# Function to split data and prepare DataLoaders
def split_and_prepare_data(data, target_column, batch_size=32, split_ratio=0.2):
    """
    Split data into training and validation sets and prepare DataLoaders.
    Args:
        data (DataFrame): Preprocessed dataset.
        target_column (str): Name of the target column.
        batch_size (int): Batch size for DataLoader.
        split_ratio (float): Proportion of data for validation (0.2 for 80-20 split).
    Returns:
        tuple: DataLoaders for training and validation sets.
    """
    # Extract features and labels
    X = data.drop(columns=[target_column, "id"]).values
    y = data[target_column].values

    # Shuffle indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Split indices
    split_point = int(len(X) * (1 - split_ratio))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    # Split data
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    # Determine correct type for targets
    if set(y) == {0, 1}:  # Binary classification
        y_train = torch.tensor(y_train, dtype=torch.float32)  # Float for BCELoss
        y_val = torch.tensor(y_val, dtype=torch.float32)
    else:  # Multi-class classification
        y_train = torch.tensor(y_train, dtype=torch.long)  # Long for CrossEntropyLoss
        y_val = torch.tensor(y_val, dtype=torch.long)

    # Prepare DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Function to train and validate a model
def train_and_validate(model, train_loader, val_loader, loss_fn, optimizer,scheduler=None, epochs=20):
    """
    Train and validate the model, tracking loss and accuracy for plotting.
    """
    model.train()
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        train_loss = 0
        for X_batch, y_batch in train_loader:
            # Adjust target type
            if isinstance(loss_fn, nn.BCELoss):
                y_batch = y_batch.float()
            elif isinstance(loss_fn, nn.CrossEntropyLoss):
                y_batch = y_batch.long()

            # Forward pass
            predictions = model(X_batch)
            loss = loss_fn(predictions.squeeze() if isinstance(loss_fn, nn.BCELoss) else predictions, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # Adjust target type
                if isinstance(loss_fn, nn.BCELoss):
                    y_batch = y_batch.float()
                elif isinstance(loss_fn, nn.CrossEntropyLoss):
                    y_batch = y_batch.long()

                predictions = model(X_batch)
                val_loss += loss_fn(predictions.squeeze() if isinstance(loss_fn, nn.BCELoss) else predictions, y_batch).item()

                # Calculate accuracy
                if predictions.shape[1] == 2:  # For classification with logits
                    predicted = torch.argmax(predictions, dim=1)
                else:  # For binary classification with probabilities
                    predicted = (predictions > 0.5).long()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct / total)

        # Step the scheduler (if provided)
        if scheduler:
            scheduler.step()

    return train_losses, val_losses, val_accuracies

# Function to perform hyperparameter tuning
def hyperparameter_tuning(train_data, target_column, model_type, param_grid, epochs=5):
    """
    Perform hyperparameter tuning for logistic regression or neural network.
    Args:
        train_data (DataFrame): Training dataset.
        target_column (str): Name of the target column.
        model_type (str): Type of model ("logistic" or "neural_net").
        param_grid (dict): Dictionary of hyperparameters to test.
        epochs (int): Number of epochs for each configuration.
    Returns:
        dict: Best hyperparameters and their validation loss.
    """
    X = train_data.drop(columns=[target_column]).values
    y = train_data[target_column].values

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32 if len(set(y)) == 2 else torch.long)

    # Split into training and validation sets
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_point = int(len(X) * 0.8)
    train_indices, val_indices = indices[:split_point], indices[split_point:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    # Ensure correct data type for targets
    y_train = torch.tensor(y_train, dtype=torch.long if model_type == "neural_net" else torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long if model_type == "neural_net" else torch.float32)

    best_params = None
    best_loss = float('inf')

    # Iterate over parameter combinations
    for learning_rate in param_grid['learning_rate']:
        for batch_size in param_grid['batch_size']:  # Pass batch_size as an integer
            # Define DataLoaders with the current batch_size
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

            # Define model
            if model_type == "logistic":
                model = nn.Sequential(
                    nn.Linear(X.shape[1], 1),
                    nn.Sigmoid()
                )
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                loss_fn = nn.BCELoss()
            elif model_type == "neural_net":
                model = nn.Sequential(
                    nn.Linear(X.shape[1], 32),
                    nn.ReLU(),
                    nn.Linear(32, 2)
                )
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                loss_fn = nn.CrossEntropyLoss()

            # Train and validate
            _, val_loss, _ = train_and_validate(model, train_loader, val_loader, loss_fn, optimizer, epochs)
            print(f"Learning Rate: {learning_rate}, Batch Size: {batch_size}, Validation Loss: {val_loss}")

            # Track best parameters
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = {'learning_rate': learning_rate, 'batch_size': batch_size}

    return {'best_params': best_params, 'best_loss': best_loss}

# Function to make predictions and generate submission file
def generate_submission(model, test_data, file_name="submission.csv"):
    """
    Generate predictions on test data and save the submission file.
    Args:
        model: Trained model (neural network).
        test_data (DataFrame): Preprocessed and aligned test dataset.
        file_name (str): Name of the submission file.
    """
    # Extract IDs and features from test data
    test_ids = test_data["id"].values
    test_features = test_data.drop(columns=["id"])

    # Convert features to PyTorch tensor
    test_features_tensor = torch.tensor(test_features.values, dtype=torch.float32)

    # Put model in evaluation mode
    model.eval()
    with torch.no_grad():
        # Get predictions
        predictions = model(test_features_tensor)

        # For binary classification, use sigmoid threshold (0.5) for predictions
        if predictions.shape[1] == 1:
            predictions = (predictions.squeeze() > 0.5).long()
        else:
            predictions = torch.argmax(predictions, dim=1)

    # Map predictions to labels
    label_mapping = {1: "Graduate", 0: "Dropout"}
    predicted_labels = [label_mapping[pred.item()] for pred in predictions]

    # Create submission DataFrame
    submission = pd.DataFrame({
        "id": test_ids,
        "Target": predicted_labels
    })

    # Save to CSV
    submission.to_csv(file_name, index=False)
    print(f"Submission file saved as {file_name}")

# Main script
if __name__ == "__main__":
    # Preprocess data:
    train_data, test_data, input_size = preprocess_and_get_input_size(TRAIN_PATH, TEST_PATH)
    # Define models
    logistic_model, neural_net = define_models(input_size)
    # Define loss functions
    logistic_loss, neural_net_loss = define_loss_functions()
    # Define optimizers
    logistic_optimizer, neural_net_optimizer = define_optimizers(logistic_model, neural_net, learning_rate=0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(neural_net_optimizer, step_size=5, gamma=0.5)
    
    # Split data into training and validation sets
    train_loader, val_loader = split_and_prepare_data(train_data, target_column="Target", batch_size=128)


    # Commented codes used for debugging and testing

    '''# Print the models
    print("Logistic Regression Model:")
    print(logistic_model)

    print("\nNeural Network:")
    print(neural_net)

    # Print the loss functions
    print("Loss Function for Logistic Regression Model:")
    print(logistic_loss)

    print("\nLoss Function for Neural Network:")
    print(neural_net_loss)

    # Print the optimizers
    print("Optimizer for Logistic Regression Model:")
    print(logistic_optimizer)

    print("\nOptimizer for Neural Network:")
    print(neural_net_optimizer)

    # Verify the data loaders
    print("\nTraining DataLoader:")
    for X_batch, y_batch in train_loader:
        print("X_batch shape:", X_batch.shape)
        print("y_batch shape:", y_batch.shape)
        break

    print("\nValidation DataLoader:")
    for X_batch, y_batch in val_loader:
        print("X_batch shape:", X_batch.shape)
        print("y_batch shape:", y_batch.shape)
        break

    # Perform hyperparameter tuning
    param_grid = {
        'learning_rate': [0.01, 0.001],
        'batch_size': [16, 32]
    }

    print("\nHyperparameter Tuning for Logistic Regression:")
    best_logistic_params = hyperparameter_tuning(train_data, target_column="Target", model_type="logistic", param_grid=param_grid)
    print(f"Best Logistic Regression Parameters: {best_logistic_params}")

    print("\nHyperparameter Tuning for Neural Network:")
    best_neural_net_params = hyperparameter_tuning(train_data, target_column="Target", model_type="neural_net", param_grid=param_grid)
    print(f"Best Neural Network Parameters: {best_neural_net_params}")'''

    # Train and track logistic regression
    logistic_train_losses, logistic_val_losses, logistic_val_accuracies = train_and_validate(
        logistic_model, train_loader, val_loader, logistic_loss, logistic_optimizer, epochs=20
    )

    # Train and track neural network
    neural_net_train_losses, neural_net_val_losses, neural_net_val_accuracies = train_and_validate(
        neural_net, train_loader, val_loader, neural_net_loss, neural_net_optimizer, scheduler=scheduler, epochs=20
    )

    # Plot convergence for training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(logistic_train_losses, label="Logistic Regression - Train Loss")
    plt.plot(logistic_val_losses, label="Logistic Regression - Val Loss")
    plt.plot(neural_net_train_losses, label="Neural Network - Train Loss")
    plt.plot(neural_net_val_losses, label="Neural Network - Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    # Plot Validation Accuracy
    epochs = 20
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), neural_net_val_accuracies, label="Neural Network - Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("Neural Network Validation Accuracy")
    plt.legend()
    plt.show()

    # Generate submission file
    generate_submission(neural_net, test_data, file_name="submission.csv")