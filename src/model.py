from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def train_model(X_train, y_train):
    # Define the model
    MLP = MLPClassifier(random_state=123, verbose=True)
    
    # Define the hyperparameters to search
    params = {
        'batch_size': [20, 30, 40, 50],
        'hidden_layer_sizes': [(2,), (3,), (3, 2)],
        'max_iter': [50, 70, 100]
    }
    
    # Set up GridSearchCV
    grid = GridSearchCV(MLP, params, cv=10, scoring='accuracy')
    
    # Fit the model using GridSearchCV
    grid.fit(X_train, y_train)
    
    # Return the best model
    return grid.best_estimator_
