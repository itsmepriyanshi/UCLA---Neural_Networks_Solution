# src/evaluation.py
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Accuracy: {accuracy}')
    return accuracy

import matplotlib.pyplot as plt

def plot_loss_curve(model):
    # Ensure the model has been fitted
    if hasattr(model, 'loss_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_, label='Loss', color='blue')
        plt.title('Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("The model has not been fitted yet or does not have loss_ attribute.")
