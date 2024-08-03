from src.data_preprocessing import load_and_preprocess_data
from src.model import train_model
from src.evaluation import evaluate_model, plot_loss_curve
from src.utils import setup_logging, log_error

def main():
    try:
        setup_logging()
        X_train, X_test, y_train, y_test = load_and_preprocess_data('data/Admission.csv')
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        plot_loss_curve(model)
        print(f'Model trained with accuracy: {accuracy}')
    except Exception as e:
        log_error(str(e))
        raise

if __name__ == "__main__":
    main()
