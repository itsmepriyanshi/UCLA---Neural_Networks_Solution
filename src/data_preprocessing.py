import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Admit_Chance'] = (data['Admit_Chance'] >= 0.8).astype(int)
    data = pd.get_dummies(data, columns=['University_Rating', 'Research'])
    X = data.drop('Admit_Chance', axis=1)
    y = data['Admit_Chance']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
