from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold

def split_data(df):
    x = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    return train_test_split(x, y, test_size=0.2, random_state=123)

def scale_data(xtrain, xtest):
    scaler = MinMaxScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)
    return xtrain_scaled, xtest_scaled

def train_logistic_regression(xtrain, ytrain):
    lrmodel = LogisticRegression().fit(xtrain, ytrain)
    return lrmodel

def train_random_forest(xtrain, ytrain):
    rfmodel = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_features='sqrt')
    rfmodel.fit(xtrain, ytrain)
    return rfmodel

def evaluate_model(model, xtest, ytest):
    ypred = model.predict(xtest)
    accuracy = accuracy_score(ytest, ypred)
    conf_matrix = confusion_matrix(ytest, ypred)
    return accuracy, conf_matrix

def cross_validate_model(model, xtrain, ytrain, n_splits=5):
    kfold = KFold(n_splits=n_splits)
    scores = cross_val_score(model, xtrain, ytrain, cv=kfold)
    return scores.mean(), scores.std()