import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from statistics import mean


def main(X,y):
    #Importing dataset
    #diamonds = pd.read_csv('test.csv')

    #Feature and target matrices
    #X = diamonds[['Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge']]
    #y = diamonds[['Total']]

    #Normalizing X
    X = normalize(X.to_numpy())

    #Training and testing split, with 25% of the data reserved as the test set
    [X_train, X_test, y_train, y_test] = train_test_split(X, y,random_state=101)

    #Define the range of lambda to test
    lmbda = np.concatenate((np.arange(0.1,1.1,0.1), np.arange(1.5, 11, 0.5), np.arange(11, 101, 1)))

    MODEL = []
    MSE = []
    for l in lmbda:
        #Train the regression model using a regularization parameter of l
        model = train_model(X_train,y_train,l)


        #Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(error(X_test,y_test,model))

    #Find best value of lmbda in terms of MSE
    ind = MSE.index(min(MSE))
    MSE_best= MSE[ind]
    model_best =MODEL[ind]
    print('MSE: ' + str(MSE[ind]))

    return MODEL[ind]


#Function that normalizes features to zero mean and unit variance.
#Input: Feature matrix X.
#Output: X, the normalized version of the feature matrix.
def normalize(X):
    X_norm = np.empty(np.shape(X))
    for i in range(len(X[1])):
        X_norm[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
    X = X_norm

    return X


#Function that trains a ridge regression model on the input dataset with lambda=l.
#Input: Feature matrix X, target variable vector y, regularization parameter l.
#Output: model, a numpy object containing the trained model.
def train_model(X,y,l):
    model = linear_model.Ridge(alpha=l, fit_intercept=True)
    model.fit(X,y)

    return model


#Function that calculates the mean squared error of the model on the input dataset.
#Input: Feature matrix X, target variable vector y, numpy model object
#Output: mse, the mean squared error
def error(X,y,model):
    y = y.to_numpy()
    y_pred = model.predict(X)

    mse = np.mean((y - y_pred) ** 2)

    return mse


if __name__ == '__main__':
    diamonds = pd.read_csv('data.csv')
    X=diamonds[['Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge']]
    y=diamonds[['Total']]
    print("*************Brooklyn Manhattan Williamsburg*************")
    model_best = main(X,y)
    print(model_best.coef_)
    print(model_best.intercept_)

    X = diamonds[['Brooklyn Bridge','Manhattan Bridge', 'Queensboro Bridge']]
    y = diamonds[['Total']]
    print("*************Brooklyn Manhattan Queensboro*************")
    model_best = main(X, y)
    print(model_best.coef_)
    print(model_best.intercept_)

    X = diamonds[['Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']]
    y = diamonds[['Total']]
    print("*************Manhattan Williamsburg Queensboro*************")
    model_best = main(X, y)
    print(model_best.coef_)
    print(model_best.intercept_)

    X = diamonds[['Brooklyn Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']]
    y = diamonds[['Total']]
    print("*************Brooklyn Williamsburg Queensboro*************")
    model_best = main(X, y)
    print(model_best.coef_)
    print(model_best.intercept_)

    X = diamonds[['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge','Total']]
    X=np.array(X)
    Brooklyn_Avg=[]
    for i in range(len(X)):
       Brooklyn_Avg.append((X[i][0]/X[i][4])*100)
    avg=mean(Brooklyn_Avg)
    print("Brooklyn " + str(avg))