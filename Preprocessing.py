import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


print("Fill na and set index: FillSet")

class FillNA(TransformerMixin, BaseEstimator):
    def __init__(self, ind, cat, num, col='Cabin'):
        self.ind = ind
        self.cat = cat
        self.num = num
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X = X.set_index(self.ind)
        X[self.cat] = X[self.cat].fillna('Missing')
        X[self.num] = X[self.num].fillna(X[self.num].median())
        X[self.col] = X[self.col].apply(lambda x: x[0])

        return X


print("Title")

class Title(TransformerMixin, BaseEstimator):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        title = []
        for i in X[self.col]:
            if 'Miss.' in i:
                title.append('Miss.')
            elif 'Mrs.' in i:
                title.append('Mrs.')
            elif 'Mr.' in i:
                title.append('Mr.')
            else:
                title.append('No')

        X[self.col] = title

        return X


print("Standardize: Scaler")
class Scaler(TransformerMixin, BaseEstimator):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        minmax = MinMaxScaler()
        X[self.col] = minmax.fit_transform(X[self.col])

        return X

print("One hot decoder imported: Decoder")

class Decoder(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        super().__init__()

        self.columns = columns
        self.onehot = OneHotEncoder(drop='first')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        one = self.onehot.fit_transform(X[self.columns]).toarray()
        col_names = self.onehot.get_feature_names()

        return pd.concat([X.drop(self.columns, axis=1), pd.DataFrame(one, index=X.index, columns=col_names)], axis=1)


print("LabelEncoder imported: LDecoder")

class LDecoder(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        super().__init__()

        self.columns = columns
        self.le = LabelEncoder()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = self.le.fit_transform(X[col]).toarray()

        return X #pd.concat([X.drop(self.columns, axis=1), pd.DataFrame(one, index=X.index, columns=col_names)], axis=1)
    
    

print("Select imported: SelectCol")

class SelectCol(TransformerMixin, BaseEstimator):
    def __init__(self, drop_cols, target, col_str):
        self.drop_cols = drop_cols
        self.target = target
        self.col_str = col_str

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.col_str is not None:
            for i in self.col_str:
                X.apply(lambda x: str(x))

        X = X.drop(self.drop_cols, axis=1)

        train = X[~X.Survived.isna()]
        test = X[X.Survived.isna()]

        return train.drop(self.target, axis=1), train[self.target], test.drop(self.target, axis=1)

    
class DTypeConverter(TransformerMixin, BaseEstimator):
    def __init__(self, cat_cols, num_cols, float_cols):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.float_cols = float_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for col in self.cat_cols:
            X[col] = X[col].astype('category')
        
        for col in self.num_cols:
            X[col] = X[col].astype('int')
        
        for col in self.float_cols:
            X[col] = X[col].astype('float')
        
        return X


print("Grid Search train: GSCV")

def GSCV(pipe, params, X, y, test, d, m, scoring, cv=5):
    grid = GridSearchCV(estimator=pipe,
                        param_grid=params,
                        cv=cv,
                        iid=False,
                        return_train_score=False,
                        refit=True,
                        scoring=scoring
                       )
    grid.fit(X, y)
    name = d + "pred" + m + "_" + scoring
    pd.DataFrame(grid.best_estimator_.predict(test), index=test.index, columns=['Survived']).to_csv(name+".csv")
    with open(name + ".pickle", 'wb') as handle:
        pickle.dump(grid.best_estimator_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return grid.best_score_, grid.best_params_, grid.best_estimator_.predict(test)



print("Ensemble function: EnsemblePropensity")

def EnsemblePropensity(directory, folder):
    d = directory + folder
    count = 0
    df = pd.DataFrame()
    for i in os.listdir(d):
        if 'csv' in i:
            score = pd.read_csv(d + i)
            df = pd.concat([df, score.iloc[:, 1]], axis=1)
            count += 1
            index = score.iloc[:, 0]

    df = pd.DataFrame(np.sum(df, axis=1), columns=['Probability'])
    df['Survived'] = [1 if i >= 2 else 0 for i in df.Probability]
    df = df.iloc[:, 1]
    df.to_csv(d + "Ensemble_" + str(count) + ".csv")

    print("Ensemble complete")


print("Test set concat training: TrainGridCV")

def TrainGridCV(pipe, params, X, y, test, d, m, cv=5):
    grid = GridSearchCV(estimator=pipe,
                        param_grid=params,
                        cv=cv,
                        iid=False,
                        return_train_score=False,
                        refit=True
                        )
    grid.fit(X, y)
    name = d + m + "_pred"
    test_pred = grid.best_estimator_.predict(test)
    test_prob = grid.predict_proba(test)

    pd.DataFrame(test_pred, columns=['Survived']).to_csv(name + ".csv")
    with open(name + ".pickle", 'wb') as handle:
        pickle.dump(grid.best_estimator_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return grid.best_score_, grid.best_params_, test_pred, test_prob