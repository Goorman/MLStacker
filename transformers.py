from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from itertools import product, combinations
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm_notebook
from tqdm import tqdm


#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

class identity_transformer():
    def __init__(self):
        pass
    
    def fit_transform(self, train):
        return train
        
    def transform(self, test):
        return test

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

class logreg_prediction():
    def __init__(self, feature_list, target, clf_params, tag):
        self.feature_list = feature_list
        self.target = target
        self.tag = tag
        self.clf_params = clf_params
        
    def fit_transform(self, train):
        self.transformer = LogisticRegression(**self.clf_params)
        
        self.transformer.fit(train[self.feature_list], train[self.target])
        
        return train
        
    def transform(self, test):
        test[self.tag] = self.transformer.predict_proba(test[self.feature_list])[:,0]
        
        return test

    def get_column_list(self):
        return [self.tag]

class logreg_stacking():
    def __init__(self, feature_list, target, clf_params, fold_number, tag, verbose = True):
        self.feature_list = feature_list
        self.target = target
        self.tag = tag
        self.clf_params = clf_params
        self.fold_number = fold_number
        self.verbose = verbose
        
    def fit_transform(self, train):
        #TODO:folding and stacking
        kfold = StratifiedKFold(n_splits = self.fold_number, shuffle = True)
        cv = list(kfold.split(train.index, train[self.target]))

        stack_feature_list = []
        if self.verbose:
        	iterator = tqdm_notebook(list(enumerate(cv)))
        else:
        	iterator = enumerate(cv)

        for i, fold in iterator:
            fold_X_train = train[self.feature_list].iloc[fold[0]]
            fold_Y_train = train[self.target].iloc[fold[0]]
            fold_X_test = train[self.feature_list].iloc[fold[1]]
            fold_Y_test = train[self.target].iloc[fold[1]]

            fold_transformer = LogisticRegression(**self.clf_params)

            fold_transformer.fit(fold_X_train, fold_Y_train)

            fold_y_predict = fold_transformer.predict_proba(fold_X_test)[:,0]
            
            stack_feature_list.append(pd.Series(index = fold[1], data = fold_y_predict))

        new_feature = pd.concat(stack_feature_list)

        train[self.tag] = new_feature

        self.transformer = LogisticRegression(**self.clf_params)
        
        self.transformer.fit(train[self.feature_list], train[self.target])
        
        return train
        
    def transform(self, test):
        test[self.tag] = self.transformer.predict_proba(test[self.feature_list])[:,0]
        
        return test

    def get_column_list(self):
        return [self.tag]

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

class xgb_prediction():
    def __init__(self, feature_list, target, clf_params, tag):
        self.feature_list = feature_list
        self.target = target
        self.tag = tag
        self.clf_params = clf_params

    def fit_transform(self, train):
        self.transformer = XGBClassifier(**self.clf_params)
        
        self.transformer.fit(train[self.feature_list], train[self.target], 
                             eval_set = [(train[self.feature_list], train[self.target])],
                             eval_metric = "logloss")
        
        return train

    def transform(self, test):
        test[self.tag] = self.transformer.predict_proba(test[self.feature_list])[:,0]
        
        return test

    def get_column_list(self):
        return [self.tag]

class xgb_stacking():
    def __init__(self, feature_list, target, clf_params, tag):
        self.feature_list = feature_list
        self.target = target
        self.tag = tag
        self.clf_params = clf_params

    def fit_transform(self, train):
        #TODO:folding and stacking
        self.transformer = XGBClassifier(**self.clf_params)
        
        self.transformer.fit(train[self.feature_list], train[self.target], 
                             eval_set = [(train[self.feature_list], train[self.target])],
                             eval_metric = "logloss")
        
        return train

    def transform(self, test):
        test[self.tag] = self.transformer.predict_proba(test[self.feature_list])[:,0]
        
        return test

    def get_column_list(self):
        return [self.tag]

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

class onehot_transformer():
    def __init__(self, feature_list, tag):
        self.feature_list = feature_list
        self.tag = tag

    def fit_transform(self, train):
        pass

    def transform(self, test):
        pass

    def get_column_list(self):
        pass

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

class logical_transformer():
	def _init__(self, feature_list, tag):
		pass

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

class multiplication_transformer():
    def __init__(self, feature_list, degree_list, feature_num, tag):
        self.feature_list = feature_list
        self.degree_list = degree_list
        self.feature_num = feature_num
        self.tag = tag
        self.__info__ = dict()

        i = 1
        for features in combinations(self.feature_list, self.feature_num):
            for degrees in product(self.degree_list, repeat = self.feature_num):
                features = list(features)
                feature_name = "{tag}_{index}".format(tag = self.tag, index = str(i)) 
                self.__info__[feature_name] = {"features":features, "degrees":degrees} 
                i += 1

    def fit_transform(self, train):
        new_train = train.copy()

        for feature_name in self.__info__:
            features, degrees = self.__info__[feature_name]["features"], self.__info__[feature_name]["degrees"]
            new_train[feature_name] = np.power(new_train[features].astype(np.float), degrees).prod(axis = 1)

        return new_train

    def transform(self, test):
        new_test = test.copy()
        
        for feature_name in self.__info__:
            features, degrees = self.__info__[feature_name]["features"], self.__info__[feature_name]["degrees"]
            new_test[feature_name] = np.power(new_test[features].astype(np.float), degrees).prod(axis = 1)

        return new_test

    def get_column_list(self):
        return self.__info__.keys()

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

class additive_transformer():
    def __init__(self, feature_list, feature_num, tag):
        self.feature_list = feature_list
        self.feature_num = feature_num
        self.tag = tag
        self.__info__ = dict()

        i = 1
        for features in combinations(self.feature_list, self.feature_num):
                features = list(features)
                feature_name = "{tag}_{index}".format(tag = self.tag, index = str(i)) 
                self.__info__[feature_name] = features
                i += 1

    def fit_transform(self, train):
        new_train = train.copy()

        for feature_name in self.__info__:
            features = self.__info[feature_name]
            new_train[feature_name] = new_train[features].sum()

        return new_train

    def transform(self, test):
        new_test = test.copy()
        
        for feature_name in self.__info__:
            features = self.__info[feature_name]
            new_test[feature_name] = new_test[features].sum()

        return new_test

    def get_column_list(self):
        return self.__info__.keys()


#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

class scaling_transformer():
    def __init__(self, feature_list, tag):
        self.feature_list = feature_list
        self.tag = tag
        self.scaler = StandardScaler()
        self.columns = ["{tag}_{feature}".format(tag = self.tag, feature = feature) for feature in self.feature_list]

    def fit_transform(self, train):    
        scaled_data = self.scaler.fit_transform(train[self.feature_list])
        new = pd.DataFrame(scaled_data, columns = self.columns, index = train.index)

        return pd.concat([train, new], axis = 1)

    def transform(self, test):
        scaled_data = self.scaler.transform(test[self.feature_list])
        new = pd.DataFrame(scaled_data, columns = self.columns, index = test.index)

        return pd.concat([test, new], axis = 1)

    def get_column_list(self):
        return self.columns

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

class custom_transformer():
    def __init__(self, func, argnames, tag):
        self.func = func
        if isinstance(argnames, list):
            self.argnames = argnames
        elif isinstance(argnames, basestring):
            self.argnames = [argnames]
        else:
            raise Exception("Unknown argnames list")
        self.tag = tag
        self.__info__ = dict()

        if len(argnames) == 1:
            feature_name = "{tag}".format(tag = self.tag)
            self.__info__[feature_name] = {"argset" : argnames[0]}
        else:
            for i, feature_set in enumerate(argnames):
                feature_name = "{tag}_{index}".format(tag = self.tag, index = i)
                self.__info__[feature_name] = {"argset" : feature_set}

    def fit_transform(self, train):
        for feature_name in self.__info__:
            feature_set = self.__info__[feature_name]["argset"]
            train[feature_name] = train[feature_set].apply(self.func, axis = 1)

        return train

    def transform(self, test):
        for feature_name in self.__info__:
            feature_set = self.__info__[feature_name]["argset"]
            test[feature_name] = test[feature_set].apply(self.func, axis = 1)

        return test

    def get_column_list(self):
        return self.__info__.keys()

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

class convex_combination():
    pass