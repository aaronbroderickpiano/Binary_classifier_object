import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

class Classification_Model:
    
    def __init__(self, df):
        self.df = df
        
    def explain(self):
    
    	print ('This is what I can do')
        
    def target_variable(self, y_var_string):
        
        if y_var_string in list(self.df):
            self.y_var = y_var_string
            
            x_var = list(self.df)
            x_var.remove(self.y_var)
            
            self.x_var = x_var
        else:
            print("Y variable not in df")
            
    def select_k_best(self, K):
        
        X_columns = SelectKBest(f_classif, k = K).fit(self.df[self.x_var], self.df[self.y_var])
        indices = X_columns.get_support(indices=True)
        d = list(self.df[self.x_var])
        column_names = []
    
        for k in range(len(indices)):
            column_names.append(d[indices[k]])
            
        self.x_var = column_names
        
    def label_encode(self):
        
        label_encoder = LabelEncoder()
        
        df = self.df
        
        columns = list(df)

        for i in range(len(columns)):
            
            df[columns[i]] = label_encoder.fit_transform(df[columns[i]].astype(str))
            
        self.df = df

    def remove_x_column(self, column_name_string):
        
        x_var = self.x_var
        x_var.remove(column_name_string)
        self.x_var = x_var
        
    def show_x_columns(self):
        
        print(self.x_var)
        
    def show_y_columns(self):
        
        print(self.y_var)
        
    def make_train_test_split(self, test_size):
        
        x_train, x_test, y_train, y_test = train_test_split(self.df[self.x_var], 
                                                            self.df[self.y_var],
                                                           test_size = test_size)
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
    def make_model(self, model_type):
        
        self.model_type = model_type
        
        if model_type == 'rf':

            model = RandomForestClassifier(n_estimators = 200)

            model.fit(self.x_train, self.y_train)

            self.model = model
            
        elif model_type == 'lr':
            
            model = LogisticRegression(penalty = 'l2')
            
            model.fit(self.x_train, self.y_train)

            self.model = model
            
        elif model_type == 'gbc':
            
            model = GradientBoostingClassifier(n_estimators = 500)
            
            model.fit(self.x_train, self.y_train)

            self.model = model
            
        else:
            print('Sorry, don\'t know that one')
        
    def model_get_feature_importances(self):
        
        if self.model_type in ['rf','gbr']:
        
            feature_importance = pd.DataFrame()

            feature_importance['feature'] = self.x_var
            feature_importance['importance'] = list(self.model.feature_importances_)

            return feature_importance.sort_values(by = ['importance'], ascending = False)
        
        else:
             print('That feature is not availble for this model')
        
    def model_get_confusion_matrix(self):
        
        y_pred = self.model.predict(self.x_test)
        
        self.y_pred = y_pred
        
        return pd.crosstab(self.y_test, self.y_pred)
    
    def model_get_accuracy(self):
        
        return self.model.score(self.x_test, self.y_test)