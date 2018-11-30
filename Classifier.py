class Classification_Model:
    
    import pandas as pd
    
    def __init__(self, df):
        
        self.df = df
        
    def target_variable(self, y_var_string):
        
        if y_var_string in list(self.df):
            self.y_var = y_var_string
            
            x_var = list(self.df)
            x_var.remove(self.y_var)
            
            self.x_var = x_var
        else:
            print("Y variable not in df")
            
    def pp_select_k_best(self, K):
        
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_classif
        
        X_columns = SelectKBest(f_classif, k = K).fit(self.df[self.x_var], self.df[self.y_var])
        indices = X_columns.get_support(indices=True)
        d = list(self.df[self.x_var])
        column_names = []
    
        for k in range(len(indices)):
            column_names.append(d[indices[k]])
            
        self.x_var = column_names
        
    def pp_remove_x_column(self, column_name_string):
        
        x_var = self.x_var
        x_var.remove(column_name_string)
        self.x_var = x_var
        
    def pp_filter_cat_cols(self, datatype, lower_bound, upper_bound, avoid_cols = [], explain = False):
        
        columns = self.x_var

        good_columns = []
        bad_columns = [] 

        avoid_columns = avoid_cols

        for i in range(len(columns)):

            if columns[i] in avoid_columns:
            
                good_columns.append(columns[i])
                
            elif df[columns[i]].dtype != datatype :
                
                good_columns.append(columns[i])

            elif df[columns[i]].dtype == datatype and df[columns[i]].count() > lower_bound and df[columns[i]].count() < upper_bound:

                good_columns.append(columns[i])

            else:
                bad_columns.append(columns[i])
                
        self.x_var = good_columns
        
        if explain == True:
        
            print('I took these columns out ' + str(bad_columns))
        
    def pp_filter_null_cols(self, null_upper_bound, avoid_cols = [], explain=False):
        
        columns = self.x_var

        good_columns = []
        bad_columns = [] 

        avoid_columns = avoid_cols


        for i in range(len(columns)):

            if self.df[columns[i]].isnull().sum()/len(self.df) < null_upper_bound or columns[i] in avoid_cols:
                
                good_columns.append(columns[i])

            else:
                bad_columns.append(columns[i])
                
        self.x_var = good_columns
        
        if explain == True:
        
            print('I took these columns out ' + str(bad_columns))
        
    def pp_label_encode_x_var(self):
        
        from sklearn.preprocessing import LabelEncoder
        
        label_encoder = LabelEncoder()
        
        df = self.df
        
        columns = self.x_var

        for i in range(len(columns)):
            
            if df[columns[i]].dtype == 'object':
            
                df[columns[i]] = label_encoder.fit_transform(df[columns[i]].astype(str))
                
            else:

                df[columns[i]] = df[columns[i]]

            
        self.df = df
        
    def pp_fillna_col(self, col_string, value):
        
        self.df[col_string] = self.df[col_string].fillna(value)
        
    def pp_fillna_x_var(self, value):
        
        for i in range(len(self.x_var)):
            
            self.df[self.x_var[i]] = self.df[self.x_var[i]].fillna(value)
        
    def model_tt_split(self, test_size):
        
        from sklearn.model_selection import train_test_split
        
        x_train, x_test, y_train, y_test = train_test_split(self.df[self.x_var], 
                                                            self.df[self.y_var],
                                                           test_size = test_size)
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
    def model_create_eval(self):
        
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import matthews_corrcoef
        import datetime
        
        result = pd.DataFrame()
        
        cross_val = cross_val_score(self.model, self.x_test, self.y_test, cv = 5)
        y_pred = self.model.predict(self.x_test)

        result['Date'] = [datetime.datetime.now()]
        result['Model_Type'] = [self.model_type]
        result['Accuracy'] = [cross_val.mean()]
        result['Acc_Std_Dev'] = [cross_val.std()]
        result['MCC'] = [matthews_corrcoef(self.y_test, y_pred)]
        result['Datapoints'] = [len(self.df)]
        
        percent_max = self.df[self.y_var].value_counts().max()/len(self.df)
        result['Percent_max'] = [percent_max]
        
        result['K'] = [len(self.x_var)]
        result['X_var'] = [self.x_var]

        self.model_eval = result
        
        
    def model_run(self, model_type):
        
        self.model_type = model_type
        
        if model_type == 'rf':
        
            from sklearn.ensemble import RandomForestClassifier
            

            model = RandomForestClassifier(n_estimators = 200)

            model.fit(self.x_train, self.y_train)
            
            self.model = model
            
            self.model_create_eval()

            
        elif model_type == 'lr':
            
            from sklearn.linear_model import LogisticRegression
            
            model = LogisticRegression(penalty = 'l2')
            
            model.fit(self.x_train, self.y_train)

            self.model = model
            
            self.model_create_eval()
            
        elif model_type == 'gbc':
            
            from sklearn.ensemble import GradientBoostingClassifier
            
            model = GradientBoostingClassifier(n_estimators = 500)
            
            model.fit(self.x_train, self.y_train)

            self.model = model
            
            self.model_create_eval()
            
        else:
            print('Sorry, don\'t know that one')
        
    def eval_feature_importances(self):
        
        if self.model_type in ['rf','gbr']:
        
            feature_importance = pd.DataFrame()

            feature_importance['feature'] = self.x_var
            feature_importance['importance'] = list(self.model.feature_importances_)

            return feature_importance.sort_values(by = ['importance'], ascending = False)
        
        else:
             print('That feature is not availble for this model')
        
    def eval_conf_matrix(self):
        
        y_pred = self.model.predict(self.x_test)
        
        self.y_pred = y_pred
        
        return pd.crosstab(self.y_test, self.y_pred)