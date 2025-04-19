import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier as xgb

from imblearn.over_sampling import SMOTE

############################################################

def data():
    #Load Data
    df = pd.read_csv("C:/Users/rjsch/Desktop/Police Report Time/file1.csv")

    #Drop Bin
    df.drop('MinutesTakenBin', axis=1, inplace=True)

    #Drop Majority of Outliers
    #df.drop(df[df['MinutesTaken'] >= 180].index, axis=0, inplace=True)

    #print(df['MinutesTaken'].describe())

    #Bins for Predict
    bins = [0, 3, 12, 40, 9834]
    labels = [0, 1, 2, 3]
    df['MinutesTakenBin'] = pd.cut(df['MinutesTaken'], bins=bins, labels=labels, right=False)

    #Change Data Type
    df['Zip'] = df['Zip'].astype(int)

    #Change Boolean to Int Values
    df['BusinessHours'] = df['BusinessHours'].astype(int)
    df['CallDuringRushHourMorning'] = df['CallDuringRushHourMorning'].astype(int)
    df['CallDuringRushHourAfternoon'] = df['CallDuringRushHourAfternoon'].astype(int)

    #Change Time to Datetime
    df['TimeCreate'] = pd.to_datetime(df['TimeCreate'])

    #Gather Seasons called in
    def get_season(date):
        year = date.year
        spring = pd.Timestamp(f'{year}-03-20')
        summer = pd.Timestamp(f'{year}-06-21')
        fall = pd.Timestamp(f'{year}-09-22')
        winter = pd.Timestamp(f'{year}-12-21')
        
        if spring <= date < summer:
            return 'Spring'
        elif summer <= date < fall:
            return 'Summer'
        elif fall <= date < winter:
            return 'Fall'
        else:
            return 'Winter'

    # Apply function to the date column
    df['WeatherSeason'] = df['TimeCreate'].apply(get_season)

    def get_shift(timestamp):
        time = timestamp.time()
        
        morning_start = pd.to_datetime('06:00').time()
        afternoon_start = pd.to_datetime('12:00').time()
        evening_start = pd.to_datetime('18:00').time()
        night_start = pd.to_datetime('00:00').time()
        
        if morning_start <= time < afternoon_start:
            return 'Morning'
        elif afternoon_start <= time < evening_start:
            return 'Afternoon'
        elif evening_start <= time or time < night_start:
            return 'Evening'
        else:
            return 'Night'
    df['ShiftBlock'] = df['TimeCreate'].apply(get_shift)

    #Encode data
    encoder = LabelEncoder()
    df['InitialTypeText'] = encoder.fit_transform(df['InitialTypeText'])
    df['WeatherSeason'] = encoder.fit_transform(df['WeatherSeason'])
    df['ShiftBlock'] = encoder.fit_transform(df['ShiftBlock'])

    model(df)



def model(df):

    df.to_csv('Work_With_Data_File.csv', index=False)
    print('Saved File')

    #Get X and y
    X = df.drop(['MinutesTakenBin', 'TimeCreate', 'TimeArrive', 'TimeTaken', 'SecondsTaken', 'MinutesTaken'], axis=1)
    y= df['MinutesTakenBin']

    #Gather splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=30)
    print('Splited Data')

    #SMOTE Data
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    print('Data Smoted')

    # param_distributions = {
    #     'n_estimators': [75, 100, 125, 150, 200],
    #     'max_depth': [5, 6, 7, 8, 9],
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'subsample': [0.7, 0.8, 0.9, 1.0],
    #     'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    #     'min_child_weight': [1, 3, 5],
    # }

    # random_search = RandomizedSearchCV(xgb(objective='multi:softprob', 
    #                                        num_class=5), 
    #                                        param_distributions=param_distributions, 
    #                                        n_iter=125, 
    #                                        cv=5, 
    #                                        verbose=1, 
    #                                        n_jobs=-1, 
    #                                        scoring='f1_macro', 
    #                                        random_state=30,

    #                                        )   
    
    # random_search.fit(X_train, y_train)

    # print(random_search.best_params_, random_search.best_score_)

    #Create Model
    model = xgb(objective='multi:softprob',
                num_class=4,
                colsample_bytree = 0.9,
                learning_rate = 0.1,
                max_depth = 9,
                n_estimators = 150,
                subsample = .7,
                gamma = 0,
                min_child_weight = 1
                )
    print('Model Formed')
    
    #Fit Model
    model.fit(X_train, y_train)
    print('Model Fitted')

    #Gather Prediction Ability
    yhat = model.predict(X_test)
    print('YHAT Created')

    #Check accuracy
    print(f'Accuracy {accuracy_score(y_test, yhat)}')
    print()
    print(classification_report(y_test, yhat))
    print()
    print(cross_val_score(model, X_test, y_test, cv=5))


data()

