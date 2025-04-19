import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from xgboost import XGBClassifier as xgb

from imblearn.over_sampling import SMOTE

from sentence_transformers import SentenceTransformer



###########################

#Set print option to display max
pd.set_option('display.max_rows', None, 'display.max_columns', None)

def load():

    #Load data
    df_ = pd.read_csv("C:/Users/rjsch/Desktop/Police Report Time/Calls_for_Service_2024.csv")

    df1 = pd.read_csv("C:/Users/rjsch/Downloads/Calls_for_Service_2022.csv")

    df2 = pd.read_csv("C:/Users/rjsch/Downloads/Calls_for_Service_2023.csv")

    df = pd.concat([df_, df1, df2], ignore_index=True)

    #Form data to only be Initiated by 911 call
    df = df[df['SelfInitiated'] == 'N']

    #Drop Rows wihtout Dispatch Time
    df.dropna(subset=['TimeCreate', 'TimeArrive'], axis=0, inplace=True)

    #Drop values that have consistant NaN and are spotted not useful
    df.drop(['Location', 'NOPD_Item', 'InitialType'], axis=1, inplace=True)

    further_adjust(df)



def further_adjust(df):

    #Convert Times to time format
    df['TimeCreate'] = pd.to_datetime(df['TimeCreate'], format='%m/%d/%Y %I:%M:%S %p')
    df['TimeArrive'] = pd.to_datetime(df['TimeArrive'], format='%m/%d/%Y %I:%M:%S %p')

    #Get time taken
    df['TimeTaken'] = df['TimeArrive'] - df['TimeCreate']
    df['SecondsTaken'] = df['TimeTaken'].dt.total_seconds()

    df.drop(df[df['TimeTaken'].dt.total_seconds() < .1].index, axis=0, inplace=True)

    #Get hour block
    df['HourBlockCreate'] = df['TimeCreate'].dt.hour

    #Drop values that come from the future if predicting
    df.drop(['TimeDispatch', 'TimeClosed', 'Disposition', 'DispositionText', 'Beat', 'Type', 'TypeText', 'Priority', 'PoliceDistrict'], axis=1, inplace=True)

    #Drop SelfInitiated column as its only nonself initiated (Only calls)
    df.drop('SelfInitiated', axis=1, inplace=True)

    #Change Seconds to Correct Data Type
    df['SecondsTaken'] = df['SecondsTaken'].astype(int)

    create_columns(df)


def create_columns(df):
    df['MinutesTaken'] = df['SecondsTaken'] / 60

    #print(df.describe())

    #df.drop(df[df['SecondsTaken'] >= 3599].index, axis=0, inplace=True)

    bins = [0, 3, 8, 25, 45, 10070]
    labels = ['0-2', '2-5', '5-25', '25-45', '45+']
    df['MinutesTakenBin'] = pd.cut(df['MinutesTaken'], bins=bins, labels=labels, right=False)

    #Encode 
    label_encoder = LabelEncoder()
    #df['InitialTypeText'] = label_encoder.fit_transform(df['InitialTypeText'])
    df['InitialPriority'] = label_encoder.fit_transform(df['InitialPriority'])
    df['BLOCK_ADDRESS'] = label_encoder.fit_transform(df['BLOCK_ADDRESS'])
    #df['InitialTypeText'] = label_encoder.fit_transform(df['InitialTypeText'])


    df[['MapX', 'MapY']] = df[['MapX', 'MapY']].replace(0, np.nan)
    df['MapX'] = label_encoder.fit_transform(df['MapX'])
    df['MapY'] = label_encoder.fit_transform(df['MapY'])

    df.dropna(axis=0, inplace=True)

    #Get day of week
    df['DayOfWeek'] = df['TimeCreate'].dt.weekday
    df['MonthOfYear'] = df['TimeCreate'].dt.month
    df['Weekday'] = df['TimeCreate'].dt.dayofweek

    df['Weekend'] = df['Weekday'] > 4
    df['Weekend'] = df['Weekend'].astype(int)
    
    df['BusinessHours'] = (df['HourBlockCreate'] <= 7) & (df['HourBlockCreate'] >= 17)
    df['CallDuringRushHourMorning'] = df['HourBlockCreate'].isin(range(7, 10))
    df['CallDuringRushHourAfternoon'] = df['HourBlockCreate'].isin(range(15, 17))


    transform_InitialTypeText(df)



def  transform_InitialTypeText(df):
    #Step 1: Get non-null texts
    texts = df['InitialTypeText'].dropna().tolist()

    # Step 2: Encode using Sentence Transformers
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)

    # Step 3: Cluster
    k = min(75, len(texts))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)

    # Step 4: Assign cluster labels directly to the original DataFrame
    df['Cluster_InitialTypeText'] = pd.NA
    df.loc[df['InitialTypeText'].notna(), 'Cluster_InitialTypeText'] = kmeans.labels_

    analyze(df)


def analyze(df):


    #print(df['TimeTaken'].min())

    df.to_csv('file1.csv', index=False)
    print('done')
    #Count Values to get input\
    #for features in df[['InitialTypeText']]:
        #print(f'{df[features].value_counts()}\n')


    #Use Scatterplot to check out corelations
    def scatter():
        for feature in df.columns:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=df[feature], y=df['SecondsTaken'], data=df)
            plt.xlabel(feature)
            plt.ylabel('SecondsTaken')
            plt.title(f'Comparing SecondsTaken with {feature}')
            plt.show()
    #scatter()

    def boxplot():
        plt.figure(figsize=(8, 5))

    xgbclas(df)



def decision_tree(df):

    #df = pd.read_csv("C:/Users/rjsch/Desktop/Police Report Time/file1.csv")

    encoder = LabelEncoder()
    df['MinutesTakenBin'] = encoder.fit_transform(df['MinutesTakenBin'])
    df['InitialTypeText'] = encoder.fit_transform(df['InitialTypeText'])

    df.to_csv('Encoded_With_Cluster.csv', index=False)

    #Get X and y
    X = df.drop(['MinutesTakenBin', 'TimeCreate', 'TimeArrive', 'TimeTaken', 'SecondsTaken', 'MinutesTaken'], axis=1)
    y= df['MinutesTakenBin']

    #Get Test and Train Groups
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=30)

    #Parameter Grid
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 4, 10],
        'max_features': ['auto', 'sqrt', 'log2'],
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, verbose=2)
    #grid_search.fit(X_train, y_train)

    #print("Best Parameters found: ", grid_search.best_params_)

    #Fit Model
    model = DecisionTreeClassifier(
        criterion = 'gini',
        max_depth = 10,
        max_features = 'sqrt',
        min_samples_leaf = 1,
        min_samples_split = 10
    )
    model.fit(X_train, y_train)

    #Get YHAT
    yhat = model.predict(X_test)

    #Check accuracy
    print(f'Accuracy {accuracy_score(y_test, yhat)}')

    print(classification_report(y_test, yhat))


def xgbclas(df):

    df = pd.read_csv("C:/Users/rjsch/Desktop/Police Report Time/file1.csv")

    encoder = LabelEncoder()
    df['MinutesTakenBin'] = encoder.fit_transform(df['MinutesTakenBin'])
    df['InitialTypeText'] = encoder.fit_transform(df['InitialTypeText'])

    df.to_csv('Encoded_With_Cluster.csv', index=False)

    #Get X and y
    X = df.drop(['MinutesTakenBin', 'TimeCreate', 'TimeArrive', 'TimeTaken', 'SecondsTaken', 'MinutesTaken'], axis=1)
    y= df['MinutesTakenBin']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=30)

    X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    param_distributions = {
        'n_estimators': [75, 100, 125, 150, 200],  # You can add more options
        'max_depth': [5, 6, 7, 8, 9],
        'learning_rate': [0.01, 0.05, 0.1],  # Add more learning rates
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }

    random_search = RandomizedSearchCV(xgb(objective='multi:softprob', 
                                           num_class=7), 
                                           param_distributions=param_distributions, 
                                           n_iter=100, 
                                           cv=5, 
                                           verbose=1, 
                                           n_jobs=-1, 
                                           scoring='f1_macro', 
                                           random_state=40,

                                           )   
    
    # random_search.fit(X_train, y_train)

    # print(random_search.best_params_, random_search.best_score_)

    model = xgb(objective='multi:softprob',
                num_class=5,
                colsample_bytree = 0.9,
                learning_rate = 0.1,
                max_depth = 9,
                n_estimators = 150,
                subsample = .7,
                gamma = 0,
                min_child_weight = 1
                )
    
    model.fit(X_train, y_train)

    yhat = model.predict(X_test)

    #Check accuracy
    print(f'Accuracy {accuracy_score(y_test, yhat)}')
    print()
    print(classification_report(y_test, yhat))
    print()
    print(cross_val_score(model, X_test, y_test, cv=5))



load()