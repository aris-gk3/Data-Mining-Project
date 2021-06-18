import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split , KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, precision_score, recall_score,precision_recall_fscore_support,balanced_accuracy_score,confusion_matrix
import matplotlib.pyplot as plot

# Evaluating Metrics for RandomForest
def count_metrics(y_test,y_pred):
    f1 = f1_score(y_test, y_pred,average='weighted')
    precision = precision_score(y_test, y_pred,average='weighted')
    recall = recall_score(y_test, y_pred,average='weighted')
    bal_accuracy = balanced_accuracy_score(y_test, y_pred,adjusted=True)
    print("Precision Score : ",precision)
    print("Recall Score : ",recall)
    print("f1 Score : ",f1)
    print("Balanced Accuracy Score : ",bal_accuracy)
    print(precision_recall_fscore_support(y_test, y_pred))
    cm1 = confusion_matrix(y_test,y_pred)
    sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    print('Sensitivity : ', sensitivity1 )
    specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    print('Specificity : ', specificity1)
    
    return f1,precision,recall,bal_accuracy

# Handle Missong Values based on the chosen parameter
def handle_mv(method,df):
    # Delete all columns with mv
    if method == 'delete':
        new_df = Delete_Column(df)
    # Fill with Mean
    elif method == 'mean':
        new_df = Fill_with_Mean(df)
    # Fill with Linear Regression
    elif method == 'lregression':
        new_df = Linear_Regression(df)
    # Fill with KNN
    elif method == 'knn':
        new_df = knn(df)
    # Fill with Linear Regression and KNN
    elif method == 'lregression-knn': 
        new_df = knn(df)
        new_df = Linear_Regression(df)
    return new_df
    
#####  Functions to handle missing values  #####
def Delete_Column(df):
    df = df.drop(columns="smoking_status")
    df = df.dropna(axis=1)
    return df

def Fill_with_Mean(df):
    df = df.drop(columns="smoking_status")
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
    return df

# Fill numerical mv with Linear Regression
def Linear_Regression(df):
    mask = df['bmi'].isnull()
    y = pd.DataFrame(df['bmi']).to_numpy()
    x = pd.DataFrame(df['avg_glucose_level']).to_numpy()
    y = y.reshape(-1,1)
    x = x.reshape(-1,1)
    x_train = x[~mask]
    y_train = y[~mask]
    reg = LinearRegression().fit(x_train,y_train)
    x_pred = x[mask]
    y_pred = reg.predict( x_pred )
    y_pred = y_pred.reshape(-1,)
    df['bmi'][mask] = y_pred
    return df

# Fill categoricla mv with KNN
def knn(df):
    # Label Encoding for Nominal/Categorical Attributes
    df = df.drop(columns='bmi')
    mask = df['smoking_status'] == 'Unknown'
    y_train = LabelEncoder().fit_transform(df['smoking_status'][~mask])
    neigh = KNeighborsClassifier(n_neighbors=3)
    x = df.drop(columns='smoking_status')
    x = label_encoding(x)
    x = StandardScaler().fit_transform(x)
    neigh.fit(x[~mask], y_train)
    y_pred = neigh.predict(x[mask])
    df['smoking_status'][mask]= neigh.predict(x[mask])
    return df

# 5-fold tests on KNN



def label_encoding(data):
    for i in range(0,len(data.columns)):
        if isinstance(data[data.columns[i]][0], str): # values of attribute are categorical
            # data[data.columns[i]] = LabelEncoder().fit_transform(data[data.columns[i]])
            enc = OneHotEncoder()
            arr = np.asarray(data[data.columns[i]]).reshape(-1,1)
            enc.fit(arr)
            data[data.columns[i]] = enc.transform(arr).toarray()
    return data

def main():    
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    # Checks ID column to see if we have duplicate entries and deletes (Not necessary)
    df[df.duplicated(['id'])].drop_duplicates(inplace=True, keep='first')
    # Delete id column
    df = df.drop(columns='id')
    # Create new set depending on the method of handling the missing values
    data = handle_mv('lregression',df)
    # Label Encoding
    coded_data = label_encoding(data)
    y = coded_data['stroke'].to_numpy()
    coded_data = coded_data.drop(columns = 'stroke')
    x = np.asarray(coded_data)
    clf = RandomForestClassifier(criterion= 'gini',bootstrap=True,oob_score=True ,class_weight='balanced')
    f1_list = list()
    precision_list = list()
    recall_list = list()
    bal_accuracy_list = list()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Use for 5-Fold cross validation
    for train_index, test_index in kf.split(x):
        x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]     
        clf.fit(x_train, y_train )
        y_pred = clf.predict(x_test)
        f1,precision,recall,bal_accuracy = count_metrics(y_test, y_pred)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        bal_accuracy_list.append(bal_accuracy)
if __name__ == "__main__":
    main()
