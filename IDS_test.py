#TEAM Bruteforce Submission
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pickle import load
import sys

#Defining all the constants needed for functions for prediction
avg_sample_rate = 24
per_feature_thresh =   [2.96110123e-01,3.80024403e-01 ,1.83377922e-01 ,2.69305954e-01,2.62430489e-01,7.47453173e-02 ,3.28176513e-01,3.23344644e+00,1.23824169e+01, 1.32992387e-01, 6.54913142e+00 ,1.86247361e-01,1.54632303e-02 ,1.28829544e-02 ,6.54951803e+00, 1.31368895e-01,6.93846846e+00 ,1.95856338e-01 ,4.50224459e+00 ,1.47374449e-01,1.38822521e+01, 1.64865429e-01 ,1.86519827e-03 ,1.79917240e+00,9.84161139e-01 ,3.37036578e+00 ,1.00562270e+00, 3.28040234e+00, 1.89252583e+00 ,4.20245425e+00 ,1.84981211e+00 ,2.72862689e+00,1.63724756e+00, 9.83080070e-01]

n_past = 12
n_future = 1
n_features = 34

#Defining function needed for reshaping to have n_past data points
def split_series(series, n_past, n_future):
    """
    Takes a series and returns two arrays with past and future data 
    # n_past ==> no of past observations
    # n_future ==> no of future observations
    """   
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
    # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)

##DEFINING MAIN

def main(argv):
    if len(argv) != 1:
        print("Please enter a valid path while calling IDS_test.py in the command line")
        return
    
    print(argv[0])
    path=argv[0]
    
 
    
    ## PREPROCESSING DATA
    
    #importing the csv as dataframe
    df1 = pd.read_csv(path)
    df = df1.drop(['S_PU1', 'S_PU3', 'F_PU3', 'F_PU5', 'S_PU5', 'F_PU9', 'F_PU11', 'S_PU11', 'S_PU9', 'INDEX(TIME_IN_HOURS)'], 1)
    
    #importing the RobustScaler object and implementing
    trans = load(open('trans.pkl', 'rb'))
    test = trans.transform(df)
    test = pd.DataFrame(test)
    
    #Reshaping the Data
    X_test, y_test = split_series(test.values,n_past, n_future)
    print(X_test.shape, y_test.shape)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
    
    
    ## IMPLEMENTING MODEL
    
    #loading the trained keras model
    model = keras.models.load_model('keras_prob2.h5')
    
    #Predict the test Data using our trained model
    pred = model.predict(X_test)
    
    
    ## PREPARING OUR CLASSIFICATION ARRAY 
    
    #Transforming/Inverse our prediction and converting into Dataframe
    pred_test2_np_tmp = np.vstack(pred)
    pred_test2_inv = trans.inverse_transform(pred_test2_np_tmp)
    pred_test2_df = pd.DataFrame(pred_test2_inv, columns=df.columns)
    test2_trim = df.iloc[n_past:].reset_index(drop=True)
    
    #Classification Array
    
    pred_np2 = pred_test2_df.to_numpy()
    actual_np2 = test2_trim.to_numpy()
    dist_array2=[]
    classification_array2=[]
    moving_avg2=[]

    threshold2 = 6

    for row in range(len(pred_np2)):
        dist = np.abs(np.subtract(pred_np2[row], actual_np2[row]))


        dist_array2.append(dist)

        if len(dist_array2)-avg_sample_rate >= 0:
            #print(len(dist_array2),dist_array2)
            avg_dist = sum(dist_array2[len(dist_array2)-avg_sample_rate:len(dist_array2)])/avg_sample_rate
            moving_avg2.append(avg_dist)
        else:
            moving_avg2.append(dist)


        diff = np.sum(np.array(per_feature_thresh - moving_avg2[len(moving_avg2)-1]) <= 0, axis = 0)
        if diff > threshold2:
            classification_array2.append(-1)
        else:
            classification_array2.append(1)
            
    
    ## OUTPUT AS CSV
    
    #Prepare df
    
    
    class_array= ['ATTACK'  if i==-1 else 'NORMAL' for i in classification_array2]
    ATTACK_NORMAL= ['NORMAL' if class_array[0]==1 else 'ATTACK' for i in range(n_past)] #Because we can't predict our first 12 data points make them equal to the 13th point.
    ATTACK_NORMAL = ATTACK_NORMAL + class_array
    final_df = pd.DataFrame(columns=['TIME','LABEL']) 
    final_df['TIME']= df1['INDEX(TIME_IN_HOURS)']
    final_df['LABEL'] =ATTACK_NORMAL
    
    #Use Neighbor information to fix minor anomalies
    window_size =20

    for i in range(0, len(final_df),window_size) :
        block=[]
        if i+window_size>len(final_df):
            break

        for j in range(i , i+window_size):
            if len(block)==0 or block[-1][0] != final_df['LABEL'].iloc[j]:
                block.append([final_df['LABEL'].iloc[j],j])

        if len(block)<=2 :
            continue
    
        if len(block)>=3 and block[2][1]-block[1][1]<12:
            for k in range(block[1][1],block[2][1]):
                final_df["LABEL"].iloc[k] = block[0][0]
                block[1][0] = block[0][0]
        
        if len(block) >= 4 and block[3][1]-block[2][1]<12:
            for k in range(block[2][1],block[3][1]):
                final_df["LABEL"].iloc[k] = block[1][0]
                block[2][0] = block[1][0]
        
        if len(block) >= 5 and block[4][1]-block[3][1]<12:
            for k in range(block[3][1],block[4][1]):
                final_df["LABEL"].iloc[k] = block[2][0]
                block[3][0] = block[2][0]
       

            
    #Export CSV
    
    final_df.to_csv('result.csv',index=False)
   
    


if __name__ == "__main__":
    main(sys.argv[1:])
    
    