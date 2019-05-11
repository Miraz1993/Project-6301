import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import traceback
from PIL import Image
from attacks import fast_gradient
from keras.models import Sequential
from keras.layers import Dense
import csv
from tensorflow import keras
from keras import layers
from keras.layers import InputLayer
'''img_size = 28
img_chan = 1
n_classes = 10'''

def get_weights(model):
    weights=[]
    for i in range(1,300):
        weights.append(model.get_variable_value('linear/linear_model/f'+str(i)+'/weights')[0][0])
    weights.append(model.get_variable_value('linear/linear_model/bias_weights')[0])
    return weights

def train_model(x,y,feat_cols):
    model = tf.estimator.LinearRegressor(feature_columns=feat_cols)
    #model = tf.estimator.DNNRegressor(hidden_units=[6,10], feature_columns=feat_cols)
    #layers.Dense(10, activation=tf.nn.relu),
    input_func = tf.estimator.inputs.pandas_input_fn(x, y, batch_size=10, num_epochs=1000, shuffle=True)
    model.train(input_fn=input_func, steps=1000)
    #model.predict(x)
    #print([(v, model.get_variable_value(v)) for v in model.get_variable_names()])
    weights=get_weights(model)
    #for v in model.get_variable_names():
        #print(v," ", model.get_variable_value(v))
    #print(model.get_variable_value('linear/linear_model/neighbourhood_cleansed/weights').flatten())
    train_metrics = model.evaluate(input_fn=input_func, steps=1000)


    #optimizer = tf.keras.optimizers.RMSprop(0.001)

    '''model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])'''
    #model.fit(x, y, epochs=1000, verbose=0)
    return model,weights

def train_keras_model(x,y,feat_cols):
    model = Sequential()
    model.add(Dense(13, input_dim=len(feat_cols), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(x, y, epochs=1500, batch_size=50, verbose=1, validation_split=0.2)

    return model

def evaluate( y_pred, y_eval):
    cnt=0
    if isinstance(y_eval, pd.DataFrame):
        y_eval=y_eval[0].values.tolist()
    else:
        y_eval=y_eval.tolist()
    for i in range(len(y_pred)):
        min=y_eval[i]-30
        max = y_eval[i] + 30
        if(y_pred[i]>=min and y_pred[i]<=max):
            cnt+=1
    acc=float(cnt)/float(len(y_pred))
    print("acc ",acc)
    return acc

def predict(x,y,model):
    # Now to predict values we do the following
    pred_input_func = tf.estimator.inputs.pandas_input_fn(x=x, y=y, batch_size=10, num_epochs=1,
                                                          shuffle=False)

    preds = model.predict(input_fn=pred_input_func)
    #ynew = preds.values
    print(type(preds))
    print(preds)
    predictions = list(preds)
    final_pred = []
    for pred in predictions:
        # print(pred)
        final_pred.append(pred["predictions"][0])
    print(type(final_pred))
    #ynew = final_pred.values
    return final_pred

def make_fgsm(sess, model,X_data,df2,weights,Y_data,Y_pred,predict, eps=0.02,epochs=1):
    #model, X_eval, y_eval, y1, predict

    X_adv=fast_gradient.fgm_reg(sess,model, X_data,df2,weights, Y_data,Y_pred, predict, eps, epochs, sign=True, clip_min=0., clip_max=1.)


    return X_adv

def getImage(A,h,w):
    for j in range(len(A)):
        A[j] = A[j] * 255
    im = Image.new("RGB", (h, w))
    pix = im.load()
    ind = 0
    for i in range(h):
        for j in range(w):
            pix[i, j] = (int(A[ind]), int(A[ind+1]), int(A[ind+2]))
            ind+=3

    return im

def rearrange(x_adv):
    x_adv= x_adv.values
    x_adv_new=[]
    for i in range(len(x_adv)):
        A=x_adv[i]
        for j in range(len(A)):
            A[j]=A[j]*255
        B = np.reshape(A, (100, 100, 3))
        print(B)
        x_adv_new.append(B)
    return x_adv_new

df=pd.read_csv("data_input11.csv",low_memory=False)

df2=pd.read_csv("data_input12.csv",low_memory=False)
y_val= df["label"]
x_data=df.drop("label",axis=1)
print(x_data)
print('--------------------')
print(df2.shape)
print(df.shape)
X_train, X_eval,y_train,y_eval=train_test_split(x_data,y_val,test_size=0.3,random_state=101)
y_eval.to_csv('y_eval.csv')
np.savetxt("index.csv", X_eval.index.values, delimiter=",")
scaler_model = MinMaxScaler()
scaler_model.fit(X_train)

X_train=pd.DataFrame(scaler_model.transform(X_train),columns=X_train.columns,index=X_train.index)
index_array=X_eval.index.values
df2_eval=df2.iloc[index_array]

df2_eval_new = df2_eval.values
print("in image data")
'''for i in range(len(df2_eval_new)):
    img = getImage(df2_eval_new[i],100,100)
    img.save('ImageData\\img' + str(i) + '.png')
    if (i == 0):
        img.show()'''
print("out image data")
scaler_model.fit(X_eval)

X_eval=pd.DataFrame(scaler_model.transform(X_eval),columns=X_eval.columns,index=X_eval.index)


scaler_model.fit(df2_eval)

df2_eval=pd.DataFrame(scaler_model.transform(df2_eval),columns=df2_eval.columns,index=df2_eval.index)

# Creating Feature Columns
feat_cols = []
for cols in df.columns[:-1]:
    column = tf.feature_column.numeric_column(cols)
    feat_cols.append(column)
model,weights=train_model(X_train,y_train,feat_cols)
y1 = predict(X_eval,y_eval,model)
print('\nEvaluating on clean data')

evaluate(y1, y_eval)

print('\nGenerating adversarial data')
sess = tf.InteractiveSession()
try:
    X_adv = make_fgsm(sess, model, X_eval, df2_eval,weights, y_eval, y1, predict, eps=0.02, epochs=1)

    #X_adv = X_adv.eval(session=sess)
    cols = list(df2_eval.columns.values)
    xadv_df = pd.DataFrame(X_adv, columns=cols)


    print('\nEvaluating on adversarial data')
    print(xadv_df)
    '''print("y_eval")
    print(type(y_eval))
    print(y_eval)'''
    y_eval_list = y_eval.tolist()
    y_eval = pd.DataFrame(y_eval_list)
    #y2 = predict(xadv_df, y_eval, model)
    #evaluate(y2, y_eval)
    y_pred = predict(xadv_df, y_eval, model)
    with open('y_pred.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(y_pred)
    print('\nShowing images')

    #x_adv_new = rearrange(xadv_df)
    #x_adv_new = df2_eval
    #df2_eval
    x_adv_new = xadv_df.values
    '''for i in range(len(x_adv_new)):
        img = getImage(x_adv_new[i],100,100)
        img.save('NewAdvData\\img' + str(i) + '.png')
        if (i == 0):
            img.show()'''

except Exception as e:
    traceback.print_exc()
    f = open('log2.txt', 'w')
    f.write(traceback.format_exc())
    f.write('An exceptional thing happed - %s' % e)

    f.close()
