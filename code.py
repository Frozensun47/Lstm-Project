
import numpy as np
import pandas
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


def prepare_data(dataset, features_count):
	X, y =[],[]
	for i in range(len(dataset)):
		end_ix = i + features_count
		if end_ix > len(dataset)-1:
			break
		seq_x, seq_y = dataset[i:end_ix], dataset[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

dataframe = pandas.read_csv('dataset4.csv', usecols=[0], engine='python')
dataset = dataframe.values
step_count = 25
X, y = prepare_data(dataset, step_count)

features_count = 1
X = X.reshape((X.shape[0], X.shape[1], features_count))

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(step_count, features_count)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=10, verbose=1)

X_input = np.asarray(dataset[len(dataset)-25:len(dataset),:]).astype('float32')
T_input=list(X_input)
Output=[]
i=0
while(i<27):
    
    if(len(T_input)>step_count):
        X_input=np.asarray(T_input[1:]).astype('float32')
        print("{} day input {}".format(i,X_input))
        X_input = X_input.reshape((1, step_count, features_count))
        yhat = model.predict(X_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        T_input.append(yhat[0][0])
        T_input=T_input[1:]
        Output.append(yhat[0][0])
        i=i+1
    else:
        X_input = X_input.reshape((1, step_count, features_count))
        yhat = model.predict(X_input, verbose=0)
        print(yhat[0])
        T_input.append(yhat[0][0])
        Output.append(yhat[0][0])
        i=i+1
    

print(Output)

pevious_data=np.arange(1,len(dataset[len(dataset)-500:len(dataset),:])+1)
new_data=np.arange(len(dataset[len(dataset)-500:len(dataset),:])-3,len(dataset[len(dataset)-500:len(dataset),:])+27-3)

plt.plot(pevious_data,dataset[len(dataset)-500:len(dataset),:])
plt.plot(new_data,Output)
plt.show()
