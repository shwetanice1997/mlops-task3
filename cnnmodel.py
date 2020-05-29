from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

from keras.utils import to_categorical
#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
model = Sequential()
model.add(Conv2D(2, kernel_size=3, activation="relu", input_shape=(28,28,1)))
i=1
n=4
for i in range(i):
    model.add(Conv2D(filters=n,kernel_size=3,activation="relu"))
    n=n*2
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train,epochs=1)

pred1=model.evaluate(X_test,y_test)

print("Accuracy is : ",pred1[1]*100)

try:
    f=open("/newdir/o.txt","w")
    f.write(str(int(pred1[1]*100)))
except:
    print(end="")
finally:
    f.close(