import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.model_selection import train_test_split

np.random.seed(0)

# 0. 학습 데이터 생성
mnist = datasets.fetch_mldata('MNIST original', data_home='.')
n = len(mnist.data)
N = 10000
indices = np.random.permutation(range(n))[:N]

X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)] # 1-of-k

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)

# 1. 모델 설정
n_in = len(X[0]) # 784
n_hidden = 200
n_out = len(Y[0])

# 1. 모델 설정
model = Sequential()

model.add(Dense(n_hidden, input_dim = n_in))
model.add(Activation('sigmoid'))

model.add(Dense(n_hidden))
model.add(Activation('sigmoid'))

model.add(Dense(n_hidden))
model.add(Activation('sigmoid'))

model.add(Dense(n_out))
model.add(Activation('softmax'))

# 2. 오차 함수 정의 및 최적화 기법 적용
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr = 0.01), metrics=['accuracy'])

# 3. 학습
epochs = 100
batch_size = 100

model.fit(X, Y, epochs = epochs, batch_size = batch_size)

# 4. 결과
loss_and_metrics = model.evaluate(X_test, Y_test)

print('loss_and_metrics:')
print(loss_and_metrics)