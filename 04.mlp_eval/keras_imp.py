import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(0) # 난수 시드

# 0. 학습 데이터 생성
N = 300

X, y = datasets.make_moons(N, noise = 0.3) # noise = 깔끔하지 않은 부분
Y = y.reshape(N, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)

# 1. 모델 설정
model = Sequential()
# 입력 - 은닉층
model.add(Dense(3, input_dim = 2))
model.add(Activation('sigmoid'))
# 은닉 - 출력층
model.add(Dense(1))
model.add(Activation('sigmoid'))

# 2. 오차 함수 정의 및 최적화 기법 적용
model.compile(loss='binary_crossentropy', optimizer=SGD(lr = 0.1), metrics=['accuracy'])

# 3. 학습
model.fit(X, Y, epochs = 500, batch_size = 20)

# 4. 결과
loss_and_metrics = model.evaluate(X_test, Y_test)

print('loss_and_metrics:')
print(loss_and_metrics)