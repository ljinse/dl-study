import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn.utils import shuffle

np.random.seed(0) # 난수 시드

# 0. 학습 데이터 생성
M = 2 # 입력 데이터의 차원
K = 3 # 클래스 수
n = 100 # 각 클래스 데이터 수
N = n * K # 전체 데이터 수

# 0. 학습 데이터 생성
X1 = np.random.rand(n, M) + np.array([0, 10])
X2 = np.random.rand(n, M) + np.array([5, 5])
X3 = np.random.rand(n, M) + np.array([10, 0])

Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

X = np.concatenate((X1, X2, X3), axis = 0)
Y = np.concatenate((Y1, Y2, Y3), axis = 0)

# 1. 모델 설정
model = Sequential()
model.add(Dense(input_dim = M, units = K))
model.add(Activation('softmax'))

# 2. 오차 함수 정의 및 최적화 기법 적용
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr = 0.1))

# 3. 학습
minibatch_size = 50
model.fit(X, Y, epochs = 20, batch_size = minibatch_size)

# 4. 결과
X_, Y_ = shuffle(X, Y)

classified = model.predict_classes(X_[0:10], batch_size = minibatch_size)

prob = model.predict_proba(X_[0:10], batch_size = minibatch_size)

print('classified:')
print(classified)
print('prop:')
print(prob)