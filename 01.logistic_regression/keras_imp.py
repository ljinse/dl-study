import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

np.random.seed(0) # 난수 시드

# 0. 학습 데이터 생성
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

# 1. 모델 설정
model = Sequential([
    Dense(input_dim = 2, units = 1),
    Activation('sigmoid')
])

# 2. 오차 함수 정의 및 최적화 기법 적용
model.compile(loss='binary_crossentropy', optimizer=SGD(lr = 0.1))

# 3. 학습
model.fit(X, Y, epochs = 200, batch_size = 1)

# 4. 결과

classes = model.predict_classes(X, batch_size = 1)
prob = model.predict_proba(X, batch_size = 1)
print('classified:')
print(Y == classes)
print('prop:')
print(prob)