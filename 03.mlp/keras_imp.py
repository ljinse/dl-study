import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

np.random.seed(0) # 난수 시드

# 0. 학습 데이터 생성
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# 1. 모델 설정
model = Sequential()
# 입력 - 은닉층
model.add(Dense(2, input_dim = 2)) # == model.add(Dense(2, input_dim = 2))
model.add(Activation('sigmoid'))
# 은닉 - 출력층
model.add(Dense(1)) # == model.add(Dense(input_dim = 2, units = 1))
model.add(Activation('sigmoid'))

# 2. 오차 함수 정의 및 최적화 기법 적용
model.compile(loss='binary_crossentropy', optimizer=SGD(lr = 0.1))

# 3. 학습
model.fit(X, Y, epochs = 4000, batch_size = 4)

# 4. 결과
classified = model.predict_classes(X, batch_size = 4)

prob = model.predict_proba(X, batch_size = 4)

print('classified:')
print(classified)
print('prop:')
print(prob)