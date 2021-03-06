import numpy as np
import tensorflow as tf

# 0. 학습 데이터 생성
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

# 1. 모델 설정
tf.set_random_seed(0) # 난수 시드

w = tf.Variable(tf.zeros([2,1]))
b = tf.Variable(tf.zeros([1]))

x = tf.placeholder(tf.float32, shape = [None, 2])
t = tf.placeholder(tf.float32, shape = [None, 1])

y = tf.nn.sigmoid(tf.matmul(x, w) + b)

# 2. 오차 함수 정의
cross_entropy = -tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))

# 3. 최적화 기법 정의
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

# 4. 세션 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 5. 학습
for epoch in range(200) :
    sess.run(train_step, feed_dict = {x: X, t: Y})

classified = correct_prediction.eval(session=sess, feed_dict = {x: X, t: Y})
prob = y.eval(session=sess, feed_dict = {x: X})

print('classified:')
print(classified)
print('prob:')
print(prob)