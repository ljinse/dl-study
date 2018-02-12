import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(0)
tf.set_random_seed(0)

# 0. 학습 데이터 생성
N = 300

X, y = datasets.make_moons(N, noise = 0.3) # noise = 깔끔하지 않은 부분
Y = y.reshape(N, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)

# 1. 모델 설정
num_hidden = 3

x = tf.placeholder(tf.float32, shape = [None, 2])
t = tf.placeholder(tf.float32, shape = [None, 1])

W = tf.Variable(tf.truncated_normal([2, num_hidden]))
b = tf.Variable(tf.zeros([num_hidden]))
h = tf.nn.sigmoid(tf.matmul(x, W) + b)

V = tf.Variable(tf.truncated_normal([num_hidden, 1])) # 오차역전파법을 사용할 때 파라미터가 0이면 오차 전달 안됨
c = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(h, V) + c)

# 2. 오차 함수 정의
cross_entropy = -tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))

# 3. 최적화 기법 정의
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 4. 세션 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 5. 학습
batch_size = 20
n_batches = N // batch_size

for epoch in range(500) :
    X_, Y_ = shuffle(X_train, Y_train)

    for i in range(n_batches) :
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict = {x: X_[start:end], t: Y_[start:end]})

# 6. 평가
accuracy_rate = accuracy.eval(session = sess, feed_dict = {x: X_test, t: Y_test})

print('accuracy:')
print(accuracy_rate)
