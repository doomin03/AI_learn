import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# MINST 데이터셋 임포트
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리
x_train, x_test = x_train / 255.0, x_test / 255.0

# 모델 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train)

# 모델 정확도 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('테스트 정확도', test_acc)

model.save("MNIST_test.h5")

print(x_test[0])
