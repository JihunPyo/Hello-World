import numpy as np

# ReLU 함수
def relu(z):
    return np.maximum(0, z)

# MSE 계산 함수
def mse(y, y_hat):
    return 0.5 * np.sum((y - y_hat) ** 2)

# 입력 벡터
x = np.array([1.0, 0.0])
y_target = np.array([0.0, 1.0])

# 가중치 및 bias 정의
W1 = np.array([[1.0, 1.2], [-1.0, -1.1]])
b1 = np.array([-0.3, 1.6])

W2 = np.array([[1.0, -1.0], [0.5, 1.0]])
b2 = np.array([1.0, 0.7])

# 변경 전
W3_before = np.array([[-0.8, 1.0], [0.3, 0.4]])
# 변경 후 (1.0 → 0.9)
W3_after = np.array([[-0.8, 0.9], [0.3, 0.4]])
b3 = np.array([0.5, -0.1])

W4 = np.array([[0.1, -0.2], [1.3, 0.4]])
b4 = np.array([1.0, -0.2])

def forward(x, W1, b1, W2, b2, W3, b3, W4, b4):
    a1 = relu(W1 @ x + b1)
    a2 = relu(W2 @ a1 + b2)
    a3 = relu(W3 @ a2 + b3)
    y = relu(W4 @ a3 + b4)
    return y

# 출력 계산
y_before = forward(x, W1, b1, W2, b2, W3_before, b3, W4, b4)
y_after = forward(x, W1, b1, W2, b2, W3_after, b3, W4, b4)

# 오차 계산
error_before = mse(y_before, y_target)
error_after = mse(y_after, y_target)

# 결과 출력
print("변경 전 출력 y:", y_before)
print("변경 후 출력 y:", y_after)
print("MSE 변경 전:", error_before)
print("MSE 변경 후:", error_after)