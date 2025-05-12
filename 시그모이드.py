import numpy as np

# 시그모이드 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 입력 벡터 (bias 포함)
x = np.array([1, 1, 0])  # x0 = 1 (bias), x1 = 1, x2 = 0

# 가중치 행렬 정의
U1 = np.array([[-0.3, 1.0, 1.2],
               [1.6, -1.0, -1.1]])

U2 = np.array([[1.0, 1.0, -1.0],
               [0.7, 0.5, 1.0]])

U3 = np.array([[0.5, -0.8, 0.9],
               [-0.1, 0.3, 0.4]])

U4 = np.array([[1.0, 0.1, -0.2],
               [-0.2, 1.3, 0.4]])

# 순전파 계산
def forward(x, weights):
    for W in weights:
        z = W @ x
        a = sigmoid(z)
        x = np.insert(a, 0, 1)  # bias 추가
    return a

# 가중치 목록
weights = [U1, U2, U3, U4]

# 출력 계산
output = forward(x, weights)
print("출력 y:", output)