import numpy as np                           # 행렬 계산을 위한 numpy 모듈
import matplotlib.pyplot as plt              # 각종 그래프, 이미지 출력용 모듈
import pickle                                # weight 저장/불러오기를 위한 모듈
from tensorflow.keras.datasets import mnist  # MNIST 데이터셋을 불러오기 위한 모듈


# Sigmoid 활성화 함수
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Sigmoid 활성화 함수의 도함수 (주의: 활성화 함수의 출력값(y)를 인자로 줘야 함)
def sigmoid_prime(y):
    return y * (1 - y)

# ReLU 활성화 함수
def relu(z):
    return np.maximum(z, 0)

# ReLU 활성화 함수의 도함수 (주의: 활성화 함수의 입력값(z)를 인자로 줘야 함)
def relu_prime(z):
    return (z >= 0) * 1.0

# 분류를 위한 Softmax 활성화 함수 (출력함수)
def softmax(z):
    # 지수가 너무 커지는 것을 방지하기 위해 최대값으로 분모, 분자를 나눔
    e_z = np.exp(z - z.max(axis=1, keepdims=True))
    return e_z / e_z.sum(axis=1, keepdims=True)

# Softmax 함수의 Cross Entropy Loss Function
def cross_entropy(y_pred, y_true):
    return (-y_true * np.log(y_pred)).sum() / y_true.shape[0]


# 하이퍼 파라메터 정의
nx = 784    # input layer node 수 : 변경하지 말 것
nh1 = 64    # hidden layer node 수
learning_rate = 0.00001
r1 = 0.9
r2 = 0.999
e = 1E-5
epoch = 30
nbatch = 32
sample_plot = True     # 데이터 확인용
training_mode = True


# 인공신경망 클래스 정의
class SimpleAutoEncoder:
    def __init__(self):
        # xavier initialization
        self.weight = {'W1': np.random.randn(nx, nh1) / np.sqrt((nx + nh1) / 2),
                       'W2': np.random.randn(nh1, nx) / np.sqrt((nh1 + nx) / 2)}

        self.layer = {}
        self.gradient = {}

        self.momentum = {'W1': np.zeros((nx, nh1)),
                         'W2': np.zeros((nh1, nx))}
        self.rms = {'W1': np.zeros((nx, nh1)),
                    'W2': np.zeros((nh1, nx))}
        self.iter = 0

    def save_weights(self):
        with open('weight_ae_backup.pkl', 'wb') as fo:
            pickle.dump(self.weight, fo)

    def restore_weights(self):
        with open('weight_ae_backup.pkl', 'rb') as fi:
            self.weight = pickle.load(fi)

    def feedforward(self, x):       # x: input 행렬, shape: (n, 784), n: 데이터 개수
        W1 = self.weight['W1']
        W2 = self.weight['W2']

        z1 = np.dot(x, W1)
        h1 = sigmoid(z1)
        z2 = np.dot(h1, W2)
        y = sigmoid(z2)

        # 학습 때 backpropagation에 사용하기 위해 layer에 저장
        self.layer = {'z1': z1, 'h1': h1, 'z2': z2}
        return y

    # 학습 함수
    def train_step(self, x):
        m = x.shape[0]

        # feedforward 1회 실시
        y_dec = self.feedforward(x)

        # loss 계산  (확인용. 실제 학습에 이 값이 쓰이지는 않음)
        loss = np.linalg.norm(x - y_dec) / m

        # 가독성을 위해 변수명을 짧게 (copy가 아닌 reference. 메모리 차지 (거의) 없음)
        W2 = self.weight['W2']
        h1 = self.layer['h1']

        # backpropagation 통해 gradient 구하기
        dLdz2 = (y_dec - x) / m
        gradW2 = np.dot(h1.T, dLdz2)

        dLdh1 = np.dot(dLdz2, W2.T)
        dLdz1 = dLdh1 * sigmoid_prime(h1)
        gradW1 = np.dot(x.T, dLdz1)

        self.gradient = {'W1': gradW1, 'W2': gradW2}

        # weight update
        self.adam_opt()     # ADAM optimization
        # self.sgd_opt()    # stochastic gradient descent optimization

        return loss

    def sgd_opt(self):
        for W in self.weight:
            self.weight[W] = self.weight[W] - learning_rate * self.gradient[W]

    def adam_opt(self):
        for W in self.weight:
            self.momentum[W] = self.momentum[W] * r1 + self.gradient[W]
            self.rms[W] = self.rms[W] * r2 + (self.gradient[W] ** 2) * (1 - r2)

            self.iter = self.iter + 1
            V_hat = self.momentum[W] / (1 - r1 ** self.iter)
            G_hat = self.rms[W] / (1 - r2 ** self.iter)

            self.weight[W] = self.weight[W] - learning_rate * V_hat / np.sqrt(G_hat + e)


# 프로그램 시작점
if __name__ == '__main__':

    # MNIST 데이터 로드
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # 입력 이미지 X_train, X_test reshape 및 normalize
    X_train = X_train.reshape((-1, 784)) / 255.0
    X_test = X_test[:100].reshape((-1, 784)) / 255.0   # 정확도 측정하는 것이 아니므로 test는 100개만.

    # MLP network 생성
    net = SimpleAutoEncoder()

    if training_mode:
        # 학습 곡선 표출용
        loss_list = list()
        acc_list = list()

        # 학습 데이터 개수
        N = X_train.shape[0]
        idx = np.arange(N)

        # 학습 실행
        for k in range(epoch):
            # 각 epoch 시작시 mini-batch 순서를 무작위로 섞어서 사용
            np.random.shuffle(idx)

            # 각 epoch 마다 평균 loss 출력용
            loss_total = 0.0

            # 학습 iteration (mini-batch 사용)
            for i in range(0, N, nbatch):
                # 랜덤하게 mini-batch를 가져오기 위한 인덱스 배열
                bi = idx[i: i+nbatch]
                bx = X_train[bi]

                loss = net.train_step(bx)

                loss_total = loss_total + loss * bx.shape[0]
                if (i / nbatch) % 5 == 0:
                    loss_list.append(loss)

            print('epoch:', k, ', average_loss:', loss_total / N)

        # 그래프 그리기
        plt.plot(loss_list, lw=0.5)
        plt.show()

        # 학습된 weight 저장
        net.save_weights()

    else:
        net.restore_weights()

    # 이제 학습된 모델로 추론을 해보자
    encoded_img = list()    # 코드화 된 이미지 (hidden layer) 저장용
    decoded_img = list()    # 디코드 된 이미지 (output layer) 저장용

    for x in X_test:
        x = x.reshape((-1, 784))
        output = net.feedforward(x)
        encoded_img.append(net.layer['h1'])
        decoded_img.append(output)

    print(len(decoded_img), 'images are decoded.')

    if sample_plot:
        img_org = X_test[0].reshape((28, 28))
        img_enc = encoded_img[0].reshape((8, 8))    # hidden layer node 수가 64이므로
        img_dec = decoded_img[0].reshape((28, 28))
        plt.imshow(img_org, cmap=plt.cm.binary)     # 원본 이미지 출력
        plt.show()
        plt.imshow(img_enc, cmap=plt.cm.binary)     # 코드화된 이미지 출력: 알아볼 수 있는가?
        plt.show()
        plt.imshow(img_dec, cmap=plt.cm.binary)     # 디코드된 이미지 출력: 원본이랑 비슷한가?
        plt.show()

    # noise 제거 테스트
    img = X_test[25]
    plt.imshow(img.reshape((28, 28)), cmap=plt.cm.binary)
    plt.show()

    noisy_img = img + 0.15 * np.random.randn(1, 784)    # 0.15만큼 가우시안 노이즈 추가
    plt.imshow(noisy_img.reshape((28, 28)), cmap=plt.cm.binary)
    plt.show()

    img_dec = net.feedforward(noisy_img)
    plt.imshow(img_dec.reshape((28, 28)), cmap=plt.cm.binary)
    plt.show()
