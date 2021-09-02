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
nh1 = 200   # 첫번째 hidden layer node 수
nh2 = 32    # 두번째 hidden layer node 수
ny = 10     # output layer node 수 : 변경하지 말 것
learning_rate = 0.001   # ADAM은 SGD에 비해 10^-2 배 정도 작은 learning rate 셋팅
r1 = 0.9
r2 = 0.999
e = 1E-5
epoch = 10
nbatch = 20
sample_plot = True     # 데이터 확인용
training_mode = True


# 인공신경망 클래스 정의
class SimpleMLP:
    def __init__(self):
        # xavier initialization
        self.weight = {'W1': np.random.randn(nx, nh1) / np.sqrt((nx + nh1) / 2),
                       'W2': np.random.randn(nh1, nh2) / np.sqrt((nh1 + nh2) / 2),
                       'W3': np.random.randn(nh2, ny) / np.sqrt((nh2 + ny) / 2)}

        self.layer = {}
        self.gradient = {}

        self.momentum = {'W1': np.zeros((nx, nh1)),
                         'W2': np.zeros((nh1, nh2)),
                         'W3': np.zeros((nh2, ny))}
        self.rms = {'W1': np.zeros((nx, nh1)),
                    'W2': np.zeros((nh1, nh2)),
                    'W3': np.zeros((nh2, ny))}
        self.iter = 0

    def save_weights(self):
        with open('weight_backup.pkl', 'wb') as fo:
            pickle.dump(self.weight, fo)

    def restore_weights(self):
        with open('weight_backup.pkl', 'rb') as fi:
            self.weight = pickle.load(fi)

    def feedforward(self, x):       # x: input 행렬, shape: (n, 784), n: 데이터 개수
        W1 = self.weight['W1']
        W2 = self.weight['W2']
        W3 = self.weight['W3']

        z1 = np.dot(x, W1)          # z1: 첫번째 hidden layer의 입력 행렬
        h1 = relu(z1)               # h1: 첫번째 hidden layer의 출력 행렬 (활성화)
        z2 = np.dot(h1, W2)         # z2: 두번째 hidden layer의 입력 행렬
        h2 = relu(z2)               # h2: 두번째 hidden layer의 출력 행렬 (활성화)
        z3 = np.dot(h2, W3)         # z3: output layer의 입력 행렬(logits), shape: (n, 10)
        # feedforward에서는 softmax 출력함수 생략 (computation 낭비)
        # softmax 출력은 학습 과정에서 cross entropy loss를 구할 때에 사용

        # 학습 때 backpropagation에 사용하기 위해 layer에 저장
        self.layer = {'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'z3': z3}
        return z3

    # 학습 함수
    def train_step(self, x, y_true):
        m = x.shape[0]

        # feedforward 1회 실시
        z3 = self.feedforward(x)

        # softmax 출력
        y_pred = softmax(z3)

        # loss 계산  (확인용. 실제 학습에 이 값이 쓰이지는 않음)
        loss = cross_entropy(y_pred, y_true)
        acc = np.count_nonzero(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)) / m

        # 가독성을 위해 변수명을 짧게 (copy가 아닌 reference. 메모리 차지 (거의) 없음)
        W2, W3 = self.weight['W2'], self.weight['W3']
        z1, h1, z2, h2 = [self.layer[key] for key in ['z1', 'h1', 'z2', 'h2']]

        # backpropagation 통해 gradient 구하기
        dLdz3 = (y_pred - y_true) / m
        gradW3 = np.dot(h2.T, dLdz3)

        dLdh2 = np.dot(dLdz3, W3.T)
        dLdz2 = dLdh2 * relu_prime(z2)
        gradW2 = np.dot(h1.T, dLdz2)

        dLdh1 = np.dot(dLdz2, W2.T)
        dLdz1 = dLdh1 * relu_prime(z1)
        gradW1 = np.dot(x.T, dLdz1)

        self.gradient = {'W1': gradW1, 'W2': gradW2, 'W3': gradW3}

        # weight update
        self.adam_opt()     # ADAM optimization
        # self.sgd_opt()    # stochastic gradient descent optimization

        return loss, acc

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

    # 위 코드 에러발생 시, 주석처리하고 아래 방법 이용
    #  1) 주소창에 https://s3.amazonaws.com/img-datasets/mnist.npz 입력하여 .npz 파일 직접 다운로드
    #  2) 다운받은 npz 파일을 본 프로젝트 폴더에 복사
    #  3) 아래 코드 두 줄 활성화 하여 사용
    # path = 'mnist.npz'
    # (X_train, Y_train), (X_test, Y_test) = mnist.load_data(path)

    if sample_plot:
        # 데이터 개수 및 행렬 형태 확인
        print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
        # -> 출력 결과: (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

        # 데이터 확인용으로 샘플의 값과 이미지 출력
        img = X_train[0]  # shape: (28, 28)
        gt = Y_train[0]
        print(img)
        print('정답:', gt)
        plt.imshow(img)
        plt.show()

    # 입력 이미지 X_train, X_test reshape 및 normalize
    X_train = X_train.reshape((-1, 784)) / 255.0
    X_test = X_test.reshape((-1, 784)) / 255.0

    # 학습용 Y_train one-hot encoding
    n = Y_train.size
    Y_train_onehot = np.zeros((n, 10))
    Y_train_onehot[np.arange(n), Y_train] = 1

    if sample_plot:
        # reshape, normalize 확인
        print(X_train[0])

        # one-hot encoding 확인
        print(Y_train_onehot[0])

    # MLP network 생성
    net = SimpleMLP()

    # 학습 없이 추론을 해보자
    success = 0
    failure = 0

    for x, gt in zip(X_test, Y_test):
        x = x.reshape((-1, 784))
        logits = net.feedforward(x)
        if np.argmax(logits) == gt:
            success += 1
        else:
            failure += 1

    print('Classification accuracy before training: ', success / (success + failure) * 100, '%')
    # 10% 근처가 나오면 정상 (10개 중 1개 찍는 확률)

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

            # 각 epoch 마다 평균 loss, accuracy 출력용
            loss_total = 0.0
            acc_total = 0.0

            # 학습 iteration (mini-batch 사용)
            for i in range(0, N, nbatch):
                # 랜덤하게 mini-batch를 가져오기 위한 인덱스 배열
                bi = idx[i: i+nbatch]

                bx = X_train[bi]
                by = Y_train_onehot[bi]

                loss, acc = net.train_step(bx, by)

                loss_total = loss_total + loss * bx.shape[0]
                acc_total = acc_total + acc * bx.shape[0]

                if (i / nbatch) % 5 == 0:
                    loss_list.append(loss)
                    acc_list.append(acc)

            print('epoch:', k, ', average_loss:',
                  loss_total / N, 'accuracy:', acc_total / N)

        # 그래프 그리기
        plt.plot(loss_list, lw=0.5)
        plt.show()
        plt.plot(acc_list, lw=0.5)
        plt.show()

        # 학습된 weight 저장
        net.save_weights()

    else:
        net.restore_weights()

    # 이제 학습된 모델로 추론을 해보자
    success = 0
    failure = 0
    failed_data = list()     # 틀린 이미지 저장용

    for x, gt in zip(X_test, Y_test):
        x = x.reshape((-1, 784))
        logits = net.feedforward(x)
        if np.argmax(logits) == gt:
            success += 1
        else:
            failure += 1
            failed_data.append((x.reshape((28, 28)), gt, np.argmax(logits)))

    print('Classification accuracy after training: ', success / (success + failure) * 100, '%')
    # 정확도는 얼마나 나오나요?
    # 앞선 예제(sigmoid, SGD)와 비교해 봅시다.
    # 비교를 위해 다른 하이퍼 파라메터는 일치시키되 learning rate는 ADAM에서 훨씩 작은 값을 사용하세요.
    # 예: SGD: 10^-1 --> ADAM: 10^-3

    print(len(failed_data), 'images are misclassified.')
    if sample_plot:
        img, gt, pred = failed_data[0]
        print('label:', gt)
        print('classification:', pred)
        plt.imshow(img)
        plt.show()
