#%% サンプル 5-2-1
import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt(
    'click.csv', delimiter=',', skiprows=1
)
train_x = train[:,0]
train_y = train[:,1]


plt.plot(train_x, train_y,'o')
plt.show()

#%% サンプル 5-2-2

# パラメータ初期化
theta0 = np.random.rand()
theta1 = np.random.rand()

# 予測関数
def f(x):
    return theta0 + theta1 * x

# 目的関数
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

# 標準化
mu = train_x.mean()

print('train_x %s' % train_x)
print('mu: %s' % mu)

sigma = train_x.std()

print('sigma: %s' % sigma)
# 正規分布
# http://gihyo.jp/dev/serial/01/java-calculation/0055

def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)

print('train_z: %s' % train_z)

plt.plot(train_z, train_y, 'o')
plt.show()


#%% サンプル 5-2-5

# 学習率
ETA = 1e-3

# 誤差の差分
diff = 1

# 更新回数
count = 0

# 学習を繰り返す
error = E(train_z, train_y)
while diff > 1e-2:
    tmp0 = theta0 - ETA * np.sum((f(train_z) - train_y))
    tmp1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)

    # Parameter更新
    theta0 = tmp0
    theta1 = tmp1 

    print('theta0: %s' % theta0)
    print('theta1: %s' % theta1)

    # 全開の誤差との差分計算
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error

    # ログの出力
    count += 1
    log = '{}回目: theta0 = {: .3f}, theta1 = {:.3f}, 差分 = {:.4f}'
    print(log.format(count, theta0, theta1, diff))

#%% 5-2-6
x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')
plt.plot(x, f(x))

print('x: %s' % x)
print('f(x): %s' % f(x))
plt.show()


#%% クリック数予測
click = f(standardize(100))

print('click数予測値: %s' % click)