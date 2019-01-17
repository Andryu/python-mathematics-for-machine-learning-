#%% Section 5-2-1
import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt(
    'click.csv', delimiter=',', skiprows=1
)
train_x = train[:,0]
train_y = train[:,1]


plt.plot(train_x, train_y,'o')
plt.show()

#%% Section 5-2-2

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