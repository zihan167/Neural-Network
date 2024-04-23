import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 目標函數
def func(x):
    x1, x2 = x
    return 0.26*(x1**2 + x2**2) + 2*x1

# 梯度
def grad(x):
    x1, x2 = x
    return np.array([0.52*x1 + 2, 0.52*x2])

# 海森矩陣
def hessian(x):
    return np.array([[0.52, 0], [0, 0.52]])

# 牛頓法
def newton_method(x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        g = grad(x)
        H = hessian(x)
        d = np.linalg.solve(H, -g)
        x = x + d
        if np.linalg.norm(g) < tol:
            break
    return x

# 初始點
x0 = np.array([1, 1])

# 執行牛頓法
x_opt = newton_method(x0)
print(f"最小值點為: x1 = {x_opt[0]:.4f}, x2 = {x_opt[1]:.4f}")
print(f"最小值為: f(x) = {func(x_opt):.4f}")

# 製圖
x1 = np.arange(-10, 10, 0.1)
x2 = np.arange(-10, 10, 0.1)
X1, X2 = np.meshgrid(x1, x2)
Z = func(np.array([X1, X2]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='pink')

# 標註最小值點
ax.scatter(x_opt[0], x_opt[1], func(x_opt), c='r', marker='o', s=50)

# 設置顏色條
cbar = fig.colorbar(ax.plot_surface(X1, X2, Z, cmap='pink'), ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Function Value')

ax.set_title('Newton: f(x1, x2) = 0.26(x1^2 + x2^2) + 2x1')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
plt.show()
