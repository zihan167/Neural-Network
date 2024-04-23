import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 目標函數
def func(x):
    x1, x2 = x
    return 0.26*(x1**2 + x2**2) + 2*x1

# 科西法
def cauchy_method(x0, tol=1e-6, max_iter=100, alpha=0.5, beta=0.5):
    x = x0
    for i in range(max_iter):
        # 計算函數值和步長
        f_x = func(x)
        t = 1
        f_xt = func(x + t)
        
        # 找到最小函數值對應的步長
        while f_xt < f_x:
            t *= alpha
            f_x = f_xt
            f_xt = func(x + t)
        
        # 更新搜索方向和步長
        g = t * beta * np.random.randn(2)
        x_new = x + g
        f_xnew = func(x_new)
        
        # 如果函數值減小,則接受新點
        if f_xnew < f_x:
            x = x_new
        else:
            t /= alpha
        
        # 檢查終止條件
        if np.linalg.norm(g) < tol:
            break
    
    return x

# 初始點
x0 = np.array([1, 1])

# 執行科西法
x_opt = cauchy_method(x0)
print(f"最小值點為: x1 = {x_opt[0]:.4f}, x2 = {x_opt[1]:.4f}")
print(f"最小值為: f(x) = {func(x_opt):.4f}")

# 製圖
x1 = np.arange(-10, 10, 0.1)
x2 = np.arange(-10, 10, 0.1)
X1, X2 = np.meshgrid(x1, x2)
Z = func(np.array([X1, X2]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis')  # 彩色曲面
ax.scatter(x_opt[0], x_opt[1], func(x_opt), c='r', marker='o', s=50)  # 紅色最小值點
ax.set_title('Cauchy: f(x1, x2) = 0.26(x1^2 + x2^2) + 2x1')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
plt.show()
