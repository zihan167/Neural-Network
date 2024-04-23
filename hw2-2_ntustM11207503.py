import numpy as np

# 輸入數據
years = np.array([1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995])
hourly_wages = np.array([11.18, 11.46, 11.74, 12.07, 12.37, 12.78, 13.17, 13.49, 13.91])

# 計算最小平方誤差法的係數 a 和 b
n = len(years)
sum_x = np.sum(years)
sum_y = np.sum(hourly_wages)
sum_x_squared = np.sum(years ** 2)
sum_xy = np.sum(years * hourly_wages)

a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
b = (sum_y - a * sum_x) / n

print("回歸線方程式： y =", a, "t +", b)

# 使用回歸線預測1998年的平均時薪
year_1998 = 1998
predicted_wage_1998 = a * year_1998 + b
print("1998年的預測平均時薪：", predicted_wage_1998)
