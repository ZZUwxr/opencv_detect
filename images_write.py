import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置字体为微软雅黑
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 'Microsoft YaHei' 是微软雅黑的英文名称
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号



# 数据
N_values = [100, 100, 100, 1000, 1000, 1000, 10000, 10000, 10000, 10000, 100000, 100000, 100000, 100000]
processes = [1, 2, 4, 1, 2, 4, 1, 2, 4, 8, 1, 2, 4, 8]
times = [0.000014, 0.000564, 0.001723, 0.000017, 0.000800, 0.001539, 0.000038, 0.001091, 0.001606, 0.002557, 0.000350, 0.001133, 0.001207, 0.003177]
pi_values = [3.131593, 3.131593, 3.131593, 3.140593, 3.140593, 3.140593, 3.141493, 3.141493, 3.141493, 3.141493, 3.141583, 3.141583, 3.141583, 3.141583]

# 创建新的图表
plt.figure(figsize=(12, 6))

# 执行时间图表
plt.subplot(1, 2, 1)
for p in set(processes):
    plt.plot([N_values[i] for i in range(len(processes)) if processes[i] == p],
             [times[i] for i in range(len(processes)) if processes[i] == p],
             marker='o', label=f'{p} 进程')
plt.xlabel('N')
plt.ylabel('执行时间 (秒)')
plt.title('不同N值和进程数的执行时间')
plt.legend()

# π 值图表
plt.subplot(1, 2, 2)
for p in set(processes):
    plt.plot([N_values[i] for i in range(len(processes)) if processes[i] == p],
             [pi_values[i] for i in range(len(processes)) if processes[i] == p],
             marker='o', label=f'{p} 进程')
plt.xlabel('N')
plt.ylabel('π 值')
plt.title('不同N值和进程数的π 值')
plt.legend()

plt.tight_layout()
plt.show()
