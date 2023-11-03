import numpy as np


# 用来返回方差是否大于threshold
def is_variance_above_threshold(data, threshold, fps, num=3):
    # 统计每一秒内特定区域出现红色的数量
    cycle_count = 0
    counts_per_cycle = []
    count = 0
    for key, value in data.items():
        if int(key) % fps == 0:
            if cycle_count > 0:
                counts_per_cycle.append(count)
                count = 0
            cycle_count += 1
        if value == 1:
            count += 1

    # 添加最后一个循环的统计结果
    counts_per_cycle.append(count)
    # 将3组方差存放在数组中
    variances = []

    group_number = 1
    for i in range(0, len(counts_per_cycle), num):
        subset = counts_per_cycle[i:i + num]
        if len(subset) == num:  # 确保每组都有num个元素
            variance = np.var(subset)
            variances.append(variance)
            group_number += 1  # 更新组号
    # 计算所选数据的方差
    # print(variances)

    if variances[0] >= threshold and variances[1] >= threshold:
        return 1
    elif (variances[0] == 0.0 and variances[1] >= threshold) or \
            (variances[1] == 0.0 and variances[0] >= threshold):
        return 1
    else:
        return 0


# 处理传回的字典数据
def data_processing(detect_data, fps, index):
    # 返回4秒内所有数据的方差处理结果
    flag = is_variance_above_threshold(detect_data, threshold=10, fps=fps)
    warning(flag, index)


# 预警函数
def warning(flag, index):
    if flag == 1:
        print(f"第{index}个视频：Warning!!!")
        print(f"----------------------------")
    else:
        print(f"第{index}个视频：无")
        print(f"----------------------------")