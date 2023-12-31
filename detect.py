import json
import numpy as np
import cv2
import os
from tqdm import tqdm


# 处理原始视频获取视频信息
def video_detect(video_file):
    # 定义视频文件路径和区域坐标
    x1, y1, x2, y2 = 700, 0, 800, 300

    # 创建VideoCapture对象
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建一个字典保存每一帧的检测结果
    detection_results = {}

    # 使用tqdm来循环遍历视频帧，并显示进度条
    for frame_count in tqdm(range(total_frames), desc="Detecting", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        # 获取视频帧的指定区域
        roi = frame[y1:y2, x1:x2]

        # 将帧转换为HSV颜色空间
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 定义红色范围
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        # 将两个掩码相加
        mask = mask1 + mask2

        # 检查是否检测到红色
        if np.sum(mask) > 0:
            detection_results[frame_count] = 1
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            detection_results[frame_count] = 0

        # 显示结果
        cv2.imshow('Frame', frame)

        # 按q键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # 计算进度并在控制台上显示
    progress = (frame_count + 1) / total_frames * 100
    print(f"Detecting... {progress:.2f}% completed", end='\r')

    frame_count += 1
    # 释放视频对象
    cap.release()
    cv2.destroyAllWindows()

    # 将检测结果保存到 JSON 文件中
    with open('save/config/detection_results.json', 'w') as file:
        json.dump(detection_results, file)


# 用来统计方差大于20的组号
def nums_sum(fps, data, num):
    # 统计每25个循环内键的值为1的数量
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
    # 计算每num个数据的方差并找出方差大于20的组号
    variances = []
    # 用来存储方差大于20的组号
    groups_with_high_variance = []
    # 组号从1开始
    group_number = 1

    for i in range(0, len(counts_per_cycle), num):
        subset = counts_per_cycle[i:i + num]
        if len(subset) == num:  # 确保每组都有4个元素
            variance = np.var(subset)
            variances.append(variance)

            if variance > 20:
                groups_with_high_variance.append(group_number)

            group_number += 1  # 更新组号

    # 返回方差大于20的组号
    # print(groups_with_high_variance)
    return groups_with_high_variance


# 用于修饰数组数据，输出最终的时间段
def process_array(nums, time_interval):
    if not nums:
        return [], []

    start_times = []
    end_times = []
    start = nums[0]
    end = nums[0]
    count = 1  # 用于跟踪数字段的长度

    for i in range(1, len(nums)):
        diff = nums[i] - nums[i - 1]

        if diff <= 2:
            end = nums[i]
            count += 1
        else:
            if count >= 3:  # 检查该数字段的长度是否大于或等于3
                start_times.append(start)
                end_times.append(end)
            start = nums[i]
            end = nums[i]
            count = 1

    if count >= 3:  # 检查最后一个数字段的长度
        start_times.append(start)
        end_times.append(end)

    # 检查相邻的start_times元素之间的差值
    i = 1
    while i < len(start_times):
        if start_times[i] - start_times[i - 1] <= time_interval:
            del start_times[i]
            del end_times[i]
        else:
            i += 1

    return start_times, end_times


def result_output(video_path, start_frames, end_frames, num):
    start_times = [st * num for st in start_frames]
    end_times = [et * num for et in end_frames]
    i = 0
    while i < len(start_times):
        s_minutes = start_times[i] // 60
        s_seconds = start_times[i] % 60
        e_minutes = end_times[i] // 60
        e_seconds = end_times[i] % 60
        print("Warning！")
        print(f"警告时间段: {s_minutes}min{s_seconds}s - {e_minutes}min{e_seconds}s")
        print("------------------------------------")
        i += 1
    print(f"{video_path}视频处理完成！")
    print("------------------------------------")


def main():
    with open('config.json', 'r') as file:
        config_data = json.load(file)

    # 读取config文件
    video_path = config_data["video_path"]
    output_path = config_data["output_path"]
    array_num = config_data["array_num"]
    fps = config_data["video_fps"]
    time_interval = config_data["time_interval"] * 60 / array_num

    # 处理视频，存储相关信息
    video_detect(video_path)

    # 读取JSON文件
    with open('save/config/detection_results.json', 'r') as file:
        detect_data = json.load(file)

    # 返回待处理数组
    array = nums_sum(fps, detect_data, array_num)
    # 返回特定时间段，包含开始时间和结束时间的两个数组
    start_times, end_times = process_array(array, time_interval)
    # 控制台输出结果
    result_output(video_path, start_times, end_times, array_num)

