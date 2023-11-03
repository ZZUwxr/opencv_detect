import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math


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
    all_variance = np.var(counts_per_cycle)
    variances = []

    group_number = 1
    for i in range(0, len(counts_per_cycle), num):
        subset = counts_per_cycle[i:i + num]
        if len(subset) == num:  # 确保每组都有num个元素
            variance = np.var(subset)
            variances.append(variance)
            group_number += 1  # 更新组号
    # 计算所选数据的方差
    print(variances)

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


def get_variances(detection_results, fps, show):
    data = detection_results
    # 统计每一秒内key的value为1的数量
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
    # print(counts_per_cycle)
    # 生成折线图
    if show:
        x = range(1, len(counts_per_cycle) + 1)
        y = counts_per_cycle
        plt.plot(x, y)
        plt.xlabel('second')
        plt.ylabel('Count')
        plt.title('frames per second')
        plt.show()

    np_arr = np.array(counts_per_cycle)


def detect():
    # 视频文件列表
    video_files = ['./videos/input/input_1.mp4', './videos/input/input_2.mp4', './videos/input/input_3.mp4']

    second = 6

    # 当前视频索引
    current_video_index = 0

    # 上一次读取的帧数
    last_frame_index = [0, 0, 0]

    # 特定区域的坐标 (x1, y1, x2, y2)
    x1, y1, x2, y2 = (700, 0, 840, 300)

    while True:
        # 打开当前视频文件
        video = cv2.VideoCapture(video_files[current_video_index])
        # 当前视频帧率
        fps = video.get(cv2.CAP_PROP_FPS)
        # 设置视频的当前帧
        video.set(cv2.CAP_PROP_POS_FRAMES, last_frame_index[current_video_index])
        frame_count = 0
        results = {}
        while video.isOpened():
            ret, frame = video.read()
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
                # 如果检测到红色，将该帧的帧数作为key，value为1
                results[f"{frame_count}"] = 1
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                # 如果没有检测到红色，将该帧的帧数作为key，value为0
                results[f"{frame_count}"] = 0

            cv2.imshow('Video', frame)
            frame_count += 1
            if frame_count == fps * second:
                data_processing(results, fps, current_video_index + 1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_count >= (8 * int(video.get(cv2.CAP_PROP_FPS))):  # 8秒后切换视频
                break

        last_frame_index[current_video_index] = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        video.release()
        # print(results)  # 输出当前视频的结果

        current_video_index = (current_video_index + 1) % len(video_files)
        # get_variances(results, 25, False)
        results.clear()
        # frame_count = 0
        # print(train_data)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect()

# ------------------------------------------------------------------------------------------------
# x1, y1, x2, y2 = (700, 0, 800, 300)
# while True:
#     for video_file in video_files:
#         cap = cv2.VideoCapture(video_file)
#         frame_count = 0
#         results = {}
#
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             # 获取视频帧的指定区域
#             roi = frame[y1:y2, x1:x2]
#             # 将帧转换为HSV颜色空间
#             hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#             # 定义红色范围
#             lower_red = np.array([0, 50, 50])
#             upper_red = np.array([10, 255, 255])
#             mask1 = cv2.inRange(hsv, lower_red, upper_red)
#
#             lower_red = np.array([170, 50, 50])
#             upper_red = np.array([180, 255, 255])
#             mask2 = cv2.inRange(hsv, lower_red, upper_red)
#             # 将两个掩码相加
#             mask = mask1 + mask2
#             # 检查是否检测到红色
#             if np.sum(mask) > 0:
#                 results[frame_count] = 1
#                 contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 for contour in contours:
#                     x, y, w, h = cv2.boundingRect(contour)
#                     cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
#             else:
#                 results[frame_count] = 0
#
#             cv2.imshow('Frame', frame)
#
#             # result = detect_red(frame, roi)
#             # results[frame_count] = result
#
#             cv2.imshow('Video', frame)
#             frame_count += 1
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#             if frame_count >= (8 * int(cap.get(cv2.CAP_PROP_FPS))):  # 8秒后切换视频
#                 break
#
#         cap.release()
#         print(results)  # 输出当前视频的结果
#         get_variances(results, 25, True)
#         results.clear()
#         frame_count = 0
#
#     cv2.destroyAllWindows()


