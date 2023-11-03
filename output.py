import os
import json
import detect
import cv2


def main():

    config_file_path = 'config.json'

    # 读取config.json文件
    with open(config_file_path, 'r') as json_file:
        config_data = json.load(json_file)

    folder_path = config_data["input_path"]

    config_data['processed_files'] = []

    # 列出文件夹中的所有文件
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for file in files:
        # 如果该文件已经被处理过，跳过
        if file in config_data['processed_files']:
            continue

        # 替换"video_path"的值
        config_data["video_path"] = os.path.join(folder_path, file)

        videos = config_data["video_path"]

        print(f"当前检测视频为: {videos}")

        # 添加文件到已处理文件的列表
        config_data['processed_files'].append(file)

        # 将修改后的内容重新写回config.json文件中
        with open(config_file_path, 'w') as outfile:
            json.dump(config_data, outfile, indent=4)

        # 替换后执行另一个函数
        detect.main()


def switch_rtsp_streams(rtsp_urls):
    current_stream = 0
    num_streams = len(rtsp_urls)

    # 打开第一个RTSP流
    cap = cv2.VideoCapture(rtsp_urls[current_stream], cv2.CAP_FFMPEG)

    while True:
        # 读取视频帧
        ret, frame = cap.read()

        # 显示当前帧
        cv2.imshow("RTSP Stream", frame)

        # 等待用户按键
        key = cv2.waitKey(1)

        # 如果按下 'q' 键，退出循环
        if key == ord('q'):
            break

        # 如果按下 's' 键，切换到下一个RTSP流
        if key == ord('s'):
            # 关闭当前RTSP流
            cap.release()

            # 切换到下一个RTSP流
            current_stream = (current_stream + 1) % num_streams
            cap = cv2.VideoCapture(rtsp_urls[current_stream], cv2.CAP_FFMPEG)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


# RTSP流URL列表
rtsp_urls = ["rtsp://example.com/stream1", "rtsp://example.com/stream2", "rtsp://example.com/stream3"]

# 切换RTSP流
switch_rtsp_streams(rtsp_urls)


# if __name__ == '__main__':
#     main()
