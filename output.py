import os
import json
from others import detect


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


if __name__ == '__main__':
    main()
