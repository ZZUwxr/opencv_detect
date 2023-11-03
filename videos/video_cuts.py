from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from datetime import datetime, timedelta

def time_to_seconds(time_str):
    time_obj = datetime.strptime(time_str, "%H:%M:%S")
    total_seconds = timedelta(hours=time_obj.hour, minutes=time_obj.minute, seconds=time_obj.second).total_seconds()
    return total_seconds

def trim_video(input_file, output_file, start_time, end_time):
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)
    ffmpeg_extract_subclip(input_file, start_seconds, end_seconds, targetname=output_file)

# 示例参数
input_file = "Test2.MP4"  # 输入视频文件名
output_file = "output_2.mp4"  # 输出截取后的视频文件名
start_time = "00:00:01"  # 开始时间，格式为"时:分:秒"
end_time = "00:05:30"  # 结束时间，格式为"时:分:秒"

# 截取视频
trim_video(input_file, output_file, start_time, end_time)