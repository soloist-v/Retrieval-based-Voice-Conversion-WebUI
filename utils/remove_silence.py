import os
from pydub import AudioSegment, silence

# 设置音频文件的目录
directory = r'F:\Datasets\RVC\XC\NeedCut'
output_path = r"F:\Datasets\RVC\XC\Cutted"

# 创建一个空的音频段，用于后续拼接音频

index = 0
# 遍历目录中的所有文件
for root, dirs, files in os.walk(directory):
    for filename in files:
        combined = AudioSegment.empty()
        if filename.endswith(".wav"):  # 假设音频文件是mp3格式
            path = os.path.join(root, filename)
            print(f"Processing {filename}...")
            audio = AudioSegment.from_file(path)

            # 处理无声片段，将超过1秒的无声段替换为0.3秒的无声
            non_silent_parts = silence.split_on_silence(audio, min_silence_len=1000, silence_thresh=-54)
            processed_audio = AudioSegment.silent(duration=0)  # 创建一个空的音频段开始构建处理后的音频
            for chunk in non_silent_parts:
                processed_audio += chunk
                # 插入1秒的无声音
                processed_audio += AudioSegment.silent(duration=300)

            # 将处理后的音频文件添加到拼接中
            combined += processed_audio
            output_filename = f"{output_path}/{filename}"
            combined.export(output_filename, format='wav')

            print(f"{filename} completed! len: {len(combined)}")
            # 检查拼接后的音频长度是否已经达到1小时（3600000毫秒）
            # if len(combined) >= 1800000:
                # 保存当前的1小时音频段


            # 移除已经保存的部分，继续下一个小时的拼接
            # combined = AudioSegment.empty()
            # index += 1
            # else:
            #     print(f"index: {index} len: {len(combined)}")

