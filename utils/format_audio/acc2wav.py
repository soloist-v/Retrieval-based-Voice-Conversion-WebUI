import os
from pydub import AudioSegment

def convert_aac_to_wav(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".aac"):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)
            # 读取AAC文件
            audio = AudioSegment.from_file(file_path, format="aac")
            # 构建WAV文件路径
            wav_path = os.path.join(folder_path, filename[:-4] + ".wav")
            # 导出为WAV格式
            audio.export(wav_path, format="wav")
            os.remove(file_path)
            print(f"Converted {filename} to WAV.")

# 指定你的文件夹路径
folder_path = r'F:\Datasets\RVC\Feisang\origin'
convert_aac_to_wav(folder_path)
