import os
import shutil

# 源文件夹和目标文件夹的路径
source_dir = r"G:\Projects\PC\RVC\python-server\opt"
target_dir = r"F:\Datasets\RVC\XC\去伴奏混响\VocalHP5Output"

# 确保目标文件夹存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 初始化文件索引
index = 0

# 遍历源文件夹中的所有子文件夹
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # 检查文件是否是人声文件
        # if "_main_vocal" in file and file.endswith(".wav"):
        if "vocal" in file and file.endswith(".wav"):
            # 构建完整的源文件路径和目标文件路径
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, f"{index}.wav")

            # 移动并重命名文件
            shutil.move(source_file, target_file)

            # 更新文件索引
            index += 1
            print(f"移动了{index}.wav")

print("文件移动完成。")
