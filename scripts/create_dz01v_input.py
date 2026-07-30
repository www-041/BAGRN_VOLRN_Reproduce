import os
import glob
import shutil

# 创建临时目录，只包含 DZ01V
tmp_dir = 'data/input_DZ01V_only'
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir)

# 找所有 DZ01V 文件夹
for timestamp_dir in glob.glob('data/input/*'):
    for sensor_dir in glob.glob(os.path.join(timestamp_dir, 'DZ01V*')):
        # 创建软链接，保持相同的文件夹名
        link_name = os.path.join(tmp_dir, os.path.basename(sensor_dir))
        os.symlink(os.path.abspath(sensor_dir), link_name)
        print(f'链接: {sensor_dir} -> {link_name}')

v_dirs = glob.glob(os.path.join(tmp_dir, 'DZ01V*'))
print(f'完成，共 {len(v_dirs)} 个 DZ01V 文件夹')