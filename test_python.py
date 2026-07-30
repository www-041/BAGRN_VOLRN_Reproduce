print("hello world")
import os, glob, shutil

tmp_dir = 'data/input_DZ01V_only'
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir)

count = 0
for timestamp_dir in glob.glob('data/input/*'):
    for sensor_dir in glob.glob(os.path.join(timestamp_dir, 'DZ01V*')):
        link_name = os.path.join(tmp_dir, os.path.basename(sensor_dir))
        os.symlink(os.path.abspath(sensor_dir), link_name)
        count += 1
        print(f'链接: {sensor_dir}')

print(f'完成，共 {count} 个')