#!/usr/bin/env python
"""按传感器过滤输入目录"""
import os
import shutil
import sys

input_dir = sys.argv[1] if len(sys.argv) > 1 else "data/input"
output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/output/sensor_DZ01V_temp"
sensor = sys.argv[3] if len(sys.argv) > 3 else "DZ01V"

os.makedirs(output_dir, exist_ok=True)

for timestamp in os.listdir(input_dir):
    ts_path = os.path.join(input_dir, timestamp)
    if not os.path.isdir(ts_path):
        continue
    for sensor_dir in os.listdir(ts_path):
        if sensor_dir.startswith(sensor):
            src = os.path.join(ts_path, sensor_dir)
            dst = os.path.join(output_dir, sensor_dir)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"复制: {sensor_dir}")

print(f"完成: {output_dir}")