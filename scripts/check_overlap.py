import sys; sys.path.insert(0, '.')
from src.io_utils import read_geotiff
from src.overlap import get_overlap_window
import glob, os

files = sorted(glob.glob('data/input/**/*DZ01V*B5*.TIF', recursive=True))
files = [f for f in files if '_PAN' not in f.upper()]

images = []
for f in files:
    arr, tr, crs, nd = read_geotiff(f)
    while arr.ndim > 2: arr = arr[0]
    rows, cols = arr.shape
    left = tr.c; right = tr.c + tr.a * cols
    top = tr.f; bottom = tr.f + tr.e * rows
    name = os.path.basename(f)
    images.append({'name': name, 'bounds': (left, bottom, right, top), 'shape': arr.shape, 'transform': tr})
    print(f'{name}: {rows}x{cols}')

print()
for i in range(len(images)):
    for j in range(i+1, len(images)):
        bi = images[i]['bounds']
        bj = images[j]['bounds']
        win = get_overlap_window(bi, images[i]['transform'], bj, images[j]['transform'])
        if win:
            wi, wj = win
            n = (wi[1]-wi[0])*(wi[3]-wi[2])
            ni = images[i]['name'][:25]
            nj = images[j]['name'][:25]
            print(f'{ni} vs {nj}: {n} pixels')
