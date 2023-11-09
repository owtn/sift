import numpy as np
import os
import shutil
from sift.sift_v2 import detect
from glob import glob
from utils.compare import son_only
from sift.sift_v3 import detect as detect3


def check_param(path, the_picture):
    v1_files = glob(os.path.join(path, '*', '*', '*', '*'))
    v1_tails = ['/'.join(file.split('/')[-4:]) for file in v1_files]
    v1_tails = son_only(v1_tails)
    son_num = len(v1_tails)
    if the_picture in v1_tails:
        return son_num
    else:
        return None


if __name__ == '__main__':
    detect_path = f"/home/hdd1/data/wanfang/2022/IFD2022/0320"
    res_path = f"/home/hdd1/wanghaoran/sift/params"
    the_picture = 'Journal of Cancer/1/02a1258acf6a00b6da0bd0aebcf483f7/606636644f70f5ec3454ddd1ed7744d6_2.jpg'
    lowest = float('inf')
    best_r = 0
    best_t = 0
    for tt in range(80, 59, -1):
        t = tt / 100.
        no_improve = True
        for rr in range(40, 51, 1):
            r = rr / 100.
            print('=================================================')
            print('param:', t, '    ', r)
            detect(detect_path, res_path, t, r)
            count_result = check_param(res_path, the_picture)
            if os.path.exists(res_path):
                shutil.rmtree(path=res_path)
            if count_result is not None:
                no_improve = False
                if count_result < lowest:
                    best_r = r
                    best_t = t
                    lowest = count_result
            else:
                break
        if no_improve:
            break

    print('==========================================================')
    print('best t is:', best_t)
    print('best r is:', best_r)
    best = np.array([best_t, best_r])
    np.save('best_para.npy', best)

    detect_path = f"/home/hdd1/data/wanfang/2022/IFD2022/0322"
    res_path = f"/home/hdd1/data/wanfang/2022/sift_v3/0322"
    detect3(detect_path, res_path, best_t, best_r)
