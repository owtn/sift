import os
import shutil
from glob import glob
import sys


def split_task(task_path, result_path, split_num=2):
    task_num = len(os.listdir(task_path))
    base_path, task_name = os.path.split(os.path.normpath(task_path.strip('_clean')))[:2]
    split_size = (task_num + split_num - 1) // split_num
    for i in range(split_num):
        split_dir = task_name + '_' + str(i)
        split_path = os.path.join(base_path, split_dir)
        start = i * split_size + 1
        end = min(task_num, (i + 1) * split_size)
        zip_name = task_name + '_' + str(i) + '.tar'
        zip_file = os.path.join(base_path, zip_name)
        os.system('mkdir ' + split_path)
        os.system(f"ls {task_path} | sed -n '{start}, {end}p' | xargs -i cp -r "
                  f"{os.path.normpath(task_path)}/{{}} {split_path}")
        os.system(f"tar -cvf {zip_file} -C {base_path} {split_dir}")

    # 创建结果目录
    oldmask = os.umask(000)
    os.mkdir(result_path, 0o777)
    os.umask(oldmask)


def delete_splits(base_path, batch_id):
    delete_tars = glob(os.path.join(base_path, f'{batch_id}_*.tar'))
    for name in delete_tars:
        os.remove(name)
    delete_list = glob(os.path.join(base_path, f'{batch_id}_*'))
    for name in delete_list:
        shutil.rmtree(name)


if __name__ == '__main__':
    split_task('/home/hdd1/data/wanfang/2023/sift_v4/0210_clean', '/home/hdd1/data/wanfang/2023/result/0210')
