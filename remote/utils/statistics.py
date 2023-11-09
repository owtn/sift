import ntpath
import os
from glob import glob
from tqdm import tqdm
import json
import re


def count_input(file_path, logger=None):
    """
    统计下载原始数据的论文数，大图数和子图数，打印出来
    Args:
        file_path: 数据路径
        log_path: 日志路径

    Returns:
        子图数
    """
    papaer_paths = glob(os.path.join(file_path, '*', '*', '*'))
    print('上传论文数：', len(papaer_paths))
    json_paths = glob(os.path.join(file_path, '*', '*', '*', '*.json'))
    print('上传大图数：', len(json_paths))
    total = 0
    errorcount = 0
    for path in tqdm(json_paths):
        with open(path, encoding='utf-8') as load_f:
            try:
                data = json.load(load_f)
                total += len(data['outputs']['object'])
            except:
                errorcount += 1
                print('=====================')
                print('json格式有错')
    print('上传子图数：', total, '错误json数：', errorcount)
    if logger is not None:
        logger.log(file_path + '\n' +
                   '上传论文数：' + str(len(papaer_paths)) + '\t' +
                   '上传大图数：' + str(len(json_paths)) + '\t' +
                   '上传子图数：' + str(total) + '\t' +
                   '错误json数：' + str(errorcount))
    return total


def count_output(file_path, logger=None):
    """
    统计跑出的结果的论文数、大图数和子图数，打印出来
    Args:
        logger: logger
        file_path: sift结果的目录
        log_path: 日志目录

    Returns:
        子图数
    """
    paper_paths = glob(os.path.join(file_path, '*'))
    print(paper_paths[0])
    print('结果论文数：', len(paper_paths) - 1)
    img_paths = glob(os.path.join(file_path, '*', '*.jpg')) + glob(os.path.join(file_path, '*', '*.png'))
    img_hashes = list(
        set(['.'.join(path.split('/')[-1].split('.')[:-1]).split('_match')[0].split('_full')[0] for path in img_paths]))
    print('结果大图数：', len(img_hashes))
    area_imgs_num = len(glob(os.path.join(file_path, '*', '*_area.*')))
    print('结果子图数：', area_imgs_num)
    if logger is not None:
        logger.log(file_path + '\n' +
                   '结果论文数：' + str(len(paper_paths) - 1) + '\t' +
                   '结果大图数：' + str(len(img_hashes)) + '\t' +
                   '结果子图数：' + str(area_imgs_num))
    return area_imgs_num


def read_log():
    detect_path = f"/home/hdd1/data/wanfang/2022/IFD2022/1015_field"
    paper_list = os.listdir(detect_path)
    paper_list.sort()
    log_file = '/home/hdd1/data/wanfang/2022/sift_v4/1015_field/progress.txt'
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_data = f.read()
        finish_papers = re.findall("论文完成检测:(.+?)\n", log_data)
        if len(finish_papers) > 0:
            last_paper = finish_papers[-1]
            last_index = paper_list.index(last_paper)
            paper_list = paper_list[last_index + 1:]


def log_input_papers(file_path, id_file):
    """
    统计原始数据的论文id列表，输出到log_file
    Args:
        file_path: 要统计的原始数据的根目录
        id_file: 统计结果记录文件

    Returns:
        论文id列表
    """
    paper_paths = glob(os.path.join(file_path, '*', '*', '*'))
    paper_names = [path.split('/')[-1] for path in paper_paths]
    with open(id_file, 'w') as f:
        for name in paper_names:
            f.write(name + '\n')
    return paper_names


if __name__ == '__main__':
    # # # 计算上传数据的子图数，大图数，论文数
    cj_year = '2023'
    cj_batch = '0328'
    cj_path = f'/home/hdd1/data/wanfang/{cj_year}/IFD{cj_year}/{cj_batch}/'
    # count_input(cj_path)

    # # # 计算跑出结果的子图数，大图数，论文数
    ci_year = cj_year
    ci_batch = cj_batch
    ci_edition = '4'
    ci_path = f'/home/hdd1/data/wanfang/{ci_year}/sift_v{ci_edition}/{ci_batch}/'
    # count_output(ci_path)

    # # 计算筛完后的子图数、大图数、论文数
    ci_path = f'/home/hdd1/data/wanfang/{cj_year}/result/{cj_batch}/'
    # count_output(ci_path)

    # # 记录检测的论文
    log_file = f'/home/hdd1/data/wanfang/{cj_year}/sift_log/{cj_year}_{cj_batch}_paperID.txt'
    log_input_papers(cj_path, log_file)
