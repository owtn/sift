import re
import os
from glob import glob


def download(url_file, download_dir, logger=None):
    """
    下载数据
    Args:
        logger: logger
        url_file: url列表文件路径
        download_dir: 下载目标目录

    Returns:
        url_file中记录的url列表
    """
    if logger is not None:
        logger.log('开始下载数据，url文件：' + url_file)
    with open(url_file, 'r') as f:
        lines = f.readlines()
    urls = [line.strip('\n').strip('\r') for line in lines]
    for url in urls:
        os.system('wget ' + url + ' -P ' + download_dir)
    if logger is not None:
        logger.log('下载完成！')
    return urls


def unzip(source_dir, url_file, target_dir, logger=None):
    """
    批量解压数据压缩包
    Args:
        logger: logger
        source_dir: 存放压缩包目录
        url_file: url列表文件路径
        target_dir: 解压目标目录

    Returns:
        压缩包路径列表
    """
    if logger is not None:
        logger.log('开始解压数据，压缩文件目录：' + source_dir)
    with open(url_file, 'r') as f:
        lines = f.readlines()
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    zip_files = [os.path.join(source_dir, line.strip('\n').strip('\r').split('/')[-1]) for line in lines]
    for zip in zip_files:
        os.system('unzip -o ' + zip + ' -d ' + target_dir)
    if logger is not None:
        logger.log('解压完成！')
    return zip_files


if __name__ == '__main__':
    year = '2023'
    batch = '0210'
    url_name = 'url.txt'
    urls = download(f'/home/hdd1/data/wanfang/{year}/download/{batch}/{url_name}',
                          f'/home/hdd1/data/wanfang/{year}/download/{batch}/')
    zip_files = unzip(f'/home/hdd1/data/wanfang/{year}/download/{batch}/',
                      f'/home/hdd1/data/wanfang/{year}/download/{batch}/{url_name}',
                      f'/home/hdd1/data/wanfang/{year}/IFD{year}/{batch}')
