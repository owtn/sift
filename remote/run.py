from utils.download import download, unzip
from utils.reformat import get_field, delete_trash
from utils.network_util import test_login
from utils.manual_task import split_task
from utils.statistics import count_input, count_output, log_input_papers
from utils.logger import Logger
from utils.extract import extract_path
from utils.make_plag_database import make_batch_database_json

from sift.sift_platform import detect

import os
import sys

if __name__ == '__main__':
    year = sys.argv[1]
    batch = sys.argv[2]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]

    # 确保可连外网
    test_login()

    base_path = f'/home/hdd1/data/wanfang/{year}/'
    download_path = os.path.join(base_path, f'download/{batch}')
    unzip_path = os.path.join(base_path, f'IFD{year}/{batch}')
    url_file_path = os.path.join(download_path, 'url.txt')
    html_file_path = os.path.join(download_path, 'html.txt')
    field_path = os.path.join(base_path, f'IFD{year}/{batch}_field')
    res_path = os.path.join(base_path, f'sift_v4/{batch}')
    clean_result_path = os.path.join(base_path, f'sift_v4/{batch}_clean')
    log_file = os.path.join(base_path, 'sift_log/log.txt')
    ID_file = os.path.join(base_path, f'sift_log/{year}_{batch}_paperID.txt')
    result_path = os.path.join(base_path, f'result/{batch}')
    database_json_path = f'/home/hdd1/data/wanfang/database/jsons/{year}'
    logger = Logger(log_file)

    logger.log('==========================================================')

    # 从网页文字中提取出压缩包url
    s = extract_path(html_file=html_file_path, url_file=url_file_path, logger=logger)

    # 下载解压数据
    urls = download(url_file_path, download_path, logger=logger)
    zip_files = unzip(download_path, url_file_path, unzip_path, logger=logger)

    # 统计下载数据总数
    count_input(file_path=unzip_path, logger=logger)

    # 构造子图信息json
    make_batch_database_json(unzip_path, database_json_path, batch)

    # 分离出医学、材料学数据
    get_field(unzip_path, field_path, copy=True)

    # 检测
    detect(field_path, res_path, 0.65, 0.55, logger=logger, useOCR=True)

    # 统计检测结果总数
    count_output(res_path, logger=logger)

    # 清理结果太多的论文
    delete_trash(res_path, 5)

    # 分割打包
    split_task(clean_result_path, result_path, split_num=3)

    # 生成下载论文ID列表
    log_input_papers(file_path=unzip_path, id_file=ID_file)

    logger.log('人工筛选前工作完成\n' + '------------------------------------------------------')