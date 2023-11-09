import sys
import os

from utils.prepareUpload import prepareUpload
from utils.manual_task import delete_splits
from utils.logger import Logger
from utils.statistics import count_output

if __name__ == '__main__':
    year = sys.argv[1]
    batch = sys.argv[2]
    base_path = f'/home/hdd1/data/wanfang/{year}/'
    res_path = os.path.join(base_path, f'result/{batch}')
    log_file = os.path.join(base_path, 'sift_log/log.txt')
    uploadPath = os.path.join(base_path, f'upload/{batch}')
    logger = Logger(log_file)

    # 筛选后结果调整为上传格式
    prepareUpload(resultPath=res_path, uploadPath=uploadPath)

    # 统计筛选后结果
    count_output(res_path, logger=logger)

    # 删除分工产生的多余文件
    delete_splits(base_path=os.path.join(base_path, f'sift_v4'), batch_id=batch)

    logger.log('上传前工作完成\n' + '======================================================')

