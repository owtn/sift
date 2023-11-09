from sift.sift_v3 import detect as detect3
# from sift.sift_platform import detect as detect_platform
from sift.sift_platform_new import detect as detect_platform
import os
import sys

if __name__ == '__main__':
    # version = '3'
    #
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
    # os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]
    #
    # year = sys.argv[1]
    # batch = sys.argv[2]
    # detect_path = f"/home/hdd1/data/wanfang/{year}/IFD{year}/{batch}"
    # res_path = f"/home/hdd1/data/wanfang/{year}/sift_v{version}/{batch}"
    # detect3(detect_path, res_path, 0.7, 0.5)
    # field_path = '/home/hdd1/data/wanfang/2023/IFD2023/1026_temp_2'
    field_path = '/home/hdd1/maxu/dup_detect/2023_subs/1026_temp1'
    res_path = '/home/hdd1/data/wanfang/2023/sift_v4/1026_temp_evil'

    detect_platform(field_path, res_path, 0.7, 0.0, logger=None, useOCR=False)