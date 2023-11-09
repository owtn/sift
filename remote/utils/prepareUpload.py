import shutil
import os
from glob import glob
import sys


def prepareUpload(resultPath, uploadPath):
    """
    sift跑出的结果，人工筛选后，调整为平台上传格式
    Args:
        resultPath: 人工筛选后的结果的目录
        uploadPath: 调整后输出的路径

    Returns:
    """
    if not os.path.exists(uploadPath):
        os.makedirs(uploadPath)
    imgs = glob(os.path.join(resultPath, '*', "*"))
    hashCodes = list(set(['.'.join(path.split('/')[-1].split('.')[:-1]).split('_')[0] for path in imgs]))
    for hashCode in hashCodes:
        record = {}
        hashPath = os.path.join(uploadPath, hashCode)
        if not os.path.exists(hashPath):
            os.makedirs(hashPath)
        results = glob(os.path.join(resultPath, '*', hashCode + '_match*'))
        for result in results:
            img_name, appendix = result.split('/')[-1].split('_')[:2]
            small_index1, small_index2 = appendix[5:].split('&')[:2]
            small_index2 = small_index2.split('_')[0].split('.')[0]
            small_index1, small_index2 = map(int, [small_index1, small_index2])
            if img_name in record:
                record[img_name].add((small_index1, small_index2))
            else:
                record[img_name] = {(small_index1, small_index2), }
            shutil.copy(result, hashPath)
        out_str = ''
        for img_name, similiar_pairs in record.items():
            out_str += f'大图{img_name}内的编号为'
            for i, (index1, index2) in enumerate(similiar_pairs):
                if i > 0:
                    out_str += '; '
                out_str += f' {index1} 和 {index2} '
            out_str += f'的子图间存在复用行为\n'

        with open(os.path.join(hashPath, 'detail.txt'), 'w') as f:
            f.write(out_str)

if __name__ == '__main__':
    # prepareUpload
    # year = sys.argv[1]
    # batch = sys.argv[2]
    year = '2023'
    batch = '0301'
    resultPath = f'/home/hdd1/data/wanfang/{year}/result/{batch}'
    uploadPath = f'/home/hdd1/data/wanfang/{year}/upload/{batch}'
    prepareUpload(resultPath, uploadPath)
