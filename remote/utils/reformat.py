"""
对旧格式数据重新检测，生成符合万方平台格式的新结果
"""

import os
from glob import glob
import shutil
from sift.sift_platform import get_subImage
import cv2 as cv
import zipfile
import hashlib
from tqdm import tqdm
from collections import Counter


# 记录有问题的数据
def logError(logFile, content):
    with open(logFile, 'a+') as log:
        log.write(content + '\n')


# 根据旧数据结果txt文件，从原始数据中提取有问题的原图和json标注文件，按新格式组织
def extractOriginData(dataBasePath, resultFile, outputPath):
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    logFile = os.path.join(outputPath, 'log.txt')
    with open(resultFile, 'r') as f:
        allLines = f.readlines()
        for line in allLines:
            line = line.strip('\r').strip('\n')
            [paperId, imgId] = line.split('/')[-2:]
            imgName1 = imgId.replace('_full.jpg', '.jpg')
            imgName2 = imgId.replace('_full.jpg', '.png')
            img_dirs = glob(os.path.join(dataBasePath, '*', '*', '*', paperId)) + \
                       glob(os.path.join(dataBasePath, '*', '*', paperId))
            if len(img_dirs) == 0:
                print('not found img directory')
                print(line)
                logError(logFile, 'no paperID:' + line)
                continue
            img_dir = None
            for path in img_dirs:
                if len(glob(os.path.join(path, 'images', imgName1))) != 0:
                    img_dir = path
                    imgName = imgName1
                    continue
                if len(glob(os.path.join(path, 'images', imgName2))) != 0:
                    img_dir = path
                    imgName = imgName2
                    continue
            if img_dir is None:
                print('empty directory')
                print(line)
                logError(logFile, 'empty directory:' + line)
                continue
            img_path = glob(os.path.join(img_dir, 'images', imgName))
            jsonName = '.'.join(imgName.split('.')[:-1]) + '.json'
            json_path = glob(os.path.join(img_dir, 'outputs', jsonName))
            if len(json_path) == 0:
                print('no json')
                print(line)
                logError(logFile, 'no json:' + line)
                continue
            out_dir = os.path.join(outputPath, paperId)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            shutil.copyfile(img_path[0], os.path.join(out_dir, imgName))
            shutil.copyfile(json_path[0], os.path.join(out_dir, jsonName))


# 对比调整格式后的sift结果和给sift提供的有问题图，输出不一致的图片地址
def compareResult(dataPath1, dataPath2):
    imgs1 = glob(os.path.join(dataPath1, '*', '*', '*'))
    imgs2 = glob(os.path.join(dataPath2, '*', '*', '*'))
    pathMap1 = {'.'.join(img.split('/')[-1].split('.')[:-1]).split('_')[0]: img for img in imgs1}
    pathMap2 = {'.'.join(img.split('/')[-1].split('.')[:-1]).split('_')[0]: img for img in imgs2}
    hashes1 = list(pathMap1.keys())
    hashes2 = list(pathMap2.keys())
    exclude1 = [pathMap1[item] for item in list(set(hashes1) - set(hashes2))]
    exclude2 = [pathMap2[item] for item in list(set(hashes2) - set(hashes1))]
    print('1有大图', len(hashes1), '张,', '2有大图', len(hashes2), '张')
    print('2比1多:')
    for path in exclude2:
        print(path)
    print('----------------------------------------------------------')
    print('1比2多:')
    for path in exclude1:
        print(path)


# 根据筛选后的结果真值生成res文件
def generateRes(basePath, resFile):
    files = glob(os.path.join(basePath, '*', '*', '*', '*'))
    full_imgs = [path for path in files if '_full' in path.split('/')[-1]]
    with open(resFile, 'w') as f:
        for path in full_imgs:
            f.write(path + '\n')


# 给原图画框
def drawRectangle(imgPath, jsonPath, outPath):
    img_rgb = cv.imread(imgPath)
    sub_set = get_subImage(jsonPath)
    red_color = (0, 0, 255)
    out_name = jsonPath.split('/')[-1].strip('.json') + '_full.jpg'
    for p1 in sub_set:
        cv.rectangle(img_rgb, (p1[0], p1[1]), (p1[2], p1[3]), red_color, 1)
    cv.imwrite(os.path.join(outPath, out_name), img_rgb)


# 去掉原始数据中间时间戳那一级目录
def stripStamp(srcPath, rstPath):
    papers = glob(os.path.join(srcPath, '*', "*"))
    for paper in papers:
        paperName = paper.split('/')[-1]
        shutil.copytree(paper, os.path.join(rstPath, paperName))


# 提取数据压缩包中的目录并重命名
def unzip_data(zip_dir, target_dir):
    zip_paths = glob(os.path.join(zip_dir, '*.zip'))
    for zip_path in zip_paths:
        print(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            for info in zip_file.infolist():
                file_path = info.filename
                paper_name, file_name = file_path.split('/')[-2:]
                if len(paper_name) > 32:
                    md = hashlib.md5()
                    md.update(paper_name.encode('utf-8'))
                    paper_name = md.hexdigest()
                save_dir = os.path.join(target_dir, paper_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, file_name)
                with open(save_path, 'ab') as f:
                    f.write(zip_file.read(info))

# 论文一级目录名太长的截取一部分
def rename_dir(base_dir):
    dir_paths = glob(os.path.join(base_dir, '*'))
    for path in dir_paths:
        if 'log.txt' in path:
            continue
        dir_name = path.split('/')[-1]
        dir_name = dir_name[:50]
        os.rename(path, os.path.join(base_dir, dir_name))


"""
下载数据去重
"""
def removeDuplicate(base_path):
    paper_paths = glob(os.path.join(base_path, '*'))
    name_set = set()
    for paper_path in paper_paths:
        files = os.listdir(paper_path)
        for file in files:
            if file in name_set:
                os.unlink(os.path.join(paper_path, file))
            else:
                name_set.add(file)
        if len(os.listdir(paper_path)) == 0:
            os.rmdir(paper_path)


def get_field(base_path, target_path, copy=False):
    """
    调整原始数据的目录结构，并只保留医学和材料领域的数据
    调整前：
    根目录
    |---时间戳
        |---领域1
        |---领域2
            |---期刊1
                |---大图1
                |---json1
    调整后：
    |---期刊1
        |---大图1
        |---json1
        |---大图2
        |---json2
    Args:
        base_path: 原始数据路径
        target_path: 目标路径
        copy: 是否将数据复制到目标路径

    Returns:
        统计各领域文件夹数目的Counter
    """
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    field_path = glob(os.path.join(base_path, '*', '*'))
    paper_counter = Counter()
    for field in field_path:
        paper_num = len(os.listdir(field))
        paper_counter[field.split('/')[-1]] += paper_num
    print('paper counter:', paper_counter)
    # medi_field = glob(os.path.join(base_path, '*', 'чФЯчЙйхМ╗хнжщвЖхЯЯ'))
    # issu_field = glob(os.path.join(base_path, '*', 'цЭРцЦЩхнжщвЖхЯЯ'))
    # other_field = glob(os.path.join(base_path, '*', 'хЕ╢ф╗ЦхнжчзСщвЖхЯЯ')) + glob(
    #     os.path.join(base_path, '*', '其他学科领域'))
    # un_field = glob(os.path.join(base_path, '*', 'цЧа'))
    # field_name = [path.split('/')[-1] for path in field_path]
    # counter = Counter(field_name)
    # print('field counter:', counter)
    all_field = glob(os.path.join(base_path, '*', '*'))
    if copy:
        # copy_field = medi_field + issu_field
        copy_field = all_field
        for field in copy_field:
            papers = os.listdir(field)
            print('copy ', field)
            for paper in tqdm(papers):
                os.system('cp -r ' + os.path.join(field, paper) + ' ' + target_path)
    return


def delete_trash(base_path, thresh):
    """
    将sift结果中匹配图片数过多的大图结果删除
    Args:
        base_path: sift结果路径
        thresh: 清理后保存的路径

    Returns:
    """
    cleaned_path = os.path.normpath(base_path) + '_clean'
    shutil.copytree(base_path, cleaned_path)
    log_files = glob(os.path.join(cleaned_path, '*.txt'))
    for file in log_files:
        os.system('rm -r ' + file)
    papers = glob(os.path.join(cleaned_path, '*'))
    papers.sort()
    for paper in papers:
        imgs = os.listdir(paper)

        sub_results = [img.split('_match')[0] for img in imgs if '_area' in img]
        counter = Counter(sub_results)
        total = len(counter)
        for name, num in counter.items():
            if num > thresh:
                os.system('rm ' + os.path.join(cleaned_path, paper, name + '*'))
                total -= 1
        if total == 0:
            os.system('rm -r '+ os.path.join(cleaned_path, paper))



if __name__ == '__main__':
    # extractOriginData
    # year = 2022
    # dataBase = '/home/hdd1/data/wanfang/{0}/IFD{0}'.format(year)
    # resultFile = '/home/hdd1/data/wanfang/{0}/IFD{0}_res_0601.txt'.format(year)
    # resultFile = '/home/hdd1/data/wanfang/2022/result/Missing_biomedical/res.txt'
    # outputFile = '/home/hdd1/data/wanfang/{0}/IFD{0}/reformat/0'.format(year)
    # outputFile = '/home/hdd1/data/wanfang/2022/IFD2022/Missing_reformat/0'
    # extractOriginData(dataBase, resultFile, outputFile)

    # compareResult
    # dataPath1 = '/home/hdd1/data/wanfang/2020/sift_v4/reformat/'
    # datapath2 = '/home/hdd1/data/wanfang/2020/IFD2020/reformat/'
    # compareResult(dataPath1, datapath2)

    # generateRes
    # base_path = '/home/hdd1/data/wanfang/2022/result/Missing_biomedical/'
    # resFile = os.path.join(base_path, 'res.txt')
    # generateRes(base_path, resFile)

    # drawRectangle
    # outPath = '/home/hdd1/wanghaoran/sift/temp'
    # inputPath = '/home/hdd1/data/wanfang/2019/segTest/0222/Am J Transl Res/9/6490a948f567e4ebd547365cc4635130/images/8446c5a6f48b05790e6d1b043c1d7e96.jpg'

    # stripStamp
    # srcPath = '/home/hdd1/data/wanfang/2022/IFD2022/0825_old'
    # rstPath = '/home/hdd1/data/wanfang/2022/IFD2022/0825'
    # stripStamp(srcPath, rstPath)

    # unzip_rename
    # zip_dir = '/home/hdd1/wanghaoran/sift/data/zip/'
    # unzip_dir = '/home/hdd1/wanghaoran/sift/data/zip_target'
    # unzip_data(zip_dir, unzip_dir)

    # base_path = '/home/hdd1/data/wanfang/2022/sift_v4/0825_old'
    # rename_dir(base_path)

    # removeDuplicate('/home/hdd1/data/wanfang/2022/IFD2022/0905/')

    # base_path = '/home/hdd1/data/wanfang/2023/IFD2023/0210'
    # target_path = '/home/hdd1/data/wanfang/2023/IFD2023/0210_field'
    # counter = get_field(base_path, target_path, copy=True)

    base_path = '/home/hdd1/data/wanfang/2023/sift_v4/0210'
    delete_trash(base_path, 5)