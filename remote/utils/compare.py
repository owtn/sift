import json
import shutil
from glob import glob
import os
from tqdm import tqdm


def big_only(tails):
    result = []
    for tail in tails:
        file_name = tail.split('/')[-1].split('.')[-2]
        name_split = file_name.split('_')
        if len(name_split) == 1:
            result.append(tail)
    return result

def son_only(tails):
    result = []
    for tail in tails:
        file_name = tail.split('/')[-1].split('.')[-2]
        name_split = file_name.split('_')
        if len(name_split) == 2 and name_split[-1] != 'full':
            result.append(tail)
    return result


def compare_files(path1, path2, target_path, name1, name2):
    v1_papers = glob(os.path.join(path1, '*', '*', '*'))
    print('{0}目录有{1}篇论文'.format(name1, len(v1_papers)))
    v1_files = glob(os.path.join(path1, '*', '*', '*', '*'))
    v2_files = glob(os.path.join(path2, '*', '*', '*', '*'))
    v1_tails = ['/'.join(file.split('/')[-4:]) for file in v1_files]
    v2_tails = ['/'.join(file.split('/')[-4:]) for file in v2_files]
    v1_hash_code = big_only(v1_tails)
    v2_hash_code = big_only(v2_tails)
    common_big = list(set(v1_hash_code).intersection(set(v2_hash_code)))
    v1_exclu_big = list(set(v1_hash_code) - set(v2_hash_code))
    v2_exclu_big = list(set(v2_hash_code) - set(v1_hash_code))
    print('{0}目录有{1}大图，多{2}个；{3}目录有{4}大图，多{5}个；相同结果{6}'.format(name1, len(v1_hash_code), len(v1_exclu_big),
                                                       name2, len(v2_hash_code), len(v2_exclu_big), len(common_big)))
    v1_tails = son_only(v1_tails)
    v2_tails = son_only(v2_tails)
    common_tails = list(set(v1_tails).intersection(set(v2_tails)))
    v1_exclu_tails = list(set(v1_tails) - set(v2_tails))
    v2_exclu_tails = list(set(v2_tails) - set(v1_tails))
    common_tails.sort()
    v1_exclu_tails.sort()
    v2_exclu_tails.sort()
    target_dir = os.path.join(target_path, '_'.join([name1, name2]))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open(os.path.join(target_dir, 'common_{0}.txt'.format(len(common_big))), 'w') as file:
        for tail in common_big:
            file.write(os.path.join(path1, tail) + '\n')
    with open(os.path.join(target_dir, '{0}_{1}.txt'.format(name1, len(v1_exclu_big))), 'w') as file:
        for tail in v1_exclu_big:
            file.write(os.path.join(path1, tail) + '\n')
    with open(os.path.join(target_dir, '{0}_{1}.txt'.format(name2, len(v2_exclu_big))), 'w') as file:
        for tail in v2_exclu_big:
            file.write(os.path.join(path2, tail) + '\n')
    print('{0}目录多{1}个子图，共{2}个；{3}目录多{4}个子图，共{5}个；相同结果{6}个子图'.format(name1, len(v1_exclu_tails),
                                                        len(v1_tails), name2, len(v2_exclu_tails), len(v2_tails),
                                                                    len(common_tails)))


def compare_paper(path1, path2, name1, name2):
    papers1 = glob(os.path.join(path1, '*/*/*'))
    papers2 = glob(os.path.join(path2, '*/*/*'))
    v1_tails = ['/'.join(file.split('/')[-3:]) for file in papers1]
    v2_tails = ['/'.join(file.split('/')[-3:]) for file in papers2]
    common_tails = list(set(v1_tails).intersection(set(v2_tails)))
    v1_exclu_tails = list(set(v1_tails) - set(v2_tails))
    v2_exclu_tails = list(set(v2_tails) - set(v1_tails))
    print('{0}目录多{1}篇论文，{2}目录多{3}篇论文，相同结果{4}篇论文'.format(name1, len(v1_exclu_tails),
                                                        name2, len(v2_exclu_tails), len(common_tails)))


def clean_paper(path1, path2, name1, name2):
    papers1 = glob(os.path.join(path1, '*/*/*'))
    papers2 = glob(os.path.join(path2, '*/*/*'))
    v1_tails = ['/'.join(file.split('/')[-3:]) for file in papers1]
    v2_tails = ['/'.join(file.split('/')[-3:]) for file in papers2]
    common_tails = list(set(v1_tails).intersection(set(v2_tails)))
    cleans = [os.path.join(path2, tail) for tail in common_tails]
    for paper in cleans:
        shutil.rmtree(paper)
    print('已删除{0}目录中{1}个论文目录'.format(path2, len(cleans)))


def compare_result(base_path, target_path, year_name, batch_name, version1, version2):
    if not version1 in [1, 2, 3, 3.1, 3.2, 3.3, 4, 4.1] or not version2 in [1, 2, 3, 3.1, 3.2, 3.3, 4, 4.1]:
        print("correct versions please!")
        return
    elif version2 == version1:
        print("different versions please!")
        return
    if version2 < version1:
        compare_result(base_path, target_path, year_name, batch_name, version2, version1)
        return
    version2 = str(version2).replace('.', '_')
    version2 = str(version2).replace('.', '_')
    v1_base = os.path.join(base_path, year_name, 'sift_v{0}'.format(version1), batch_name)
    v2_base = os.path.join(base_path, year_name, 'sift_v{0}'.format(version2), batch_name)
    compare_files(v1_base, v2_base, os.path.join(target_path, year_name, batch_name), 'v{0}'.format(version1),
                  'v{0}'.format(version2))


def count_json(file_path):
    paper_paths = glob(os.path.join(file_path, '*', '*', '*'))
    print('论文数：', len(paper_paths))
    json_paths = glob(os.path.join(file_path, '*', '*', '*', 'outputs', '*.json'))
    print('大图数：', len(json_paths))
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
    print('子图数：', total)
    return total


if __name__ == '__main__':
    # compare_result参数
    # base_path = '/home/hdd1/data/wanfang/'
    # result_path = '/home/hdd1/wanghaoran/sift/diff/'
    # year_name = '2022'
    # batch_name = '0507_Biomedical'
    # version1 = 2
    # version2 = 3
    # compare_result(base_path, result_path, year_name, batch_name, version1, version2)

    # compare_files参数
    # compare_files_year = '2022'
    # compare_files_batch = 'seg_test'
    # name1 = 'manu_seg'
    # name2 = 'maxu_seg'
    # path1 = '/home/hdd1/data/wanfang/2019/sift_v4/0222'
    # path2 = '/home/hdd1/data/wanfang/2022/sift_v4/segTest_maxu'

    # compare_files参数
    # compare_files_year = '2022'
    # compare_files_batch = 'seg_test'
    # name1 = 'maxu_seg'
    # name2 = 'manu'
    # path1 = '/home/hdd1/data/wanfang/2022/sift_v4/segTest_maxu'
    # path2 = '/home/hdd1/data/wanfang/2019/sift_v4/0222'
    # target_path = '/home/hdd1/wanghaoran/sift/diff/{0}/{1}'.format(compare_files_year, compare_files_batch)
    # compare_files(path1, path2, target_path, name1, name2)

    # count_json参数
    ori_base = '/home/hdd1/data/wanfang/'
    ori_year = '2019'
    ori_batch = '0222'
    count_json(os.path.join(ori_base, ori_year, 'IFD'+ori_year, ori_batch))

    # compare_paper参数
    # paper_path1 = '/home/hdd1/data/wanfang/2020/IFD2022/0320/'
    # paper_path2 = '/home/hdd1/data/wanfang/2022/IFD2022/0322/'
    # paper_name1 = '0320'
    # paper_name2 = '0322'
    # compare_paper(paper_path1, paper_path2, paper_name1, paper_name2)


    # clean_paper(paper_path1, paper_path2, paper_name1, paper_name2)

