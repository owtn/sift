import os
from glob import glob
import json
from tqdm import tqdm

def make_database_json_2019_2020_2021(year):
    base_path = f'/home/hdd1/data/wanfang/{year}/IFD{year}'
    record = {}
    count = 0
    images_dirs = glob(os.path.join(base_path, '*', '*', '*', 'images')) + glob(os.path.join(base_path, '*', '*', '*', '*', 'images'))
    for images_dir in images_dirs:
        paper_path = '/'.join(images_dir.split('/')[:-1])
        paper_name = paper_path.split('/')[-1]
        jsons = glob(os.path.join(paper_path, 'outputs', '*.json'))
        for json_path in jsons:
            image_name = json_path.split('/')[-1].split('.')[0]
            image_path = glob(os.path.join(paper_path, 'images', image_name + '.*'))
            if len(image_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        json_content = json.load(f)
                        for i, box in enumerate(json_content['outputs']['object']):
                            record[image_name + '_' + str(i)] = {
                                'big_img_path': image_path[0],
                                'classification': box['name'],
                                'paper': paper_name,
                                'big_img': image_name,
                                'xmin': box['bndbox']['xmin'],
                                'ymin': box['bndbox']['ymin'],
                                'xmax': box['bndbox']['xmax'],
                                'ymax': box['bndbox']['ymax']
                            }
                            count += 1
                    except:
                        print('\nskip with json error!')
                        print(json_path)
    with open(f'/home/hdd1/wanghaoran/sift/data/database/jsons/{year}.json', 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)
    print('总子图数：', count)


def make_database_json_2022():
    year = '2022'
    base_path = f'/home/hdd1/data/wanfang/{year}/IFD{year}'
    record = {}
    count = 0
    images_dirs = glob(os.path.join(base_path, '*', '*', '*', 'images')) + glob(os.path.join(base_path, '*', '*', '*', '*', 'images'))
    for images_dir in images_dirs:
        paper_path = '/'.join(images_dir.split('/')[:-1])
        paper_name = paper_path.split('/')[-1]
        jsons = glob(os.path.join(paper_path, 'outputs', '*.json'))
        for json_path in jsons:
            image_name = json_path.split('/')[-1].split('.')[0]
            image_path = glob(os.path.join(paper_path, 'images', image_name+'.*'))
            if len(image_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        json_content = json.load(f)
                        for i, box in enumerate(json_content['outputs']['object']):
                            record[image_name + '_' + str(i)] = {
                                'big_img_path': image_path[0],
                                'classification': box['name'],
                                'paper': paper_name,
                                'big_img': image_name,
                                'xmin': box['bndbox']['xmin'],
                                'ymin': box['bndbox']['ymin'],
                                'xmax': box['bndbox']['xmax'],
                                'ymax': box['bndbox']['ymax']
                            }
                            count += 1
                    except:
                        print('\nskip with json error!')
                        print(json_path)

    new_format_dirs = ['0616', '0708', '0825', '0831', '0905']
    for batch_name in new_format_dirs:
        json_paths = glob(os.path.join(base_path, batch_name, '*', '*.json'))
        for json_path in json_paths:
            paper_name = json_path.split('/')[-2]
            image_name = json_path.split('/')[-1].split('.')[0]
            image_path = glob(os.path.join(os.path.dirname(json_path), image_name + '.*'))
            if len(image_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        json_content = json.load(f)
                        for i, box in enumerate(json_content['outputs']['object']):
                            record[image_name + '_' + str(i)] = {
                                'big_img_path': image_path[0],
                                'classification': box['name'],
                                'paper': paper_name,
                                'big_img': image_name,
                                'xmin': box['bndbox']['xmin'],
                                'ymin': box['bndbox']['ymin'],
                                'xmax': box['bndbox']['xmax'],
                                'ymax': box['bndbox']['ymax']
                            }
                            count += 1
                    except:
                        print('\nskip with json error!')
                        print(json_path)

    new_format_dirs = ['0915', '1001', '1015', '1101', '1115', '1201', '1205', '1227']
    for batch_name in new_format_dirs:
        json_paths = glob(os.path.join(base_path, batch_name, '*', '*', '*', '*.json'))
        for json_path in json_paths:
            paper_name = json_path.split('/')[-2]
            image_name = json_path.split('/')[-1].split('.')[0]
            image_path = glob(os.path.join(os.path.dirname(json_path), image_name + '.*'))
            if len(image_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        json_content = json.load(f)
                        for i, box in enumerate(json_content['outputs']['object']):
                            record[image_name + '_' + str(i)] = {
                                'big_img_path': image_path[0],
                                'classification': box['name'],
                                'paper': paper_name,
                                'big_img': image_name,
                                'xmin': box['bndbox']['xmin'],
                                'ymin': box['bndbox']['ymin'],
                                'xmax': box['bndbox']['xmax'],
                                'ymax': box['bndbox']['ymax']
                            }
                            count += 1
                    except:
                        print('\nskip with json error!')
                        print(json_path)

    with open(f'/home/hdd1/wanghaoran/sift/data/database/jsons/{year}.json', 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)
    print('总子图数：', count)


def make_database_json_2023():
    year = '2023'
    base_path = f'/home/hdd1/data/wanfang/{year}/IFD{year}'
    trg_base_path = f'/home/hdd1/wanghaoran/sift/data/database/jsons/2023'
    new_format_dirs = ['0110', '0210', '0310', '0328', '0425', '0510', '0525', '0615', '0625', '0710', '0810', '0825', '0910']
    count = 0
    for batch_name in new_format_dirs:
        src_base_path = os.path.join(base_path, batch_name)
        count += make_batch_database_json(src_base_path, trg_base_path, batch_name)
    print('总子图数：', count)


def make_batch_database_json(src_base_path, trg_base_path, batch_name):
    count = 0
    record = {}
    json_paths = glob(os.path.join(src_base_path, '*', '*', '*', '*.json'))
    for json_path in tqdm(json_paths):
        paper_name = json_path.split('/')[-2]
        image_name = json_path.split('/')[-1].split('.')[0]
        image_path = glob(os.path.join(os.path.dirname(json_path), image_name + '.*'))
        if len(image_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    json_content = json.load(f)
                    for i, box in enumerate(json_content['outputs']['object']):
                        record[image_name + '_' + str(i)] = {
                            'big_img_path': image_path[0],
                            'classification': box['name'],
                            'paper': paper_name,
                            'big_img': image_name,
                            'xmin': box['bndbox']['xmin'],
                            'ymin': box['bndbox']['ymin'],
                            'xmax': box['bndbox']['xmax'],
                            'ymax': box['bndbox']['ymax']
                        }
                        count += 1
                except:
                    print('\nskip with json error!')
                    print(json_path)
    with open(os.path.join(trg_base_path, f'{batch_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)
    os.system('chmod 777 ', )
    return count


def make_temp_database_json(src_base_path, trg_base_path, batch_name):
    count = 0
    record = {}
    json_paths = glob(os.path.join(src_base_path, '*.json'))
    for json_path in tqdm(json_paths):
        paper_name = json_path.split('/')[-2]
        image_name = '.'.join(json_path.split('/')[-1].split('.')[:-1])
        image_path = glob(os.path.join(os.path.dirname(json_path), image_name + '.*'))
        if len(image_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    json_content = json.load(f)
                    for i, box in enumerate(json_content['outputs']['object']):
                        record[image_name + '_' + str(i)] = {
                            'big_img_path': image_path[0],
                            'classification': box['name'],
                            'paper': paper_name,
                            'big_img': image_name,
                            'xmin': box['bndbox']['xmin'],
                            'ymin': box['bndbox']['ymin'],
                            'xmax': box['bndbox']['xmax'],
                            'ymax': box['bndbox']['ymax']
                        }
                        count += 1
                except:
                    print('\nskip with json error!')
                    print(json_path)
    with open(os.path.join(trg_base_path, f'{batch_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)
    os.system('chmod 777 ', )
    return count


if __name__ == '__main__':
    # make_database_json_2019_2020_2021('2021')
    # make_database_json_2022()
    # unzip_path = os.path.join('/home/hdd1/data/wanfang/2023', f'IFD2023/0925')
    # make_batch_database_json(unzip_path, '/home/hdd1/data/wanfang/database/jsons/2023', '0925')

    make_temp_database_json('/home/hdd1/data/wanfang/2023/IFD2023/1026_temp_2/0', '/home/hdd1/data/wanfang/2023/IFD2023/1026_temp_2', '1026_temp_2')


