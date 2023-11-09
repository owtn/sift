import os
import json

def read_json(filename, buffer_size=2048):
    with open(filename, 'r', encoding='utf-8') as f:
        buffer = f.read(buffer_size)
        while buffer:
            yield buffer
            buffer = f.read(buffer_size)

if __name__ == '__main__':
    json_path = '/home/hdd1/wanghaoran/sift/data/database/jsons/2022.json'
    buffer = ''
    file_count = 0
    total_count = 0
    item_count = 0
    file_dict = {}
    for seg in read_json(json_path, 500000):
        buffer += seg
        end_of_block = buffer.find('},')
        while end_of_block != -1:
            try:
                item = json.loads(buffer[:end_of_block+1] + '}')
            except:
                print(buffer[:end_of_block+1] + '}')
                raise Exception
            buffer = '{' + buffer[end_of_block+2:]
            item_count += 1
            file_dict = {**file_dict, **item}
            if item_count >= 100000:
                with open(f'/home/hdd1/wanghaoran/sift/data/database/jsons/2022/{file_count}.json', 'w') as f:
                    json.dump(file_dict, f, ensure_ascii=False)
                file_dict = {}
                total_count += item_count
                item_count = 0
                file_count += 1
            end_of_block = buffer.find('},')

