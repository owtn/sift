import os
year = 2023
batch = 1026
base_path = f'/home/hdd1/data/wanfang/{year}/'
res_path = os.path.join(base_path, f'sift_v4/{batch}')
log_file = os.path.join(res_path,'log.txt')
#print(log_file)
sub_set = [10,20,15,30]
p1 = sub_set[0]

p1 = [max(cord, 0) for cord in p1[:4]]
print(p1)