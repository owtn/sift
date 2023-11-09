import requests
import os

def test_login():
    try:
        res = requests.get(url='https://www.baidu.com/', timeout=2)
    except requests.exceptions.ConnectTimeout as e:
        os.system('python3 /home/hdd1/share/auto-login/Login.py login zy2106250 whrg2032106250')
        res = requests.get(url='https://www.baidu.com/', timeout=2)
    code = res.status_code
    return code


if __name__ == '__main__':
    test_login()

