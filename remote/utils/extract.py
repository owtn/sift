import re


def extract_path(html_file, url_file, logger=None):
    """
    从包含数据压缩包url的内容中提取url列表
    Args:
        html_file: 平台网页复制内容文件
        url_file: 生成的url目标文件路径
        logger: logger

    Returns:
        提取到的url数量
    """
    with open(html_file, encoding='utf-8') as f:
        data = f.read()
        s_http = re.findall("http://phocheck.wanfangdata.com.cn/temporary_file/download/testing/(.+?).zip", data)
        s_https = re.findall("https://phocheck.wanfangdata.com.cn/temporary_file/download/testing/(.+?).zip", data)
    s_http = list(set(s_http))
    s_https = list(set(s_https))
    s_http.sort()
    s_https.sort()
    s = ["http://phocheck.wanfangdata.com.cn/temporary_file/download/testing/" + it + ".zip" for it in s_http] + \
        ["https://phocheck.wanfangdata.com.cn/temporary_file/download/testing/" + it + ".zip" for it in s_https]
    with open(url_file, 'w', encoding='utf-8') as f:
        for url in s:
            f.write(url + '\n')
    if logger is not None:
        logger.log(f'提取到{len(s)}条压缩包url')
    return s


if __name__ == '__main__':
    s = extract_path("/home/hdd1/data/wanfang/2023/download/0310/html.txt",
                     '/home/hdd1/data/wanfang/2023/download/0310/url.txt')
    print(len(s))
