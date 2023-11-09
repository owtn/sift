import time


class Logger:
    def __init__(self, file_path):
        self.path = file_path

    def log(self, content):
        with open(self.path, 'a+') as f:
            f.write('\n' + time.asctime(time.localtime()) + '\n')
            f.write(content + '\n')
