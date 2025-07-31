import os
from datetime import datetime

class Logger():
    """
    这个Python函数定义了一个名为Logger的类，该类用于记录日志。
    在初始化函数__init__中，接收三个参数：
    filename表示日志文件名，
    is_debug表示是否启用调试模式，
    path表示日志文件保存路径，默认为指定路径。/home/dzh/Projects2
    在logging函数中，将输入的日志信息打印出来，并根据是否启用调试模式决定是否将日志写入文件。/data/dzh/Project2
    """
    def __init__(self, filename, is_debug, path='/home/dzh/Projects2/MHGR/MMSSL_9/logs/'):
        self.filename = filename
        self.path = path
        self.log_ = not is_debug
    def logging(self, s):
        s = str(s)
        print(datetime.now().strftime('%Y-%m-%d %H:%M: '), s)
        if self.log_:
            with open(os.path.join(os.path.join(self.path, self.filename)), 'a+') as f_log:
                f_log.write(str(datetime.now().strftime('%Y-%m-%d %H:%M:  ')) + s + '\n')
