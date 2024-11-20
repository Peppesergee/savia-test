import sys
import os


class Logger:
    def __init__(self, logfile="logs.txt", logfolder="./logs"):
        if not os.path.exists(logfolder):
            os.mkdir(logfolder)

        self.stdout = sys.stdout
        self.log = open(os.path.join(logfolder, logfile), "w")

    def write(self, text):
        self.stdout.write(text)
        self.log.write(text)

    #        self.log.flush()

    def close(self):
        self.stdout.close()
        self.log.close()

    def flush(self):
        pass
