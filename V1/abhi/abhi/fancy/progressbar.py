import time
import sys

class ProgressBar:
    
    def __init__(self, total_len):
        self.total_len = total_len

    def progress(self, iteration=0, text=""):
        if self.total_len >= iteration:
            time.sleep(0.01)
            barLength = 100
            percent = (round((iteration / self.total_len) * 100, 2))
            nb_bar_fill = int(round((barLength * percent) / 100))
            bar_fill = '█' * nb_bar_fill
            bar_empty = ' ' * (barLength - nb_bar_fill)
            sys.stdout.write("\r" + str(text) + " |{0}| {1}% ({2}/{3})".format(str(bar_fill + bar_empty), percent, iteration, self.total_len))
            if percent < 100:
                sys.stdout.flush()
            else:
                print("\n")