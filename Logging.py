import datetime
import csv
from operator import contains
import os


class Logger():
    def __init__(self):
        self.log_file_writer = None

    def get_current_time_timestamp(self):
        return str(datetime.datetime.now()).split('.')[0]


    def get_csv_writer(self):
        if self.log_file_writer is None:

            has_culomns = 'log.csv' in os.listdir(None)
                

            file = open('log.csv', 'a')
            self.log_file_writer = csv.writer(file)

            if not has_culomns:
                self.log_file_writer.writerow(['part', 'train_accuracy', 'test_accuracy',
                                                        'train_top5_accuracy', 'test_top5_accuracy',
                                                     'train_loss', 'test_loss', 'time'])

        return self.log_file_writer

    def log_model(self,part, history):
        train_accuracy, test_accuracy, train_top5_acc, test_top5_acc, train_loss, test_loss = history
        timestamp = str(datetime.datetime.now()).split('.')[0]
        writer = self.get_csv_writer()
        writer.writerow([part] + history + [timestamp])
        return timestamp
