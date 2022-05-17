import pandas as pd
import numpy as np


class data:

    def __init__(self, path_to_csv=None, file_name=None):
        self.path_to_csv = path_to_csv
        self.file_name = file_name

    def download(self):
        frame = pd.read_csv(self.path_to_csv + self.file_name, header=None)
        # frame.columns = ['S'+str(i) for i in range(len(list(frame)))]
        # frame.index = np.arange(len(frame))

        return frame

    def upload(self, matrix, columns=None, name=None):
        frame = pd.DataFrame(matrix)

        if columns is not None:
            frame.columns = columns

        frame.to_csv(self.path_to_csv + name)

    def convert_datatime(self):
        #example covert ELECTRICITY dataset to year
        series_ELD = pd.read_csv('data/LD2011_2014.txt', sep=';', decimal=',')
        series_ELD.rename(columns={'Unnamed: 0': 'Datatime'}, inplace=True)
        series_ELD['Datetime'] = pd.to_datetime(series_ELD.Datatime, format='%Y-%m-%d %H:%M:%S')
        series_ELD['year'] = series_ELD.Datetime.dt.year
        series_ELD['month'] = series_ELD.Datetime.dt.month
        series_ELD['day'] = series_ELD.Datetime.dt.day
        series_ELD['Hour'] = series_ELD.Datetime.dt.hour
        ayear = series_ELD.groupby('year').sum()

        return ayear


