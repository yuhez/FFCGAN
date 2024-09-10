""""
Read error files using pandas
"""
import pandas as pd

class Read_Error():
    def __init__(self,dpath=[]):
        self.dpath = dpath
        self.phasegan_name = 'phasegan.txt'
        self.cyclegan_name = 'cyclegan.txt'
        self.paired_name = 'paired.txt'
        self.pix2pix_name = 'pix2pix.txt'
        self.name_list = [self.phasegan_name,self.cyclegan_name,self.paired_name,self.pix2pix_name]
        if self.dpath:
            self.name_list = [self.dpath + '/' + name for name in self.name_list]
    def read_data(self,filename):
        df = pd.read_csv(filename,sep=' ',header=None)
        df.columns = ["L2", "DSSIM", "FRCM"]
        return df
    def get_error(self):
        phasegan,cyclegan,paired,pix2pix = [self.read_data(i) for i in self.name_list]
        return phasegan,cyclegan,paired,pix2pix

# Filtering by multiple conditionals
# df[(df.rain_octsep < 1000) & (df.outflow_octsep < 4000)] # Can't use the keyword 'and'
# Getting a row via a numerical index
# df.iloc[30]