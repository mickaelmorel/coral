mypath = "../sample_code"
from sys import argv, path
from os.path import abspath
path.append(abspath(mypath))
import seaborn as sns; sns.set()
import pandas as pd
import data_manager

class DataManager(data_manager.DataManager):
       
#    def __init__(self, basename="", input_dir=""):
#        ''' New contructor.'''
#        DataManager.__init__(self, basename, input_dir)
        # So something here
    
    def toDF(self, set_name):
        DF = pd.DataFrame(self.data['X_'+set_name])
        if set_name == 'train':
            Y = self.data['Y_train']
            DF = DF.assign(target=Y)          
        return DF

    def DataStats(self, set_name):
        ''' Display simple data statistics'''
        DF = self.toDF(set_name)
        print DF.describe()
    
    def ShowScatter(self, var1, var2, set_name):
        ''' Show scatter plots.'''
        DF = self.toDF(set_name)
        if set_name == 'train':
            sns.pairplot(DF.ix[:, [var1, var2, "target"]], hue="target")
        else:
            sns.pairplot(DF.ix[:, [var1, var2]])

    

if __name__=="__main__":
    if len(argv)==1:
        input_dir = "../public_data"
        output_dir = "../res"
    else:
        input_dir = argv[1]
        output_dir = argv[2];
        
    print("Using input_dir: " + input_dir)
    print("Using output_dir: " + output_dir)
    
    basename = 'crime'
    D = DataManager(basename, input_dir)
    print D
    
    D.DataStats('train')
    D.ShowScatter(1, 2, 'train')
