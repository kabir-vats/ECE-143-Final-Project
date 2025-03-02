import data_loader
import pandas as pd
def preprocess_raw(new_path='../raw'):
    """read raw data, remove NaN columns and save to given path

    Args:
        new_path (str, optional): path to save cleanned files. Defaults to 'raw'.
    """    
    path=data_loader.download()
    fake_raw=pd.read_csv(path+'/Fake.csv')
    fake_cleaned=fake_raw.dropna()
    fake_cleaned.to_csv(new_path+'/Fake.csv')
    true_raw=pd.read_csv(path+'/True.csv')
    true_cleaned=true_raw.dropna()
    true_cleaned.to_csv(new_path+'/True.csv')