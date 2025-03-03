import data_loader
import pandas as pd
def preprocess_raw(new_path='../raw'):
    """read raw data, remove NaN columns and save to given path

    Args:
        new_path (str, optional): path to save cleanned files. Defaults to 'raw'.
    """    
    path=data_loader.download()
    fake_raw = pd.read_csv(path+'/Fake.csv', keep_default_na=False) 
    fake_raw['text'] = fake_raw['text'].astype(str).str.strip() 
    fake_raw['text'] = fake_raw['text'].replace(r'^\s*$', float('nan'), regex=True)  #
    fake_raw.dropna(subset=["text"], inplace=True) 
    fake_raw.to_csv(new_path+'/Fake.csv', index=False)
    true_raw = pd.read_csv(path+'/True.csv', keep_default_na=False)
    true_raw['text'] = true_raw['text'].astype(str).str.strip()
    true_raw['text'] = true_raw['text'].replace(r'^\s*$', float('nan'), regex=True)
    true_raw.dropna(subset=["text"], inplace=True)
    true_raw.to_csv(new_path+'/True.csv', index=False)