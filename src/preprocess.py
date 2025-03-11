import data_loader
import pandas as pd


def preprocess_raw(new_path='../raw'):
    """read raw data, remove NaN columns and drop duplicates from text OR title columns, then save to given path

    Args:
        new_path (str, optional): path to save cleaned files. Defaults to 'raw'.
    """
    path = data_loader.download()

    # Process Fake
    fake_raw = pd.read_csv(path+'/Fake.csv', keep_default_na=False)
    fake_raw['text'] = fake_raw['text'].astype(str).str.strip()

    fake_raw['text'] = fake_raw['text'].apply(lambda x: x[x.find(" - ")+3:] if isinstance(x, str) and x.find(" - ") != -1 and x.find(" - ") < 100 else x)
    
    fake_raw['text'] = fake_raw['text'].replace(r'^\s*$', float('nan'), regex=True)
    
    
    fake_raw.dropna(subset=["text"], inplace=True)

    fake_raw['title'] = fake_raw['title'].astype(str).str.strip()
    fake_raw['title'] = fake_raw['title'].replace(r'^\s*$', float('nan'), regex=True)

    fake_raw = fake_raw.drop_duplicates(subset=['text'])
    fake_raw = fake_raw.drop_duplicates(subset=['title'])

    fake_raw.to_csv(new_path+'/Fake.csv', index=False)

    # Process True
    true_raw = pd.read_csv(path+'/True.csv', keep_default_na=False)

    true_raw['text'] = true_raw['text'].astype(str).str.strip()

    true_raw['text'] = fake_raw['text'].apply(lambda x: x[x.find(" - ")+3:] if isinstance(x, str) and x.find(" - ") != -1 and x.find(" - ") < 100 else x)
    
    true_raw['text'] = true_raw['text'].replace(r'^\s*$', float('nan'), regex=True)
    true_raw.dropna(subset=["text"], inplace=True)

    true_raw['title'] = true_raw['title'].astype(str).str.strip()
    true_raw['title'] = true_raw['title'].replace(r'^\s*$', float('nan'), regex=True)

    true_raw = true_raw.drop_duplicates(subset=['text'])
    true_raw = true_raw.drop_duplicates(subset=['title'])

    true_raw.to_csv(new_path+'/True.csv', index=False)