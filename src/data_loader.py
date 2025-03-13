import pandas as pd
from sklearn.model_selection import train_test_split

def download()->str:
    """download dataset raw data from kaggle

    Returns:
        str: path of downloaded file
    """
    import kagglehub # Lazy loaded because it can't import on some jupyter environments
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
    return path


def trim_string(x, max_len=200) -> str:
    """Trim string to max_len words

    Args:
        x (str): string to trim
        max_len (int, optional): maximum number of words to keep. Defaults to 200.
    
    Returns:
        str: trimmed string
    """
    x = x.split(maxsplit=max_len)
    x = ' '.join(x[:max_len])

    return x


def dataset_split(path='raw/',ratio=(0.7,0.15,0.15))->pd.DataFrame:
    """split raw csv files into train, validation and test sets

    Args:
        path (str, optional): path of raw files. Defaults to '../raw/'.
        ratio (tuple, optional): splitting ratio. Defaults to (0.7,0.15,0.15).

    Returns:
        tuple: Contains (train_df, val_df, test_df) as pandas DataFrames
    """
    assert sum(ratio)==1.0 and len(ratio)==3, "ratio error"
    true_df = pd.read_csv(path+'True.csv')
    fake_df = pd.read_csv(path+'Fake.csv')
    true_df["label"] = 1
    fake_df["label"] = 0
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df['titletext'] = df['title'] + ". " + df['text']
    df['text'] = df['text'].apply(trim_string)
    df['titletext'] = df['titletext'].apply(trim_string)    
    df = df.reindex(columns=['label', 'title', 'text', 'titletext'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df, temp_df = train_test_split(df, test_size=ratio[2]+ratio[1], random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=ratio[2]/(ratio[2]+ratio[1]), random_state=42, stratify=temp_df["label"])
    return train_df, val_df, test_df
