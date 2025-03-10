import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
def download()->str:
    """download dataset raw data from kaggle

    Returns:
        str: path of downloaded file
    """
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
    return path

def dataset_split(path='raw/',ratio=(0.7,0.15,0.15))->pd.DataFrame:
    """split raw csv files into train, validation and test sets

    Args:
        path (str, optional): path of raw files. Defaults to '../raw/'.
        ratio (tuple, optional): splitting ratio. Defaults to (0.7,0.15,0.15).

    Returns:
        pd.DataFrame: train, validation and test sets
    """    
    assert sum(ratio)==1.0 and len(ratio)==3, "ratio error"
    true_df = pd.read_csv(path+'Fake.csv')
    fake_df = pd.read_csv(path+'True.csv')
    true_df["label"] = 1
    fake_df["label"] = 0
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df, temp_df = train_test_split(df, test_size=ratio[2]+ratio[1], random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=ratio[2]/(ratio[2]+ratio[1]), random_state=42, stratify=temp_df["label"])
    return train_df,val_df,test_df