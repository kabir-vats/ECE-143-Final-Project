import kagglehub
def download()->str:
    """download dataset raw data from kaggle

    Returns:
        str: path of downloaded file
    """
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
    return path

