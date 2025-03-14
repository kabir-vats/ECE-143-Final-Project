import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
import spacy
import nltk
import string

from nltk.corpus import stopwords
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD, PCA, LatentDirichletAllocation
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from util import  RAW_DATA_DIR, VISUALIZATION_DIR

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class DataVisualizer:
    """
    A class for visualizing text-based datasets, particularly for fake news detection.
    It supports subject distribution, text length analysis, word frequency, 
    TF-IDF analysis, Word2Vec, and LDA topic modeling.
    """

    def __init__(self, is_save = True, raw_dir = RAW_DATA_DIR, output_dir=VISUALIZATION_DIR):
        """
        Initializes the DataVisualizer.

        Args:
            df (pd.DataFrame): The input DataFrame containing at least 'text', 'subject', and 'true/false' columns.
            output_dir (str): Directory to save generated plots.
        """
        self.is_save = is_save
        self.output_dir = output_dir
        self.df_true = pd.read_csv(raw_dir+'/Fake.csv')
        self.df_false = pd.read_csv(raw_dir+'/True.csv')

        self.df_false["label"] = False
        self.df_true["label"] = True
        self.df_false['title_length'] = self.df_false['title'].apply(lambda x : len(x.strip().split()))
        self.df_true['title_length'] = self.df_true['title'].apply(lambda x : len(x.strip().split()))
        self.df_false['text_length'] = self.df_false['text'].apply(lambda x: len(x.split()))
        self.df_true['text_length'] = self.df_true['text'].apply(lambda x: len(x.split()))
        self.df_false['date'] = pd.to_datetime(self.df_false['date'], errors='coerce')
        self.df_true['date'] = pd.to_datetime(self.df_true['date'], errors='coerce')
        self.df = pd.concat([self.df_true, self.df_false], ignore_index=True)
        

        for label in ["Overall", "Compare","True", "False"]:
            os.makedirs(os.path.join(self.output_dir, label), exist_ok=True)

        self.nlp = spacy.load('en_core_web_lg', disable=["parser", "ner", "textcat"])

    def save_plot(self, filename: str, label='Overall'):
        """
        Saves the current plot to the specified output directory.

        Args:
            filename (str): The filename to save the plot.
            label (str): The dataset type ("Overall", "True", or "False").
        """
        assert label in ["Overall","Compare", "True", "False"], "label must be 'Overall', 'True', or 'False'."
        
        filepath = os.path.join(self.output_dir, label, filename)
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved plot: {filepath}")
        plt.close()

    def plot_title_length_distribution(self):
        plt.figure(figsize=(12,6))
        sns.histplot(self.df_true['title_length'], 
                    kde=False, label='Fake', bins=50, color='blue', alpha=0.5)
        sns.histplot(self.df_false['title_length'], 
                    kde=False, label='True', bins=50, color='orange',alpha=0.5)
        plt.xlabel('Title Length')
        plt.title('Length of title comparison')
        if self.save_plot:
            self.save_plot("title_length_distribution.png", 'Compare')
        else:
            plt.show()
       
        

    def plot_subject_distribution(self):
        """Plots and saves the distribution of subjects separately for True, False, and Overall data."""
        for label, data in zip(["Overall", "True", "False"], [self.df, self.df_true, self.df_false]):
            plt.figure(figsize=(12, 6))
            sns.countplot(y=data['subject'], order=data['subject'].value_counts().index, palette='viridis')
            plt.xlabel("Count")
            plt.ylabel("Subject")
            plt.title(f"Distribution of Subjects ({label})")
            self.save_plot("subject_distribution.png", label)
        plt.figure(figsize=(12, 6))
        sns.countplot(y=self.df_true['subject'], order=self.df_true['subject'].value_counts().index,color='blue', alpha=0.5)
        sns.countplot(y=self.df_false['subject'], order=self.df_false['subject'].value_counts().index,color='orange', alpha=0.5)
        plt.xlabel("Count")
        plt.ylabel("Subject")
        plt.title(f"Distribution of Subjects (Compare)")
        if self.save_plot:
            self.save_plot("subject_distribution.png", 'Compare')
        else:
            plt.show()
    def plot_text_length_distribution(self):
        """Plots and saves the distribution of text lengths separately for True, False, and Overall data."""
        plt.figure(figsize=(12,6))
        sns.histplot(self.df_true['text_length'], 
                    kde=False, label='Fake', bins=50, color='blue', alpha=0.5)
        sns.histplot(self.df_false['text_length'], 
                    kde=False, label='True', bins=50, color='orange',alpha=0.5)
        plt.xlabel('Text Length')
        plt.title('Length of Text comparison')
        if self.save_plot:
            self.save_plot("Text_length_distribution.png", 'Compare')
        else:
            plt.show()

    def plot_word_frequency(self, top_n=20):
        """
        Plots and saves the most common words separately for True, False, and Overall data.

        Args:
            top_n (int): Number of top words to display.
        """
        assert isinstance(top_n, int) and top_n > 0, "top_n must be a positive integer."

        for label, data in zip(["Overall", "True", "False"], [self.df, self.df_true, self.df_false]):
            words = []
            for text in data['text']:
                words.extend(self.clean_text(text))

            word_freq = Counter(words)
            common_words = word_freq.most_common(top_n)

            df_freq = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Frequency', y='Word', data=df_freq, palette='viridis')
            plt.title(f"Top {top_n} Most Common Words ({label})")
            if self.save_plot:
                self.save_plot("word_frequency.png", label)
            else:
                plt.show()  

    def generate_wordcloud(self):
        """Generates and saves a word cloud visualization separately for True, False, and Overall data."""
        for label, data in zip(["Overall", "True", "False"], [self.df, self.df_true, self.df_false]):
            text = " ".join(data['text'])
            wordcloud = WordCloud(stopwords=stop_words, background_color="white", width=800, height=400).generate(text)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Word Cloud ({label})")
            if self.save_plot:
                self.save_plot("word_cloud.png", label)
            else:
                plt.show()

    def plot_time_series(self):
        """Plot the number of articles over time for True and Fake news in the same plot."""
        df_time = self.df.groupby([self.df['date'].dt.to_period('M'), 'label']).size().reset_index(name='count')
        # df_time.info()
        df_time["date"] = df_time["date"].dt.to_timestamp() 
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_time, x="date", y="count", hue="label", marker="o")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.title("Article Count Over Time (True vs. Fake)")
        if self.save_plot:
            self.save_plot("Time_series.png", 'Compare')
        else:
            plt.show()

    def plot_radar_chart(self):
        """
        Plot the radar chart
        """
        categories = ["Word Count", "Unique Word Count", "Sentence Count", "Avg Word Length", "Avg Sentence Length"]
        stats = {}

        for label, data in zip([ "True", "False"], [ self.df_true, self.df_false]):
            if data.empty:
                continue

            word_counts = data["text"].apply(lambda x: len(x.split()))
            unique_word_counts = data["text"].apply(lambda x: len(set(x.split())))
            sentence_counts = data["text"].apply(lambda x: x.count(".") + x.count("!") + x.count("?"))
            avg_word_length = data["text"].apply(lambda x: np.mean([len(word) for word in x.split()]) if x else 0)
            avg_sentence_length = word_counts / sentence_counts.replace(0, np.nan)

            stats[label] = [
                word_counts.mean(),
                unique_word_counts.mean(),
                sentence_counts.mean(),
                avg_word_length.mean(),
                avg_sentence_length.mean()
            ]

        max_values = np.max(list(stats.values()), axis=0)
        normalized_stats = {k: np.array(v) / max_values for k, v in stats.items()}

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1] 

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        color_dict = {'True':'blue','False':'orange',}

        for label, values in normalized_stats.items():
            values = values.tolist()
            values += values[:1]  
            ax.fill(angles, values, color=color_dict[label], alpha=0.25)
            ax.plot(angles, values, label=label, marker="o", linestyle="-")
            


        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.title("Text Feature Radar Chart")
        plt.legend()
        plt.savefig(os.path.join(VISUALIZATION_DIR, "Overall", "text_feature_radar_chart.png"))
        if self.save_plot:
            self.save_plot("Radar_chart.png", 'Compare')
        else:
            plt.show()

    def plot_tfidf_top_words(self, top_n=20):
        """Plots the top TF-IDF words."""
        vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
        X_tfidf = vectorizer.fit_transform(self.df['text'])
        feature_names = vectorizer.get_feature_names_out()
        scores = X_tfidf.toarray().sum(axis=0)

        df_tfidf = pd.DataFrame({'Word': feature_names, 'TF-IDF Score': scores})
        df_tfidf = df_tfidf.sort_values(by='TF-IDF Score', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(x='TF-IDF Score', y='Word', data=df_tfidf, palette='plasma')
        plt.title(f"Top {top_n} TF-IDF Words")
        if self.save_plot:
            self.save_plot("tfidf_top_words.png", 'Overall')
        else:
            plt.show()

    def train_word2vec(self):
        """Trains Word2Vec and visualizes vector representations."""
        sentences = [text.split() for text in self.df['text']]
        model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)

        words = list(model.wv.index_to_key)[:20]
        vectors = [model.wv[word] for word in words]

        plt.figure(figsize=(12, 6))
        sns.heatmap(vectors, annot=False, cmap='coolwarm')
        plt.yticks(ticks=range(len(words)), labels=words, rotation=0)
        plt.title("Word2Vec Vector Representations (Top Words)")
        if self.save_plots:
            self.save_plot("word2vec_vectors.png", 'Overall')
        else:
            plt.show()

    @staticmethod
    def clean_text(text):
        """
        Cleans text by removing punctuation and stopwords.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of cleaned words.
        """
        assert isinstance(text, str), "Input text must be a string."

        words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        return [word for word in words if word not in stop_words]
    

if __name__ == "__main__":
    

    visualizer = DataVisualizer()
    # visualizer.plot_subject_distribution()
    visualizer.plot_title_length_distribution()
    visualizer.plot_text_length_distribution()
    visualizer.plot_word_frequency()
    visualizer.generate_wordcloud()
    visualizer.plot_time_series()
    visualizer.plot_radar_chart()
    # visualizer.plot_tfidf_top_words()
    # visualizer.train_word2vec()