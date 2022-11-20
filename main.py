import re
import warnings

from gensim.corpora import BleiCorpus
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt

from preprocessing import process_article

warnings.filterwarnings('ignore', category=DeprecationWarning)


def main():
    plot_models(data_directory=['data', 'new_data'],
                alpha=['symmetric', 'asymmetric'],
                num_topics=[5, 20, 50])


def plot_models(data_directory: list[str], alpha: list[str], num_topics: list[int]):
    """Plots data using models with parameters from cartesian product of all parameters used."""
    for directory in data_directory:
        for a in alpha:
            for num in num_topics:
                corpus = BleiCorpus(f'{directory}/ap.dat', f'{directory}/vocabulary.txt')
                model = LdaModel(corpus, id2word=corpus.id2word, alpha=a, num_topics=num)
                save_topic_frequency_plot(model, corpus, filename=f'img/{directory}/{a}_{num}.jpg')
                print(f'{directory=}, alpha={a}, num_topics={num}')


def save_topic_frequency_plot(model: LdaModel, corpus: BleiCorpus, filename: str):
    """Saves plot with topic frequency to filename (must include dir path)."""
    number_topics_used = [len(model[doc]) for doc in corpus]
    plt.hist(number_topics_used)
    plt.xlabel('number of topics')
    plt.ylabel('number of articles')
    plt.savefig(filename)
    plt.close()


def parse_articles(txt_path: str) -> list[str]:
    """Returns list of articles."""
    with open(txt_path, 'r') as f:
        articles = f.read()
    articles = re.findall(r'<TEXT>\n (.+)\n </TEXT>', articles)
    return articles


def create_new_data():
    """Create word frequency info dat file and vocabulary.txt
    using preprocessed articles.
    """
    articles = parse_articles('data/ap.txt')
    articles = [process_article(article) for article in articles]
    vocabulary = create_vocabulary(articles)
    create_dat(articles, vocabulary, 'new_data/ap.dat')
    save_vocabulary_to_txt(vocabulary, 'new_data/vocabulary.txt')


def create_vocabulary(articles: list[list[str]]):
    """articles — list of lists of preprocessed words.

    Returns dict with words from articles and their indices.
    """
    vocabulary = {}
    index = 0
    for article in articles:
        for word in article:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
    return vocabulary


def create_dat(articles: list[list[str]],
               vocabulary: dict[str, int],
               dat_path: str):
    """Creates dat file with words info from articles.

    Each line contains number of unique words from vocabulary used in article
    and word indices with frequency of those words in article in a form of
    "index:frequency" separated by space.
    """
    with open(dat_path, 'w') as f:
        for article in articles:
            article_words = create_word_frequency_dict(article, vocabulary)
            f.write(f'{format_word_frequency(article_words)}\n')


def create_word_frequency_dict(article: list[str],
                               vocabulary: dict[str, int]) -> dict[int, int]:
    """Returns dict of word frequencies in article
    where key is word index from vocabulary.
    """
    article_words = {}
    for word in article:
        if vocabulary[word] not in article_words:
            article_words[vocabulary[word]] = 1
        else:
            article_words[vocabulary[word]] += 1
    return article_words


def format_word_frequency(article_words: dict[int, int]) -> str:
    """Returns formatted word frequency info in article."""
    info = f'{len(article_words)}'
    for word, frequency in article_words.items():
        info = f'{info} {word}:{frequency}'
    return info


def save_vocabulary_to_txt(vocabulary: dict[str, int], txt_path: str):
    """Writes words from vocabulary in txt file. One word per line."""
    with open(txt_path, 'w') as f:
        for word in vocabulary:
            f.write(f'{word}\n')


if __name__ == '__main__':
    main()
    # create_new_data()
