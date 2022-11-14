from nltk import SnowballStemmer, WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords

stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()


def process_article(article: str) -> list[str]:
    """Returns list of stemmed words from article without stopwords."""
    words = tokenize_article(article)
    words = lemmatize_words(words)
    words = remove_stopwords(words)
    return stem_words(words)


def tokenize_article(article: str) -> list[str]:
    """Returns list of words from article."""
    words = word_tokenize(article)
    return [word for word in words if word.isalpha()]


def lemmatize_words(words: list[str]) -> list[str]:
    """Returns list of lemmatized words."""
    return [lemmatizer.lemmatize(word) for word in words]


def remove_stopwords(words: list[str]) -> list[str]:
    """Returns list of words filtered by stopwords."""
    return [word for word in words if word not in stopwords.words('english')]


def stem_words(words: list[str]) -> list[str]:
    """Returns list of stemmed words."""
    return [stemmer.stem(word) for word in words]
