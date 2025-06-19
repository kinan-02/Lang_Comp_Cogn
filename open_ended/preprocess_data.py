import spacy
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import random
import difflib
import re

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


nlp = spacy.load("en_core_web_sm")
def lemmatize_word(word):
    doc = nlp(word)
    return doc[0].lemma_

def get_polysemy_count(word):
    return len(wn.synsets(word))

def clean_token(token):
    return ' '.join(re.findall(r'\b[a-z]+\b', token.lower()))

def select_columns(df):
    selected_columns = ['DATA_FILE', 'IA_ID', 'IA_LABEL', 'IA_DWELL_TIME', 'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_FIX_PROGRESSIVE', 'IA_FIRST_RUN_DWELL_TIME',
                        'sentence', 'answered_correctly', 'WORD_NORM', 'FREQ_WEB', 'WORD_LEN', 'SURP_KENLM', 'SURP_LSTM', 'SURP_GPT2']
    df = df[selected_columns]
    return df

def harmonize_emotions(celer, emotions):
    def get_ratings(row):
        # Assign ratings back to celer
        return pd.Series(sentence_to_word_ratings[row['sentence']][row['WORD_NORM']])

    stop_words = set(stopwords.words('english'))
    # lemmatizer = WordNetLemmatizer()
    df2 = emotions.set_index('Word')

    sentence_to_word_ratings = {}
    for sentence in celer['sentence'].unique():
        # Filter rows corresponding to the current sentence
        sentence_rows = celer[celer['sentence'] == sentence]
        words = sentence_rows['WORD_NORM'].tolist()

        # Compute sentence-level average for fallback
        sentence_avg = []
        for word in words:
            lemma = word
            if lemma in df2.index:
                sentence_avg.append(df2.loc[lemma][['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']])
            else:
                if isinstance(word, str):
                    lemma = lemmatize_word(word)
                if lemma in df2.index:
                    sentence_avg.append(df2.loc[lemma][['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']])
        if sentence_avg:
            sentence_avg = pd.DataFrame(sentence_avg).mean().tolist()
        else:
            sentence_avg = [5.0, 5.0, 5.0]

        # Compute rating for each word
        word_to_rating = {}
        for word in words:
            if word in stop_words:
                word_to_rating[word] = [5.0, 5.0, 5.0]
            else:
                lemma = word
                if lemma in df2.index:
                    word_to_rating[word] = df2.loc[lemma][['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']].tolist()
                else:
                    if isinstance(word, str):
                        lemma = lemmatize_word(word)
                    if lemma in df2.index:
                        word_to_rating[word] = df2.loc[lemma][['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']].tolist()
                    else:
                        word_to_rating[word] = sentence_avg
        sentence_to_word_ratings[sentence] = word_to_rating
    celer[['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']] = celer.apply(get_ratings, axis=1)

def harmonize_orth_size(df, orth_size):
    def find_closest_word(word, candidates, n=1):
        # Function to get closest word(s) from the dictionary
        matches = difflib.get_close_matches(word, candidates, n=n, cutoff=0.0)
        return random.choice(matches) if matches else None

    orth_size['word'] = orth_size['word'].str.lower()
    orth_dict = orth_size.set_index('word')['orth_size'].to_dict()
    word_list = [w for w in orth_size['word'] if isinstance(w, str) and w != 'nan']

    def fill_missing_orth_size(row):
        # Fill missing using nearest neighbor
        orth = row['orth_size']
        word = str(row['WORD_NORM']).lower()

        if pd.isna(orth):
            if isinstance(word, str):
                word = clean_token(word)
                if len(word.split()) > 1:
                    return 0
                closest = find_closest_word(word, word_list)
                if closest:
                    return orth_dict[closest]
            return 0  # fallback if word is not valid or no close match
        else:
            return orth

    df['orth_size'] = df['WORD_NORM'].str.lower().map(orth_dict)
    df['orth_size'] = df.apply(fill_missing_orth_size, axis=1)

if __name__ == "__main__":
    df = pd.read_csv("../data/celer/sent_ia.tsv", sep='\t')
    df = select_columns(df)
    df = df.dropna(subset=['sentence', 'WORD_NORM'])
    emotions_df = pd.read_csv("../data/emotions_dataset/BRM-emot-submit.csv")
    harmonize_emotions(df, emotions_df)
    df.to_csv("../data/data_with_emotions.csv", index=False)

    df = pd.read_csv('../data/data_with_emotions.csv')
    orth_size = pd.read_csv("../data/Orthographic/englishCPdatabase2.txt", delimiter='\t', encoding='ISO-8859-1', header=None)
    orth_size = orth_size[[0, 2]]
    orth_size.columns = ['word', 'orth_size']
    harmonize_orth_size(df, orth_size)
    df.to_csv("../data/data_with_orth_size.csv", index=False)

    df = pd.read_csv('../data/data_with_orth_size.csv')
    # Harmonize Polysemy Count
    df['polysemy_count'] = df['WORD_NORM'].apply(get_polysemy_count)
    df.to_csv('../data/data_with_polysemy.csv', index=False)