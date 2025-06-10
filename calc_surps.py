import pandas as pd
import os
import sys
import kenlm
import numpy as np
import nltk
nltk.download('punkt')
from minicons import scorer
import torch
from tqdm import tqdm
from nltk.tokenize import word_tokenize

def get_data():
    df = pd.read_csv('data/ia_Paragraph_ordinary.csv')
    keep_columns= ['participant_id','paragraph', 'IA_LABEL', 'IA_ID', 'IA_DWELL_TIME','IA_REGRESSION_PATH_DURATION','IA_FIRST_FIX_PROGRESSIVE','IA_FIRST_FIXATION_DURATION','IA_FIRST_RUN_DWELL_TIME', 'word_length', 'word_length_no_punctuation',
    'wordfreq_frequency','subtlex_frequency', 'gpt2_surprisal']
    df = df[keep_columns]
    return df

def calc_surps_trigram(df):
    MODEL_PATH = "/trigram_model/model.bin"

    model = kenlm.Model(MODEL_PATH)
    print(f"Created {MODEL_PATH}, size: {os.path.getsize(MODEL_PATH) / (1024 * 1024):.2f} MB")

    # Function to compute surprisal
    def get_surprisal(word, context_words):
        state = kenlm.State()
        next_state = kenlm.State()
        model.BeginSentenceWrite(state)
        for w in context_words:
            model.BaseScore(state, w, next_state)
            state, next_state = next_state, state
        log10_prob = model.BaseScore(state, word, next_state)
        surprisal = -log10_prob * np.log(10)
        return surprisal

    # Compute surprisal for each word
    surprisals = []
    for idx, row in df.iterrows():
        word = str(row['IA_LABEL']).lower()  # Adjust column name as needed
        # Get context: previous two words
        paragraph = row['paragraph'].split(" ")
        index = row['IA_ID']
        if index == 1:
            context_words = []
        elif index == 2:
            context_words = [paragraph[index - 2]]
        else:
            context_words = [paragraph[index - 3], paragraph[index - 2]]

        # Skip non-alphanumeric or handle as needed
        if not word.isalnum():
            surprisals.append(np.nan)
            continue

        surprisal = get_surprisal(word, context_words)
        surprisals.append(surprisal)

    # Add surprisal to dataframe
    df['trigram_surprisal'] = surprisals
    df = df.sort_values(by=['participant_id', 'paragraph', 'IA_ID'])
    output_path = "data/ia_Paragraph_with_trigram_surprisal.csv"
    df.to_csv(output_path, index=False)

def calc_surps_pythia(df):
    model = scorer.IncrementalLMScorer("EleutherAI/pythia-70m", device="cuda" if torch.cuda.is_available() else "cpu")
    df = df.sort_values(by=['paragraph', 'IA_ID'])
    surprisal_lookup = {}

    # Group unique paragraphs
    unique_paragraphs = df[['paragraph', 'IA_ID', 'IA_LABEL']].drop_duplicates(subset=['paragraph', 'IA_ID'])
    paragraph_map = unique_paragraphs.groupby('paragraph').apply(lambda x: [word for _, word in sorted(zip(x['IA_ID'], x['IA_LABEL']))]).to_dict()

    # Loop through each paragraph
    for para_text, word_list in tqdm(paragraph_map.items()):
        try:
            # Getting scores
            word_scores = model.word_score_tokenized(
                para_text,
                tokenize_function=lambda s: s.split(),
                surprisal=True,
                base_two=True
            )[0]

            if len(word_scores) == len(word_list):
                for i, (_, surprisal) in enumerate(word_scores):
                    surprisal_lookup[(para_text, i)] = surprisal
            else:
                print(f"Token mismatch in paragraph: {para_text[:50]}")
                for i in range(len(word_list)):
                    surprisal_lookup[(para_text, i)] = None

        except Exception as e:
            print(f"Error in paragraph: {para_text[:50]} → {e}")
            for i in range(len(word_list)):
                surprisal_lookup[(para_text, i)] = None

    df['pythia_surprisal'] = df.apply(lambda row: surprisal_lookup.get((row['paragraph'], row['IA_ID'])), axis=1)
    df = df.sort_values(by=['participant_id','paragraph', 'IA_ID'])
    df.to_csv("ia_Paragraph_with_pythia_surprisal.csv", index=False)

if __name__ == "__main__":
    df = get_data()
    calc_surps_trigram(df)
    calc_surps_pythia(df)