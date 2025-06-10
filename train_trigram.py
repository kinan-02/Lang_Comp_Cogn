import os
import requests
from datasets import load_dataset
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import re
import sys

def get_data():
    os.makedirs('wikipedia_trigram', exist_ok=True)
    os.chdir('wikipedia_trigram')
    os.system('apt-get update')
    os.system('apt-get install -y build-essential cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev')
    os.system('pip install wikiextractor kenlm nltk pandas numpy')

    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

        # Save to wiki_raw.txt
        print("Saving to wiki_raw.txt...")
        with open('wiki_raw.txt', 'w', encoding='utf-8') as f:
            for article in dataset:
                text = article['text'].strip()
                if text:
                    f.write(text + '\n')
    except Exception as e:
        print(f"Error with Hugging Face dataset: {e}")

    if not os.path.exists('wiki_raw.txt') or os.path.getsize('wiki_raw.txt') == 0:
        print("Error: wiki_raw.txt is missing or empty.")
        raise SystemExit(1)
    print(f"Created wiki_raw.txt, size: {os.path.getsize('wiki_raw.txt') / (1024 * 1024):.2f} MB")


def clean_text():
    with open('/wikipedia_trigram/wikipedia_trigram/wiki_raw.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Tokenize and lowercase
    tokens = [word.lower() for word in word_tokenize(text) if word.isalnum()]
    with open('wiki_clean.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(tokens) + '\n')

def train_model():
    clean_file = '/Users/kinanibrahim/PycharmProjects/PythonProject2/wikipedia_trigram/wikipedia_trigram/wiki_clean.txt'
    print(f"{clean_file} size: {os.path.getsize(clean_file) / (1024 * 1024):.2f} MB")

    # Train trigram model with verbose output
    print("Training trigram model with KenLM...")
    train_cmd = 'kenlm/build/bin/lmplz -o 3 --discount_fallback --verbose_header < /Users/kinanibrahim/PycharmProjects/PythonProject2/wikipedia_trigram/wikipedia_trigram/wiki_clean.txt > model.arpa'
    if os.system(train_cmd) != 0:
        print("Error: KenLM training failed. Check wiki_clean.txt or kenlm installation.")
        sys.exit(1)

    arpa_file = 'trigram_model/model.arpa'
    if not os.path.exists(arpa_file) or os.path.getsize(arpa_file) == 0:
        print(f"Error: {arpa_file} is missing or empty.")
        sys.exit(1)
    print(f"Created {arpa_file}, size: {os.path.getsize(arpa_file) / (1024 * 1024):.2f} MB")
    os.system(f'head -n 20 {arpa_file}')

    print("Converting to binary format...")
    bin_cmd = 'kenlm/build/bin/build_binary model.arpa model.bin'
    if os.system(bin_cmd) != 0:
        print("Error: Failed to convert to binary format.")
        sys.exit(1)

    bin_file = 'trigram_model/model.bin'
    if not os.path.exists(bin_file) or os.path.getsize(bin_file) == 0:
        print(f"Error: {bin_file} is missing or empty.")
        sys.exit(1)
    print(f"Created {bin_file}, size: {os.path.getsize(bin_file) / (1024 * 1024):.2f} MB")

def install_kenlm():
    import subprocess
    kenlm_path = "./kenlm"
    subprocess.run(f"rm -rf {kenlm_path}", shell=True)
    subprocess.run(f"git clone https://github.com/kpu/kenlm.git {kenlm_path}", shell=True)
    build_path = os.path.join(kenlm_path, "build")
    os.makedirs(build_path, exist_ok=True)
    subprocess.run(f"cd {build_path} && cmake .. && make -j4", shell=True)

if __name__ == '__main__':
    get_data()
    clean_text()
    install_kenlm()
    train_model()