import click
import datetime
from lxml import etree
import pathlib
import pandas as pd
import random
import re
import requests
from nltk.corpus import stopwords
from stemming.porter2 import stem
from transformers import pipeline

FAST_GEO = pd.read_csv("data/geo_uri_label_utf8.csv", names=["URI", "Label"])
FAST_TOPICS = pd.read_csv("data/topic_uri_label_utf8.csv",
                          names=["URI", "Label"])

for fast_df in [FAST_GEO, FAST_TOPICS]:
    fast_df["stemmed"] = fast_df["Label"].apply(lambda x: stem(x.lower()))

NER = pipeline("ner")
special_char_re = re.compile(r'[^a-zA-Z]')
stop_words_list = stopwords.words('english')

def cleanup(term: str) -> str:
    cleaned = []
    for char in term.split():
        cleaned_char = special_char_re.sub(' ', char).lower()
        if cleaned_char in stop_words_list:
            continue
        cleaned.append(cleaned_char)
    return ' '.join(cleaned)

def create_datasets(etd_path: str):
    etds = pathlib.Path(etd_path)
    etd_paths = [etd for etd in etds.iterdir()]
    # Using 70-15-15 ratio
    training, validation, testing = [], [], []
    for etd in etd_paths:
        cutoff = random.random()
        if cutoff <= 0.70:
            training.append(etd)
        elif cutoff <= 0.90:
            validation.append(etd)
        else:
            testing.append(etd)


def classify_abstract(abstract: str) -> list:
    """Classifies an ETD abstract using Named Entity Recognition (NER)

    param -- abstract
    """
    fast_entities = []
    for row in NER(abstract):
        if row["entity"].startswith("I-LOC"):
            fast_entities.extend(search_fast(row["word"], FAST_GEO))
        elif row["entity"] in ["I-MISC", "I-ORG"]:
            fast_entities.extend(search_fast(row["word"], FAST_TOPICS))
    return fast_entities


def get_abstract(druid: str) -> str:
    """Retrieves xml from Stanford PURL URI, parses, and returns the MODS
    abstract.

    param -- druid
    """
    purl_url = f"https://purl.stanford.edu/{druid}.xml"
    purl_result = requests.get(purl_url)
    purl_xml = etree.XML(purl_result.text.encode())
    abstract = purl_xml.find(
        "mods:mods/mods:abstract",
        namespaces={"mods": "http://www.loc.gov/mods/v3"}
    )
    return abstract.text


def search_fast(term: str, vocab: pd.DataFrame) -> list:
    stemmed_term = stem(term.lower())
    options = []
    for row in vocab[vocab["stemmed"].str.contains(stemmed_term)].itertuples(
        index=True
    ):
        options.append((row.URI, row.Label))
        if len(options) >= 151:
            break
    return options


if __name__ == "__main__":
    click.echo(f"AI-ETD Workflow {datetime.datetime.utcnow()}")
