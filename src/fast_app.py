__author__ = "Jeremy Nelson"
# """Proof-of-concept for active training of cataloging theses and dissertations"""

import argparse
import datetime
import numpy as np
import os
import pandas as pd
import pathlib
import random
import requests
import spacy
import streamlit as st
import sys

from lxml import etree
from hugs import SUMMARIZER
from stemming.porter2 import stem
from spacy import displacy
from spacy_lookup import Entity

parser = argparse.ArgumentParser(description='AI-ETD Cataloging Utility')
parser.add_argument('--etd', help='Path to full-text ETDs')
parser.add_argument('--data', default='../data', help='Path to data directory')

try:
    args = parser.parse_args()
except SystemExit as e:
    os._exit(e.code)

etds = pathlib.Path(args.etd)
etd_paths = [etd for etd in etds.iterdir()]
data = pathlib.Path(args.data)

ns = { "mods": "http://www.loc.gov/mods/v3" }



@st.cache
def load_fast_vocabs():
    fast_topics = load_fast(data/'topic_uri_label_utf8.csv')
    fast_geo = load_fast(data/'geo_uri_label_utf8.csv')    
    fast_chronology = load_fast(data/'chron_uri_label_utf8.csv')
    return fast_topics, fast_geo, fast_chronology


def load_fast(csv_location: str):
    fast_df = pd.read_csv(csv_location, names=['URI', 'Label'])
    fast_df['stemmed'] = fast_df['Label'].apply(lambda x: stem(x.lower()))
    return fast_df


def search_fast(term: str, vocab: pd.DataFrame) -> str:
    stemmed_term = stem(term.lower())
    options = []
    for row in vocab[vocab['stemmed'].str.contains(stemmed_term)].itertuples(index=True):
        options.append((row.Label))
        if len(options) >= 151:
            break
    return options


@st.cache
def generate_labels(fast_df):
    labels = {}
    for row in fast_df.iterrows():
        labels[row[1]['URI']] = [row[1]['Label'],]
    return labels


def fast_spacy(fast_df):
    entity = Entity(keywords_dict=generate_labels(fast_df), label="FAST_TOPIC")
    nlp = spacy.load('en_core_web_md')
    nlp.add_pipe(entity)
    nlp.remove_pipe("ner")
    return nlp, entity

def summary(thesis):
    return SUMMARIZER(thesis)


st.title("Stanford Theses and Dissertations")
fast_load_state = st.sidebar.text("Loading FAST...")
fast_topics, fast_geo, fast_chronology = load_fast_vocabs() 
spacy_ner, topic_entity = fast_spacy(fast_topics)
#spacy_ner.add_pipe(Entity(keywords_dict=generate_labels(fast_geo), label="FAST_GEOGRAPHIC"))
fast_load_state.text("Finished loading FAST")


if st.button('Random Thesis'):
    position = random.randint(0, len(etd_paths))
    thesis = etd_paths[position]
    purl_url = f"https://purl.stanford.edu/{thesis.name[:-4]}.xml"
    purl_result = requests.get(purl_url)
    purl_xml = etree.XML(purl_result.text.encode())
    title = purl_xml.find("mods:mods/mods:titleInfo/mods:title", namespaces=ns)
    st.subheader(f"{title.text}")
    st.write(f"Thesis Selected {position} Druid: {thesis.name}\n{purl_url}")
    names = purl_xml.findall("mods:mods/mods:name", namespaces=ns)
    for name in names:
        namePart = name.find("mods:namePart", namespaces=ns)
        role = name.find("mods:role/mods:roleTerm", namespaces=ns)
        if role is not None:
            st.write(f"\t{namePart.text}, {role.text}")
        else:
            st.write(f"\t{namePart.text}")
    st.subheader("Abstract")
    abstract = purl_xml.find("mods:mods/mods:abstract", namespaces=ns)
    st.write(abstract.text)
    st.subheader("BERT Summary")
    st.write(summary(abstract.text)[0]['summary_text'])
    st.subheader("Named Entities using spaCy")
    doc = spacy_ner(abstract.text)
    st.write(displacy.render(doc, style='ent'), unsafe_allow_html=True)
    # st.write(summary(thesis))
    # st.write(NER(thesis.read_text()))
    # for row in NER(abstract.text):
    #    if row['entity'].startswith("I-LOC"):
    #        st.subheader(f"FAST Geo results for {row['word']}")
    #        geo_etd = search_fast(row['word'], fast_geo)
    #        st.write(geo_etd)
    #    elif row['entity'].startswith("I-MISC") or  row['entity'].startswith("I-ORG"):
    #        st.subheader(f"FAST Topic results for {row['word']}, entity type {row['entity']}")
    #        topic_etd = search_fast(row['word'], fast_topics)
    #        st.write(topic_etd)
    #    else:
    #        st.write(row)

    st.sidebar.subheader("FAST Topics")
    entities = []
    for ent in doc.ents:
        if ent.text not in entities:
            entities.append(ent.text)
    for text in entities:
        st.sidebar.markdown(f"{text} {topic_entity.keyword_processor.get_keyword(text)}")
#topic_input = st.sidebar.text_input("Enter topic terms")
#if topic_input is not None:
#    topic_output = search_fast(topic_input, fast_topics)
#    st.sidebar.multiselect(options=topic_output, label='Select FAST Topics')
#geo_input = st.sidebar.text_input("Enter geographic terms")
#if geo_input is not None:
#    geo_output = search_fast(geo_input, fast_geo)
#    st.sidebar.multiselect(options=geo_output, label='Select FAST Geographic')
#cron_input = st.sidebar.text_input("Enter time period")
#if cron_input is not None:
#    chron_output = search_fast(topic_input, fast_chronology)
#    st.sidebar.multiselect(options=chron_output, label='Select FAST Chronology')
    # st.text(topic_output)
print(f"Started at {datetime.datetime.utcnow()}, now running in the terminal")
