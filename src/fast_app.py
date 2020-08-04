import streamlit as st
import pandas as pd
import numpy as np
from stemming.porter2 import stem
import pathlib
import random
from lxml import etree
import requests
from hugs import NER, SUMMARIZER


etds = pathlib.Path("../../tmp/etds/")
etd_paths = [etd for etd in etds.iterdir()]
ns = { "mods": "http://www.loc.gov/mods/v3" }


@st.cache
def load_fast(csv_location: str)-> pd.DataFrame:
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
    

def summary(thesis):
    return SUMMARIZER(thesis.read_text())


st.title("Stanford Theses and Dissertations")
fast_load_state = st.sidebar.text("Loading FAST...")
fast_topics = load_fast('../data/topic_uri_label_utf8.csv')
fast_geo = load_fast('../data/geo_uri_label_utf8.csv')
fast_chronology = load_fast('../data/chron_uri_label_utf8.csv')
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
    st.subheader("Abstract")
    abstract = purl_xml.find("mods:mods/mods:abstract", namespaces=ns)
    st.write(abstract.text)
    st.subheader("Summary of Abstract")
#    st.write(summary(abstract.text)[0].get("summary_text))
    # st.write(summary(thesis))
    st.write(NER(thesis.read_text()))

st.sidebar.subheader("FAST Topics")
topic_input = st.sidebar.text_input("Enter topic terms")
if topic_input is not None:
    topic_output = search_fast(topic_input, fast_topics)
    st.sidebar.multiselect(options=topic_output, label='Select FAST Topics')
geo_input = st.sidebar.text_input("Enter geographic terms")
if geo_input is not None:
    geo_output = search_fast(topic_input, fast_geo)
    st.sidebar.multiselect(options=geo_output, label='Select FAST Geographic')
cron_input = st.sidebar.text_input("Enter time period")
if cron_input is not None:
    chron_output = search_fast(topic_input, fast_chronology)
    st.sidebar.multiselect(options=chron_output, label='Select FAST Chronology')
    # st.text(topic_output)

