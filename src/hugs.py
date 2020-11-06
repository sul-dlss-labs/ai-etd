from transformers import pipeline

SUMMARIZER = pipeline("summarization")
NER = pipeline("ner")
