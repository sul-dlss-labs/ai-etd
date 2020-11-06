__license__ = "Apache 2"

from zipfile import ZipFile

import numpy as np
import pandas as pd

fast_uri_list = []
for row in fast_bio['URI']:
    fast_uri_list.append(row)

def process_page(page_text: str, fast_uris) -> list:
    record = np.zeros(len(fast_bio))
    for uri in fast_uris:
        position = fast_uri_list.index(uri)
        record[position] = 1.
    record = list(record)
    record.insert(0, page_text)
    return record    

def generate_classes(druid: str) -> list:
    output = []
    found_fast = google_books_df.loc[google_books_df['druid'] == f"druid:{druid}"]
    for row in found_fast.iterrows():
        label = row[1]['fast_label']
        found_bio = fast_bio.loc[fast_bio['Label'] == label]
        if len(found_bio) > 0:
            output.append(found_bio['URI'].item())
        else:
            print(f"{label} not found in fast_bio for druid {druid}")
    return output

def batch(full_text: str) -> list:
    batches = []
    for i in range(0, len(full_text), 512):
        batches.append(full_text[i:i+512])
    return batches

def process_book(path: pathlib.Path, chunk: bool=True) -> list:
    druid = path.name
    fast_uris = generate_classes(druid)
    output = []
    zip_path = None
    for row in path.iterdir():
        if row.suffix.startswith(".zip"):
            zip_path = row
    with ZipFile(zip_path) as zip_file:
        for zip_info in zip_file.infolist():
            if zip_info.file_size < 1:
                continue
            with zip_file.open(zip_info) as zip_extract:
                full_text = zip_extract.read().decode()
                if len(full_text) < 2: # Removes any single pages ie just contines a new line character
                    continue
                if chunk is True:
                    chunks = batch(full_text)
                    for chunk_str in chunks:
                        record = [druid,]
                        record.extend(process_page(chunk_str, fast_uris))
                        output.append(record)
                else:
                   record = [druid,]
                   record.extend(process_page(full_text, fast_uris))
                   output.append(record)
    return output
