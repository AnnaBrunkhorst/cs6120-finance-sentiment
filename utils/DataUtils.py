import json
import os
import re
from datetime import datetime, timedelta

import requests
import spacy as sp
from spacy.tokens import Doc


class DataUtils:
    def __init__(self):
        return

    """
    This function reads all json files from the given dir and returns a json array
    Parameters:
        dir `str`: directory to be scanned
    
        fname_filter `str`: Filter files and select only those json files that start with fname_filter
    """

    def read_json_files(dir: str, fname_filter=None) -> list:
        json_fnames = []
        for (_, _, fnames) in os.walk(dir):
            json_fnames.extend(fnames)
            break  # ensure only top level json files are read.

        filter_predicate = lambda name: name.startswith(fname_filter) and name.endswith(
            '.json') if fname_filter else lambda \
                name: name.startswith(filter)
        json_fnames = list(filter(filter_predicate, json_fnames))

        json_data = []
        for fname in json_fnames:
            with open(f'{dir}/{fname}', 'r') as json_file:
                json_data.append(json.load(json_file))
        return json_data

    """
    Returns a list of start and end timestamp for use with Alpha Vantage API
    Parameters:
        start_date: start date of timestamps list e.g. '20240801T0000'
        end_date: last date of timestamps 
    """

    def generate_timestamps(self, start_date, end_date, interval_days=1):
        DATETIME_FORMAT = '%Y%m%dT%H%M'

        start_datetime = datetime.strptime(start_date, DATETIME_FORMAT) + timedelta(minutes=1)
        end_datetime = datetime.strptime(start_date, DATETIME_FORMAT) + timedelta(days=1)
        timestamps = []  # a list of start and end timestamps (a tuple for each day)

        while end_datetime <= datetime.strptime(end_date, DATETIME_FORMAT):
            timestamps.append((start_datetime.strftime(DATETIME_FORMAT), end_datetime.strftime(DATETIME_FORMAT)))
            start_datetime += timedelta(days=1)
            end_datetime += timedelta(days=1)
        return timestamps

    """
    Downloads and return json responses from Alpha Vantage API. 
    Requires API key to be present in env file in the same directory
    Parameters:
        timestamps: a list of start and end timestamps e.g. ('20230101T0001', '20230102T0000') 
                 the time difference should be 1 day to work efficiently with API free tier
        returns a list of json objects
    """

    def download_data(self, timestamps: list, api_key: str) -> list:
        # Vantage news API Key

        json_arr = []
        # 30 days of financial news (FREE tier only allows 25 requests/day, hence, only 25 days of data is collected).
        for t in timestamps:
            time_from, time_to = t[0], t[1]

            # ticker = 'MSFT'
            url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&time_from={time_from}&time_to={time_to}&limit=1000&apikey={api_key}'
            r = requests.get(url, headers={
                'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.75 Safari/537.36",
                'Accept-Language': 'en-US,en;h=0.9'})
            if r.status_code == 200:
                print(f'{time_from} - {time_to}')
                data = r.json()
            else:
                print(f"Error: {r.status_code}")
                break

            json_arr.append(data)
        return json_arr

    def __clean_doc(self, doc: Doc, max_words) -> list[str]:
        # OPT-2: Remove stop words & lemmatize
        # txt = [token.lemma_ for token in doc if not token.is_stop]
        # OPT-2: Lemmatize but keep stopwords
        txt = [token.lemma_ for token in doc]
        return txt[:max_words]

    def clean_data(self, data, max_words=64, batch_size=5000):
        # Preprocess a list of documents
        nlp = sp.load('en_core_web_sm', disable=['ner', 'parser'])  # NER not required for this task

        brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in data)
        cleaned_data = [self.__clean_doc(doc, max_words)
                        for doc in nlp.pipe(brief_cleaning, batch_size=batch_size, n_process=-1)]
        return cleaned_data