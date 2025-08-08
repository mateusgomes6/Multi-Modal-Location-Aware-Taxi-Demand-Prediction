import calendar
from os.path import isfile
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime
from config import max_count
from utils import get_all_lines, pad_sentences, build_vocab

# Constants
EVENTS_FILE = "data/barclays_events.csv"
PREPROCESSED_DESC_FILE = "data/barclays_events_description_preprocessed.csv"
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_TIME = "07:00 PM"

def validate_files() -> None:
    required_files = [EVENTS_FILE, PREPROCESSED_DESC_FILE]
    for file in required_files:
        if not isfile(file):
            raise FileNotFoundError(file)

def load_and_preprocess_data() -> Tuple[pd.DataFrame, List[str], Dict]:
    df = pd.read_csv(EVENTS_FILE, sep=", ")
    descriptions = get_all_lines(PREPROCESSED_DESC_FILE)[1:]  # Pula o cabeçalho
    
    padded_descriptions = pad_sentences(descriptions)
    vocabulary, vocabulary_inv, word_counts = build_vocab(padded_descriptions)
    
    print(len(vocabulary))
    return df, descriptions, vocabulary

def parse_event_time(row: pd.Series) -> Tuple[datetime, datetime]:
    month_abr, date_x, year = row["date"].split()
    month_num = list(calendar.month_abbr).index(month_abr)
    day = int(date_x)
    
    time_str = DEFAULT_TIME if "TBA" in row["time"] else row["time"]
    hour = int(time_str.split(":")[0]) + 12
    
    base_time = datetime(int(year), month_num, day, hour)
    return (
        base_time.strftime(TIME_FORMAT),
        base_time.replace(minute=1).strftime(TIME_FORMAT)
    )

def encode_description(description: str, vocabulary: Dict[str, int]) -> List[int]:
    padded_sentence = [0] * max_count
    words = description.split()
    for i in range(min(max_count, len(words))):
        padded_sentence[i] = vocabulary.get(words[i].strip(), 0)
    return padded_sentence

def get_all_events() -> Dict[str, Dict]:
    validate_files()
    df, descriptions, vocabulary = load_and_preprocess_data()
    
    events_info = {}
    for idx, row in df.iterrows():
        try:
            time_str1, time_str2 = parse_event_time(row)
            encoded_desc = encode_description(descriptions[idx], vocabulary)
            
            event_data = {
                "name": row["event_title"],
                "description": descriptions[idx],
                "padded_description": encoded_desc
            }
            
            events_info[time_str1] = event_data
            events_info[time_str2] = event_data
            
        except Exception as e:
            print(f"{idx}: {str(e)}")
            continue
    
    return events_info

if __name__ == '__main__':
    events = get_all_events()
    print(len(events))
    sample_key = next(iter(events))
    print(sample_key)
    print(events[sample_key])