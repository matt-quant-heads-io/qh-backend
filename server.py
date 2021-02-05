import sys,os
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)

from config.config import Config
from library.feeds.polygon_feed import PolygonFeed

import argparse
import datetime as dt
from typing import List
import time
import ast
import pandas as pd


DATA_DICT = {'datetime': [], 'symbol': [], 's': [], 'c': [], 'i': [], 'x': [], 'r': []}


def process_message(message):
    message_dict = ast.literal_eval(message)[0]
    message_dict['t'] /= 1000
    message_dict['t'] = dt.datetime.utcfromtimestamp(message_dict['t']).strftime('%Y-%m-%d %H:%M:%S')
    DATA_DICT['datetime'].append(message_dict['t'])
    DATA_DICT['symbol'].append(message_dict['pair'])
    DATA_DICT['price'].append(message_dict['p'])
    DATA_DICT['s'].append(message_dict['s'])
    DATA_DICT['c'].append(message_dict['c'])
    DATA_DICT['i'].append(message_dict['i'])
    DATA_DICT['x'].append(message_dict['x'])
    DATA_DICT['r'].append(message_dict['r'])
    print(DATA_DICT)
    return DATA_DICT


def process_message(messages: List[str]):
    """
        Custom processing function for incoming streaming messages.
    """
    def add_message_to_list(message):
        """
            Simple function that parses dict objects from incoming message.
        """
        messages.append(ast.literal_eval(message))

    return add_message_to_list


parser = argparse.ArgumentParser(description='Price Feed client (from polyon.io feed) and server (for UI)')
parser.add_argument('--product_type', type=str, default='crypto')
parser.add_argument('--symbol', type=str, default='XT.BTC-USD')


def main():
    config = Config()
    args = parser.parse_args()
    product_type = args.product_type
    symbol = args.symbol
    api_key = config["api_creds"]["polygon"]["api_key"]

    if not api_key:
        msg = 'api_key cannot be None type'
        raise Exception(msg)
    else:
        while True:
            messages = []
            poly_feed = PolygonFeed(cluster=product_type, auth_key=api_key, process_message=process_message(messages))
            poly_feed.run_async()
            poly_feed.subscribe(symbol)

            time.sleep(0.5)
            poly_feed.close_connection()
            df = pd.DataFrame(messages)
            # print(f"df is {df.head()}")
            df = df.iloc[5:, 0].to_frame()
            print(f"df is {df.head()}")
            df.columns = ["data"]
            df["data"] = df["data"].astype("str")

            df = pd.json_normalize(df["data"].apply(lambda x: dict(eval(x))))
            dt_stamps = []

            for dt_stamp in df['t'].values:
                dt_val = dt_stamp / 1000
                dt_stamps.append(dt.datetime.utcfromtimestamp(dt_val).strftime('%Y-%m-%d %H:%M:%S'))

            df['t'] = dt_stamps
            df.set_index('t', inplace=True)
            df.drop('r', axis=1, inplace=True)
            print(df.head())

            # export data to sqlite
            # with sqlite3.connect("realtime_crypto.sqlite") as conn:
            #     df.to_sql("data", con=conn, if_exists="append", index=False)


if __name__ == "__main__":
    main()


