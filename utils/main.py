import time
from typing import List
from library.feeds import PolygonFeed
import ast

def my_custom_process_message(messages: List[str]):
    """
        Custom processing function for incoming streaming messages.
    """
    def add_message_to_list(message):
        """
            Simple function that parses dict objects from incoming message.
        """
        messages.append(ast.literal_eval(message))

    return add_message_to_list


def my_custom_error_handler(ws, error):
    print("this is my custom error handler", error)


def my_custom_close_handler(ws):
    print("this is my custom close handler")


def main():
    key = 'RzpEOwfKImP4AQOyJhCBPfwpIl4N_iY5'
    messages=[]
    my_client = PolygonFeed("crypto", key, my_custom_process_message(messages))
    my_client.run_async()
    print("Hi")
    my_client.subscribe("XT.BTC-USD")
    print(messages)
    time.sleep(1)
    print("Hi2")
    my_client.close_connection()


if __name__ == "__main__":
    main()
