"""Miscellaneous Utilities Module
Due to a dependency cycle, these were unable to be placed in helper routines,
and contain utilities for dealing with the json config files.

"""
import os
import json
def read_json(fname):
    """
    Function reads json string from a file, sanitizes it,
    and returns a python dictionary with the contents.

    Takes:
        fname: string, file path to json
    Returns:
        dic: dict, de-serialized json
    """
    with open(fname) as f:
        st = f.read()
    json_acceptable_string = st.replace("'", "\"")
    dic = json.loads(json_acceptable_string)
    return dic

root = os.path.dirname(os.path.abspath(__file__))

config = read_json(os.path.join(root, "config.json"))
