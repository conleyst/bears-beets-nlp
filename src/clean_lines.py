#!/usr/bin/env python

import pandas as pd
import json
import re
import itertools


# define functions to be used to clean raw json
def speaker_line_format(conversation):
    """Turn list of strings into a list of tuples of the form (character, character-line).

    Args:
        conversation: list of strings
    Returns a list of tuples containing character -- character-line pairs.
    """

    # if conversation is from deleted scene, it breaks the pattern we use
    # remove the deleted scene flag and following non-break space
    # if error, it means the conversation is empty
    try:
        if 'Deleted' in conversation[0]:
            conversation = conversation[2:]
    except IndexError:
        return []

    conv_str = ''.join(conversation)  # concat all strings
    conv_str = conv_str.replace('\n', '')  # remove newline character
    # regex looks for sequences of capitalized words w/ space between, all followed by :
    # remove first entry -- will be either empty or bracketed action describing scene
    split_list = re.split('([A-Z](?:[a-z]+\s[A-Z])*[a-z]+:)', conv_str)[1:]
    split_list = list(map(lambda s: s.replace(':', ''), split_list))  # remove colon from speaker
    split_list = list(map(lambda s: re.sub('(\[.*?\])', '', s),
                          split_list))  # remove bracketed descriptions of actions, leave empty string
    split_list = list(map(lambda s: s.strip(), split_list))  # remove white space padding
    split_list = list(map(lambda s: s.replace('  ', ' '), split_list))  # replace double space with single space

    assert len(split_list) % 2 == 0, "list does not seem to be in character, character-line format"
    pair_list = list(zip(*[iter(split_list)] * 2))  # turn list into a list of pairs
    pair_list = list(
        filter(lambda t: len(t[1]) > 0, pair_list))  # remove pairs with empty line (e.g. only action description)

    return pair_list


def get_season_ep(url):
    """Return a list containing the season and episode numbers from the URL.

    Args:
        url -- string containing the url in question
    Return a tuple of integers
    """

    str_list = url[-8:-4].split('-')
    int_pair = tuple(map(lambda s: int(s), str_list))

    return int_pair


def create_df(scraped_json):
    """Return a pandas df with rows corresponding to a spoken line and rows (character, line, season, episode)."""

    char_line_pairs = list(map(lambda x: speaker_line_format(x['conversation']),
                               scraped_json))  # list of lists return by speaker_line_format
    all_info = list(map(lambda x: get_season_ep(x['url']),
                        scraped_json))  # (season, ep) at entry i corresponds to entry i of char_line_pairs
    zipped = zip(char_line_pairs, all_info)
    row_lists = [[line + pair[1] for line in pair[0]] for pair in
                 zipped]  # concat (season, ep) at entry i to every pair in entry i of char_line_pairs
    rows = list(
        itertools.chain.from_iterable(row_lists))  # concat all lists in row_lists to get single list of rows for df

    col_names = ['character', 'line', 'season', 'episode']
    df = pd.DataFrame(rows, columns=col_names)

    return df


if __name__ == '__main__':

    # import raw json
    with open('../data/raw_lines.json', 'r') as f:
        lines = json.load(f)

    # create df
    script_df = create_df(lines)
    script_df = script_df.sort_values(['season', 'episode'])

    # export to csv
    script_df.to_csv("../data/lines.csv", index=False)
