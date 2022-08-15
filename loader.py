import os
from typing import Tuple
from random import choice
import random

import pandas as pd
import numpy as np

def load_data(root_dir: str, load_data=False):
    if not load_data:
        artists_path = os.path.join(root_dir, 'artists.csv')
        tracks_path = os.path.join(root_dir, 'tracks.csv')
    else:
        artists_path = os.path.join(root_dir, 'artists_clean.csv')
        tracks_path = os.path.join(root_dir, 'tracks_clean.csv')
        
    # read data
    df_artists = pd.read_csv(artists_path)
    df_tracks = pd.read_csv(tracks_path)
    
    # do preprocessing on the data if needed
    if not load_data:
        print('preprocessing begin')
        df_tracks, df_artists = _preprocess_data(df_tracks, df_artists)
    
    return df_tracks, df_artists

def _preprocess_data(df_tracks: pd.DataFrame, df_artists: pd.DataFrame):    
    # drop unneccesery columns
    print('Dropping unneccesery columns...')
    df_tracks.drop(['id', 'artists'], axis=1, inplace=True)
    
    # rename columns
    print('Renaming columns...')
    df_artists.rename({'popularity': 'artist_popularity', 'name': 'artist_name', 'id': 'id_artists'}, axis=1, inplace=True)
    df_tracks.rename({'release_date': 'year', 'name': 'song_name'}, axis=1, inplace=True)
    
    # Remove duplicated from data
    print('Removing duplicates...')
    tracks_rows = _remove_duplicates(df_tracks)
    artists_rows = _remove_duplicates(df_artists)
    
    if tracks_rows + artists_rows == 0:
        print('\n\033[1mInference:\033[0m The dataset doesn\'t have any duplicates')
    else:
        print(f'\n\033[1mInference:\033[0m Number of duplicates dropped/fixed from tracks ---> {tracks_rows}')
        print(f'\n\033[1mInference:\033[0m Number of duplicates dropped/fixed from artists ---> {artists_rows}')
    
    # Fill/Drop missing data
    print('Filling/Dropping missing data...')
    df_tracks, df_artists = _fill_missing_data(df_tracks, df_artists)
    
    # Convert lists to individual items
    print('Expanding lists...')
    df_tracks, df_artists = _convert_lists(df_tracks, df_artists)
    
    # Convert release-date to year
    df_tracks.year = df_tracks.year.apply(lambda x: int(x.split('-')[0]))
    
    # remove time signature of 0 (impossible)
    df.drop(df[df["time_signature"] == 0].index, axis=1, inplace=True)
    
    return df_tracks, df_artists
    
def _remove_duplicates(df):
    rows,cols = df.shape

    df.drop_duplicates(inplace=True)

    if df.shape==(rows,cols):
        return 0
    else:
        return rows-df.shape[0]
    
def _fill_missing_data(df_tracks, df_artists):
    df_tracks.dropna(inplace=True, subset=['song_name'])
    df_artists.genres = df_artists.genres.map(eval)
    df_artists.genres = df_artists.genres.map(lambda x: x if len(x) > 0 else 'N/A')
    
    return df_tracks, df_artists
    
def _convert_lists(df_tracks, df_artists):
    df_tracks.id_artists = df_tracks.id_artists.apply(eval)
    df_tracks = df_tracks.explode('id_artists')
    
    return df_tracks, df_artists

def convert_genres(df):
    def _get_genres(df):
        genlist = df['genres'].dropna().apply(eval).tolist()

        genset = set()
        for gen in genlist:
            if type(gen) == list:
                for g in gen:
                    genset.add(g.lower())
            else:
                genset.add(gen.lower)
        genres = sorted(genset)
        
        return genres
    
    def _replace_genall(input):
        input = eval(input)
        if len(input) > 2 or len(input) == 0:
            return np.nan
        else:
            return choice(input)
        
    random.seed(42)
    
    genres = _get_genres(df)
    
    df["genres"] = df["genres"].apply(replace_genall)
    df = df.dropna(subset = ['genres'])
    
    top_gen = small_df["genres"].value_counts()[2:102].index.tolist()
    top_gen.append("UNK")
    df["genres"] = df["genres"].apply(replace_genall)
    df["genres"].fillna("UNK", inplace = True)
    df = df[df["genres"].isin(top_gen)]
    
    return df