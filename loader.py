import os
from typing import Tuple

import pandas as pd
import numpy as np

def load_data(root_dir: str, preprocessing=False):
    if not preprocessing:
        artists_path = os.path.join(root_dir, 'artists.csv')
        tracks_path = os.path.join(root_dir, 'tracks.csv')
    else:
        artists_path = os.path.join(root_dir, 'artists_clean.csv')
        tracks_path = os.path.join(root_dir, 'tracks_clean.csv')
        
    # read data
    df_artists = pd.read_csv(artists_path)
    df_tracks = pd.read_csv(tracks_path)
    
    # do preprocessing on the data if needed
    if not preprocessing:
        print('preprocessing begin')
        df_tracks, df_artists = _preprocess_data(df_tracks, df_artists)
    
    return df_tracks, df_artists

def _preprocess_data(df_tracks: pd.DataFrame, df_artists: pd.DataFrame):
    # drop unneccesery columns
    df_tracks.drop(['id', 'artists'], axis=1, inplace=True)
    
    # rename columns
    df_artists.rename({'popularity': 'artist_popularity', 'name': 'artist_name'}, axis=1, inplace=True)
    df_tracks.rename({'release_date': 'year', 'name': 'song_name'}, axis=1, inplace=True)
    
    # Remove duplicated from data
    tracks_rows = _remove_duplicates(df_tracks)
    artists_rows = _remove_duplicates(df_artists)
    
    if tracks_rows + artists_rows == 0:
        print('\n\033[1mInference:\033[0m The dataset doesn\'t have any duplicates')
    else:
        print(f'\n\033[1mInference:\033[0m Number of duplicates dropped/fixed from tracks ---> {tracks_rows}')
        print(f'\n\033[1mInference:\033[0m Number of duplicates dropped/fixed from artists ---> {artists_rows}')
    
    # Fill/Drop missing data
    df_tracks, df_artists = _fill_missing_data(df_tracks, df_artists)
    
    # Convert lists to individual items
    df_tracks, df_artists = _convert_lists(df_tracks, df_artists)
    
    # Convert release-date to year
    df_tracks.year = df_tracks.year.apply(lambda x: int(x.split('-')[0]))
    
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