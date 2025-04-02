import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.title("Music Recommendation")
with open('dataset1.pkl', 'rb') as f:
    df = pickle.load(f)
with open('track (1).pkl', 'rb') as file:
    track = pickle.load(file)
song_names = df['track_name'].tolist()
selected_song = st.selectbox("Select a song", song_names)

with open('tracks1.pkl', 'rb') as f:
    track1= pickle.load(f)
with open('tracks2.pkl', 'rb') as f:
    track2 = pickle.load(f)

with open('scaled_data.pkl', 'rb') as f:
    df2 = pickle.load(f)
with open('rf_classifier2.pkl', 'rb') as f:
    rf_classifier2 = pickle.load(f)
index_number = df[df['track_name'] == selected_song].index[0]
selected_row = df2.iloc[index_number]


prediction = rf_classifier2.predict(selected_row.values.reshape(1, -1))
if st.button("Predict Playlist"):
    st.write("Predicted Playlist:","Feel Good" if prediction[0]==0 else "Moody")
song_vectorizer = CountVectorizer()
song_vectorizer.fit(track1['track_genre'].astype(str))
song_vectorizer = CountVectorizer()
song_vectorizer.fit(track2['track_genre'].astype(str))
def get_similarities(song_name, data):

  # Getting vector for the input song.
  text_array1 = song_vectorizer.transform(data[data['track_name']==song_name]['track_genre']).toarray()
  num_array1 = data[data['track_name']==song_name].select_dtypes(include=np.number).to_numpy()

  # We will store similarity for each row of the dataset.
  sim = []
  for idx, row in data.iterrows():
    name = row['track_name']

    # Getting vector for current song.
    text_array2 = song_vectorizer.transform(data[data['track_name']==name]['track_genre']).toarray()
    num_array2 = data[data['track_name']==name].select_dtypes(include=np.number).to_numpy()

    # Calculating similarities for text as well as numeric features
    text_sim = cosine_similarity(text_array1, text_array2)[0][0]
    num_sim = cosine_similarity(num_array1, num_array2)[0][0]
    sim.append(text_sim + num_sim)

  return sim
def recommend_songs1(song_name, data=track1):
  # Base case
  if track1[track1['track_name'] == song_name].shape[0] == 0:
    print('This song is either not so popular or you\
    have entered invalid_name.\n Some songs you may like:\n')

    for song in data.sample(n=5)['track_name'].values:
      print(song)
    return

  track1['similarity_factor'] = get_similarities(song_name, track1)

  data.sort_values(by=['similarity_factor', 'popularity'],
                   ascending = [False, False],
                   inplace=True)

  # First song will be the input song itself as the similarity will be highest.
  st.table(data[['track_name', 'artists']][2:7])
def recommend_songs2(song_name, data=track2):
  # Base case
  if track2[track2['track_name'] == song_name].shape[0] == 0:
    print('This song is either not so popular or you\
    have entered invalid_name.\n Some songs you may like:\n')

    for song in data.sample(n=5)['track_name'].values:
      print(song)
    return

  track2['similarity_factor'] = get_similarities(song_name, track2)

  data.sort_values(by=['similarity_factor', 'popularity'],
                   ascending = [False, False],
                   inplace=True)

  # First song will be the input song itself as theif st.button("Predict Playlist"):similarity will be highest.
  st.table(data[['track_name', 'artists']][2:7])
if st.button('Recommend'):
   song_name=selected_song
   st.write(recommend_songs1(song_name) if prediction[0]==0 else recommend_songs2(song_name))