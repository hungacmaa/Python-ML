# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:01:29 2022

@author: HungNguyen
"""
import pandas as pd
import sqlite3
# triplet_dataset = pd.read_csv(filepath_or_buffer='train_triplets.txt', nrows=10000,sep='\t', header=None, names=['user','song','play_count'])
# print(triplet_dataset)
# output_dict = {}
# with open('train_triplets.txt') as f:
#     for line_number, line in enumerate(f):
#         user = line.split('\t')[0]
#         play_count = int(line.split('\t')[2])
#         if user in output_dict:
#             play_count +=output_dict[user]
#             output_dict.update({user:play_count})
#         output_dict.update({user:play_count})
# output_list = [{'user':k,'play_count':v} for k,v in output_dict.items()]
# play_count_df = pd.DataFrame(output_list)
# play_count_df = play_count_df.sort_values(by = 'play_count', ascending = False)
# play_count_df.to_csv(path_or_buf='user_playcount_df.csv', index = False)
play_count_df = pd.read_csv('user_playcount_df.csv')
# print(play_count_df.shape)
# output_dict = {}
# with open('train_triplets.txt') as f:
#     for line_number, line in enumerate(f):
#         song = line.split('\t')[1]
#         play_count = int(line.split('\t')[2])
#         if song in output_dict:
#             play_count +=output_dict[song]
#             output_dict.update({song:play_count})
#         output_dict.update({song:play_count})
# output_list = [{'song':k,'play_count':v} for k,v in output_dict.items()]
# song_count_df = pd.DataFrame(output_list)
# song_count_df = song_count_df.sort_values(by = 'play_count', ascending = False)
# song_count_df.to_csv(path_or_buf='song_playcount_df.csv', index = False)

song_count_df = pd.read_csv("song_playcount_df.csv")

# total_play_count = sum(song_count_df["play_count"])
# a = (float(play_count_df.head(n=100000).play_count.sum())/total_play_count)*100
play_count_subset = play_count_df.head(n=100000)
user_subset = play_count_subset['user']

song_playcount_subset = song_count_df.head(30000)
song_subset = song_playcount_subset['song']

# triplet_dataset = pd.read_csv(filepath_or_buffer='train_triplets.txt',sep='\t', header=None, names=['user','song','play_count'])
# triplet_dataset_sub = triplet_dataset[triplet_dataset.user.isin(user_subset)]
# del(triplet_dataset)
# triplet_dataset_sub_song = triplet_dataset_sub[triplet_dataset_sub.song.isin(song_subset)]
# del(triplet_dataset_sub)
# triplet_dataset_sub_song.to_csv(path_or_buf='triplet_dataset_sub_song. csv', index = False)

triplet_dataset_sub_song = pd.read_csv('triplet_dataset_sub_song. csv')

# conn = sqlite3.connect('track_metadata.db')
# track_metadata_df_sub = pd.read_sql_query("select * from songs", conn)
# conn.close()

# merge data
# del(track_metadata_df_sub['track_id'])
# del(track_metadata_df_sub['artist_mbid'])
# track_metadata_df_sub = track_metadata_df_sub.drop_duplicates(['song_id'])
# triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song, track_metadata_df_sub, how='left', left_on='song', right_on='song_id')
# triplet_dataset_sub_song_merged.rename(columns={'play_count':'listen_count'},inplace=True)
# del(triplet_dataset_sub_song_merged['song_id'])
# del(triplet_dataset_sub_song_merged['artist_id'])
# del(triplet_dataset_sub_song_merged['duration'])
# del(triplet_dataset_sub_song_merged['artist_familiarity'])
# del(triplet_dataset_sub_song_merged['artist_hotttnesss'])
# del(triplet_dataset_sub_song_merged['track_7digitalid'])
# del(triplet_dataset_sub_song_merged['shs_perf'])
# del(triplet_dataset_sub_song_merged['shs_work'])

# triplet_dataset_sub_song_merged.to_csv(path_or_buf='triplet_dataset_sub_song_merged.csv', index = False)

triplet_dataset_sub_song_merged = pd.read_csv("triplet_dataset_sub_song_merged.csv")

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

popular_songs = triplet_dataset_sub_song_merged[['title','listen_count']].groupby('title').sum().reset_index()
popular_songs_top_20 = popular_songs.sort_values('listen_count', ascending=False).head(n=20)
objects = (list(popular_songs_top_20['title']))
y_pos = np.arange(len(objects))
performance = list(popular_songs_top_20['listen_count'])

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation='vertical')
plt.ylabel('Item count')
plt.title('Most popular songs')
plt.show()

popular_artists = triplet_dataset_sub_song_merged[["artist_name", "listen_count"]].groupby("artist_name").sum().reset_index()
popular_artists_top_20 = popular_artists.sort_values('listen_count', ascending=False).head(n=20)

x = list(popular_artists_top_20["artist_name"])
y = list(popular_artists_top_20["listen_count"])
pos = np.arange(20)

plt.bar(pos, y, align='center', alpha=0.5, color='green')
plt.xticks(pos, x, rotation='vertical')
plt.ylabel('Item count')
plt.title("Most popular artists")
plt.show()

user_song_count_distribution = triplet_dataset_sub_song_merged[['user','title']].groupby('user').count().reset_index().sort_values(by='title', ascending = False)
print(user_song_count_distribution.title.describe())

x = user_song_count_distribution.title
n, bins, patches = plt.hist(x, 50, facecolor='green', alpha=0.75)
plt.xlabel('Play Counts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ User\ Play\ Count\ Distribution}\ $')
plt.grid(True)
plt.show()

def create_popularity_recommendation(train_data, user_id, item_id):
    #Get a count of user_ids for each unique song as recommendation score
    train_data_grouped = train_data.groupby([item_id]).agg({user_id: 'count'}).reset_index() # danh sách bài hát và số lượng người nghe tương ứng
    train_data_grouped.rename(columns = {user_id: 'score'},inplace=True)
    #Sort the songs based upon recommendation score
    train_data_sort = train_data_grouped.sort_values(['score', item_id], ascending = [0,1])
    #Generate a recommendation rank based upon score
    train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first') #xếp hạng các bài hát theo số lượng người nghe giảm dần
    #Get the top 10 recommendations
    popularity_recommendations = train_data_sort.head(20) # lấy ra 20 bài đầu tiên
    return popularity_recommendations

recommendations = create_popularity_recommendation(triplet_dataset_sub_song_merged,'user','title')

song_count_subset = song_count_df.head(n=5000)
user_subset = list(play_count_subset.user)
song_subset = list(song_count_subset.song)
triplet_dataset_sub_song_merged_sub = triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.song.isin(song_subset)]
print(triplet_dataset_sub_song_merged_sub['title'].unique().shape)

from sklearn.model_selection import train_test_split
import Recommenders
train_data, test_data = train_test_split(triplet_dataset_sub_song_merged_sub, test_size =0.30, random_state=0)
is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user', 'title')
user_id = list(train_data.user)[1]
user_items = is_model.get_user_items(user_id)
xyz = is_model.recommend(user_id)

triplet_dataset_sub_song_merged_sum_df = triplet_dataset_sub_song_merged[['user','listen_count']].groupby('user').sum().reset_index()
triplet_dataset_sub_song_merged_sum_df.rename(columns={'listen_count':'total_listen_count'},inplace=True)
triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song_merged,triplet_dataset_sub_song_merged_sum_df)
triplet_dataset_sub_song_merged['fractional_play_count'] = triplet_dataset_sub_song_merged['listen_count']/triplet_dataset_sub_song_merged['total_listen_count']
my_db = triplet_dataset_sub_song_merged[['user', 'song', 'listen_count', 'total_listen_count', 'fractional_play_count']]


from scipy.sparse import coo_matrix
small_set = triplet_dataset_sub_song_merged
user_codes = small_set.user.drop_duplicates().reset_index()
song_codes = small_set.song.drop_duplicates().reset_index()
user_codes.rename(columns={'index':'user_index'}, inplace=True)
song_codes.rename(columns={'index':'song_index'}, inplace=True)
song_codes['so_index_value'] = list(song_codes.index)
user_codes['us_index_value'] = list(user_codes.index)
small_set = pd.merge(small_set,song_codes,how='left')
small_set = pd.merge(small_set,user_codes,how='left')
mat_candidate = small_set[['us_index_value','so_index_value','fractional_play_count']]
data_array = mat_candidate.fractional_play_count.values
row_array = mat_candidate.us_index_value.values
col_array = mat_candidate.so_index_value.values
data_sparse = coo_matrix((data_array, (row_array, col_array)),dtype=float)

import math as mt
from scipy.sparse.linalg import * #used for matrix multiplication
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix


def compute_svd(urm, K):
    U, s, Vt = svds(urm, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i,i] = mt.sqrt(s[i])

    U = csc_matrix(U, dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)
    
    return U, S, Vt

def compute_estimated_matrix(urm, U, S, Vt, uTest, K, test):
    rightTerm = S*Vt 
    max_recommendation = 250
    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    recomendRatings = np.zeros(shape=(MAX_UID,max_recommendation ), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :]*rightTerm
        estimatedRatings[userTest, :] = prod.todense()
        recomendRatings[userTest, :] = (-estimatedRatings[userTest, :]).argsort()[:max_recommendation]
    return recomendRatings

K=50
#Initialize a sample user rating matrix
urm = data_sparse
MAX_PID = urm.shape[1]
MAX_UID = urm.shape[0]

#Compute SVD of the input user ratings matrix
U, S, Vt = compute_svd(urm, K)
uTest = [27513]
#Get estimated rating for test user

print("Predicted ratings:")
uTest_recommended_items = compute_estimated_matrix(urm, U, S, Vt, uTest, K, True)
for user in uTest:
    print("Recommendation for user with user id {}". format(user))
    rank_value = 1
    for i in uTest_recommended_items[user,0:10]:
        song_details = small_set[small_set.so_index_value == i].drop_duplicates('so_index_value')[['title','artist_name']]
        print("The number {} recommended song is {} BY {}".format(rank_value, list(song_details['title'])[0],list(song_details['artist_name'])[0]))
        rank_value+=1