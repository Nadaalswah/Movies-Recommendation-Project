import pandas as pd
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

"""# Loading Data Sets"""

movies = pd.read_csv(r'ml-latest-small\movies.csv')
ratings = pd.read_csv(r'ml-latest-small\ratings.csv')

movies.head()

titles=movies['title'].unique()
len(titles)

ratings.head()

"""# Mergeing movies and ratings"""

movies_ratings = pd.merge( movies,ratings, on='movieId')
movies_ratings.head()

utility_matrix = movies_ratings.pivot_table(index='userId',columns='movieId',values='rating').fillna(0)
utility_matrix.head()

scaler=MinMaxScaler()
utility_matrix_scaled=scaler.fit_transform(utility_matrix)

item_item_similarity=cosine_similarity(utility_matrix_scaled.T)

similarity_df=pd.DataFrame(item_item_similarity,index=utility_matrix.columns,columns=utility_matrix.columns)
similarity_df.head()

similarity_df = pd.DataFrame(item_item_similarity, index=utility_matrix.columns, columns=utility_matrix.columns)
similarity_df.to_csv('similarity_df.csv')

# with open('similarity_df.pkl', 'wb') as f:
#     pickle.dump(similarity_df, f)

def recommender(movie , similarity_mat , movies_data , k):
  index=movies_data[movies_data['title']==movie].index[0]
  distances = sorted(enumerate(similarity_mat[index]),reverse=True,key = lambda x: x[1])
  recommended_movies = []
  for i in distances[:k]:
    recommended_movies.append(movies_data.iloc[i[0]].title)

  return recommended_movies

# test:
movie = 'Jurassic Park (1993)'
recommended_movies = recommender(movie, similarity_df.values, movies,10)

# Print recommended movies and their posters
for movie in recommended_movies:
    print(f"Recommended Movie: {movie}")
    print()


