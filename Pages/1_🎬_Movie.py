import streamlit as st
import pandas as pd
import requests
import os.path

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Streamlit app configuration
st.set_page_config(layout="wide", 
                   page_title="Movie Based Recommendation", 
                   page_icon="🎬")

###################################### Data Loading ######################################
@st.cache_resource()
def load_data() -> tuple[pd.DataFrame]:
    movies = pd.read_csv(r'ml-latest-small\movies.csv')
    ratings = pd.read_csv(r'ml-latest-small\ratings.csv')
    tags = pd.read_csv(r'ml-latest-small\tags.csv')
    links = pd.read_csv(r'ml-latest-small\links.csv')

    movies_ratings = pd.merge( movies,ratings, on='movieId')
    users = movies_ratings['userId'].values
    movie_titles = movies['title'].values

    return movies, ratings, tags, links, movies_ratings, users, movie_titles

@st.cache_resource()
def load_sim_mat(movies_ratings) -> pd.DataFrame:
    if os.path.exists("similarity_df.csv"):
        return pd.read_csv("similarity_df.csv").set_index('movieId')
    
    utility_matrix = movies_ratings.pivot_table(index='userId',columns='movieId',values='rating').fillna(0)

    scaler=MinMaxScaler()
    utility_matrix_scaled=scaler.fit_transform(utility_matrix)

    item_item_similarity = cosine_similarity(utility_matrix_scaled.T)

    similarity_df=pd.DataFrame(item_item_similarity,index=utility_matrix.columns,columns=utility_matrix.columns)

    similarity_df = pd.DataFrame(item_item_similarity, index=utility_matrix.columns, columns=utility_matrix.columns)
    similarity_df.to_csv('similarity_df.csv')
    return pd.read_csv("similarity_df.csv").set_index('movieId')

movies, ratings, tags, links, movies_ratings, users, movie_titles = load_data()
sim_mat = load_sim_mat(movies_ratings)

######################################### Processing #########################################
@st.cache_resource()
def recommenderMovie(movie , similarity_mat , movies_data , k) -> list:
  index=movies_data[movies_data['title']==movie].index[0]

  distances = sorted(enumerate(similarity_mat[index]),reverse=True,key = lambda x: x[1])

  recommended_movies = []
  for idx, _ in distances[1: k+1]:
    recommended_movies.append(movies_data.iloc[idx]['movieId'])

  return recommended_movies

@st.cache_resource()
def get_movie_details(movie_id, df_movies, df_ratings, df_links):
    try:
        # Get IMDb ID and TMDB ID for the movie
        imdb_id = df_links[df_links['movieId'] == movie_id]['imdbId'].values[0]
        tmdb_id = df_links[df_links['movieId'] == movie_id]['tmdbId'].values[0]

        # Get movie data from movies_df
        movie_data = df_movies[df_movies['movieId'] == movie_id].iloc[0]

        # Extract genres from the genres column (assuming genres are separated by '|')
        genres = movie_data['genres'].split('|') if 'genres' in movie_data else []

        # Fetch poster image URL using TMDB API
        api_key = 'b8c96e534866701532768a313b978c8b'   # Replace with your TMDB API key
        response = requests.get(f'https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}')

        response = response.json()
        poster_url = response.get('poster_path', '')
        full_poster_url = f'https://image.tmdb.org/t/p/w500{poster_url}' if poster_url else ''

        # Return movie details as a dictionary
        return {
            "title": response['original_title'],
            "genres": genres,
            "avg_rating": round(response['vote_average'], 1),
            "num_ratings": response['vote_count'],
            "imdb_id": imdb_id,
            "tmdb_id": tmdb_id,
            "poster_url": full_poster_url,
            "overview" : response['overview'],
            "release_date" : response['release_date']
        }
    except Exception as e:
        st.error(f"Error fetching details for movie ID {movie_id}: {e}")
    return None


######################################### Streamlit #########################################
st.title(":clapper: Movie Based Recommendation :clapper:")

form1 = st.form(key='options')

col1, col2 = form1.columns(2, gap= 'medium')

movieTitle = col1.selectbox("Select a Movie", movie_titles)
topK = col2.number_input('Number of Recommendations', min_value=1, max_value=20, value=10)

btn = form1.form_submit_button('Recommend:popcorn:!', use_container_width=True, type='primary',)
                        #  on_click=recommenderMovie, args=[movieTitle, sim_mat.values, movies, topK])

if btn:
    recommendations_ids = recommenderMovie(movieTitle, sim_mat.values, movies, topK)

    st.subheader(f"Top {topK} recommendations for the movie: {movieTitle}")

    nCols = 3 
    nRows = (topK // nCols) + (1 if topK % nCols > 0 else 0)

    for row in range(nRows):
        cols = st.columns(nCols)
        for col in range(nCols):
            index = row * nCols + col
            if index < topK:
                movie_id = recommendations_ids[index]
                movie_data = get_movie_details(movie_id, movies, ratings, links)
                if movie_data is None:
                    continue
                movie_title = movie_data.get('title')
                poster_url = movie_data.get('poster_url')

                with cols[col]:
                    # st.image(poster_url, width=150)
                    st.markdown(f"""<img src='{poster_url}' width='200' style='display: block; 
                                margin: 0 auto; margin-bottom: 25px'>""" , unsafe_allow_html=True)
                    
                    # st.write(movie_title)  # Display movie title above the poster
                    st.markdown(f"<h5 style='text-align: center; color: white;'>{movie_title}</h5>", unsafe_allow_html=True)