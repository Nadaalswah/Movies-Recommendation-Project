import streamlit as st
import pandas as pd
import requests

# Streamlit app configuration
st.set_page_config(layout="wide", 
                   page_title="Movie Recommendation System", 
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
def load_popular(movies_ratings, k) -> pd.DataFrame:
    movies_ratings_count = pd.DataFrame(movies_ratings.groupby(['title', 'movieId'])['rating'].mean())

    movies_ratings_count['rating_count'] = pd.DataFrame(movies_ratings.groupby(['title', 'movieId'])['rating'].count())
    movies_ratings_count['rating'] = round(movies_ratings_count['rating'], 1)

    # movies_ratings_count = 
    movies_ratings_count = (movies_ratings_count[(movies_ratings_count['rating'] > 3.5) & 
                        (movies_ratings_count['rating_count'] > 100)]
                        .sort_values('rating', ascending=False))

    movies_ratings_count = movies_ratings_count.reset_index()

    return movies_ratings_count.head(k)

movies, ratings, tags, links, movies_ratings, users, movie_titles = load_data()

######################################### Processing #########################################
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
st.title(":clapper: Movie recommendation System :clapper:")
st.subheader("Popular Movies")
topK = 10

recommendations_ids = load_popular(movies_ratings, topK)['movieId'].values

nCols = 5
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
                # st.markdown(f"<h5 style='text-align: center; color: white;'>{movie_title}</h5>", unsafe_allow_html=True)

                # st.image(poster_url, width=150)
                st.markdown(f"""<img src='{poster_url}' width='200' style='display: block; 
                            margin: 0 auto; margin-bottom: 25px'>""" , unsafe_allow_html=True)
                st.write(movie_title)  # Display movie title above the poster
######################################### Explorer #########################################
st.subheader("Movie Explorer")
form1 = st.form(key='options')

col1, col2 = form1.columns(2, gap= 'medium')

movieTitle = form1.selectbox("Select a Movie", movie_titles)

btn = form1.form_submit_button('Recommend:popcorn:!', use_container_width=True, type='primary',)

if btn:
    movie_id = movies[movies['title']==movieTitle]['movieId'].values[0]
    movie_data = get_movie_details(movie_id, movies, ratings, links)

    col1, col2 = st.columns(2)
    with col1:
        st.image(movie_data['poster_url'])

    with col2:
        # st.subheader(movie_data['title'], divider=False)
        st.subheader(movie_data['title'], divider=True)
        st.write(f"Genre: {movie_data['genres']}")
        st.write(f"Overview:")
        st.caption(f"{movie_data['overview']}")

        st.write(f"Release Date: {movie_data['release_date']}")

        col21, col22 = col2.columns(2)
        with col21:
            st.write(f"Rating: {movie_data['avg_rating']}/10")

        with col22:
            st.write(f"No. Ratings: {movie_data['num_ratings']}")
