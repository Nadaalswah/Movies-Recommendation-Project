# Movies Recommendation System Project 🎬
This project presents a comprehensive movie recommendation system utilizing the MovieLens dataset. Leveraging advanced recommendation algorithms, the system offers personalized movie suggestions and computes item similarities. Additionally, it includes an interactive Streamlit dashboard that allows users to engage with and explore their personalized recommendations and discover similar movies effortlessly.
<br>

# User Interface Components
## User Page

*  User Selection: Dropdown to select users.
*  User History: Displays user interactions.
*  Top-N Recommendations: List of top-N items recommended for the user.
*  Navigation: Navigate through recommendations.

## Item Page

*  Item Selection: Dropdown to select items.
*  Item Profile: Displays item metadata.
*  Top-N Similar Items: List of top-N similar items.
*   Navigation: Navigate through similar items.

# Recommender Models
## User Recommender
### Models Used:
* Singular Value Decomposition (SVD)
* Neural Collaborative Filtering (NCF)
* LightFM (Final Model)
### Training Process:
The models were trained using the MovieLens dataset to predict user preferences.
Hyperparameter tuning was performed to optimize the performance of each model.
The final model, LightFM, was selected based on its performance and accuracy.

## Item Similarity
Calculation Method:
Item similarities are calculated on demand when an item is selected.
Cosine similarity is used to find the top-N similar items.

# Dataset
LINK: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
## Dataset Files
* ratings.csv: User ratings for movies.
* movies.csv: Metadata about the movies.
* links.csv: Identifiers that can be used to link to other sources of movie data.
* tags.csv: User-generated tags for movies.
  
# Contributors
1. Nada Abd Alfatah
2. Adel Mamoun
3. Heba Abdelhady
4. Nadin Ahmed
