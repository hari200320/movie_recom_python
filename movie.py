import numpy as np
import pandas as pd

# Sample Data
def load_data():
    # Movie data
    movies_data = {
        'movie_id': [1, 2, 3, 4, 5],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E']
    }
    # Ratings data
    ratings_data = {
        'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        'movie_id': [1, 2, 1, 3, 2, 4, 3, 5, 4, 5],
        'rating': [5, 3, 4, 2, 5, 3, 4, 5, 2, 1]
    }
    return pd.DataFrame(movies_data), pd.DataFrame(ratings_data)

# Create a User-Movie Matrix
def create_user_movie_matrix(ratings):
    user_movie_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')
    return user_movie_matrix.fillna(0)  # Replace NaNs with 0s

# Compute Cosine Similarity
def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Compute Movie Similarities
def compute_movie_similarities(user_movie_matrix):
    num_movies = user_movie_matrix.shape[1]
    similarities = np.zeros((num_movies, num_movies))
    
    for i in range(num_movies):
        for j in range(num_movies):
            similarities[i, j] = cosine_similarity(
                user_movie_matrix.iloc[:, i],
                user_movie_matrix.iloc[:, j]
            )
    return similarities

# Predict Rating
def predict_rating(user_id, movie_id, user_movie_matrix, movie_similarities):
    # Get rated movies by the user
    rated_movies = user_movie_matrix.loc[user_id, user_movie_matrix.loc[user_id] > 0].index
    if len(rated_movies) == 0:
        return 0  # No rated movies to base prediction on
    
    # Get similarities between the movie to predict and rated movies
    similarities = movie_similarities[movie_id - 1, [movie - 1 for movie in rated_movies]]
    ratings = user_movie_matrix.loc[user_id, rated_movies]
    
    # Calculate the weighted average rating
    weighted_sum = np.dot(similarities, ratings)
    sum_of_similarities = np.sum(np.abs(similarities))
    
    if sum_of_similarities == 0:
        return 0  # Avoid division by zero
    
    return weighted_sum / sum_of_similarities

# Display movie titles and ratings
def display_movie_info(movie_id, movies_df):
    movie_title = movies_df[movies_df['movie_id'] == movie_id]['title'].values
    return movie_title[0] if len(movie_title) > 0 else "Unknown Movie"

# Main Execution
if __name__ == "__main__":
    # Load data
    movies, ratings = load_data()
    
    # Display the data tables
    print("Movies Data:")
    print(movies.to_string(index=False))  # Display movies data table
    
    print("\nRatings Data:")
    print(ratings.to_string(index=False))  # Display ratings data table
    
    # Create user-movie matrix
    user_movie_matrix = create_user_movie_matrix(ratings)
    
    # Compute movie similarities
    movie_similarities = compute_movie_similarities(user_movie_matrix)
    
    # Define a list of movies to predict ratings for
    movie_ids_to_predict = [1, 2, 3, 4, 5]  # IDs of the movies for which we want predictions
    user_id = 1  # Example user ID
    
    # Predict and display ratings for the list of movies
    print("\nPredicted Ratings:")
    for movie_id in movie_ids_to_predict:
        movie_title = display_movie_info(movie_id, movies)
        predicted_rating = predict_rating(user_id, movie_id, user_movie_matrix, movie_similarities)
        print(f"- {movie_title}: {predicted_rating:.2f}")
    
    # Example: Predict rating for a specific movie
    specific_movie_id = 3  # Movie ID to predict
    specific_movie_title = display_movie_info(specific_movie_id, movies)
    predicted_rating_for_specific_movie = predict_rating(user_id, specific_movie_id, user_movie_matrix, movie_similarities)
    
    print(f"\nPredicted rating for '{specific_movie_title}' by User {user_id}: {predicted_rating_for_specific_movie:.2f}")
