
"""
Hybrid Recommender System
--------------------------

This project combines both user-based and item-based collaborative filtering
techniques to generate personalized movie recommendations using the MovieLens dataset.

Steps:
1. Data Preparation
2. User-Based Collaborative Filtering
3. Item-Based Collaborative Filtering
4. Final Hybrid Recommendation

"""

#############################################
# PROJECT: Hybrid Recommender System
#############################################

# For the user with the given ID, make predictions using both item-based and user-based recommender methods.
# Take 5 recommendations from the user-based model and 5 from the item-based model, and finally provide 10 recommendations from both models.

#############################################
# Task 1: Data Preparation
#############################################

import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 300)

# 1. Read the Movie and Rating datasets.
# Dataset containing movieId, movie title, and genre information
movie = pd.read_csv('/Users/betulcoklu/PycharmProjects/pythonProject/datasets/movie.csv')
movie.head()
movie.shape

# Dataset containing userID, movie title, rating, and timestamp information
rating = pd.read_csv('/Users/betulcoklu/PycharmProjects/pythonProject/datasets/rating.csv')
rating.head()
rating.shape
rating["userId"].nunique()


# 2. Add the movie names and genres to the rating dataset using the movie dataset.
# Only the movie IDs are present in the ratings; add movie names and genres from the movie dataset.
df = movie.merge(rating, how="left", on="movieId")
df.head()
df.shape


# 3. Calculate the total number of users who rated each movie. Remove movies with less than 1000 ratings from the dataset.
# Calculate the total number of users who rated each movie.
comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts

# Store the names of movies with less than 1000 ratings in rare_movies and remove them from the dataset.
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape


# 4. Create a pivot table where the index is userID, columns are movie titles, and values are ratings.

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()


# 5. Convert all the above steps into a function
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


#############################################
# Task 2: Determining the Movies Watched by the User to Make Recommendations
#############################################

# 1. Select a random user id.
random_user = 108170

# 2. Create a new dataframe called random_user_df consisting of the observations for the selected user.
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()
random_user_df.shape

# 3. Assign the movies rated by the selected user to a list called movies_watched.
movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list()
movies_watched

movie.columns[movie.notna().any()].to_list()

#############################################
# Task 3: Accessing the Data and IDs of Other Users Who Watched the Same Movies
#############################################

# 1. Select the columns of the movies watched by the selected user from user_movie_df and create a new dataframe called movies_watched_df.
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape

# 2. Create a new dataframe called user_movie_count that contains the number of movies watched by each user from the selected user's watched movies.
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head(5)

# 3. Users who have watched at least 60% of the movies rated by the selected user are considered similar users.
# Create a list called users_same_movies consisting of these user IDs.
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
len(users_same_movies)



#############################################
# Task 4: Determining the Most Similar Users to the User for Recommendation
#############################################

# 1. Filter the movies_watched_df dataframe to include only users in the users_same_movies list who are similar to the selected user.
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()
final_df.shape

# 2. Create a new dataframe corr_df containing the correlations between users.
corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

#corr_df[corr_df["user_id_1"] == random_user]



# 3. Filter users with high correlation (above 0.65) with the selected user and create a new dataframe called top_users.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape
top_users.head()

# 4. Merge the top_users dataframe with the rating dataset
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].unique()
top_users_ratings.head()



#############################################
# Task 5: Calculating the Weighted Average Recommendation Score and Keeping the Top 5 Movies
#############################################

# 1. Create a new variable called weighted_rating which is the product of each user's correlation and rating values.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

# 2. Create a new dataframe called recommendation_df containing the movie id and the mean of weighted ratings for each movie.
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# 3. Select the movies in recommendation_df with weighted rating greater than 3.5 and sort by weighted rating.
# Save the top 5 observations as movies_to_be_recommend.
recommendation_df[recommendation_df["weighted_rating"] > 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)


# 4. Get the names of the 5 recommended movies.
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]

# 0    Mystery Science Theater 3000: The Movie (1996)
# 1                               Natural, The (1984)
# 2                             Super Troopers (2001)
# 3                         Christmas Story, A (1983)
# 4                       Sonatine (Sonachine) (1993)



#############################################
# 6. Item-Based Recommendation
#############################################

# Make item-based recommendations based on the most recently watched and highest rated movie by the user.
user = 108170

# 1. Read the movie and rating datasets.
movie = pd.read_csv('Modül_4_Tavsiye_Sistemleri/datasets/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')

# 2. For the user to be recommended, get the id of the most recently rated movie with a rating of 5.
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# 3. Filter the user_movie_df dataframe (created in the user-based recommendation section) by the selected movie id.
movie[movie["movieId"] == movie_id]["title"].values[0]
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df

# 4. Using the filtered dataframe, find and sort the correlations between the selected movie and other movies.
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# Function that performs the last two steps
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

# 5. Recommend the top 5 movies other than the selected movie itself.
movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)
# From 1 to 6. The movie itself is at 0, so we exclude it.
movies_from_item_based[1:6].index


# 'My Science Project (1985)',
# 'Mediterraneo (1991)',
# 'Old Man and the Sea,
# The (1958)',
# 'National Lampoon's Senior Trip (1995)',
# 'Clockwatchers (1997)']



