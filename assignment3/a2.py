# %% [markdown]
# # CS/INFO 5304 Assignment 3: Recommender Systems (Part A - C)
# 
# **Author**: Yufan Zhang (yz2894)
# 
# ---

# %%
import pandas as pd
import numpy as np

# Load the datasets
businesses = pd.read_csv('data/business.csv', header=None, names=['business'])
ratings = pd.read_csv('data/user-business.csv', header=None)

# %%
businesses

# %%
ratings

# %%
def cosine_similarity_(A, B):
    """
    Compute the cosine similarity between two matrices A and B.
    
    A and B must have the same number of features (columns), but they can have
    different numbers of observations (rows).
    """
    # Normalize the rows of A and B
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    
    # Compute the cosine similarity
    cosine_similarities = np.dot(A_norm, B_norm.T)
    
    return cosine_similarities


# %% [markdown]
# ## Part A: user – user recommender system

# %%
def user_user_predictor(user_index, ratings, num_businesses=100):
    # Exclude the first num_businesses businesses for similarity calculation
    ratings_excluded = ratings.copy()
    ratings_excluded.iloc[:, :num_businesses] = 0
    
    # Target user's ratings
    target_user_ratings = ratings_excluded.iloc[user_index].values.reshape(1, -1)
    
    # Calculate cosine similarity between the target user and all other users
    cos_similarities = cosine_similarity_(target_user_ratings, ratings_excluded.values)[0]
    
    # Calculate the rAlex, b for the first num_businesses businesses
    r_user_b = np.dot(cos_similarities.reshape(1, -1), ratings.values[:, :num_businesses]).flatten()
    
    return r_user_b

# %%
r_alex_b = user_user_predictor(3, ratings, 100)

# Get the top 5 businesses with the highest similarity scores
top_5_indices = np.argsort(r_alex_b)[-5:][::-1]
top_5_businesses = businesses.iloc[top_5_indices].values.flatten()
top_5_scores = r_alex_b[top_5_indices]

# Convert to a dataframe for better visualization
result = pd.DataFrame({'business': top_5_businesses, 'score': top_5_scores})
print("Top 5 businesses recommended to Alex using user-user collaborative filtering:")
print(result)

# %% [markdown]
# ## Part B: item – item recommender system

# %%
def item_item_predictor(user_index, ratings, num_businesses=100):
    # Transpose the ratings matrix to work with businesses as rows for the item-item system
    ratings_transposed = ratings.T

    # Exclude Alex's ratings for similarity calculation
    ratings_excluded = ratings_transposed.copy()
    ratings_excluded = np.delete(ratings_excluded.values, user_index, axis=1)

    # Calculate cosine similarity between businesses
    business_cos_similarities = cosine_similarity_(ratings_excluded, ratings_excluded)
    np.fill_diagonal(business_cos_similarities, 0)  # Exclude self-similarity

    # Target user's ratings for items
    user_ratings_for_items = ratings.iloc[user_index, :].values

    # Calculate the rAlex, b for the first num_businesses businesses
    r_item_b = np.dot(
        business_cos_similarities[:num_businesses], user_ratings_for_items
    )
    return r_item_b

# %%
r_alex_b = item_item_predictor(3, ratings, 100)

# Find the top 5 businesses with the highest rAlex,b values
top_5_indices = np.argsort(r_alex_b)[-5:][::-1]
top_5_businesses = businesses.iloc[top_5_indices]['business'].values
top_5_scores = r_alex_b[top_5_indices]

# Convert to a dataframe for better visualization
result = pd.DataFrame({'business': top_5_businesses, 'score': top_5_scores})
print("Top 5 businesses recommended to Alex using item-item collaborative filtering:")
print(result)

# %% [markdown]
# ## Part C: Latent hidden model recommender system

# %%
from scipy.linalg import svd


def latent_factor_predictor(user_index, ratings, num_features=10, num_businesses=100):
    # Perform Singular Value Decomposition (SVD)
    U, sigma, VT = svd(ratings.values, full_matrices=False)

    # Keep only the top k features for k = 10
    U_k = U[:, :num_features]
    sigma_k = np.diag(sigma[:num_features])
    VT_k = VT[:num_features, :]

    # Estimate R* using the lower rank approximation
    R_star = np.dot(U_k, np.dot(sigma_k, VT_k))
    return R_star[:, :num_businesses][user_index]

# %%
alex_ratings_estimated = latent_factor_predictor(3, ratings, 10, 100)

# Get the top 5 businesses with the highest estimated ratings for Alex
top_5_indices_latent = np.argsort(alex_ratings_estimated)[-5:][::-1]
top_5_businesses = businesses.iloc[top_5_indices_latent]["business"].values
top_5_scores = alex_ratings_estimated[top_5_indices_latent]

# Convert to a dataframe for better visualization
result = pd.DataFrame({'business': top_5_businesses, 'score': top_5_scores})
print("Top 5 businesses recommended to Alex using latent factor collaborative filtering:")
print(result)


