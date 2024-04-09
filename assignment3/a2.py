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
# Exclude the first 100 businesses for similarity calculation
ratings_excluded = ratings.copy()
ratings_excluded.iloc[:, :100] = 0

# Calculate cosine similarity between Alex (4th user) and all users
alex_ratings = ratings_excluded.iloc[3].values.reshape(1, -1)  # Alex's ratings with first 100 businesses excluded
users_ratings = ratings_excluded.values
cos_similarities = cosine_similarity_(alex_ratings, users_ratings)[0]

# Calculate rAlex,b for the first 100 businesses
r_alex_b = np.dot(cos_similarities.reshape(1, -1), ratings.values[:, :100]).flatten()

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
# Transpose the ratings matrix to work with businesses as rows for the item-item system
ratings_transposed = ratings.T

# Calculate cosine similarity between businesses
business_cos_similarities = cosine_similarity_(ratings_transposed.values, ratings_transposed.values)
np.fill_diagonal(business_cos_similarities, 0)  # Zero out diagonal to exclude self-similarity

# Alex's ratings for items
alex_ratings_for_items = ratings.iloc[3, :].values

# Calculate rAlex,b for each of the first 100 businesses
r_alex_b = np.dot(business_cos_similarities[:100], alex_ratings_for_items)

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

# Perform Singular Value Decomposition (SVD)
U, sigma, VT = svd(ratings.values, full_matrices=False)

# Keep only the top k features for k = 10
k = 10
U_k = U[:, :k]
sigma_k = np.diag(sigma[:k])
VT_k = VT[:k, :]

# Estimate R* using the lower rank approximation
R_star = np.dot(U_k, np.dot(sigma_k, VT_k))

# Alex's estimated ratings for the first 100 businesses
alex_ratings_estimated = R_star[3, :100]

# Get the top 5 businesses with the highest estimated ratings for Alex
top_5_indices_latent = np.argsort(alex_ratings_estimated)[-5:][::-1]
top_5_businesses = businesses.iloc[top_5_indices_latent]["business"].values
top_5_scores = alex_ratings_estimated[top_5_indices_latent]

# Convert to a dataframe for better visualization
result = pd.DataFrame({'business': top_5_businesses, 'score': top_5_scores})
print("Top 5 businesses recommended to Alex using latent factor collaborative filtering:")
print(result)

# %%



