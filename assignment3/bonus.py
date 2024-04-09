# %% [markdown]
# # CS/INFO 5304 Assignment 3: Recommender Systems (Part D)
# 
# **Author**: Yufan Zhang (yz2894)
# 
# ---

# %%
import pandas as pd
import numpy as np
from scipy.linalg import svd

# Load the datasets
businesses = pd.read_csv('data/business.csv', header=None, names=['business'])
ratings = pd.read_csv('data/user-business.csv', header=None)
ratings_test = pd.read_csv('data/user-business_test.csv', header=None)


# %%
# Function to compute the cosine similarity between two matrices
def cosine_similarity(A, B):
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


# %%
# Wrap the recommendation systems into functions
def user_user_predictor(user_index, ratings, num_businesses=100):
    ratings_excluded = ratings.copy()
    ratings_excluded.iloc[:, :num_businesses] = 0  # Exclude first `num_businesses` for similarity
    
    target_user_ratings = ratings_excluded.iloc[user_index].values.reshape(1, -1)
    cos_similarities = cosine_similarity(target_user_ratings, ratings_excluded.values)[0]
    
    r_user_b = np.dot(cos_similarities.reshape(1, -1), ratings.values[:, :num_businesses]).flatten()
    return r_user_b


def item_item_predictor(user_index, ratings, num_businesses=100):
    ratings_transposed = ratings.T
    
    business_cos_similarities = cosine_similarity(ratings_transposed.values, ratings_transposed.values)
    np.fill_diagonal(business_cos_similarities, 0)  # Exclude self-similarity
    
    user_ratings_for_items = ratings.iloc[user_index, :].values
    
    r_item_b = np.dot(business_cos_similarities[:num_businesses], user_ratings_for_items)
    return r_item_b


def latent_factor_predictor(user_index, ratings, num_features=10, num_businesses=100):
    U, sigma, VT = svd(ratings.values, full_matrices=False)
    
    U_k = U[:, :num_features]
    sigma_k = np.diag(sigma[:num_features])
    VT_k = VT[:num_features, :]
    
    R_star = np.dot(U_k, np.dot(sigma_k, VT_k))
    return R_star[:, :num_businesses][user_index]


# %%
def convert_predictions_to_binary_by_threshold(predictions, threshold):
    """
    Convert continuous predictions into binary predictions (1s and 0s)
    based on a specified threshold.
    """
    binary_predictions = (predictions > threshold).astype(int)
    return binary_predictions


def convert_predictions_to_binary_by_top_k(predictions, k):
    """
    Convert continuous predictions into binary predictions (1s and 0s)
    by selecting the top k businesses with the highest predicted ratings.
    """
    top_k_indices = np.argsort(predictions)[-k:]
    binary_predictions = np.zeros_like(predictions)
    binary_predictions[top_k_indices] = 1
    return binary_predictions.astype(int)


def ensemble_predictor(user_index, business_indices, ratings_train, weights=None):
    """
    Ensemble predictor that combines predictions from multiple methods.
    
    :param user_index: Index of the user for whom predictions are made
    :param business_indices: Indices of businesses to predict ratings for
    :param ratings_train: Training ratings DataFrame
    :param weights: Weights for each predictor's contribution. If None, equal weighting is used.
    :return: Predicted ratings for the specified businesses for the specified user
    """
    # Placeholder for implementing the individual prediction functions
    prediction_user_user = user_user_predictor(user_index, ratings_train, business_indices)
    prediction_item_item = item_item_predictor(user_index, ratings_train, business_indices)
    prediction_latent_factor = latent_factor_predictor(user_index, ratings_train, 10, business_indices)
    
    # Normalize predictions
    prediction_user_user = (prediction_user_user - prediction_user_user.min()) / (prediction_user_user.max() - prediction_user_user.min())
    prediction_item_item = (prediction_item_item - prediction_item_item.min()) / (prediction_item_item.max() - prediction_item_item.min())
    prediction_latent_factor = (prediction_latent_factor - prediction_latent_factor.min()) / (prediction_latent_factor.max() - prediction_latent_factor.min())
    
    # If no weights provided, default to equal weighting
    if weights is None:
        weights = [1/3, 1/3, 1/3]
        
    # Combine predictions
    combined_prediction = (
        weights[0] * prediction_user_user +
        weights[1] * prediction_item_item +
        weights[2] * prediction_latent_factor
    )
    
    # Normalize combined prediction
    combined_prediction = (combined_prediction - combined_prediction.min()) / (combined_prediction.max() - combined_prediction.min())
    
    return combined_prediction


# %%
from sklearn.metrics import f1_score

# Split the data into training and validation sets
# validation_indices = range(5, len(ratings), 1000)
validation_indices = [5, 6, 7, 8, 9]

ratings_train = ratings
ratings_validation = ratings.iloc[validation_indices]


# Test the ensemble predictor on the validation set
business_indices = 100
threshold = 0.5
top_k = 10
f1_scores = []

for user_index in validation_indices:
    predictions = ensemble_predictor(user_index, business_indices, ratings_train)
    # print(predictions)
    
    predictions = convert_predictions_to_binary_by_top_k(predictions, top_k)
    # predictions = convert_predictions_to_binary_by_threshold(predictions, threshold)
    
    labels = ratings_validation.loc[user_index][:business_indices].values
    f1 = f1_score(labels, predictions)
    
    f1_scores.append(f1)
    
    print(f"Predictions for user {user_index}:")
    print(predictions)
    print(f"Labels for user {user_index}:")
    print(labels)
    print(f"F1 Score for user {user_index}:", f1)
    print("\n")
    
    
print("\nAverage F1 Score:", np.mean(f1_scores))

# %%
ratings_all = pd.concat([ratings, ratings_test])
ratings_all.reset_index(drop=True, inplace=True)

ratings_all

# %%
test_user_indices = list(ratings_all.index)[-5:]
test_user_indices

# %%
ratings_train = ratings_all

# Test the ensemble predictor on the validation set
business_indices = 100
threshold = 0.5
top_k = 10
test_predictions = []

for user_index in test_user_indices:
    predictions = ensemble_predictor(user_index, business_indices, ratings_train)
    predictions = convert_predictions_to_binary_by_top_k(predictions, top_k)

    test_predictions.append(predictions)
    
    
with open('bonus_submission.csv', 'w') as f:
    for predictions in test_predictions:
        f.write(','.join(map(str, predictions)))
        f.write('\n')

# %%



