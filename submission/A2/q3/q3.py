# %% [markdown]
# # Question 3: Outlier Detection
# 
# CS 5304 - Data Science in the Wild, Assignment 2
# 
# **Author**: Yufan Zhang (yz2894)
# 

# %%
import pandas as pd
import numpy as np

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from itertools import combinations


# %%
# Load the data
df = pd.read_csv('data/prog_book.csv')

# Convert 'Reviews' from object to numeric
df['Reviews'] = df['Reviews'].str.replace(',', '')
df['Reviews'] = df['Reviews'].astype(int)

df.head()

# %% [markdown]
# ## Task 1: Univariate Outlier detection

# %%
# Numerical features
features = ["Rating", "Reviews", "Number_Of_Pages", "Price"]

# Generate box plots for each feature
fig = make_subplots(rows=1, cols=4)

for feature in features:
    fig.add_trace(
        go.Box(y=df[feature], name=feature),
        row=1,
        col=features.index(feature) + 1,
    )

fig.update_layout(height=600, width=1000, title_text="Box Plots for Each Feature")
fig.write_html("plots/task1.html")
fig.show()

# %% [markdown]
# ## Task 2: Multivariate Outlier detection

# %%
# Encoding 'Type' column with LabelEncoder
df_encoded = df.copy()
df_encoded['Type'] = LabelEncoder().fit_transform(df['Type'])

df_encoded.head()

# %%
# Selecting the needed features
features = ['Price', 'Number_Of_Pages', 'Rating', 'Reviews', 'Type']
X = df_encoded[features]

# Scaling the features
X_scaled = StandardScaler().fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

X_scaled_df

# %% [markdown]
# ### Task 2.1: Bivariate Outlier Detection
# 
# To view the interactive plot, please visit the following link: [yufanbruce.com/dsw/posts/a2](https://yufanbruce.com/dsw/posts/a2).

# %%
# All the bivariate combinations of the features
bivariate_combinations = list(combinations(features, 2))

outliers_bivariate = {}  # Store the outliers for each bivariate combination

fig = make_subplots(
    rows=5,
    cols=2,
    subplot_titles=[f"{x[0]} vs {x[1]}" for x in bivariate_combinations],
    vertical_spacing=0.05,
)

for i, combination in enumerate(bivariate_combinations):
    feature1, feature2 = combination
    X_subset = X_scaled_df[[feature1, feature2]]
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.4, min_samples=7).fit(X_subset)
    labels = dbscan.labels_

    # Store the outliers
    outlier_indices = np.where(labels == -1)[0]
    outliers_bivariate[combination] = df.iloc[outlier_indices][[feature1, feature2]]
    
    outliers = X_subset[labels == -1]
    outliers_bivariate[combination] = outliers

    # Plot
    fig.add_trace(
        go.Scatter(
            x=X_subset[feature1],
            y=X_subset[feature2],
            mode="markers",
            marker=dict(color=labels, colorscale="Viridis", showscale=True),
            text=labels,
            showlegend=False,
        ),
        row=i // 2 + 1,
        col=i % 2 + 1,
    )
    fig.update_xaxes(title_text=feature1, row=i // 2 + 1, col=i % 2 + 1)
    fig.update_yaxes(title_text=feature2, row=i // 2 + 1, col=i % 2 + 1)


fig.update_layout(height=2000, width=1200, title_text="Bivariate DBSCAN Analysis")
fig.update_traces(marker_showscale=False)

fig.write_html("plots/task2_1.html")
fig.show()

# %% [markdown]
# ### Task 2.2: Trivariate Outlier Detection
# 
# To view the interactive plot, please visit the following link: [yufanbruce.com/dsw/posts/a2](https://yufanbruce.com/dsw/posts/a2).

# %%
trivariate_combinations = list(combinations(features, 3))

outliers_trivariate = {}  # Store the outliers for each trivariate combination

total_plots = len(trivariate_combinations)
cols = 1
rows = total_plots

fig = make_subplots(
    rows=rows,
    cols=cols,
    specs=[[{"type": "scatter3d"}] for _ in range(total_plots)],
    subplot_titles=[f"{x[0]}, {x[1]}, {x[2]}" for x in trivariate_combinations],
    vertical_spacing=0.04,
)

for i, combination in enumerate(trivariate_combinations):
    feature1, feature2, feature3 = combination
    X_subset = X_scaled_df[[feature1, feature2, feature3]]

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.7, min_samples=12).fit(X_subset)
    labels = dbscan.labels_

    # Store the outliers
    outlier_indices = np.where(labels == -1)[0]
    outliers_trivariate[combination] = df.iloc[outlier_indices][list(combination)]

    # Create a 3D scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=X_subset[feature1],
            y=X_subset[feature2],
            z=X_subset[feature3],
            mode="markers",
            marker=dict(
                size=3,
                color=labels,  # Color points by cluster labels
                colorscale="Viridis",  # Choose a color scale
                opacity=0.8,
            ),
            showlegend=False,
        ),
        row=i + 1,
        col=1,
    )

    fig.update_scenes(
        dict(xaxis_title=feature1, yaxis_title=feature2, zaxis_title=feature3),
        row=i + 1,
        col=1,
    )

fig.update_layout(height=600 * rows, width=600, title_text="Trivariate DBSCAN Analysis")
fig.update_traces(marker_showscale=False)

# Show the plot
fig.write_html("plots/task2_2.html")
fig.show()

# %%



