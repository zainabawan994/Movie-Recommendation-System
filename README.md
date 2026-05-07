# 🎬 Movie Recommendation System using Collaborative Filtering

A Machine Learning based Movie Recommendation System developed using **Python**, **Pandas**, and **Scikit-learn**.  
This project recommends movies to users based on their ratings and movie preferences using **Collaborative Filtering** techniques.

---

# 📌 Project Overview

Recommendation systems are widely used in platforms like:

- Netflix
- Amazon Prime
- Spotify
- YouTube
- E-commerce websites

This project uses:

- User-Based Collaborative Filtering
- Movie-Based Collaborative Filtering
- Cosine Similarity

to recommend movies.

The system analyzes user ratings and finds patterns between users and movies to generate personalized recommendations.

---

# 🚀 Features

✅ User-Based Recommendation System  
✅ Movie-Based Recommendation System  
✅ Cosine Similarity Implementation  
✅ User-Movie Rating Matrix  
✅ Sparse Matrix Handling  
✅ Personalized Recommendations  
✅ Data Preprocessing with Pandas  
✅ Similar Users Detection  
✅ Similar Movies Detection  

---

# 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Scikit-learn | Machine Learning |
| Jupyter Notebook | Development Environment |

---

# 📂 Dataset

The dataset contains:

- User IDs
- Movie IDs
- Movie Titles
- Ratings

Files used:

```bash
movies.csv
ratings.csv
```

---

# ⚙️ Project Workflow

## 1️⃣ Import Libraries

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
```

---

## 2️⃣ Load Dataset

```python
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
```

---

## 3️⃣ Merge Datasets

```python
df = movies.merge(ratings, on='movieId')
```

Merged dataset contains:

- userId
- movieId
- title
- rating

---

## 4️⃣ Create Pivot Table

User-Movie Matrix:

```python
p_table = df.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)
```

### Matrix Representation

| userId | movie1 | movie2 | movie3 |
|---|---|---|---|
| 1 | 5 | NaN | 3 |
| 2 | 4 | 2 | NaN |

---

## 5️⃣ Handle Missing Values

Since every user does not rate every movie, the matrix becomes sparse.

```python
p_table = p_table.fillna(0)
```

---

# 📊 Cosine Similarity

Cosine similarity measures similarity between users or movies.

## Formula

```math
\cos(\theta)=\frac{A \cdot B}{||A|| ||B||}
```

Where:

- A = User/Movie Vector
- B = Another User/Movie Vector

---

# 👥 User-Based Collaborative Filtering

This method finds users with similar movie tastes.

## Calculate User Similarity

```python
user_similarity = cosine_similarity(p_table)
```

## Create Similarity DataFrame

```python
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=p_table.index,
    columns=p_table.index
)
```

---

# 🎥 Movie-Based Collaborative Filtering

This method finds movies similar to a selected movie.

## Calculate Movie Similarity

```python
movie_similarity = cosine_similarity(p_table.T)
```

### Why Transpose is Used

- Original Matrix = Users × Movies
- Transposed Matrix = Movies × Users

---

## Create Movie Similarity DataFrame

```python
movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=p_table.columns,
    columns=p_table.columns
)
```

---

# 🎯 Movie Recommendation Function

```python
def recommend_movies(movie_id, n=5):

    similar_movies = movie_similarity_df[movie_id]\
        .sort_values(ascending=False)

    top_movies = similar_movies.iloc[1:n+1]

    return top_movies
```

---

# ▶️ Example Usage

```python
recommend_movies(1)
```

### Example Output

| movieId | similarity |
|---|---|
| 3114 | 0.82 |
| 356 | 0.79 |
| 480 | 0.76 |

---

# 👤 Similar Users Example

```python
similar_users = user_similarity_df[1]\
    .sort_values(ascending=False)

print(similar_users.head())
```

### Output

| userId | similarity |
|---|---|
| 1 | 1.000 |
| 4505 | 0.211 |
| 5062 | 0.210 |

---

# 📈 Evaluation Metrics

## Precision

Precision measures how many recommended movies were actually relevant.

```math
Precision=\frac{Relevant\ Recommended}{Total\ Recommended}
```

---

## Recall

Recall measures how many relevant movies were successfully recommended.

```math
Recall=\frac{Relevant\ Recommended}{Total\ Relevant}
```

---

# 🔥 Challenges

## Sparse Matrix Problem

Most users rate only a few movies, causing many missing values.

### Solution

```python
fillna(0)
```

---

## Cold Start Problem

New users have no ratings history.

### Possible Solutions

- Recommend popular movies
- Use content-based filtering

---

# 📁 Project Structure

```bash
Movie-Recommendation-System/
│
├── movies.csv
├── ratings.csv
├── recommendation_system.ipynb
├── README.md
└── requirements.txt
```

---

# 📦 Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn
```

---

# ▶️ How to Run Project

## Step 1 — Clone Repository

```bash
git clone <repository-link>
```

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3 — Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```bash
recommendation_system.ipynb
```

---

# 💡 Future Improvements

- Streamlit Web App
- Deep Learning Recommendation System
- Hybrid Recommendation Engine
- Content-Based Filtering
- Real-Time Recommendations
- Deployment on Cloud

---

# 🌟 Applications

- Netflix-style movie recommendations
- OTT platforms
- Music recommendation systems
- Product recommendation systems
- E-commerce personalization

---

# 📚 Learning Outcomes

This project demonstrates:

- Data preprocessing
- Pivot tables
- Sparse matrices
- Collaborative filtering
- Cosine similarity
- Recommendation systems
- Similarity-based machine learning

---

# 👨‍💻 Author

Developed using Machine Learning and Collaborative Filtering techniques with Python.

---
