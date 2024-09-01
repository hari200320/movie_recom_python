# Movie Recommender System

## Overview

This is a simple movie recommender system implemented in Python. It predicts movie ratings for a specific user based on their previous ratings and the similarity between movies. The system uses cosine similarity to measure how similar movies are to each other.

## Requirements

To run this script, you need Python installed with the following libraries:

- `numpy`
- `pandas`

You can install these libraries using pip:

```bash
pip install numpy pandas

Output

Movies Data:
 movie_id     title
        1  Movie A
        2  Movie B
        3  Movie C
        4  Movie D
        5  Movie E

Ratings Data:
 user_id  movie_id  rating
       1         1       5
       1         2       3
       2         1       4
       2         3       2
       3         2       5
       3         4       3
       4         3       4
       4         5       5
       5         4       2
       5         5       1

Predicted Ratings:
- Movie A: 4.12
- Movie B: 3.08
- Movie C: 2.84
- Movie D: 3.56
- Movie E: 2.11

Predicted rating for 'Movie C' by User 1: 2.84
