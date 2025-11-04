# Frontend

# Backend

# Database

# Recommendation

## Baseline

## Content-based filtering

## Collaborative filtering

```python
from cf import CfSystem

# use cached value
USE_PICKLE = True
# use top k similar items
TOP_K = 50
TOP_N = 10

cf = CfSystem()
cf.initialize(TOP_K, USE_PICKLE)

# get top n movie recommendations as json
recommendations = cf.get_recommendations(user_id, TOP_N)
```

```python
# JSON
{
  "user_id": 1,
  "recommendations": [
    {
      "movie_id": 10001,
      "predicted_rating": 9.2,
    },
    {
      "movie_id": 10002,
      "predicted_rating": 8.8,
    },
  ]
}
```

## Deep learning based
