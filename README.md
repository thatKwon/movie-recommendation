# Frontend



# Backend




# Database



# Recommendation
## Baseline

## Content-based filtering

## Collaborative filtering
```
from cf import CfSystem

# use cached value
USE_PICKLE = True
# use top k similar items
TOP_K = 50
TOP_N = 10

cf = CfSystem()
cf.initialize(TOP_K, USE_PICKLE)

# get top n movie recommendations (top_n=10 as default)
recommendations = cf.get_recommendations(user_id, TOP_N)
```

## Deep learning based
