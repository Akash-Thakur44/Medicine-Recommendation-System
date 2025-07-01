```markdown
# ML-Enhanced Pharmaceutical Recommendation System

An interactive medicine recommendation system powered by machine learning. Users enter symptoms or medicine names to get personalized recommendations from a curated database.

## Features

- **ML Techniques:** Content-based filtering (TF-IDF + cosine similarity), K-Means clustering, and classification (Decision Tree, Naive Bayes, K-NN)[^4].
- **Hybrid Engine:** Combines multiple ML approaches for robust recommendations[^4].
- **Web Interface:** Real-time search, price/rating filtering, and intuitive results display[^1][^2].
- **Performance:** Highest accuracy with K-NN (51.8%) and Naive Bayes (51.5%), outperforming Decision Tree (29.4%)[^3][^4].

## ML Algorithm Performance

| Algorithm      | CV Accuracy (%) |
| -------------- | -------------- |
| Decision Tree  | 29.4           |
| Naive Bayes    | 51.5           |
| K-NN           | 51.8           |

## Quick Start

```

from ml_pharmaceutical_system import PharmaceuticalMLSystem

ml_system = PharmaceuticalMLSystem()
ml_system.load_data()
ml_system.setup_content_based_filtering()
ml_system.setup_clustering()
ml_system.setup_classification()

recommendations = ml_system.get_comprehensive_recommendations(
user_query="fever headache",
max_price=100,
min_rating=4.0
)
print(recommendations)

```
_See `ml_usage_guide.md` for more details.[^5]_

## Author

Akash Thakur
```

[^1]: app.js

[^2]: index.html

[^3]: ml_algorithm_performance.jpg

[^4]: ml_pharmaceutical_system.py

[^5]: ml_usage_guide.md

[^6]: simple_medicines.csv

[^7]: simple_recommender.py

[^8]: simple-medicine-recommender.md

[^9]: style.css

