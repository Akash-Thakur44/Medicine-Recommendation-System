
# ML-Enhanced Pharmaceutical Recommendation System Usage Guide

## Quick Start
```python
# Import and initialize
from ml_pharmaceutical_system import PharmaceuticalMLSystem

ml_system = PharmaceuticalMLSystem()
ml_system.load_data()
ml_system.setup_content_based_filtering()
ml_system.setup_clustering()
ml_system.setup_classification()

# Get recommendations
recommendations = ml_system.get_comprehensive_recommendations(
    user_query="fever headache", 
    max_price=100, 
    min_rating=4.0
)
print(recommendations)
```

## Features Implemented
1. **Content-Based Filtering**: Uses TF-IDF and cosine similarity
2. **K-Means Clustering**: Groups medicines by price/rating patterns  
3. **Classification**: Predicts categories from symptoms
4. **Hybrid System**: Combines all approaches for best results

## Integration with Web App
The system is designed to integrate easily with the existing web application:
- Replace simple keyword matching with ML recommendations
- Add the PharmaceuticalMLSystem as a backend service
- Use comprehensive_recommendations() for main search functionality
- Add specialized functions for different recommendation types

## Learning Outcomes
- Practical experience with scikit-learn
- Understanding of different ML paradigms
- Real-world application of text processing
- System integration skills
- Performance evaluation techniques
