
"""
Machine Learning Enhanced Pharmaceutical Recommendation System

Features:
1. Content-Based Filtering with TF-IDF and Cosine Similarity
2. K-Means Clustering for Medicine Grouping  
3. Classification Algorithms (Decision Tree, Naive Bayes, K-NN)
4. Hybrid Recommendation System
5. Web Interface Integration Ready

Author: Akash Thakur
Dataset Size: 31 Medicines
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class PharmaceuticalMLSystem:
    def __init__(self):
        """Initialize the ML-enhanced pharmaceutical recommendation system"""
        self.medicines_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim_matrix = None
        self.kmeans_model = None
        self.best_classifier = None
        self.indication_tfidf = None

    def load_data(self):
        """Loading the pharmaceutical dataset"""
        medicines_data = {
            'name': [
                'Paracetamol', 'Ibuprofen', 'Aspirin', 'Amoxicillin', 'Azithromycin',
                'Omeprazole', 'Metformin', 'Amlodipine', 'Atorvastatin', 'Lisinopril',
                'Cetirizine', 'Loratadine', 'Vitamin D3', 'Vitamin B12', 'Iron Tablets',
                'Calcium Carbonate', 'Ranitidine', 'Domperidone', 'Insulin', 'Salbutamol',
                'Prednisolone', 'Dexamethasone', 'Diclofenac', 'Tramadol', 'Codeine',
                'Ciprofloxacin', 'Doxycycline', 'Fluconazole', 'Acyclovir', 'Pantoprazole',
                'Levothyroxine'
            ],
            'category': [
                'Analgesic', 'Analgesic', 'Analgesic', 'Antibiotic', 'Antibiotic',
                'Antacid', 'Antidiabetic', 'Antihypertensive', 'Cholesterol', 'Antihypertensive',
                'Antihistamine', 'Antihistamine', 'Vitamin', 'Vitamin', 'Supplement',
                'Supplement', 'Antacid', 'Digestive', 'Antidiabetic', 'Bronchodilator',
                'Steroid', 'Steroid', 'Anti-inflammatory', 'Analgesic', 'Analgesic',
                'Antibiotic', 'Antibiotic', 'Antifungal', 'Antiviral', 'Antacid',
                'Thyroid'
            ],
            'indication': [
                'fever headache pain', 'pain inflammation fever', 'pain fever blood_clot_prevention',
                'bacterial_infections respiratory_tract', 'bacterial_infections pneumonia',
                'acid_reflux stomach_ulcer', 'diabetes blood_sugar_control', 'high_blood_pressure',
                'high_cholesterol heart_disease', 'high_blood_pressure heart_failure',
                'allergies hay_fever itching', 'allergies seasonal rhinitis',
                'bone_health vitamin_deficiency', 'anemia vitamin_deficiency', 'iron_deficiency anemia',
                'calcium_deficiency bone_health', 'heartburn acid_reflux', 'nausea vomiting digestive',
                'diabetes insulin_dependent', 'asthma breathing_difficulty',
                'inflammation allergic_reactions', 'severe_inflammation swelling',
                'pain inflammation arthritis', 'moderate_severe_pain', 'mild_moderate_pain cough',
                'bacterial_infections urinary_tract', 'bacterial_infections acne',
                'fungal_infections yeast', 'viral_infections herpes', 'acid_reflux gastritis',
                'thyroid_hormone_replacement'
            ],
            'price': [
                25, 35, 30, 120, 180, 45, 85, 95, 150, 110,
                40, 50, 65, 75, 55, 60, 35, 42, 450, 220,
                95, 125, 80, 160, 90, 140, 130, 175, 200, 55, 85
            ],
            'manufacturer': [
                'Generic', 'Generic', 'Generic', 'Cipla', 'Pfizer',
                'Sun Pharma', 'Cipla', 'Lupin', 'Atorva', 'Lupin',
                'Cipla', 'Sun Pharma', 'Healthvit', 'Neurobion', 'Ferrous',
                'Shelcal', 'Glaxo', 'Domped', 'Insulin', 'Ventolin',
                'Wysolone', 'Dexona', 'Voveran', 'Ultracet', 'Codral',
                'Ciplox', 'Doxy', 'Flucan', 'Zovirax', 'Pantop', 'Eltroxin'
            ],
            'rating': [
                4.2, 4.1, 3.9, 4.5, 4.4, 4.3, 4.6, 4.2, 4.1, 4.4,
                4.0, 4.1, 4.3, 4.5, 3.8, 4.0, 3.9, 3.7, 4.8, 4.6,
                4.2, 4.3, 4.0, 3.9, 3.8, 4.4, 4.1, 4.2, 4.3, 4.1, 4.5
            ]
        }

        self.medicines_df = pd.DataFrame(medicines_data)
        print(f"Loaded {len(self.medicines_df)} medicines into the system")

    def setup_content_based_filtering(self):
        """Setup TF-IDF vectorization and cosine similarity for content-based filtering"""
        # Combine features for content analysis
        self.medicines_df['combined_features'] = (
            self.medicines_df['category'] + ' ' + 
            self.medicines_df['indication'] + ' ' + 
            self.medicines_df['manufacturer']
        )

        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.medicines_df['combined_features'])

        # Calculate cosine similarity matrix
        self.cosine_sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        print("Content-based filtering setup complete")

    def setup_clustering(self):
        """Setup K-means clustering based on price and rating"""
        # Prepare features for clustering
        clustering_features = self.medicines_df[['price', 'rating']].copy()

        # Perform K-means clustering
        self.kmeans_model = KMeans(n_clusters=4, random_state=42)
        self.medicines_df['cluster'] = self.kmeans_model.fit_predict(clustering_features)

        print("K-means clustering setup complete")

    def setup_classification(self):
        """Setup classification models for category prediction"""
        # Prepare features for classification
        self.indication_tfidf = TfidfVectorizer(max_features=15, stop_words='english')
        indication_features = self.indication_tfidf.fit_transform(self.medicines_df['indication']).toarray()

        # Normalize numeric features
        price_norm = (self.medicines_df['price'] - self.medicines_df['price'].min()) / (
            self.medicines_df['price'].max() - self.medicines_df['price'].min())
        rating_norm = (self.medicines_df['rating'] - self.medicines_df['rating'].min()) / (
            self.medicines_df['rating'].max() - self.medicines_df['rating'].min())

        # Create feature matrix
        X = np.column_stack([indication_features, price_norm, rating_norm])
        y = self.medicines_df['category']

        # Test different classifiers
        classifiers = {
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=4),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=2)
        }

        best_score = 0
        for name, clf in classifiers.items():
            # Use cross-validation for small dataset
            cv_scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
            mean_score = cv_scores.mean()

            if mean_score > best_score:
                best_score = mean_score
                self.best_classifier = clf

        # Train best classifier on full dataset
        self.best_classifier.fit(X, y)

        print(f"Classification setup complete. Best accuracy: {best_score:.3f}")

    def get_content_recommendations(self, medicine_name, num_recommendations=5):
        """Get recommendations based on content similarity"""
        try:
            idx = self.medicines_df[self.medicines_df['name'] == medicine_name].index[0]
            sim_scores = list(enumerate(self.cosine_sim_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
            medicine_indices = [i[0] for i in sim_scores]

            recommendations = self.medicines_df.iloc[medicine_indices][
                ['name', 'category', 'indication', 'price', 'rating']
            ].copy()
            recommendations['similarity_score'] = [round(score[1], 3) for score in sim_scores]

            return recommendations
        except:
            return pd.DataFrame()

    def get_cluster_recommendations(self, target_medicine, num_recommendations=5):
        """Get recommendations from the same cluster"""
        try:
            target_cluster = self.medicines_df[
                self.medicines_df['name'] == target_medicine
            ]['cluster'].iloc[0]

            cluster_medicines = self.medicines_df[
                (self.medicines_df['cluster'] == target_cluster) & 
                (self.medicines_df['name'] != target_medicine)
            ]

            return cluster_medicines.sort_values('rating', ascending=False).head(num_recommendations)[
                ['name', 'category', 'indication', 'price', 'rating', 'cluster']
            ]
        except:
            return pd.DataFrame()

    def predict_category(self, indication_text, price_range=100, expected_rating=4.0):
        """Predict medicine category from symptoms"""
        indication_vec = self.indication_tfidf.transform([indication_text]).toarray()

        price_norm = (price_range - self.medicines_df['price'].min()) / (
            self.medicines_df['price'].max() - self.medicines_df['price'].min())
        rating_norm = (expected_rating - self.medicines_df['rating'].min()) / (
            self.medicines_df['rating'].max() - self.medicines_df['rating'].min())

        feature_vector = np.column_stack([indication_vec, [[price_norm, rating_norm]]])

        return self.best_classifier.predict(feature_vector)[0]

    def get_comprehensive_recommendations(self, user_query, max_price=None, min_rating=None):
        """Get comprehensive recommendations using all ML techniques"""
        # Content-based similarity
        query_vec = self.tfidf_vectorizer.transform([user_query])
        indication_matrix = self.tfidf_vectorizer.transform(self.medicines_df['indication'])
        similarities = cosine_similarity(query_vec, indication_matrix)[0]

        # Apply filters
        filtered_df = self.medicines_df.copy()
        if max_price:
            filtered_df = filtered_df[filtered_df['price'] <= max_price]
        if min_rating:
            filtered_df = filtered_df[filtered_df['rating'] >= min_rating]

        # Calculate hybrid scores
        final_scores = []
        for idx, medicine in filtered_df.iterrows():
            content_score = similarities[idx] if idx < len(similarities) else 0
            rating_score = (medicine['rating'] - self.medicines_df['rating'].min()) / (
                self.medicines_df['rating'].max() - self.medicines_df['rating'].min())
            price_score = 1 - ((medicine['price'] - self.medicines_df['price'].min()) / (
                self.medicines_df['price'].max() - self.medicines_df['price'].min()))

            final_score = (content_score * 0.5) + (rating_score * 0.3) + (price_score * 0.2)
            final_scores.append((idx, final_score))

        # Sort and get top recommendations
        final_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in final_scores[:5]]

        recommendations = self.medicines_df.iloc[top_indices].copy()
        recommendations['ml_score'] = [score for idx, score in final_scores[:5]]

        return recommendations[['name', 'category', 'indication', 'price', 'rating', 'ml_score']]

# Example usage
if __name__ == "__main__":
    # Initialize the system
    ml_system = PharmaceuticalMLSystem()

    # Setup all components
    ml_system.load_data()
    ml_system.setup_content_based_filtering()
    ml_system.setup_clustering()
    ml_system.setup_classification()

    # Test recommendations
    print("\n=== TESTING ML RECOMMENDATION SYSTEM ===")

    # Content-based test
    print("\nContent-based recommendations for Paracetamol:")
    content_recs = ml_system.get_content_recommendations('Paracetamol')
    print(content_recs)

    # Clustering test
    print("\nCluster-based recommendations for Paracetamol:")
    cluster_recs = ml_system.get_cluster_recommendations('Paracetamol')
    print(cluster_recs)

    # Comprehensive test
    print("\nComprehensive recommendations for 'fever pain':")
    comp_recs = ml_system.get_comprehensive_recommendations('fever pain', max_price=100)
    print(comp_recs)

    print("\n=== ML SYSTEM READY FOR WEB INTEGRATION ===")
