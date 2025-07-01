# Simple Medicine Recommendation System
# A basic implementation that could be built in a week

import pandas as pd
import re

class SimpleMedicineRecommender:
    """
    A basic medicine recommendation system for educational purposes.
    This is simplified compared to industry-level solutions.
    """

    def __init__(self, csv_file='simple_medicines.csv'):
        """Load medicine data from CSV file"""
        try:
            self.df = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(self.df)} medicines from database")
        except FileNotFoundError:
            print("❌ CSV file not found!")
            self.df = pd.DataFrame()

    def find_by_symptom(self, symptom):
        """
        Simple symptom-based recommendation using keyword matching
        Returns top 5 medicines that match the symptom
        """
        symptom_words = symptom.lower().split()
        matches = []

        for _, medicine in self.df.iterrows():
            indication = medicine['indication'].lower()

            # Count how many symptom words match
            match_count = 0
            for word in symptom_words:
                if word in indication:
                    match_count += 1

            if match_count > 0:
                matches.append({
                    'name': medicine['name'],
                    'category': medicine['category'],
                    'indication': medicine['indication'],
                    'price': medicine['price'],
                    'rating': medicine['rating'],
                    'match_score': match_count
                })

        # Sort by match score and rating
        matches.sort(key=lambda x: (x['match_score'], x['rating']), reverse=True)
        return matches[:5]

    def find_by_category(self, category):
        """Get medicines in a specific category, sorted by rating"""
        filtered = self.df[self.df['category'].str.contains(category, case=False, na=False)]
        return filtered.sort_values('rating', ascending=False).to_dict('records')

    def search_medicine(self, name):
        """Search for a specific medicine by name"""
        result = self.df[self.df['name'].str.contains(name, case=False, na=False)]
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

    def get_cheap_medicines(self, max_price=50):
        """Find medicines under a certain price"""
        cheap = self.df[self.df['price'] <= max_price]
        return cheap.sort_values(['rating', 'price'], ascending=[False, True]).to_dict('records')

# Demo usage
if __name__ == "__main__":
    print("🏥 Simple Medicine Recommendation System")
    print("=" * 50)

    # Initialize recommender
    recommender = SimpleMedicineRecommender()

    if not recommender.df.empty:
        # Demo 1: Search by symptom
        print("\n🔍 Demo 1: Finding medicines for 'headache'")
        headache_meds = recommender.find_by_symptom("headache")
        for med in headache_meds:
            print(f"  • {med['name']} - ₹{med['price']} (Rating: {med['rating']})")

        # Demo 2: Browse by category
        print("\n📋 Demo 2: Best antibiotics")
        antibiotics = recommender.find_by_category("Antibiotic")[:3]
        for med in antibiotics:
            print(f"  • {med['name']} - ₹{med['price']} (Rating: {med['rating']})")

        # Demo 3: Budget medicines
        print("\n💰 Demo 3: Budget medicines (under ₹30)")
        cheap_meds = recommender.get_cheap_medicines(30)[:3]
        for med in cheap_meds:
            print(f"  • {med['name']} - ₹{med['price']} (Rating: {med['rating']})")

        print("\n✨ System working correctly!")
    else:
        print("❌ Could not load data. Make sure 'simple_medicines.csv' exists.")
