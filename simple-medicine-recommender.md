# Simple Medicine Recommendation System

This is a basic medicine recommendation system. It's simple but functional!

## Files Included
- `simple_medicines.csv` - Basic medicine database (31 medicines)
- Web application (HTML/CSS/JS) - Interactive interface
- Python backend logic - For testing and development

## Features
✅ **Find by Symptoms** - Type symptoms, get medicine suggestions
✅ **Browse by Category** - Select category to see available medicines  
✅ **Search Medicine** - Look up specific medicines
✅ **Price Filtering** - Find medicines within budget
✅ **Rating Display** - See user ratings for medicines

## How It Works
1. **Symptom Search**: Simple keyword matching between user symptoms and medicine indications
2. **Category Browse**: Filter medicines by therapeutic category
3. **Basic Recommendations**: Sort by rating and price

## Database Structure
Each medicine has:
- Name
- Category (Analgesic, Antibiotic, etc.)
- Indication (what it treats)
- Price (in ₹)
- Manufacturer
- Rating (out of 5)

## Sample Data (31 Medicines)
- **Pain Relief**: Paracetamol, Ibuprofen, Aspirin
- **Antibiotics**: Amoxicillin, Azithromycin, Ciprofloxacin
- **Stomach**: Omeprazole, Ranitidine
- **Allergies**: Cetirizine, Loratadine
- **And more...**

## Technology Used
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Backend**: Python (for data processing)
- **Data**: CSV file with medicine information

## Limitations
- Small dataset (31 medicines vs 500+ in industry)
- Simple keyword matching (not AI-powered)
- Basic UI design
- No user accounts or history
- No real-time data updates
- No drug interaction checking

## Future Improvements
- Add more medicines to database
- Implement proper search algorithms
- Add user reviews system
- Include dosage information
- Add side effects warnings

---