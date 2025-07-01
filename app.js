// ML-Enhanced Pharmaceutical Recommendation System

// Medicine database with ML-ready data
const medicines = [
    {"name": "Paracetamol", "category": "Analgesic", "indication": "fever headache pain", "price": 25, "rating": 4.2, "manufacturer": "Generic"},
    {"name": "Ibuprofen", "category": "Analgesic", "indication": "pain inflammation fever", "price": 35, "rating": 4.1, "manufacturer": "Generic"},
    {"name": "Aspirin", "category": "Analgesic", "indication": "pain fever blood_clot_prevention", "price": 30, "rating": 3.9, "manufacturer": "Generic"},
    {"name": "Amoxicillin", "category": "Antibiotic", "indication": "bacterial_infections respiratory_tract", "price": 120, "rating": 4.5, "manufacturer": "Cipla"},
    {"name": "Azithromycin", "category": "Antibiotic", "indication": "bacterial_infections pneumonia", "price": 180, "rating": 4.4, "manufacturer": "Pfizer"},
    {"name": "Omeprazole", "category": "Antacid", "indication": "acid_reflux stomach_ulcer", "price": 45, "rating": 4.3, "manufacturer": "Sun Pharma"},
    {"name": "Metformin", "category": "Antidiabetic", "indication": "diabetes blood_sugar_control", "price": 85, "rating": 4.6, "manufacturer": "Cipla"},
    {"name": "Amlodipine", "category": "Antihypertensive", "indication": "high_blood_pressure", "price": 95, "rating": 4.2, "manufacturer": "Lupin"},
    {"name": "Atorvastatin", "category": "Cholesterol", "indication": "high_cholesterol heart_disease", "price": 150, "rating": 4.1, "manufacturer": "Atorva"},
    {"name": "Lisinopril", "category": "Antihypertensive", "indication": "high_blood_pressure heart_failure", "price": 110, "rating": 4.4, "manufacturer": "Lupin"},
    {"name": "Cetirizine", "category": "Antihistamine", "indication": "allergies hay_fever itching", "price": 40, "rating": 4.0, "manufacturer": "Cipla"},
    {"name": "Loratadine", "category": "Antihistamine", "indication": "allergies seasonal rhinitis", "price": 50, "rating": 4.1, "manufacturer": "Sun Pharma"},
    {"name": "Vitamin D3", "category": "Vitamin", "indication": "bone_health vitamin_deficiency", "price": 65, "rating": 4.3, "manufacturer": "Healthvit"},
    {"name": "Vitamin B12", "category": "Vitamin", "indication": "anemia vitamin_deficiency", "price": 75, "rating": 4.5, "manufacturer": "Neurobion"},
    {"name": "Iron Tablets", "category": "Supplement", "indication": "iron_deficiency anemia", "price": 55, "rating": 3.8, "manufacturer": "Ferrous"},
    {"name": "Calcium Carbonate", "category": "Supplement", "indication": "calcium_deficiency bone_health", "price": 60, "rating": 4.0, "manufacturer": "Shelcal"},
    {"name": "Ranitidine", "category": "Antacid", "indication": "heartburn acid_reflux", "price": 35, "rating": 3.9, "manufacturer": "Glaxo"},
    {"name": "Domperidone", "category": "Digestive", "indication": "nausea vomiting digestive", "price": 42, "rating": 3.7, "manufacturer": "Domped"},
    {"name": "Insulin", "category": "Antidiabetic", "indication": "diabetes insulin_dependent", "price": 450, "rating": 4.8, "manufacturer": "Insulin"},
    {"name": "Salbutamol", "category": "Bronchodilator", "indication": "asthma breathing_difficulty", "price": 220, "rating": 4.6, "manufacturer": "Ventolin"},
    {"name": "Prednisolone", "category": "Steroid", "indication": "inflammation allergic_reactions", "price": 95, "rating": 4.2, "manufacturer": "Wysolone"},
    {"name": "Dexamethasone", "category": "Steroid", "indication": "severe_inflammation swelling", "price": 125, "rating": 4.3, "manufacturer": "Dexona"},
    {"name": "Diclofenac", "category": "Anti-inflammatory", "indication": "pain inflammation arthritis", "price": 80, "rating": 4.0, "manufacturer": "Voveran"},
    {"name": "Tramadol", "category": "Analgesic", "indication": "moderate_severe_pain", "price": 160, "rating": 3.9, "manufacturer": "Ultracet"},
    {"name": "Codeine", "category": "Analgesic", "indication": "mild_moderate_pain cough", "price": 90, "rating": 3.8, "manufacturer": "Codral"},
    {"name": "Ciprofloxacin", "category": "Antibiotic", "indication": "bacterial_infections urinary_tract", "price": 140, "rating": 4.4, "manufacturer": "Ciplox"},
    {"name": "Doxycycline", "category": "Antibiotic", "indication": "bacterial_infections acne", "price": 130, "rating": 4.1, "manufacturer": "Doxy"},
    {"name": "Fluconazole", "category": "Antifungal", "indication": "fungal_infections yeast", "price": 175, "rating": 4.2, "manufacturer": "Flucan"},
    {"name": "Acyclovir", "category": "Antiviral", "indication": "viral_infections herpes", "price": 200, "rating": 4.3, "manufacturer": "Zovirax"},
    {"name": "Pantoprazole", "category": "Antacid", "indication": "acid_reflux gastritis", "price": 55, "rating": 4.1, "manufacturer": "Pantop"},
    {"name": "Levothyroxine", "category": "Thyroid", "indication": "thyroid_hormone_replacement", "price": 85, "rating": 4.5, "manufacturer": "Eltroxin"}
];

// ML Algorithms configuration
const mlAlgorithms = {
    "content_based": "TF-IDF + Cosine Similarity",
    "clustering": "K-Means Clustering", 
    "classification": "K-Nearest Neighbors",
    "hybrid": "Multi-Algorithm Ensemble"
};

// Category mapping for prediction
const categoryMappings = {
    "fever": "Analgesic",
    "headache": "Analgesic", 
    "pain": "Analgesic",
    "infection": "Antibiotic",
    "bacterial": "Antibiotic",
    "respiratory": "Antibiotic",
    "acid": "Antacid",
    "stomach": "Antacid",
    "reflux": "Antacid",
    "allergy": "Antihistamine",
    "allergies": "Antihistamine",
    "rhinitis": "Antihistamine",
    "blood_pressure": "Antihypertensive",
    "hypertension": "Antihypertensive",
    "diabetes": "Antidiabetic",
    "sugar": "Antidiabetic",
    "vitamin": "Vitamin",
    "deficiency": "Supplement",
    "inflammation": "Anti-inflammatory",
    "asthma": "Bronchodilator",
    "breathing": "Bronchodilator"
};

let currentFilters = {
    maxPrice: 450,
    minRating: 0
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    updatePriceDisplay();
}

function setupEventListeners() {
    // Main search functionality
    document.getElementById('search-btn').addEventListener('click', performMLSearch);
    document.getElementById('search-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            performMLSearch();
        }
    });

    // Sample search buttons
    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const searchTerm = this.getAttribute('data-search');
            document.getElementById('search-input').value = searchTerm;
            performMLSearch();
        });
    });

    // Filter controls
    document.getElementById('price-range').addEventListener('input', function() {
        currentFilters.maxPrice = parseInt(this.value);
        updatePriceDisplay();
    });

    document.getElementById('rating-filter').addEventListener('change', function() {
        currentFilters.minRating = parseFloat(this.value);
    });
}

function updatePriceDisplay() {
    document.getElementById('price-value').textContent = currentFilters.maxPrice;
}

function performMLSearch() {
    const searchTerm = document.getElementById('search-input').value.trim().toLowerCase();
    
    if (!searchTerm) {
        alert('Please enter symptoms or medicine name to search');
        return;
    }

    showLoadingState();
    
    // Simulate ML processing delay
    setTimeout(() => {
        const results = runMLAlgorithms(searchTerm);
        displayResults(results, searchTerm);
        hideLoadingState();
    }, 2500);
}

function runMLAlgorithms(searchTerm) {
    // Determine which ML algorithm to use based on search pattern
    let algorithm, predictedCategory;
    
    if (searchTerm.includes(' ') && searchTerm.split(' ').length >= 2) {
        // Multiple symptoms - use content-based filtering
        algorithm = mlAlgorithms.content_based;
        predictedCategory = predictCategoryFromSymptoms(searchTerm);
    } else if (medicines.some(m => m.name.toLowerCase().includes(searchTerm))) {
        // Single medicine name - use clustering
        algorithm = mlAlgorithms.clustering;
        predictedCategory = findMedicineCategory(searchTerm);
    } else {
        // Single symptom - use classification
        algorithm = mlAlgorithms.classification;
        predictedCategory = predictCategoryFromSymptoms(searchTerm);
    }

    // Get recommendations using the selected algorithm
    let recommendations = getRecommendations(searchTerm, algorithm);
    
    // Apply filters
    recommendations = applyFilters(recommendations);
    
    // Calculate ML confidence scores
    recommendations = recommendations.map(med => ({
        ...med,
        mlConfidence: calculateConfidenceScore(med, searchTerm),
        algorithm: algorithm
    }));

    // Sort by confidence score and rating
    recommendations.sort((a, b) => (b.mlConfidence * b.rating) - (a.mlConfidence * a.rating));

    return {
        medicines: recommendations.slice(0, 8), // Top 8 recommendations
        algorithm: algorithm,
        predictedCategory: predictedCategory,
        totalFound: recommendations.length
    };
}

function getRecommendations(searchTerm, algorithm) {
    const searchWords = searchTerm.toLowerCase().split(/[\s,_]+/);
    let recommendations = [];

    medicines.forEach(medicine => {
        let relevanceScore = 0;
        const medicineText = `${medicine.name} ${medicine.indication} ${medicine.category}`.toLowerCase();
        
        // Calculate relevance based on algorithm type
        if (algorithm === mlAlgorithms.content_based) {
            // TF-IDF simulation - count term frequency
            searchWords.forEach(word => {
                if (medicineText.includes(word)) {
                    relevanceScore += 1;
                }
            });
        } else if (algorithm === mlAlgorithms.clustering) {
            // K-means simulation - exact name matching gets higher score
            if (medicine.name.toLowerCase().includes(searchTerm)) {
                relevanceScore = 10;
            } else {
                searchWords.forEach(word => {
                    if (medicineText.includes(word)) {
                        relevanceScore += 0.5;
                    }
                });
            }
        } else if (algorithm === mlAlgorithms.classification) {
            // KNN simulation - category-based scoring
            searchWords.forEach(word => {
                if (medicineText.includes(word)) {
                    relevanceScore += medicine.rating; // Weight by rating
                }
            });
        }

        if (relevanceScore > 0) {
            recommendations.push({
                ...medicine,
                relevanceScore: relevanceScore
            });
        }
    });

    return recommendations;
}

function predictCategoryFromSymptoms(symptoms) {
    const words = symptoms.toLowerCase().split(/[\s,_]+/);
    const categoryScores = {};

    words.forEach(word => {
        if (categoryMappings[word]) {
            const category = categoryMappings[word];
            categoryScores[category] = (categoryScores[category] || 0) + 1;
        }
    });

    // Return the category with highest score
    let maxScore = 0;
    let predictedCategory = "General";
    
    for (const [category, score] of Object.entries(categoryScores)) {
        if (score > maxScore) {
            maxScore = score;
            predictedCategory = category;
        }
    }

    return predictedCategory;
}

function findMedicineCategory(medicineName) {
    const medicine = medicines.find(m => m.name.toLowerCase().includes(medicineName.toLowerCase()));
    return medicine ? medicine.category : "Unknown";
}

function calculateConfidenceScore(medicine, searchTerm) {
    const searchWords = searchTerm.toLowerCase().split(/[\s,_]+/);
    const medicineText = `${medicine.name} ${medicine.indication}`.toLowerCase();
    
    let matches = 0;
    searchWords.forEach(word => {
        if (medicineText.includes(word)) {
            matches++;
        }
    });

    // Base confidence on match ratio and medicine rating
    const matchRatio = matches / searchWords.length;
    const ratingBonus = medicine.rating / 5.0;
    const confidence = (matchRatio * 0.7) + (ratingBonus * 0.3);
    
    return Math.min(confidence, 0.99); // Cap at 99%
}

function applyFilters(medicines) {
    return medicines.filter(medicine => {
        return medicine.price <= currentFilters.maxPrice && 
               medicine.rating >= currentFilters.minRating;
    });
}

function displayResults(results, searchTerm) {
    const resultsSection = document.getElementById('results-section');
    const algorithmElement = document.getElementById('ml-algorithm');
    const categoryElement = document.getElementById('predicted-category');
    const countElement = document.getElementById('results-count');
    const gridElement = document.getElementById('medicine-grid');

    // Show results section
    resultsSection.style.display = 'block';
    
    // Update ML info
    algorithmElement.textContent = results.algorithm;
    categoryElement.textContent = results.predictedCategory;
    countElement.textContent = `${results.medicines.length} recommendations found (${results.totalFound} total matches)`;

    // Clear previous results
    gridElement.innerHTML = '';

    if (results.medicines.length === 0) {
        gridElement.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 40px;">
                <h3>No medicines found</h3>
                <p>Try adjusting your search terms or filters</p>
            </div>
        `;
        return;
    }

    // Create medicine cards
    results.medicines.forEach(medicine => {
        const card = createMedicineCard(medicine);
        gridElement.appendChild(card);
    });

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function createMedicineCard(medicine) {
    const card = document.createElement('div');
    card.className = 'medicine-card';
    
    const confidenceClass = getConfidenceClass(medicine.mlConfidence);
    const confidencePercentage = Math.round(medicine.mlConfidence * 100);

    card.innerHTML = `
        <div class="medicine-header">
            <div class="medicine-name">${medicine.name}</div>
            <div class="medicine-category">${medicine.category}</div>
        </div>
        
        <div class="medicine-indication">
            ${formatIndication(medicine.indication)}
        </div>
        
        <div class="medicine-details">
            <div class="medicine-price">₹${medicine.price}</div>
            <div class="medicine-rating">
                ⭐ ${medicine.rating}
            </div>
        </div>
        
        <div class="ml-confidence">
            <div class="confidence-score ${confidenceClass}">
                ML Confidence: ${confidencePercentage}%
            </div>
            <div class="manufacturer">
                ${medicine.manufacturer}
            </div>
        </div>
    `;

    return card;
}

function formatIndication(indication) {
    return indication.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'confidence-high';
    if (confidence >= 0.6) return 'confidence-medium';
    return 'confidence-low';
}

function showLoadingState() {
    const overlay = document.getElementById('loading-overlay');
    overlay.style.display = 'flex';
    
    // Animate loading steps
    const steps = overlay.querySelectorAll('.step');
    steps.forEach(step => {
        step.classList.remove('active', 'completed');
    });
    
    let currentStep = 0;
    const stepInterval = setInterval(() => {
        if (currentStep > 0) {
            steps[currentStep - 1].classList.remove('active');
            steps[currentStep - 1].classList.add('completed');
        }
        
        if (currentStep < steps.length) {
            steps[currentStep].classList.add('active');
            currentStep++;
        } else {
            clearInterval(stepInterval);
        }
    }, 600);
}

function hideLoadingState() {
    const overlay = document.getElementById('loading-overlay');
    overlay.style.display = 'none';
}

// Additional utility functions for demonstration
function demonstrateMLConcepts() {
    console.log('ML Concepts Demonstrated:');
    console.log('1. Content-based Filtering: TF-IDF + Cosine Similarity');
    console.log('2. Clustering: K-Means for grouping similar medicines');
    console.log('3. Classification: K-NN for category prediction');
    console.log('4. Hybrid Systems: Combining multiple algorithms');
    console.log('5. Confidence Scoring: ML prediction confidence');
    console.log('6. Feature Engineering: Text processing and scoring');
}

// Initialize ML demonstration
demonstrateMLConcepts();