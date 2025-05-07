# Data Mining Project

## Overview

This project demonstrates the implementation of three fundamental data mining techniques on different datasets:

1. Decision Tree Classification
2. Clustering
3. Sentiment Analysis

The primary objective is to showcase the practical application and understanding of:
- Supervised learning
- Unsupervised learning
- Text mining techniques

## Project Structure

The project is organized into three main directories:

1. `Decision_tree_classification/` - Implementation of Decision Tree Classifier on Financial dataset
2. `Clustering/` - Implementation of Clustering algorithms (K-Means, Hierarchical) on Wholesale Customers dataset
3. `Sentiment_analysis/` - Implementation of Sentiment Analysis on Corona virus tweets dataset

Each section contains:
- Source code
- Dataset (or dataset link)
- Analysis reports
- Visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Data-Mining-project.git
cd Data-Mining-project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Decision Tree Classification
```bash
cd Decision_tree_classification
python decision_tree.py
```

### Clustering
```bash
cd Clustering
python clustering.py
```

### Sentiment Analysis
```bash
cd Sentiment_analysis
python sentiment_analysis.py
```

## Techniques Implemented

### 1. Decision Tree Classification
- Purpose: Predict categorical labels based on input features
- Application: Financial dataset analysis

### 2. Clustering
- Purpose: Unsupervised grouping of data points based on feature similarity
- Methods: K-Means, Hierarchical Clustering
- Application: Wholesale Customers dataset analysis

### 3. Sentiment Analysis
- Purpose: Natural Language Processing for text sentiment classification
- Categories: Positive, Negative, Neutral
- Application: Corona virus tweets analysis

## Technical Stack

- **Programming Language:** Python
- **Key Libraries:**
  - Scikit-learn
  - Pandas
  - NumPy
  - Matplotlib
  - NLTK

## Implementation Details

Each technique implementation includes:

1. **Data Preprocessing**
   - Data cleaning
   - Feature engineering
   - Data normalization

2. **Model Development**
   - Model training
   - Parameter tuning
   - Performance evaluation

3. **Results Analysis**
   - Performance metrics
   - Visualizations
   - Insights and conclusions

## Results

### Decision Tree Classification
- Model achieved 99.92% accuracy (0.08% error rate) in predicting financial outcomes
- Most influential features identified for decision making
- Successfully classified customer risk levels

### Clustering
- Identified 4 distinct customer segments
- Revealed clear patterns in customer purchasing behavior
- Demonstrated effective grouping of similar customer profiles

### Sentiment Analysis
- Achieved 99.01% accuracy (0.99% error rate) in sentiment classification
- Successfully categorized tweets into positive, neutral, and negative sentiments
- Provided insights into public sentiment trends

## License

This project is licensed under the MIT License.
