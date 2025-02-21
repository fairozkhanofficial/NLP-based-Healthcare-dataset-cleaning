import pandas as pd
import numpy as np
import re
import nltk
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy import stats
import os
import io
import base64
from flask import Flask, request, jsonify, send_file

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class HealthcareDataCleaner:
    def __init__(self):
        self.cleaning_stats = {}
        self.original_df = None
        self.cleaned_df = None
        self.visualization_data = {}
    
    def load_data(self, filepath):
        """Load healthcare dataset from CSV file"""
        try:
            self.original_df = pd.read_csv(filepath)
            self.cleaning_stats['original_rows'] = len(self.original_df)
            self.cleaning_stats['original_columns'] = len(self.original_df.columns)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def detect_text_columns(self):
        """Identify columns containing text data for NLP processing"""
        text_columns = []
        for col in self.original_df.columns:
            if self.original_df[col].dtype == 'object':
                # Check if column contains text data (more than just short codes)
                sample = self.original_df[col].dropna().astype(str).iloc[0] if not self.original_df[col].dropna().empty else ""
                if len(sample.split()) > 3:
                    text_columns.append(col)
        return text_columns
    
    def clean_text(self, text):
        """Clean and normalize text data using NLP techniques"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        
        return ' '.join(cleaned_tokens)
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        # Store missing value counts before cleaning
        self.cleaning_stats['missing_values_before'] = self.original_df.isna().sum().sum()
        
        # For numerical columns, fill missing values with median
        num_cols = self.original_df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            median_value = self.original_df[col].median()
            self.cleaned_df[col] = self.original_df[col].fillna(median_value)
        
        # For categorical columns, fill with mode
        cat_cols = self.original_df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            mode_value = self.original_df[col].mode()[0] if not self.original_df[col].mode().empty else "Unknown"
            self.cleaned_df[col] = self.original_df[col].fillna(mode_value)
        
        # Store missing value counts after cleaning
        self.cleaning_stats['missing_values_after'] = self.cleaned_df.isna().sum().sum()
    
    def remove_outliers(self):
        """Remove outliers from numerical columns using z-score"""
        num_cols = self.cleaned_df.select_dtypes(include=['int64', 'float64']).columns
        
        # Count initial rows
        initial_rows = len(self.cleaned_df)
        
        # Store outliers per column for visualization
        outlier_data = {}
        
        for col in num_cols:
            z_scores = stats.zscore(self.cleaned_df[col], nan_policy='omit')
            # Identify outliers (z-score > 3)
            outliers = abs(z_scores) > 3
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                outlier_data[col] = outlier_count
                # Replace outliers with column median
                median_val = self.cleaned_df[col].median()
                self.cleaned_df.loc[outliers, col] = median_val
        
        self.cleaning_stats['outliers_removed'] = outlier_data
        self.cleaning_stats['outlier_total'] = sum(outlier_data.values())
    
    def standardize_categorical_data(self):
        """Standardize categorical values using clustering for similar terms"""
        cat_cols = self.cleaned_df.select_dtypes(include=['object']).columns
        standardization_changes = {}
        
        for col in cat_cols:
            # Skip columns with unique values greater than 100 (likely IDs or free text)
            if self.cleaned_df[col].nunique() > 100:
                continue
                
            # Get value counts
            value_counts = self.cleaned_df[col].value_counts()
            
            # Find potential misspellings or variations (low frequency items)
            candidates = value_counts[value_counts < len(self.cleaned_df) * 0.01].index.tolist()
            main_values = value_counts[value_counts >= len(self.cleaned_df) * 0.01].index.tolist()
            
            if len(candidates) > 1 and len(main_values) > 0:
                # Convert to strings
                str_candidates = [str(val).lower() for val in candidates]
                
                # Vectorize the candidates
                vectorizer = TfidfVectorizer()
                try:
                    X = vectorizer.fit_transform(str_candidates)
                    
                    # Cluster similar terms
                    n_clusters = min(5, len(candidates))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X)
                    
                    # Map each candidate to its cluster
                    cluster_map = {candidate: cluster for candidate, cluster in zip(candidates, clusters)}
                    
                    # For each cluster, find most common value or map to nearest main value
                    changes = {}
                    for cluster_id in range(n_clusters):
                        cluster_items = [item for item, c_id in cluster_map.items() if c_id == cluster_id]
                        if len(cluster_items) > 1:
                            # Find most frequent item in this cluster
                            most_common = max(cluster_items, key=lambda x: value_counts.get(x, 0))
                            
                            # Map all others to this value
                            for item in cluster_items:
                                if item != most_common:
                                    changes[item] = most_common
                    
                    # Apply changes
                    if changes:
                        standardization_changes[col] = changes
                        self.cleaned_df[col] = self.cleaned_df[col].replace(changes)
                except:
                    # Skip if vectorization fails
                    pass
        
        self.cleaning_stats['standardized_categories'] = standardization_changes
    
    def process_text_columns(self):
        """Process text columns using NLP techniques"""
        text_columns = self.detect_text_columns()
        text_cleaning_stats = {}
        
        for col in text_columns:
            # Store original word count
            original_word_count = self.original_df[col].astype(str).apply(lambda x: len(x.split())).sum()
            
            # Clean text data
            self.cleaned_df[col] = self.original_df[col].astype(str).apply(self.clean_text)
            
            # Calculate cleaned word count
            cleaned_word_count = self.cleaned_df[col].astype(str).apply(lambda x: len(x.split())).sum()
            
            text_cleaning_stats[col] = {
                'original_word_count': original_word_count,
                'cleaned_word_count': cleaned_word_count,
                'reduction_percentage': round(100 - (cleaned_word_count / original_word_count * 100), 2) if original_word_count > 0 else 0
            }
        
        self.cleaning_stats['text_cleaning'] = text_cleaning_stats
    
    def check_data_types(self):
        """Ensure correct data types for columns"""
        type_conversions = {}
        
        # Look for numerical data stored as strings
        for col in self.cleaned_df.columns:
            if self.cleaned_df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    # Check if column contains mostly numbers
                    numeric_count = pd.to_numeric(self.cleaned_df[col], errors='coerce').notna().sum()
                    if numeric_count > len(self.cleaned_df) * 0.8:  # If 80% can be converted to numeric
                        self.cleaned_df[col] = pd.to_numeric(self.cleaned_df[col], errors='coerce')
                        type_conversions[col] = f"object -> {self.cleaned_df[col].dtype}"
                except:
                    pass
                    
        self.cleaning_stats['type_conversions'] = type_conversions
    
    def detect_duplicates(self):
        """Detect and remove duplicate records"""
        # Count duplicates
        duplicates = self.cleaned_df.duplicated().sum()
        self.cleaning_stats['duplicates_removed'] = duplicates
        
        # Remove duplicates
        if duplicates > 0:
            self.cleaned_df = self.cleaned_df.drop_duplicates()
    
    def generate_visualizations(self):
        """Generate visualization data for the dashboard"""
        visualizations = {}
        
        # 1. Compare row counts
        visualizations['row_comparison'] = {
            'original': self.cleaning_stats['original_rows'],
            'cleaned': len(self.cleaned_df)
        }
        
        # 2. Missing values before/after
        visualizations['missing_values'] = {
            'before': self.cleaning_stats['missing_values_before'],
            'after': self.cleaning_stats['missing_values_after']
        }
        
        # 3. Column data types
        visualizations['data_types'] = {
            'original': self.original_df.dtypes.astype(str).to_dict(),
            'cleaned': self.cleaned_df.dtypes.astype(str).to_dict()
        }
        
        # 4. Text cleaning metrics
        if 'text_cleaning' in self.cleaning_stats:
            visualizations['text_cleaning'] = self.cleaning_stats['text_cleaning']
        
        # 5. Distribution plots for numerical columns (sample 3 columns)
        num_cols = self.cleaned_df.select_dtypes(include=['int64', 'float64']).columns[:3]
        distribution_plots = {}
        
        for col in num_cols:
            # Original distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(self.original_df[col].dropna(), color='blue', alpha=0.5, label='Original')
            sns.histplot(self.cleaned_df[col].dropna(), color='green', alpha=0.5, label='Cleaned')
            plt.title(f'Distribution of {col} - Before vs After Cleaning')
            plt.legend()
            
            # Save plot to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            distribution_plots[col] = plot_data
            
        visualizations['distribution_plots'] = distribution_plots
        
        # Store visualization data
        self.visualization_data = visualizations
        
        return visualizations
    
    def clean_data(self, filepath):
        """Main method to clean the healthcare dataset"""
        # Load data
        if not self.load_data(filepath):
            return False
        
        # Create a copy for cleaning
        self.cleaned_df = self.original_df.copy()
        
        # Apply all cleaning steps
        self.handle_missing_values()
        self.remove_outliers()
        self.standardize_categorical_data()
        self.process_text_columns()
        self.check_data_types()
        self.detect_duplicates()
        
        # Generate visualization data
        self.generate_visualizations()
        
        # Calculate final statistics
        self.cleaning_stats['final_rows'] = len(self.cleaned_df)
        self.cleaning_stats['final_columns'] = len(self.cleaned_df.columns)
        self.cleaning_stats['rows_removed'] = self.cleaning_stats['original_rows'] - self.cleaning_stats['final_rows']
        self.cleaning_stats['cleaning_percentage'] = round(
            (self.cleaning_stats['rows_removed'] + self.cleaning_stats['missing_values_before'] - 
             self.cleaning_stats['missing_values_after'] + self.cleaning_stats['outlier_total']) / 
            (self.cleaning_stats['original_rows'] * self.cleaning_stats['original_columns']) * 100, 2)
        
        return True
    
    def get_cleaning_stats(self):
        """Return cleaning statistics"""
        return self.cleaning_stats
    
    def get_visualization_data(self):
        """Return visualization data"""
        return self.visualization_data
    
    def save_cleaned_data(self, output_path):
        """Save cleaned data to CSV"""
        if self.cleaned_df is not None:
            self.cleaned_df.to_csv(output_path, index=False)
            return True
        return False

# Flask routes
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/clean', methods=['POST'])
def clean_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save the uploaded file temporarily
    temp_filepath = 'temp_upload.csv'
    file.save(temp_filepath)
    
    # Initialize the data cleaner and process the file
    cleaner = HealthcareDataCleaner()
    success = cleaner.clean_data(temp_filepath)
    
    if not success:
        os.remove(temp_filepath)
        return jsonify({'error': 'Failed to process file'}), 500
    
    # Save the cleaned file
    cleaned_filepath = 'cleaned_data.csv'
    cleaner.save_cleaned_data(cleaned_filepath)
    
    # Prepare response with statistics and visualization data
    response = {
        'status': 'success',
        'cleaning_stats': cleaner.get_cleaning_stats(),
        'visualization_data': cleaner.get_visualization_data()
    }
    
    # Clean up the temporary file
    os.remove(temp_filepath)
    
    return jsonify(response)

@app.route('/api/download', methods=['GET'])
def download_file():
    """Endpoint to download the cleaned CSV file"""
    if os.path.exists('cleaned_data.csv'):
        return send_file('cleaned_data.csv', as_attachment=True)
    else:
        return jsonify({'error': 'No cleaned file available'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)