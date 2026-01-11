import sys
import json
import os
import argparse
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# IMPORT SECTION
# ============================================================================

try:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.metrics import accuracy_score
    import joblib
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML packages not fully available: {e}", file=sys.stderr)
    ML_AVAILABLE = False

try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("Warning: boto3 not available", file=sys.stderr)

try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Warning: kagglehub not available", file=sys.stderr)

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
from urllib.parse import quote_plus
import re

try:
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not installed. Run: pip install sentence-transformers", file=sys.stderr)
    TRANSFORMERS_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SkinAnalysisConfig:
    """Configuration for skin analysis pipeline"""
    image_size: Tuple[int, int] = (224, 224)
    embedding_dim: int = 2048

    def __post_init__(self):
        self.skin_conditions = [
            'healthy', 'acne', 'eczema', 'psoriasis', 
            'rosacea', 'melanoma', 'dermatitis'
        ]
        self.skin_types = ['oily', 'dry', 'combination', 'sensitive', 'normal']
        self.severity_levels = ['none', 'mild', 'moderate', 'severe']

# ============================================================================
# S3 HELPER
# ============================================================================

def download_from_s3(bucket: str, key: str) -> str:
    """Download image from S3"""
    if not S3_AVAILABLE:
        raise ImportError("boto3 not installed")

    print(f"Downloading from s3://{bucket}/{key}...", file=sys.stderr)
    s3_client = boto3.client('s3')

    suffix = os.path.splitext(key)[1] or '.jpg'
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = temp_file.name
    temp_file.close()

    try:
        s3_client.download_file(bucket, key, temp_path)
        print(f"Downloaded to: {temp_path}", file=sys.stderr)
        return temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise Exception(f"S3 download failed: {str(e)}")

# ============================================================================
# RESNET50 EMBEDDER
# ============================================================================

class ResNetEmbedder:
    """Generate 2048-dim embeddings using ResNet50"""

    def __init__(self):
        if not ML_AVAILABLE:
            raise ImportError("ML packages not available")

        print(f"Loading ResNet50 model...", file=sys.stderr)

        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input

        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
        self.preprocess = preprocess_input

        print("✅ ResNet50 loaded (2048-dim embeddings)", file=sys.stderr)

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for ResNet50"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return self.preprocess(img_array)

    def generate_embedding(self, image_path: str) -> np.ndarray:
        """Generate 2048-dimensional embedding"""
        img_array = self.preprocess_image(image_path)
        embedding = self.model.predict(img_array, verbose=0)
        return embedding.flatten()

# ============================================================================
# MULTI-TASK CLASSIFIER
# ============================================================================

class SkinAnalysisClassifier:
    """Multi-output classifier for skin analysis"""

    def __init__(self, config: SkinAnalysisConfig):
        if not ML_AVAILABLE:
            raise ImportError("ML packages not available")

        self.config = config
        self.condition_classifier = None
        self.skin_type_classifier = None
        self.severity_classifier = None
        self.hydration_regressor = None
        self.pigmentation_regressor = None
        self.texture_regressor = None
        self.condition_encoder = LabelEncoder()
        self.skin_type_encoder = LabelEncoder()
        self.severity_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def predict(self, embedding: np.ndarray) -> Dict:
        """Generate predictions"""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        embedding_scaled = self.scaler.transform(embedding)

        cond_pred = self.condition_classifier.predict(embedding_scaled)[0]
        cond_proba = self.condition_classifier.predict_proba(embedding_scaled)[0]
        condition = self.condition_encoder.inverse_transform([cond_pred])[0]
        condition_conf = float(np.max(cond_proba))

        type_pred = self.skin_type_classifier.predict(embedding_scaled)[0]
        type_proba = self.skin_type_classifier.predict_proba(embedding_scaled)[0]
        skin_type = self.skin_type_encoder.inverse_transform([type_pred])[0]
        skin_type_conf = float(np.max(type_proba))

        sev_pred = self.severity_classifier.predict(embedding_scaled)[0]
        severity = self.severity_encoder.inverse_transform([sev_pred])[0]

        hydration = int(np.clip(self.hydration_regressor.predict(embedding_scaled)[0], 0, 100))
        pigmentation = int(np.clip(self.pigmentation_regressor.predict(embedding_scaled)[0], 0, 100))
        texture = int(np.clip(self.texture_regressor.predict(embedding_scaled)[0], 0, 100))

        clarity = 100 - (50 if condition != 'healthy' else 0)
        skin_score = int(np.mean([clarity, hydration, pigmentation, texture]))

        detected_conditions = []
        for idx, prob in enumerate(cond_proba):
            if prob > 0.1:
                cond_name = self.condition_encoder.inverse_transform([idx])[0]
                detected_conditions.append({
                    'condition': cond_name,
                    'confidence': float(prob),
                    'severity': severity if cond_name == condition else 'none'
                })
        detected_conditions = sorted(detected_conditions, key=lambda x: x['confidence'], reverse=True)

        return {
            'skinScore': skin_score,
            'scoreBreakdown': {
                'clarity': clarity,
                'hydration': hydration,
                'pigmentation': pigmentation,
                'texture': texture
            },
            'detectedConditions': detected_conditions,
            'skinType': skin_type,
            'skinTypeConfidence': round(skin_type_conf, 2),
            'primaryCondition': {
                'condition': condition,
                'confidence': round(condition_conf, 2),
                'severity': severity
            },
            'metrics': {
                'hydration': hydration,
                'pigmentation': pigmentation,
                'texture': texture
            }
        }

    def load(self, save_dir: str):
        """Load models"""
        self.condition_classifier = joblib.load(f'{save_dir}/condition_classifier.pkl')
        self.skin_type_classifier = joblib.load(f'{save_dir}/skin_type_classifier.pkl')
        self.severity_classifier = joblib.load(f'{save_dir}/severity_classifier.pkl')
        self.hydration_regressor = joblib.load(f'{save_dir}/hydration_regressor.pkl')
        self.pigmentation_regressor = joblib.load(f'{save_dir}/pigmentation_regressor.pkl')
        self.texture_regressor = joblib.load(f'{save_dir}/texture_regressor.pkl')
        self.scaler = joblib.load(f'{save_dir}/scaler.pkl')
        self.condition_encoder = joblib.load(f'{save_dir}/condition_encoder.pkl')
        self.skin_type_encoder = joblib.load(f'{save_dir}/skin_type_encoder.pkl')
        self.severity_encoder = joblib.load(f'{save_dir}/severity_encoder.pkl')
        print(f"Models loaded from {save_dir}/", file=sys.stderr)

# ============================================================================
# MOCK DATA GENERATOR
# ============================================================================

def generate_mock_result(user_id: str = None) -> Dict:
    """Generate varied mock results"""
    import random

    clarity = random.randint(70, 95)
    hydration = random.randint(60, 98)
    pigmentation = random.randint(65, 92)
    texture = random.randint(68, 94)
    skin_score = int((clarity + hydration + pigmentation + texture) / 4)

    return {
        'skinScore': skin_score,
        'scoreBreakdown': {
            'clarity': clarity,
            'hydration': hydration,
            'pigmentation': pigmentation,
            'texture': texture
        },
        'detectedConditions': [
            {'condition': 'healthy', 'confidence': 0.85, 'severity': 'none'},
        ],
        'skinType': random.choice(['normal', 'oily', 'combination', 'dry', 'sensitive']),
        'skinTypeConfidence': round(random.uniform(0.75, 0.95), 2),
        'primaryCondition': {
            'condition': 'healthy',
            'confidence': 0.85,
            'severity': 'none'
        },
        'metrics': {
            'hydration': hydration,
            'pigmentation': pigmentation,
            'texture': texture
        },
        'modelVersion': 'resnet50-v1.0',
        'userId': user_id,
        'timestamp': datetime.utcnow().isoformat()
    }

# ============================================================================
# SEMANTIC SEARCH ENGINE
# ============================================================================

class SemanticProductSearchEngine:
    """Real-time semantic search using HuggingFace Sentence Transformers"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required")
        
        print(f"\n🤖 Loading HuggingFace model: {model_name}...", file=sys.stderr)
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded ({self.embedding_dim}-dimensional embeddings)", file=sys.stderr)
    
    def create_user_query(self, analysis_result: Dict) -> str:
        """Generate semantic search query from skin analysis"""
        condition = analysis_result['primaryCondition']['condition']
        severity = analysis_result['primaryCondition']['severity']
        skin_type = analysis_result['skinType']
        metrics = analysis_result['scoreBreakdown']
        
        query_parts = [f"{skin_type} skin", f"{condition}"]
        
        if severity in ['moderate', 'severe']:
            query_parts.append(f"intensive treatment {condition}")
        
        if metrics['hydration'] < 70:
            query_parts.append("deep hydration moisturizing")
        if metrics['pigmentation'] < 70:
            query_parts.append("brightening even skin tone")
        if metrics['texture'] < 70:
            query_parts.append("smoothing texture")
        if metrics['clarity'] < 75:
            query_parts.append("clarifying pore-minimizing")
        
        ingredient_map = {
            'acne': 'salicylic acid niacinamide tea tree',
            'eczema': 'ceramides colloidal oatmeal centella',
            'rosacea': 'azelaic acid niacinamide green tea',
            'psoriasis': 'salicylic acid vitamin d',
            'dermatitis': 'hyaluronic acid ceramides squalane',
            'healthy': 'antioxidants vitamin c retinol peptides'
        }
        
        if condition in ingredient_map:
            query_parts.append(ingredient_map[condition])
        
        query = ' '.join(query_parts)
        print(f"🔍 Generated query: '{query}'", file=sys.stderr)
        return query
    
    def rank_products(self, user_query: str, products: List[Dict], top_k: int = 10) -> List[Dict]:
        """Rank products using semantic similarity"""
        print(f"\n🔄 Ranking {len(products)} products with semantic similarity...", file=sys.stderr)
        
        # Encode user query
        query_embedding = self.model.encode(user_query, convert_to_numpy=True)
        
        # Encode product descriptions
        product_texts = []
        for p in products:
            text_parts = [
                str(p.get('product_name', '')),
                str(p.get('brand', '')),
                str(p.get('description', '')),
                str(p.get('ingredients', '')),
                str(p.get('concerns', '')),
                str(p.get('benefits', ''))
            ]
            product_text = ' '.join([t for t in text_parts if t])
            product_texts.append(product_text)
        
        product_embeddings = self.model.encode(product_texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Compute cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        product_norms = product_embeddings / np.linalg.norm(product_embeddings, axis=1, keepdims=True)
        similarities = np.dot(product_norms, query_norm)
        
        # Add similarity scores
        ranked_products = []
        for idx, product in enumerate(products):
            product_copy = product.copy()
            product_copy['semantic_score'] = float(similarities[idx])
            ranked_products.append(product_copy)
        
        # Sort by similarity
        ranked_products.sort(key=lambda x: x['semantic_score'], reverse=True)
        
        print(f"✅ Ranking complete (Top score: {ranked_products[0]['semantic_score']:.3f})", file=sys.stderr)
        
        return ranked_products[:top_k]

# ============================================================================
# PRODUCT DATABASE
# ============================================================================

def get_curated_products() -> List[Dict]:
    """Curated skincare product database"""
    return [
        {
            'product_id': 'PROD_001',
            'product_name': 'Advanced Snail 96 Mucin Power Essence',
            'brand': 'COSRX',
            'price': 25.00,
            'rating': 4.6,
            'num_reviews': 8200,
            'category': 'serum',
            'concerns': 'hydration,texture,healing,dryness',
            'description': 'Lightweight essence with 96% snail secretion filtrate for deep hydration and skin repair',
            'ingredients': 'snail mucin, sodium hyaluronate, panthenol, allantoin',
            'benefits': 'intense hydration, skin repair, improves texture, soothes irritation',
            'source': 'Curated Database',
            'product_url': 'https://www.cosrx.com/products/advanced-snail-96-mucin-power-essence'
        },
        {
            'product_id': 'PROD_002',
            'product_name': 'Niacinamide 10% + Zinc 1%',
            'brand': 'The Ordinary',
            'price': 12.90,
            'rating': 4.4,
            'num_reviews': 15000,
            'category': 'serum',
            'concerns': 'acne,pores,texture,oiliness,blemishes',
            'description': 'High-strength vitamin and mineral formula to reduce blemishes and congestion',
            'ingredients': 'niacinamide, zinc pca, hyaluronic acid',
            'benefits': 'reduces breakouts, minimizes pores, controls oil, brightens',
            'source': 'Curated Database',
            'product_url': 'https://theordinary.com/en-us/niacinamide-10-zinc-1-serum-100415.html'
        },
        {
            'product_id': 'PROD_003',
            'product_name': 'C-Firma Fresh Day Serum',
            'brand': 'Drunk Elephant',
            'price': 80.00,
            'rating': 4.6,
            'num_reviews': 2890,
            'category': 'serum',
            'concerns': 'hyperpigmentation,dark spots,dullness,aging',
            'description': 'Potent vitamin C day serum that firms, brightens, and improves signs of photoaging',
            'ingredients': 'l-ascorbic acid, ferulic acid, vitamin e, pumpkin ferment extract',
            'benefits': 'brightens skin, fades dark spots, antioxidant protection, firms',
            'source': 'Curated Database',
            'product_url': 'https://www.drunkelephant.com/products/c-firma-serum'
        },
        {
            'product_id': 'PROD_004',
            'product_name': 'Retinol 0.5% in Squalane',
            'brand': 'The Ordinary',
            'price': 9.80,
            'rating': 4.3,
            'num_reviews': 8900,
            'category': 'treatment',
            'concerns': 'aging,wrinkles,texture,fine lines',
            'description': 'Moderate-strength retinol serum for visible signs of aging',
            'ingredients': 'retinol, squalane, vitamin e',
            'benefits': 'reduces wrinkles, improves texture, cell renewal, anti-aging',
            'source': 'Curated Database',
            'product_url': 'https://theordinary.com/en-us/retinol-0-5-in-squalane-100446.html'
        },
        {
            'product_id': 'PROD_005',
            'product_name': 'Protini Polypeptide Cream',
            'brand': 'Drunk Elephant',
            'price': 72.00,
            'rating': 4.7,
            'num_reviews': 4200,
            'category': 'moisturizer',
            'concerns': 'aging,firmness,hydration,dryness',
            'description': 'Protein moisturizer that restores younger, revived-looking skin',
            'ingredients': 'signal peptides, growth factors, amino acids, pygmy waterlily',
            'benefits': 'firms skin, improves elasticity, deep hydration, strengthens barrier',
            'source': 'Curated Database',
            'product_url': 'https://www.drunkelephant.com/products/protini-cream'
        },
        {
            'product_id': 'PROD_006',
            'product_name': 'Salicylic Acid 2% Solution',
            'brand': 'The Ordinary',
            'price': 8.50,
            'rating': 4.2,
            'num_reviews': 12000,
            'category': 'treatment',
            'concerns': 'acne,blackheads,pores,texture,congestion',
            'description': 'Direct acid exfoliant that helps fight blemishes and improve skin texture',
            'ingredients': 'salicylic acid, witch hazel water',
            'benefits': 'unclogs pores, reduces acne, exfoliates, controls oil',
            'source': 'Curated Database',
            'product_url': 'https://theordinary.com/en-us/salicylic-acid-2-solution-100411.html'
        },
        {
            'product_id': 'PROD_007',
            'product_name': 'Centella Calming Gel Cream',
            'brand': 'COSRX',
            'price': 18.00,
            'rating': 4.5,
            'num_reviews': 3800,
            'category': 'moisturizer',
            'concerns': 'redness,irritation,sensitivity,rosacea',
            'description': 'Lightweight gel-cream that soothes and hydrates sensitive skin',
            'ingredients': 'centella asiatica extract, niacinamide, hyaluronic acid',
            'benefits': 'calms irritation, reduces redness, hydrates, strengthens skin barrier',
            'source': 'Curated Database',
            'product_url': 'https://www.cosrx.com/products/pure-fit-cica-cream'
        },
        {
            'product_id': 'PROD_008',
            'product_name': 'Azelaic Acid 10% Suspension',
            'brand': 'The Ordinary',
            'price': 9.80,
            'rating': 4.2,
            'num_reviews': 7900,
            'category': 'treatment',
            'concerns': 'rosacea,hyperpigmentation,texture,acne,redness',
            'description': 'Brightening cream-gel formula that evens skin tone and texture',
            'ingredients': 'azelaic acid, caprylic/capric triglyceride',
            'benefits': 'brightens, evens tone, reduces blemishes, calms redness',
            'source': 'Curated Database',
            'product_url': 'https://theordinary.com/en-us/azelaic-acid-suspension-10-100410.html'
        },
        {
            'product_id': 'PROD_009',
            'product_name': 'Hyaluronic Acid 2% + B5',
            'brand': 'The Ordinary',
            'price': 8.90,
            'rating': 4.5,
            'num_reviews': 18000,
            'category': 'serum',
            'concerns': 'dryness,dehydration,fine lines,texture',
            'description': 'Multi-depth hydration serum with hyaluronic acid and vitamin B5',
            'ingredients': 'sodium hyaluronate, hyaluronic acid, panthenol',
            'benefits': 'intense hydration, plumps skin, reduces fine lines, improves texture',
            'source': 'Curated Database',
            'product_url': 'https://theordinary.com/en-us/hyaluronic-acid-2-b5-serum-100420.html'
        },
        {
            'product_id': 'PROD_010',
            'product_name': 'Good Genes All-In-One Lactic Acid Treatment',
            'brand': 'Sunday Riley',
            'price': 85.00,
            'rating': 4.5,
            'num_reviews': 5400,
            'category': 'treatment',
            'concerns': 'texture,dullness,aging,hyperpigmentation',
            'description': 'Concentrated lactic acid treatment that exfoliates and brightens',
            'ingredients': 'lactic acid, licorice extract, lemongrass, aloe',
            'benefits': 'exfoliates, brightens, improves texture, reveals radiance',
            'source': 'Curated Database',
            'product_url': 'https://www.sundayriley.com/products/good-genes-lactic-acid-treatment'
        },
        {
            'product_id': 'PROD_011',
            'product_name': 'Double Repair Face Moisturizer',
            'brand': 'CeraVe',
            'price': 19.99,
            'rating': 4.6,
            'num_reviews': 9200,
            'category': 'moisturizer',
            'concerns': 'dryness,eczema,sensitivity,barrier damage',
            'description': 'Fragrance-free moisturizer with ceramides and hyaluronic acid',
            'ingredients': 'ceramides, hyaluronic acid, niacinamide, glycerin',
            'benefits': 'restores skin barrier, intense hydration, soothes dryness',
            'source': 'Curated Database',
            'product_url': 'https://www.cerave.com/skincare/moisturizers/face-moisturizer-am-spf-30'
        },
        {
            'product_id': 'PROD_012',
            'product_name': 'Honey Halo Ceramide Face Moisturizer',
            'brand': 'Farmacy',
            'price': 48.00,
            'rating': 4.4,
            'num_reviews': 2100,
            'category': 'moisturizer',
            'concerns': 'dryness,sensitivity,dullness,dehydration',
            'description': 'Hydrating gel-cream with buckwheat honey and ceramides',
            'ingredients': 'buckwheat honey, ceramides, hyaluronic acid, echinacea',
            'benefits': 'hydrates, strengthens barrier, soothes, adds glow',
            'source': 'Curated Database',
            'product_url': 'https://www.farmacybeauty.com/products/honey-halo'
        }
    ]

# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

class HybridRecommender:
    """Hybrid recommendation system with semantic search and multi-factor ranking"""
    
    def __init__(self):
        self.products = get_curated_products()
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_engine = SemanticProductSearchEngine('sentence-transformers/all-MiniLM-L6-v2')
                self.use_semantic = True
            except Exception as e:
                print(f"Warning: Semantic engine failed to load: {e}", file=sys.stderr)
                self.use_semantic = False
        else:
            self.use_semantic = False
    
    def recommend(self, analysis_result: Dict, top_n: int = 8) -> List[Dict]:
        """Generate personalized product recommendations"""
        print("\n" + "="*70, file=sys.stderr)
        print("🚀 GENERATING PERSONALIZED RECOMMENDATIONS", file=sys.stderr)
        print("="*70, file=sys.stderr)
        
        # Step 1: Semantic ranking (if available)
        if self.use_semantic:
            user_query = self.semantic_engine.create_user_query(analysis_result)
            ranked_products = self.semantic_engine.rank_products(user_query, self.products, top_k=top_n * 2)
        else:
            print("⚠️  Using fallback ranking (no semantic search)", file=sys.stderr)
            ranked_products = self.products.copy()
            for p in ranked_products:
                p['semantic_score'] = 0.5
        
        # Step 2: Apply quality filters
        filtered_products = self._apply_filters(ranked_products, analysis_result)
        
        # Step 3: Hybrid scoring
        final_recommendations = self._hybrid_score(filtered_products, analysis_result, top_n)
        
        # Step 4: Add explanations
        for rec in final_recommendations:
            rec['why_recommended'] = self._generate_explanation(rec, analysis_result)
        
        print(f"\n✅ Generated {len(final_recommendations)} recommendations", file=sys.stderr)
        if final_recommendations:
            print(f"   Top: {final_recommendations[0]['product_name']} (score: {final_recommendations[0]['final_score']:.3f})", file=sys.stderr)
        print("="*70, file=sys.stderr)
        
        return final_recommendations
    
    def _apply_filters(self, products: List[Dict], analysis_result: Dict) -> List[Dict]:
        """Filter products by quality and relevance"""
        filtered = []
        
        for product in products:
            # Quality threshold
            if product['rating'] < 3.5 and product['num_reviews'] > 100:
                continue
            
            # Price filter for severe conditions
            if product['price'] > 150:
                if analysis_result['primaryCondition']['severity'] not in ['severe', 'moderate']:
                    continue
            
            filtered.append(product)
        
        return filtered
    
    def _hybrid_score(self, products: List[Dict], analysis_result: Dict, top_n: int) -> List[Dict]:
        """Combine multiple signals for final ranking"""
        for product in products:
            semantic_score = product.get('semantic_score', 0.5)
            
            # Quality score (Bayesian average)
            rating = product.get('rating', 0)
            num_reviews = product.get('num_reviews', 0)
            quality_score = (4.0 * 50 + rating * num_reviews) / (50 + num_reviews)
            quality_score_norm = quality_score / 5.0
            
            # Price value score
            price = product.get('price', 50)
            price_score = 1.0 / (1.0 + np.log1p(price / 10))
            
            # Concern alignment
            product_concerns = set(product.get('concerns', '').lower().split(','))
            user_condition = analysis_result['primaryCondition']['condition'].lower()
            concern_match = 1.0 if user_condition in product_concerns else 0.5
            
            # Skin type match
            skin_type = analysis_result['skinType'].lower()
            product_name_lower = product['product_name'].lower()
            skin_type_match = 1.0 if skin_type in product_name_lower else 0.8
            
            # Hybrid score
            final_score = (
                0.40 * semantic_score +
                0.25 * quality_score_norm +
                0.20 * concern_match +
                0.10 * price_score +
                0.05 * skin_type_match
            )
            
            product['final_score'] = final_score
            product['score_breakdown'] = {
                'semantic': round(semantic_score, 3),
                'quality': round(quality_score_norm, 3),
                'concern_match': round(concern_match, 3),
                'price_value': round(price_score, 3),
                'skin_type_match': round(skin_type_match, 3)
            }
        
        # Sort by final score
        products.sort(key=lambda x: x['final_score'], reverse=True)
        return products[:top_n]
    
    def _generate_explanation(self, product: Dict, analysis_result: Dict) -> str:
        """Generate human-readable explanation"""
        reasons = []
        
        # Semantic relevance
        if product.get('semantic_score', 0) > 0.7:
            reasons.append("Highly relevant for your skin profile")
        
        # Condition targeting
        condition = analysis_result['primaryCondition']['condition']
        if condition.lower() in product.get('concerns', '').lower():
            reasons.append(f"Specifically targets {condition}")
        
        # Quality
        if product.get('rating', 0) >= 4.5 and product.get('num_reviews', 0) > 1000:
            reasons.append("Highly rated by thousands of users")
        elif product.get('rating', 0) >= 4.5:
            reasons.append("Excellent user ratings")
        
        # Price value
        if product.get('price', 100) < 20:
            reasons.append("Excellent value for money")
        elif product.get('price', 100) < 40:
            reasons.append("Great value")
        
        # Ingredients
        metrics = analysis_result['scoreBreakdown']
        ingredients = product.get('ingredients', '').lower()
        
        if metrics['hydration'] < 70 and any(ing in ingredients for ing in ['hyaluronic', 'ceramide', 'squalane']):
            reasons.append("Contains key hydrating ingredients")
        
        if metrics['pigmentation'] < 70 and any(ing in ingredients for ing in ['vitamin c', 'niacinamide', 'azelaic']):
            reasons.append("Brightening formula")
        
        return " • ".join(reasons) if reasons else "Good match for your needs"

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_with_realtime_recommendations(bucket: str, key: str, user_id: str = None,
                                         use_mock: bool = False, use_cache: bool = False):
    """Main analysis with real-time recommendations"""
    
    try:
        print(f"\n🎯 Starting analysis for user: {user_id}", file=sys.stderr)
        print(f"   Bucket: {bucket}", file=sys.stderr)
        print(f"   Key: {key}", file=sys.stderr)
        print(f"   Mode: {'Mock' if use_mock else 'Real'} | Cache: {use_cache}", file=sys.stderr)
        
        # Step 1: Skin Analysis
        if use_mock or not ML_AVAILABLE:
            print("\n📊 Using mock skin analysis data", file=sys.stderr)
            analysis_result = generate_mock_result(user_id)
        else:
            print("\n📊 Performing real skin analysis...", file=sys.stderr)
            local_path = download_from_s3(bucket, key)
            
            try:
                model_dir = './models/skin_analysis'
                if Path(model_dir).exists():
                    embedder = ResNetEmbedder()
                    classifier = SkinAnalysisClassifier(SkinAnalysisConfig())
                    classifier.load(model_dir)
                    
                    embedding = embedder.generate_embedding(local_path)
                    analysis_result = classifier.predict(embedding)
                else:
                    print("⚠️  No trained models found, using mock data", file=sys.stderr)
                    analysis_result = generate_mock_result(user_id)
            finally:
                if os.path.exists(local_path):
                    os.remove(local_path)
        
        print(f"\n   Skin Score: {analysis_result['skinScore']}", file=sys.stderr)
        print(f"   Skin Type: {analysis_result['skinType']}", file=sys.stderr)
        print(f"   Primary Condition: {analysis_result['primaryCondition']['condition']}", file=sys.stderr)
        
        # Step 2: Generate recommendations
        recommender = HybridRecommender()
        recommendations = recommender.recommend(analysis_result, top_n=8)
        
        # Step 3: Build final result
        final_result = {
            **analysis_result,
            's3Bucket': bucket,
            's3Key': key,
            'userId': user_id,
            'timestamp': datetime.utcnow().isoformat(),
            'recommendations': {
                'products': recommendations,
                'count': len(recommendations),
                'recommendation_engine': 'HuggingFace Transformers + Hybrid Ranking' if TRANSFORMERS_AVAILABLE else 'Hybrid Ranking',
                'model': 'sentence-transformers/all-MiniLM-L6-v2' if TRANSFORMERS_AVAILABLE else 'fallback',
                'total_products_analyzed': len(get_curated_products())
            }
        }
        
        print("\n" + "="*70)
        print("📋 ANALYSIS COMPLETE")
        print("="*70)
        print(f"Skin Score: {final_result['skinScore']}/100")
        print(f"Recommendations: {len(recommendations)} products")
        print("="*70 + "\n")
        
        print("--- JSON OUTPUT START ---")
        print(json.dumps(final_result, indent=2))
        print("--- JSON OUTPUT END ---")
        
        sys.exit(0)
    except Exception as e:
        error_output = {
            'error': str(e),
            'type': type(e).__name__,
            'timestamp': datetime.utcnow().isoformat()
        }
        print(json.dumps(error_output), file=sys.stderr)
        sys.exit(1)

def generate_recommendations_from_json(json_path: str):
    """Generate recommendations from existing analysis JSON"""
    try:
        print(f"Reading analysis from: {json_path}", file=sys.stderr)
        
        with open(json_path, 'r') as f:
            analysis_result = json.load(f)
            
        recommender = HybridRecommender()
        recommendations = recommender.recommend(analysis_result, top_n=8)
        
        result = {
            'recommendations': {
                'products': recommendations,
                'count': len(recommendations),
                'recommendation_engine': 'HuggingFace Transformers + Hybrid Ranking' if TRANSFORMERS_AVAILABLE else 'Hybrid Ranking'
            }
        }

        print("--- JSON OUTPUT START ---")
        print(json.dumps(result, indent=2))
        print("--- JSON OUTPUT END ---")
        sys.exit(0)
        
    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skin Care Recommendations')
    parser.add_argument('--bucket', type=str, help='S3 bucket')
    parser.add_argument('--key', type=str, help='S3 key')
    parser.add_argument('--userId', type=str, help='User ID')
    parser.add_argument('--mock', action='store_true', help='Use mock data')
    parser.add_argument('--input', type=str, help='Path to input analysis JSON')

    args = parser.parse_args()

    if args.input:
        generate_recommendations_from_json(args.input)
    elif args.bucket and args.key:
        analyze_with_realtime_recommendations(args.bucket, args.key, args.userId, use_mock=args.mock)
    else:
        parser.print_help()
        sys.exit(1)

        sys.exit(1)

# ============================================================================
# MAIN CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Real-time Skin Analysis & Recommendations with HuggingFace Transformers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mock analysis with cached products (fastest)
  python recommendationController.py --bucket test --key test.jpg --userId user123 --mock --cache
  
  # Real analysis with real-time recommendations
  python recommendationController.py --bucket my-bucket --key image.jpg --userId user123
        """
    )
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket name')
    parser.add_argument('--key', type=str, required=True, help='S3 object key')
    parser.add_argument('--userId', type=str, help='User ID for tracking')
    parser.add_argument('--mock', action='store_true', help='Use mock skin analysis (faster)')
    parser.add_argument('--cache', action='store_true', help='Use cached products (faster, for testing)')
    
    args = parser.parse_args()
    
    print("="*70, file=sys.stderr)
    print("SKINCARE AI - RECOMMENDATION CONTROLLER", file=sys.stderr)
    print("="*70, file=sys.stderr)
    
    analyze_with_realtime_recommendations(
        bucket=args.bucket,
        key=args.key,
        user_id=args.userId,
        use_mock=args.mock,
        use_cache=args.cache
    )
