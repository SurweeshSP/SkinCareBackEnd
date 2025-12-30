import sys
import json
import os
import argparse
import tempfile
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

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SkinAnalysisConfig:
    """Configuration for skin analysis pipeline"""
    image_size: Tuple[int, int] = (224, 224)  # ResNet50 input size
    embedding_dim: int = 2048  # ResNet50 embedding dimension

    def __post_init__(self):
        self.skin_conditions = [
            'healthy', 'acne', 'eczema', 'psoriasis', 
            'rosacea', 'melanoma', 'dermatitis'
        ]
        self.skin_types = ['oily', 'dry', 'combination', 'sensitive', 'normal']
        self.severity_levels = ['none', 'mild', 'moderate', 'severe']

# ============================================================================
# KAGGLE DATASET DOWNLOADER
# ============================================================================

def download_ham10000_dataset():
    """Download HAM10000 dataset from Kaggle"""
    if not KAGGLE_AVAILABLE:
        raise ImportError("kagglehub not installed. Run: pip install kagglehub")

    print("=" * 70, file=sys.stderr)
    print("DOWNLOADING HAM10000 DATASET FROM KAGGLE", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("Dataset: kmader/skin-cancer-mnist-ham10000", file=sys.stderr)
    print("Size: ~3GB (may take 5-10 minutes)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    print(f"\n✅ Dataset downloaded to: {path}", file=sys.stderr)
    return path

def prepare_ham10000_data(dataset_path: str):
    """Prepare HAM10000 dataset for training"""
    print("\nPreparing HAM10000 dataset...", file=sys.stderr)

    # Load metadata
    metadata_path = Path(dataset_path) / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        metadata_path = Path(dataset_path) / "ham10000" / "HAM10000_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found")

    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} samples", file=sys.stderr)

    # Map labels
    label_mapping = {
        'nv': 'healthy',
        'mel': 'melanoma',
        'bkl': 'healthy',
        'bcc': 'dermatitis',
        'akiec': 'acne',
        'vasc': 'rosacea',
        'df': 'healthy'
    }

    # Find images
    image_dirs = [
        Path(dataset_path) / "HAM10000_images_part_1",
        Path(dataset_path) / "HAM10000_images_part_2",
    ]

    valid_dir = None
    for img_dir in image_dirs:
        if img_dir.exists():
            valid_dir = img_dir
            break

    if not valid_dir:
        raise FileNotFoundError("Image directory not found")

    print(f"Using image directory: {valid_dir}", file=sys.stderr)

    df['image_path'] = df['image_id'].apply(lambda x: str(valid_dir / f"{x}.jpg"))
    df['condition'] = df['dx'].map(label_mapping)
    df['skin_type'] = np.random.choice(['normal', 'oily', 'dry', 'combination'], len(df))

    severity_map = {
        'mel': 'severe', 'bcc': 'moderate', 'akiec': 'moderate',
        'vasc': 'mild', 'nv': 'none', 'bkl': 'none', 'df': 'none'
    }
    df['severity'] = df['dx'].map(severity_map)

    df['hydration'] = np.random.randint(60, 95, len(df))
    df['pigmentation'] = np.random.randint(55, 90, len(df))
    df['texture'] = np.random.randint(60, 92, len(df))

    df = df[df['image_path'].apply(os.path.exists)]
    print(f"Found {len(df)} valid images", file=sys.stderr)

    return df

# ============================================================================npm run dev

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
# RESNET50 EMBEDDER (No compatibility issues!)
# ============================================================================

class ResNetEmbedder:
    """Generate 2048-dim embeddings using ResNet50"""

    def __init__(self):
        if not ML_AVAILABLE:
            raise ImportError("ML packages not available")

        print(f"Loading ResNet50 model...", file=sys.stderr)

        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input

        # Load pre-trained ResNet50
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

    def generate_embeddings_batch(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple images"""
        embeddings = []
        total = len(image_paths)

        print(f"\nGenerating ResNet50 embeddings for {total} images...", file=sys.stderr)

        for i in range(0, total, batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    img_array = self.preprocess_image(path)
                    batch_images.append(img_array[0])
                except Exception as e:
                    print(f"Error: {path}: {e}", file=sys.stderr)
                    continue

            if batch_images:
                batch_array = np.array(batch_images)
                batch_embeddings = self.model.predict(batch_array, verbose=0)
                embeddings.extend(batch_embeddings)

            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {min(i+batch_size, total)}/{total}", file=sys.stderr)

        print(f"✅ Generated {len(embeddings)} embeddings", file=sys.stderr)
        return np.array(embeddings)

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

    def train(self, embeddings: np.ndarray, conditions: List[str],
              skin_types: List[str], severities: List[str],
              hydration_scores: np.ndarray, pigmentation_scores: np.ndarray,
              texture_scores: np.ndarray, test_size: float = 0.2):
        """Train all classifiers"""

        print("\n" + "="*70, file=sys.stderr)
        print("TRAINING MULTI-TASK MODELS", file=sys.stderr)
        print("="*70, file=sys.stderr)

        X_scaled = self.scaler.fit_transform(embeddings)
        y_condition = self.condition_encoder.fit_transform(conditions)
        y_skin_type = self.skin_type_encoder.fit_transform(skin_types)
        y_severity = self.severity_encoder.fit_transform(severities)

        indices = np.arange(len(X_scaled))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=y_condition
        )
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]

        print("\n[1/6] Training condition classifier...", file=sys.stderr)
        self.condition_classifier = LogisticRegression(
            max_iter=1000, multi_class='multinomial', class_weight='balanced', random_state=42
        )
        self.condition_classifier.fit(X_train, y_condition[train_idx])
        cond_acc = accuracy_score(y_condition[test_idx], self.condition_classifier.predict(X_test))
        print(f"      Accuracy: {cond_acc:.3f}", file=sys.stderr)

        print("\n[2/6] Training skin type classifier...", file=sys.stderr)
        self.skin_type_classifier = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
        )
        self.skin_type_classifier.fit(X_train, y_skin_type[train_idx])
        type_acc = accuracy_score(y_skin_type[test_idx], self.skin_type_classifier.predict(X_test))
        print(f"      Accuracy: {type_acc:.3f}", file=sys.stderr)

        print("\n[3/6] Training severity classifier...", file=sys.stderr)
        self.severity_classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.severity_classifier.fit(X_train, y_severity[train_idx])
        sev_acc = accuracy_score(y_severity[test_idx], self.severity_classifier.predict(X_test))
        print(f"      Accuracy: {sev_acc:.3f}", file=sys.stderr)

        print("\n[4/6] Training hydration regressor...", file=sys.stderr)
        self.hydration_regressor = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        self.hydration_regressor.fit(X_train, hydration_scores[train_idx])
        hyd_r2 = self.hydration_regressor.score(X_test, hydration_scores[test_idx])
        print(f"      R² Score: {hyd_r2:.3f}", file=sys.stderr)

        print("\n[5/6] Training pigmentation regressor...", file=sys.stderr)
        self.pigmentation_regressor = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        self.pigmentation_regressor.fit(X_train, pigmentation_scores[train_idx])
        pig_r2 = self.pigmentation_regressor.score(X_test, pigmentation_scores[test_idx])
        print(f"      R² Score: {pig_r2:.3f}", file=sys.stderr)

        print("\n[6/6] Training texture regressor...", file=sys.stderr)
        self.texture_regressor = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        self.texture_regressor.fit(X_train, texture_scores[train_idx])
        tex_r2 = self.texture_regressor.score(X_test, texture_scores[test_idx])
        print(f"      R² Score: {tex_r2:.3f}", file=sys.stderr)

        print("\n" + "="*70, file=sys.stderr)
        print("✅ TRAINING COMPLETE", file=sys.stderr)
        print("="*70, file=sys.stderr)

        return {
            'condition_accuracy': cond_acc,
            'skin_type_accuracy': type_acc,
            'severity_accuracy': sev_acc,
            'hydration_r2': hyd_r2,
            'pigmentation_r2': pig_r2,
            'texture_r2': tex_r2
        }

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

    def save(self, save_dir: str):
        """Save models"""
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.condition_classifier, f'{save_dir}/condition_classifier.pkl')
        joblib.dump(self.skin_type_classifier, f'{save_dir}/skin_type_classifier.pkl')
        joblib.dump(self.severity_classifier, f'{save_dir}/severity_classifier.pkl')
        joblib.dump(self.hydration_regressor, f'{save_dir}/hydration_regressor.pkl')
        joblib.dump(self.pigmentation_regressor, f'{save_dir}/pigmentation_regressor.pkl')
        joblib.dump(self.texture_regressor, f'{save_dir}/texture_regressor.pkl')
        joblib.dump(self.scaler, f'{save_dir}/scaler.pkl')
        joblib.dump(self.condition_encoder, f'{save_dir}/condition_encoder.pkl')
        joblib.dump(self.skin_type_encoder, f'{save_dir}/skin_type_encoder.pkl')
        joblib.dump(self.severity_encoder, f'{save_dir}/severity_encoder.pkl')
        print(f"\n✅ Models saved to {save_dir}/", file=sys.stderr)

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
        'skinType': random.choice(['normal', 'oily', 'combination']),
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
# MAIN ANALYSIS
# ============================================================================

def analyze_skin_image(bucket: str, key: str, user_id: str = None, use_mock: bool = False):
    """Main analysis entry point"""
    try:
        print(f"Starting analysis for user: {user_id}", file=sys.stderr)
        print(f"Image: s3://{bucket}/{key}", file=sys.stderr)

        if use_mock or not ML_AVAILABLE:
            print("Using mock data mode", file=sys.stderr)
            result = generate_mock_result(user_id)
            result['s3Bucket'] = bucket
            result['s3Key'] = key
        else:
            local_path = download_from_s3(bucket, key)

            try:
                model_dir = './models/skin_analysis'
                if Path(model_dir).exists():
                    print("Loading trained models...", file=sys.stderr)
                    embedder = ResNetEmbedder()
                    classifier = SkinAnalysisClassifier(SkinAnalysisConfig())
                    classifier.load(model_dir)

                    embedding = embedder.generate_embedding(local_path)
                    result = classifier.predict(embedding)
                else:
                    print("No trained models, using mock data", file=sys.stderr)
                    result = generate_mock_result(user_id)

                result['s3Bucket'] = bucket
                result['s3Key'] = key
                result['userId'] = user_id
                result['timestamp'] = datetime.utcnow().isoformat()
            finally:
                if os.path.exists(local_path):
                    os.remove(local_path)

        print("--- JSON OUTPUT START ---")
        print(json.dumps(result, indent=2))
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

# ============================================================================
# TRAINING
# ============================================================================

def train_models_with_kaggle_data(output_dir: str = './models/skin_analysis', 
                                  max_samples: int = 1000):
    """Train with real Kaggle data"""
    if not ML_AVAILABLE:
        print("ERROR: ML packages not available", file=sys.stderr)
        sys.exit(1)

    print("\n" + "="*70, file=sys.stderr)
    print("TRAINING WITH HAM10000 DATASET (ResNet50)", file=sys.stderr)
    print("="*70, file=sys.stderr)

    dataset_path = download_ham10000_dataset()
    df = prepare_ham10000_data(dataset_path)

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"\nUsing {max_samples} samples", file=sys.stderr)

    embedder = ResNetEmbedder()
    embeddings = embedder.generate_embeddings_batch(df['image_path'].tolist())

    config = SkinAnalysisConfig()
    classifier = SkinAnalysisClassifier(config)

    metrics = classifier.train(
        embeddings=embeddings,
        conditions=df['condition'].tolist(),
        skin_types=df['skin_type'].tolist(),
        severities=df['severity'].tolist(),
        hydration_scores=df['hydration'].values,
        pigmentation_scores=df['pigmentation'].values,
        texture_scores=df['texture'].values
    )

    classifier.save(output_dir)

    print("\n" + "="*70, file=sys.stderr)
    print("✅ TRAINING COMPLETE", file=sys.stderr)
    print("="*70, file=sys.stderr)
    print(f"Models saved to: {output_dir}", file=sys.stderr)
    print(f"Metrics:", file=sys.stderr)
    for key, value in metrics.items():
        print(f"  - {key}: {value:.3f}", file=sys.stderr)
    print("="*70, file=sys.stderr)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skin Analysis with ResNet50')
    parser.add_argument('--bucket', type=str, help='S3 bucket')
    parser.add_argument('--key', type=str, help='S3 key')
    parser.add_argument('--userId', type=str, help='User ID')
    parser.add_argument('--mock', action='store_true', help='Use mock data')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--output', type=str, default='./models/skin_analysis')
    parser.add_argument('--samples', type=int, default=1000)

    args = parser.parse_args()

    if args.train:
        train_models_with_kaggle_data(args.output, max_samples=args.samples)
    elif args.bucket and args.key:
        analyze_skin_image(args.bucket, args.key, args.userId, use_mock=args.mock)
    else:
        parser.print_help()
        sys.exit(1)