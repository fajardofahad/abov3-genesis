"""
ABOV3 Genesis - ML-Powered Debug Engine
Advanced AI/ML debugging system with Claude-level intelligence
"""

import sys
import os
import ast
import json
import pickle
import hashlib
import logging
import numpy as np
import pandas as pd
import threading
import asyncio
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Set
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import re
from functools import lru_cache
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ML/AI imports with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class ErrorEmbedding:
    """Error representation in vector space"""
    error_id: str
    error_type: str
    message: str
    context: str
    embedding: np.ndarray
    timestamp: datetime
    severity: int
    confidence: float = 0.0


@dataclass
class CodePattern:
    """Code pattern for ML analysis"""
    pattern_id: str
    code_snippet: str
    ast_features: Dict[str, Any]
    complexity_score: float
    risk_score: float
    frequency: int = 1
    last_seen: datetime = field(default_factory=datetime.now)


@dataclass
class FixSuggestion:
    """AI-generated fix suggestion"""
    fix_id: str
    original_code: str
    fixed_code: str
    explanation: str
    confidence: float
    success_rate: float
    application_count: int = 0
    feedback_score: float = 0.0


class MLDebugLevel(Enum):
    """ML-enhanced debug levels"""
    BASIC = 1
    PATTERN_RECOGNITION = 2
    SEMANTIC_ANALYSIS = 3
    PREDICTIVE = 4
    CLAUDE_INTELLIGENCE = 5


class TransformerErrorAnalyzer:
    """Transformer-based error pattern recognition"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.model_name = "microsoft/codebert-base"
        self.initialized = False
        self.error_embeddings = {}
        self.pattern_clusters = {}
        
        if HAS_TRANSFORMERS:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize transformer model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            if HAS_TORCH and torch.cuda.is_available():
                self.device = "cuda"
                self.model.to(self.device)
            
            self.initialized = True
            logging.info(f"Transformer model initialized: {self.model_name}")
        except Exception as e:
            logging.warning(f"Failed to initialize transformer model: {e}")
            self.initialized = False
    
    def encode_error(self, error_text: str, context: str = "") -> np.ndarray:
        """Encode error message and context into embeddings"""
        if not self.initialized:
            return self._fallback_encoding(error_text, context)
        
        try:
            combined_text = f"{error_text} [SEP] {context}"
            inputs = self.tokenizer(
                combined_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            return embeddings[0]
        except Exception as e:
            logging.warning(f"Transformer encoding failed: {e}")
            return self._fallback_encoding(error_text, context)
    
    def _fallback_encoding(self, error_text: str, context: str = "") -> np.ndarray:
        """Fallback encoding using simple feature extraction"""
        if not HAS_SKLEARN:
            # Basic hash-based encoding
            combined = f"{error_text} {context}"
            hash_val = int(hashlib.md5(combined.encode()).hexdigest(), 16)
            return np.array([hash_val % 1000 / 1000.0] * 128)
        
        vectorizer = TfidfVectorizer(max_features=128, stop_words='english')
        combined_text = f"{error_text} {context}"
        
        try:
            embedding = vectorizer.fit_transform([combined_text]).toarray()[0]
            return embedding
        except:
            return np.random.random(128)
    
    def find_similar_errors(self, error_embedding: np.ndarray, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Find similar errors using embedding similarity"""
        similar_errors = []
        
        for error_id, stored_embedding in self.error_embeddings.items():
            similarity = self._cosine_similarity(error_embedding, stored_embedding.embedding)
            if similarity > threshold:
                similar_errors.append((error_id, similarity))
        
        return sorted(similar_errors, key=lambda x: x[1], reverse=True)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            return dot_product / norms if norms > 0 else 0.0
        except:
            return 0.0
    
    def store_error_embedding(self, error: ErrorEmbedding):
        """Store error embedding for future similarity searches"""
        self.error_embeddings[error.error_id] = error
    
    def cluster_errors(self, min_samples: int = 5) -> Dict[str, List[str]]:
        """Cluster errors by similarity"""
        if not self.error_embeddings or not HAS_SKLEARN:
            return {}
        
        embeddings = np.array([e.embedding for e in self.error_embeddings.values()])
        error_ids = list(self.error_embeddings.keys())
        
        try:
            clusterer = DBSCAN(eps=0.3, min_samples=min_samples)
            cluster_labels = clusterer.fit_predict(embeddings)
            
            clusters = defaultdict(list)
            for error_id, label in zip(error_ids, cluster_labels):
                if label != -1:  # -1 is noise in DBSCAN
                    clusters[f"cluster_{label}"].append(error_id)
            
            return dict(clusters)
        except Exception as e:
            logging.warning(f"Error clustering failed: {e}")
            return {}


class SemanticCodeAnalyzer:
    """Neural network-based code understanding"""
    
    def __init__(self):
        self.ast_vectorizer = None
        self.complexity_model = None
        self.risk_model = None
        self.code_patterns = {}
        self.trained = False
        
        if HAS_SKLEARN:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for code analysis"""
        try:
            # Initialize models
            self.complexity_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.risk_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            self.ast_vectorizer = CountVectorizer(max_features=500, analyzer='word')
            
            # Try to load pre-trained models
            self._load_pretrained_models()
        except Exception as e:
            logging.warning(f"Failed to initialize semantic models: {e}")
    
    def _load_pretrained_models(self):
        """Load pre-trained models if available"""
        model_dir = Path(__file__).parent / "ml_models"
        model_dir.mkdir(exist_ok=True)
        
        try:
            complexity_path = model_dir / "complexity_model.joblib"
            risk_path = model_dir / "risk_model.joblib"
            vectorizer_path = model_dir / "ast_vectorizer.joblib"
            
            if all(p.exists() for p in [complexity_path, risk_path, vectorizer_path]):
                self.complexity_model = joblib.load(complexity_path)
                self.risk_model = joblib.load(risk_path)
                self.ast_vectorizer = joblib.load(vectorizer_path)
                self.trained = True
                logging.info("Pre-trained models loaded successfully")
        except Exception as e:
            logging.warning(f"Could not load pre-trained models: {e}")
    
    def analyze_code_semantics(self, code: str, file_path: str = "") -> Dict[str, Any]:
        """Analyze code semantics using ML"""
        analysis = {
            'ast_features': self._extract_ast_features(code),
            'complexity_prediction': 0,
            'risk_score': 0.0,
            'semantic_patterns': [],
            'intent_analysis': {},
            'logic_flow': [],
            'code_quality_score': 0.0
        }
        
        try:
            # Extract AST features
            ast_features = analysis['ast_features']
            
            # Predict complexity and risk if models are trained
            if self.trained and HAS_SKLEARN:
                feature_vector = self._vectorize_ast_features(ast_features)
                
                complexity_pred = self.complexity_model.predict_proba([feature_vector])[0]
                analysis['complexity_prediction'] = complexity_pred.max()
                
                risk_pred = self.risk_model.predict_proba([feature_vector])[0]
                analysis['risk_score'] = risk_pred[1] if len(risk_pred) > 1 else risk_pred[0]
            
            # Analyze semantic patterns
            analysis['semantic_patterns'] = self._identify_semantic_patterns(code)
            
            # Intent analysis
            analysis['intent_analysis'] = self._analyze_code_intent(code, ast_features)
            
            # Logic flow analysis
            analysis['logic_flow'] = self._analyze_logic_flow(code)
            
            # Overall quality score
            analysis['code_quality_score'] = self._calculate_quality_score(analysis)
            
        except Exception as e:
            logging.warning(f"Semantic analysis failed: {e}")
        
        return analysis
    
    def _extract_ast_features(self, code: str) -> Dict[str, Any]:
        """Extract features from AST"""
        features = {
            'node_types': defaultdict(int),
            'depth': 0,
            'complexity_score': 0,
            'function_count': 0,
            'class_count': 0,
            'import_count': 0,
            'control_structures': defaultdict(int),
            'variable_usage': defaultdict(int),
            'function_calls': [],
            'error_handling': defaultdict(int)
        }
        
        try:
            tree = ast.parse(code)
            
            class ASTFeatureExtractor(ast.NodeVisitor):
                def __init__(self):
                    self.depth = 0
                    self.max_depth = 0
                
                def visit(self, node):
                    features['node_types'][type(node).__name__] += 1
                    self.depth += 1
                    self.max_depth = max(self.max_depth, self.depth)
                    
                    # Specific node handling
                    if isinstance(node, ast.FunctionDef):
                        features['function_count'] += 1
                    elif isinstance(node, ast.ClassDef):
                        features['class_count'] += 1
                    elif isinstance(node, ast.Import):
                        features['import_count'] += 1
                    elif isinstance(node, (ast.If, ast.While, ast.For)):
                        features['control_structures'][type(node).__name__] += 1
                    elif isinstance(node, ast.Name):
                        features['variable_usage'][node.id] += 1
                    elif isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            features['function_calls'].append(node.func.id)
                    elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                        features['error_handling'][type(node).__name__] += 1
                    
                    self.generic_visit(node)
                    self.depth -= 1
            
            extractor = ASTFeatureExtractor()
            extractor.visit(tree)
            features['depth'] = extractor.max_depth
            
            # Calculate complexity score
            features['complexity_score'] = (
                features['function_count'] * 2 +
                features['class_count'] * 3 +
                sum(features['control_structures'].values()) * 1.5 +
                features['depth'] * 0.5
            )
            
        except SyntaxError:
            features['syntax_error'] = True
        except Exception as e:
            logging.warning(f"AST feature extraction failed: {e}")
        
        return features
    
    def _vectorize_ast_features(self, ast_features: Dict[str, Any]) -> np.ndarray:
        """Convert AST features to vector representation"""
        feature_vector = []
        
        # Numeric features
        numeric_features = [
            'depth', 'complexity_score', 'function_count', 
            'class_count', 'import_count'
        ]
        
        for feature in numeric_features:
            feature_vector.append(ast_features.get(feature, 0))
        
        # Node type counts (top 20 most common)
        common_nodes = [
            'Name', 'Load', 'Store', 'Assign', 'Call', 'Attribute',
            'If', 'Compare', 'BinOp', 'Return', 'FunctionDef',
            'arg', 'arguments', 'Expr', 'Add', 'Str', 'Num',
            'For', 'While', 'Try'
        ]
        
        for node_type in common_nodes:
            feature_vector.append(ast_features['node_types'].get(node_type, 0))
        
        # Control structure counts
        control_types = ['If', 'While', 'For', 'Try']
        for ctrl_type in control_types:
            feature_vector.append(ast_features['control_structures'].get(ctrl_type, 0))
        
        return np.array(feature_vector, dtype=np.float32)
    
    def _identify_semantic_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Identify semantic patterns in code"""
        patterns = []
        
        # Common patterns to detect
        pattern_definitions = [
            {
                'name': 'factory_pattern',
                'regex': r'class\s+\w+Factory|def\s+create_\w+',
                'description': 'Factory pattern detected'
            },
            {
                'name': 'singleton_pattern',
                'regex': r'__new__.*cls\._instance|_instance\s*=\s*None',
                'description': 'Singleton pattern detected'
            },
            {
                'name': 'observer_pattern',
                'regex': r'def\s+(notify|subscribe|unsubscribe)',
                'description': 'Observer pattern detected'
            },
            {
                'name': 'context_manager',
                'regex': r'def\s+__enter__|def\s+__exit__|with\s+\w+',
                'description': 'Context manager pattern detected'
            },
            {
                'name': 'decorator_pattern',
                'regex': r'@\w+|def\s+\w+\(.*\):\s*def\s+wrapper',
                'description': 'Decorator pattern detected'
            },
            {
                'name': 'error_handling',
                'regex': r'try:\s|except\s+\w+:|finally:',
                'description': 'Error handling pattern detected'
            },
            {
                'name': 'logging_pattern',
                'regex': r'logging\.\w+|logger\.\w+|\.log\(',
                'description': 'Logging pattern detected'
            },
            {
                'name': 'async_pattern',
                'regex': r'async\s+def|await\s+|asyncio\.',
                'description': 'Asynchronous pattern detected'
            }
        ]
        
        for pattern_def in pattern_definitions:
            matches = re.findall(pattern_def['regex'], code, re.MULTILINE | re.IGNORECASE)
            if matches:
                patterns.append({
                    'pattern_name': pattern_def['name'],
                    'description': pattern_def['description'],
                    'occurrences': len(matches),
                    'confidence': min(len(matches) / 5.0, 1.0)
                })
        
        return patterns
    
    def _analyze_code_intent(self, code: str, ast_features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code intent and purpose"""
        intent_analysis = {
            'primary_intent': 'unknown',
            'secondary_intents': [],
            'confidence': 0.0,
            'evidence': []
        }
        
        # Intent indicators
        intent_indicators = {
            'data_processing': ['pandas', 'numpy', 'dataframe', 'array', 'process'],
            'web_service': ['flask', 'django', 'fastapi', 'request', 'response', 'route'],
            'machine_learning': ['sklearn', 'tensorflow', 'torch', 'model', 'train', 'predict'],
            'database': ['sql', 'query', 'database', 'connection', 'cursor'],
            'file_processing': ['file', 'read', 'write', 'path', 'directory'],
            'testing': ['test', 'assert', 'mock', 'fixture', 'pytest'],
            'utility': ['util', 'helper', 'tool', 'format', 'convert'],
            'configuration': ['config', 'setting', 'parameter', 'environment'],
            'authentication': ['auth', 'login', 'password', 'token', 'permission'],
            'api_client': ['client', 'request', 'api', 'endpoint', 'http']
        }
        
        code_lower = code.lower()
        intent_scores = {}
        
        for intent, keywords in intent_indicators.items():
            score = 0
            found_keywords = []
            
            for keyword in keywords:
                count = code_lower.count(keyword)
                if count > 0:
                    score += count
                    found_keywords.append(keyword)
            
            if score > 0:
                intent_scores[intent] = {
                    'score': score,
                    'keywords': found_keywords,
                    'confidence': min(score / 10.0, 1.0)
                }
        
        if intent_scores:
            # Primary intent (highest score)
            primary = max(intent_scores.items(), key=lambda x: x[1]['score'])
            intent_analysis['primary_intent'] = primary[0]
            intent_analysis['confidence'] = primary[1]['confidence']
            intent_analysis['evidence'] = primary[1]['keywords']
            
            # Secondary intents
            secondary = [intent for intent, data in intent_scores.items() 
                        if intent != primary[0] and data['confidence'] > 0.3]
            intent_analysis['secondary_intents'] = secondary
        
        return intent_analysis
    
    def _analyze_logic_flow(self, code: str) -> List[Dict[str, Any]]:
        """Analyze logic flow and control structures"""
        flow_analysis = []
        
        try:
            tree = ast.parse(code)
            
            class FlowAnalyzer(ast.NodeVisitor):
                def __init__(self):
                    self.flows = []
                    self.depth = 0
                
                def visit_If(self, node):
                    self.flows.append({
                        'type': 'conditional',
                        'depth': self.depth,
                        'complexity': len(node.orelse) > 0,
                        'line': node.lineno
                    })
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1
                
                def visit_For(self, node):
                    self.flows.append({
                        'type': 'loop',
                        'subtype': 'for',
                        'depth': self.depth,
                        'line': node.lineno
                    })
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1
                
                def visit_While(self, node):
                    self.flows.append({
                        'type': 'loop',
                        'subtype': 'while',
                        'depth': self.depth,
                        'line': node.lineno
                    })
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1
                
                def visit_Try(self, node):
                    self.flows.append({
                        'type': 'error_handling',
                        'depth': self.depth,
                        'handlers': len(node.handlers),
                        'has_finally': len(node.finalbody) > 0,
                        'line': node.lineno
                    })
                    self.generic_visit(node)
            
            analyzer = FlowAnalyzer()
            analyzer.visit(tree)
            flow_analysis = analyzer.flows
            
        except Exception as e:
            logging.warning(f"Logic flow analysis failed: {e}")
        
        return flow_analysis
    
    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall code quality score"""
        score = 1.0  # Start with perfect score
        
        # Penalize high complexity
        complexity = analysis.get('complexity_prediction', 0)
        if complexity > 0.8:
            score -= 0.2
        elif complexity > 0.6:
            score -= 0.1
        
        # Penalize high risk
        risk = analysis.get('risk_score', 0)
        score -= risk * 0.3
        
        # Reward good patterns
        patterns = analysis.get('semantic_patterns', [])
        good_patterns = ['error_handling', 'logging_pattern', 'context_manager']
        for pattern in patterns:
            if pattern['pattern_name'] in good_patterns:
                score += 0.1
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def train_models(self, training_data: List[Dict[str, Any]]):
        """Train ML models on code analysis data"""
        if not HAS_SKLEARN or not training_data:
            return
        
        try:
            # Prepare training data
            features = []
            complexity_labels = []
            risk_labels = []
            
            for data_point in training_data:
                ast_features = data_point['ast_features']
                feature_vector = self._vectorize_ast_features(ast_features)
                features.append(feature_vector)
                
                # Labels (these would come from manual annotation)
                complexity_labels.append(data_point.get('complexity_label', 0))
                risk_labels.append(data_point.get('risk_label', 0))
            
            features = np.array(features)
            
            # Train models
            if len(set(complexity_labels)) > 1:
                self.complexity_model.fit(features, complexity_labels)
            
            if len(set(risk_labels)) > 1:
                self.risk_model.fit(features, risk_labels)
            
            self.trained = True
            
            # Save trained models
            self._save_models()
            
        except Exception as e:
            logging.error(f"Model training failed: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            model_dir = Path(__file__).parent / "ml_models"
            model_dir.mkdir(exist_ok=True)
            
            joblib.dump(self.complexity_model, model_dir / "complexity_model.joblib")
            joblib.dump(self.risk_model, model_dir / "risk_model.joblib")
            joblib.dump(self.ast_vectorizer, model_dir / "ast_vectorizer.joblib")
            
            logging.info("ML models saved successfully")
        except Exception as e:
            logging.warning(f"Could not save models: {e}")


class IntelligentFixGenerator:
    """AI system that learns from successful fixes"""
    
    def __init__(self):
        self.fix_database = {}
        self.success_patterns = {}
        self.vectorizer = None
        self.fix_classifier = None
        self.trained = False
        
        if HAS_SKLEARN:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for fix generation"""
        try:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.fix_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        except Exception as e:
            logging.warning(f"Fix generator initialization failed: {e}")
    
    def generate_fix_suggestions(self, error_msg: str, code_context: str, 
                               error_type: str) -> List[FixSuggestion]:
        """Generate intelligent fix suggestions"""
        suggestions = []
        
        # Template-based fixes (immediate fallback)
        template_fixes = self._get_template_fixes(error_type, error_msg)
        suggestions.extend(template_fixes)
        
        # ML-based fixes (if trained)
        if self.trained and HAS_SKLEARN:
            ml_fixes = self._generate_ml_fixes(error_msg, code_context, error_type)
            suggestions.extend(ml_fixes)
        
        # Pattern-based fixes
        pattern_fixes = self._generate_pattern_fixes(error_msg, code_context)
        suggestions.extend(pattern_fixes)
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        return suggestions[:5]  # Top 5 suggestions
    
    def _get_template_fixes(self, error_type: str, error_msg: str) -> List[FixSuggestion]:
        """Generate template-based fixes"""
        fixes = []
        fix_id_base = f"template_{error_type}_{hash(error_msg) % 10000}"
        
        templates = {
            'AttributeError': [
                {
                    'pattern': "'NoneType' object has no attribute",
                    'fix_template': "# Add null check\nif {obj} is not None:\n    {original_code}\nelse:\n    # Handle None case\n    pass",
                    'explanation': "Add a null check to prevent AttributeError on None objects"
                },
                {
                    'pattern': "object has no attribute",
                    'fix_template': "# Use getattr with default\nvalue = getattr({obj}, '{attr}', default_value)",
                    'explanation': "Use getattr() to safely access attributes with a default value"
                }
            ],
            'KeyError': [
                {
                    'pattern': "KeyError:",
                    'fix_template': "# Use dict.get() for safe access\nvalue = {dict_name}.get('{key}', default_value)",
                    'explanation': "Use dict.get() to safely access dictionary keys"
                },
                {
                    'pattern': "KeyError:",
                    'fix_template': "# Check key existence\nif '{key}' in {dict_name}:\n    value = {dict_name}['{key}']\nelse:\n    # Handle missing key\n    value = default_value",
                    'explanation': "Check if key exists before accessing"
                }
            ],
            'IndexError': [
                {
                    'pattern': "list index out of range",
                    'fix_template': "# Check list bounds\nif 0 <= {index} < len({list_name}):\n    value = {list_name}[{index}]\nelse:\n    # Handle out of bounds\n    value = None",
                    'explanation': "Check list bounds before accessing elements"
                }
            ],
            'TypeError': [
                {
                    'pattern': "unexpected keyword argument",
                    'fix_template': "# Remove unexpected keyword argument\n{function_call}  # Remove '{arg}' argument",
                    'explanation': "Remove the unexpected keyword argument from the function call"
                }
            ],
            'ValueError': [
                {
                    'pattern': "invalid literal",
                    'fix_template': "# Add input validation\ntry:\n    result = {conversion_func}({input_value})\nexcept ValueError:\n    # Handle invalid input\n    result = default_value",
                    'explanation': "Add try-except block to handle invalid input values"
                }
            ]
        }
        
        if error_type in templates:
            for i, template in enumerate(templates[error_type]):
                if re.search(template['pattern'], error_msg, re.IGNORECASE):
                    fix_suggestion = FixSuggestion(
                        fix_id=f"{fix_id_base}_{i}",
                        original_code="# Original problematic code",
                        fixed_code=template['fix_template'],
                        explanation=template['explanation'],
                        confidence=0.7,  # Template fixes have moderate confidence
                        success_rate=0.6
                    )
                    fixes.append(fix_suggestion)
        
        return fixes
    
    def _generate_ml_fixes(self, error_msg: str, code_context: str, 
                          error_type: str) -> List[FixSuggestion]:
        """Generate ML-based fixes using trained models"""
        fixes = []
        
        # This would use the trained classifier to suggest fixes
        # For now, return placeholder
        try:
            combined_input = f"{error_type}: {error_msg} Context: {code_context}"
            
            # Vectorize input
            input_vector = self.vectorizer.transform([combined_input])
            
            # Predict fix category
            if hasattr(self.fix_classifier, 'predict_proba'):
                probabilities = self.fix_classifier.predict_proba(input_vector)[0]
                classes = self.fix_classifier.classes_
                
                # Generate fixes based on top predictions
                for prob, fix_class in sorted(zip(probabilities, classes), reverse=True)[:3]:
                    if prob > 0.1:  # Minimum confidence threshold
                        fix_suggestion = self._create_ml_fix_suggestion(
                            fix_class, error_msg, code_context, prob
                        )
                        fixes.append(fix_suggestion)
        
        except Exception as e:
            logging.warning(f"ML fix generation failed: {e}")
        
        return fixes
    
    def _create_ml_fix_suggestion(self, fix_class: str, error_msg: str, 
                                 code_context: str, confidence: float) -> FixSuggestion:
        """Create fix suggestion from ML prediction"""
        fix_id = f"ml_{fix_class}_{hash(error_msg) % 10000}"
        
        # Map fix classes to actual fixes
        fix_mappings = {
            'null_check': {
                'code': "# Add null check\nif variable is not None:\n    # Your code here\n    pass",
                'explanation': "Add null check based on ML prediction"
            },
            'exception_handling': {
                'code': "try:\n    # Your code here\n    pass\nexcept Exception as e:\n    # Handle exception\n    logging.error(f'Error: {e}')",
                'explanation': "Add exception handling based on ML analysis"
            },
            'input_validation': {
                'code': "# Validate input\nif not isinstance(input_value, expected_type):\n    raise ValueError('Invalid input type')",
                'explanation': "Add input validation based on error pattern"
            }
        }
        
        fix_info = fix_mappings.get(fix_class, {
            'code': "# ML-suggested fix\n# Review and modify as needed",
            'explanation': f"ML-suggested fix for {fix_class}"
        })
        
        return FixSuggestion(
            fix_id=fix_id,
            original_code=code_context[:200],
            fixed_code=fix_info['code'],
            explanation=fix_info['explanation'],
            confidence=confidence,
            success_rate=0.5  # Default for ML suggestions
        )
    
    def _generate_pattern_fixes(self, error_msg: str, code_context: str) -> List[FixSuggestion]:
        """Generate fixes based on learned patterns"""
        fixes = []
        
        # Look for similar patterns in success database
        for pattern_id, pattern_data in self.success_patterns.items():
            similarity = self._calculate_pattern_similarity(error_msg, pattern_data['error_pattern'])
            
            if similarity > 0.6:  # Similar enough
                fix_suggestion = FixSuggestion(
                    fix_id=f"pattern_{pattern_id}",
                    original_code=code_context[:200],
                    fixed_code=pattern_data['fix_code'],
                    explanation=f"Based on successful fix pattern: {pattern_data['description']}",
                    confidence=similarity * pattern_data['success_rate'],
                    success_rate=pattern_data['success_rate']
                )
                fixes.append(fix_suggestion)
        
        return fixes
    
    def _calculate_pattern_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between error patterns"""
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def learn_from_fix(self, error_msg: str, error_type: str, 
                      original_code: str, fixed_code: str, 
                      success: bool, user_feedback: float = 0.0):
        """Learn from applied fixes"""
        fix_id = f"learned_{hash(error_msg + fixed_code) % 100000}"
        
        if success:
            # Store successful fix
            self.fix_database[fix_id] = {
                'error_msg': error_msg,
                'error_type': error_type,
                'original_code': original_code,
                'fixed_code': fixed_code,
                'success': True,
                'feedback_score': user_feedback,
                'timestamp': datetime.now()
            }
            
            # Update success patterns
            pattern_key = f"{error_type}_{hash(error_msg) % 1000}"
            if pattern_key in self.success_patterns:
                pattern = self.success_patterns[pattern_key]
                pattern['success_count'] += 1
                pattern['total_attempts'] += 1
                pattern['success_rate'] = pattern['success_count'] / pattern['total_attempts']
            else:
                self.success_patterns[pattern_key] = {
                    'error_pattern': error_msg,
                    'error_type': error_type,
                    'fix_code': fixed_code,
                    'description': f"Fix for {error_type}",
                    'success_count': 1,
                    'total_attempts': 1,
                    'success_rate': 1.0
                }
        else:
            # Update failure statistics
            pattern_key = f"{error_type}_{hash(error_msg) % 1000}"
            if pattern_key in self.success_patterns:
                self.success_patterns[pattern_key]['total_attempts'] += 1
                pattern = self.success_patterns[pattern_key]
                pattern['success_rate'] = pattern['success_count'] / pattern['total_attempts']
    
    def retrain_models(self):
        """Retrain models based on learned fixes"""
        if not HAS_SKLEARN or not self.fix_database:
            return
        
        try:
            # Prepare training data from learned fixes
            texts = []
            labels = []
            
            for fix_data in self.fix_database.values():
                text = f"{fix_data['error_type']}: {fix_data['error_msg']} Context: {fix_data['original_code'][:500]}"
                texts.append(text)
                
                # Create label based on fix pattern
                if 'null' in fix_data['fixed_code'].lower() or 'none' in fix_data['fixed_code'].lower():
                    label = 'null_check'
                elif 'try:' in fix_data['fixed_code'] or 'except' in fix_data['fixed_code']:
                    label = 'exception_handling'
                elif 'validation' in fix_data['fixed_code'].lower() or 'isinstance' in fix_data['fixed_code']:
                    label = 'input_validation'
                else:
                    label = 'generic_fix'
                
                labels.append(label)
            
            if len(set(labels)) > 1:  # Need at least 2 classes
                # Fit vectorizer and classifier
                text_vectors = self.vectorizer.fit_transform(texts)
                self.fix_classifier.fit(text_vectors, labels)
                self.trained = True
                
                logging.info(f"Fix generator retrained with {len(texts)} examples")
        
        except Exception as e:
            logging.error(f"Fix generator retraining failed: {e}")


class PredictiveDebugger:
    """Predictive system to catch issues before they manifest"""
    
    def __init__(self):
        self.anomaly_detector = None
        self.risk_patterns = {}
        self.code_health_history = deque(maxlen=1000)
        self.prediction_models = {}
        
        if HAS_SKLEARN:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize predictive models"""
        try:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.prediction_models['error_likelihood'] = MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
            )
        except Exception as e:
            logging.warning(f"Predictive model initialization failed: {e}")
    
    def analyze_code_health(self, code: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code health and predict potential issues"""
        health_analysis = {
            'overall_score': 0.0,
            'risk_factors': [],
            'anomalies': [],
            'predictions': {},
            'recommendations': [],
            'confidence': 0.0
        }
        
        try:
            # Extract health metrics
            health_metrics = self._extract_health_metrics(code, metrics)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(health_metrics)
            health_analysis['anomalies'] = anomalies
            
            # Calculate risk factors
            risk_factors = self._identify_risk_factors(code, health_metrics)
            health_analysis['risk_factors'] = risk_factors
            
            # Make predictions
            predictions = self._make_predictions(health_metrics)
            health_analysis['predictions'] = predictions
            
            # Generate recommendations
            recommendations = self._generate_health_recommendations(health_analysis)
            health_analysis['recommendations'] = recommendations
            
            # Calculate overall health score
            health_analysis['overall_score'] = self._calculate_health_score(health_analysis)
            health_analysis['confidence'] = min(len(self.code_health_history) / 100.0, 1.0)
            
            # Store in history
            health_record = {
                'timestamp': datetime.now(),
                'metrics': health_metrics,
                'score': health_analysis['overall_score']
            }
            self.code_health_history.append(health_record)
        
        except Exception as e:
            logging.warning(f"Code health analysis failed: {e}")
        
        return health_analysis
    
    def _extract_health_metrics(self, code: str, external_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract code health metrics"""
        metrics = {}
        
        # Basic code metrics
        lines = code.splitlines()
        metrics['lines_of_code'] = len([line for line in lines if line.strip()])
        metrics['comment_ratio'] = len([line for line in lines if line.strip().startswith('#')]) / max(len(lines), 1)
        metrics['blank_line_ratio'] = len([line for line in lines if not line.strip()]) / max(len(lines), 1)
        
        # Complexity metrics
        metrics['cyclomatic_complexity'] = external_metrics.get('cyclomatic_complexity', 0)
        metrics['nesting_depth'] = external_metrics.get('nesting_depth', 0)
        metrics['function_count'] = external_metrics.get('function_count', 0)
        
        # Quality indicators
        metrics['has_docstrings'] = 1.0 if '"""' in code or "'''" in code else 0.0
        metrics['has_type_hints'] = 1.0 if '->' in code or ': ' in code else 0.0
        metrics['has_error_handling'] = 1.0 if 'try:' in code and 'except' in code else 0.0
        metrics['has_logging'] = 1.0 if 'logging.' in code or 'logger.' in code else 0.0
        
        # Anti-pattern indicators
        metrics['bare_except_count'] = code.count('except:')
        metrics['global_usage'] = code.count('global ')
        metrics['print_statements'] = code.count('print(')
        metrics['todo_fixme_count'] = code.upper().count('TODO') + code.upper().count('FIXME')
        
        # Security indicators
        metrics['eval_usage'] = code.count('eval(')
        metrics['exec_usage'] = code.count('exec(')
        metrics['shell_usage'] = code.count('shell=True')
        
        return metrics
    
    def _detect_anomalies(self, health_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect anomalies in code health metrics"""
        anomalies = []
        
        if not HAS_SKLEARN or not self.code_health_history:
            return self._detect_static_anomalies(health_metrics)
        
        try:
            # Prepare historical data
            historical_metrics = []
            for record in self.code_health_history:
                metric_vector = [record['metrics'].get(key, 0) for key in health_metrics.keys()]
                historical_metrics.append(metric_vector)
            
            if len(historical_metrics) < 10:  # Need sufficient history
                return self._detect_static_anomalies(health_metrics)
            
            # Fit anomaly detector
            self.anomaly_detector.fit(historical_metrics)
            
            # Check current metrics
            current_vector = [health_metrics.get(key, 0) for key in health_metrics.keys()]
            anomaly_score = self.anomaly_detector.decision_function([current_vector])[0]
            is_anomaly = self.anomaly_detector.predict([current_vector])[0] == -1
            
            if is_anomaly:
                anomalies.append({
                    'type': 'statistical_anomaly',
                    'description': 'Code metrics show unusual patterns compared to history',
                    'severity': 'medium',
                    'anomaly_score': anomaly_score,
                    'affected_metrics': self._identify_anomalous_metrics(health_metrics, current_vector)
                })
        
        except Exception as e:
            logging.warning(f"ML anomaly detection failed: {e}")
            return self._detect_static_anomalies(health_metrics)
        
        return anomalies
    
    def _detect_static_anomalies(self, health_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect anomalies using static thresholds"""
        anomalies = []
        
        # Define thresholds for various metrics
        thresholds = {
            'cyclomatic_complexity': {'high': 15, 'critical': 30},
            'nesting_depth': {'high': 5, 'critical': 8},
            'lines_of_code': {'high': 500, 'critical': 1000},
            'bare_except_count': {'high': 1, 'critical': 3},
            'comment_ratio': {'low': 0.1},  # Too few comments
            'print_statements': {'high': 5, 'critical': 10},
            'eval_usage': {'high': 0},  # Any eval usage is suspicious
            'exec_usage': {'high': 0},  # Any exec usage is suspicious
            'shell_usage': {'high': 0}   # Any shell=True usage is risky
        }
        
        for metric, value in health_metrics.items():
            if metric in thresholds:
                threshold_config = thresholds[metric]
                
                if 'critical' in threshold_config and value >= threshold_config['critical']:
                    anomalies.append({
                        'type': 'threshold_violation',
                        'metric': metric,
                        'value': value,
                        'threshold': threshold_config['critical'],
                        'severity': 'critical',
                        'description': f'{metric} is critically high: {value}'
                    })
                elif 'high' in threshold_config and value >= threshold_config['high']:
                    anomalies.append({
                        'type': 'threshold_violation',
                        'metric': metric,
                        'value': value,
                        'threshold': threshold_config['high'],
                        'severity': 'high',
                        'description': f'{metric} is high: {value}'
                    })
                elif 'low' in threshold_config and value <= threshold_config['low']:
                    anomalies.append({
                        'type': 'threshold_violation',
                        'metric': metric,
                        'value': value,
                        'threshold': threshold_config['low'],
                        'severity': 'medium',
                        'description': f'{metric} is too low: {value}'
                    })
        
        return anomalies
    
    def _identify_anomalous_metrics(self, health_metrics: Dict[str, float], 
                                  current_vector: List[float]) -> List[str]:
        """Identify which specific metrics are anomalous"""
        # This would use feature importance or statistical analysis
        # For now, return metrics that are significantly different from mean
        anomalous_metrics = []
        
        if len(self.code_health_history) > 10:
            historical_means = {}
            for key in health_metrics.keys():
                values = [record['metrics'].get(key, 0) for record in self.code_health_history]
                historical_means[key] = np.mean(values) if values else 0
            
            for i, (key, current_value) in enumerate(health_metrics.items()):
                historical_mean = historical_means.get(key, 0)
                if historical_mean > 0 and abs(current_value - historical_mean) / historical_mean > 0.5:
                    anomalous_metrics.append(key)
        
        return anomalous_metrics
    
    def _identify_risk_factors(self, code: str, health_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify risk factors that could lead to future issues"""
        risk_factors = []
        
        # High complexity risks
        if health_metrics.get('cyclomatic_complexity', 0) > 10:
            risk_factors.append({
                'factor': 'high_complexity',
                'description': 'High cyclomatic complexity increases bug likelihood',
                'risk_level': 'high',
                'mitigation': 'Break down complex functions into smaller, focused units'
            })
        
        # Deep nesting risks
        if health_metrics.get('nesting_depth', 0) > 4:
            risk_factors.append({
                'factor': 'deep_nesting',
                'description': 'Deep nesting makes code hard to understand and maintain',
                'risk_level': 'medium',
                'mitigation': 'Use early returns and guard clauses to reduce nesting'
            })
        
        # Security risks
        if health_metrics.get('eval_usage', 0) > 0:
            risk_factors.append({
                'factor': 'security_risk',
                'description': 'Use of eval() creates security vulnerabilities',
                'risk_level': 'critical',
                'mitigation': 'Replace eval() with safer alternatives like ast.literal_eval()'
            })
        
        # Maintainability risks
        if health_metrics.get('comment_ratio', 0) < 0.1:
            risk_factors.append({
                'factor': 'low_documentation',
                'description': 'Low comment ratio makes code hard to maintain',
                'risk_level': 'medium',
                'mitigation': 'Add comments and docstrings to explain complex logic'
            })
        
        # Error handling risks
        if health_metrics.get('has_error_handling', 0) == 0:
            risk_factors.append({
                'factor': 'no_error_handling',
                'description': 'Lack of error handling can cause unexpected failures',
                'risk_level': 'high',
                'mitigation': 'Add try-except blocks around risky operations'
            })
        
        # Code quality risks
        if health_metrics.get('bare_except_count', 0) > 0:
            risk_factors.append({
                'factor': 'bare_except',
                'description': 'Bare except clauses can hide important errors',
                'risk_level': 'medium',
                'mitigation': 'Specify exact exception types to catch'
            })
        
        return risk_factors
    
    def _make_predictions(self, health_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Make predictions about future issues"""
        predictions = {
            'error_likelihood': 0.0,
            'maintenance_difficulty': 0.0,
            'performance_issues': 0.0,
            'security_vulnerability': 0.0,
            'confidence': 0.0
        }
        
        # Rule-based predictions (when ML is not available)
        complexity = health_metrics.get('cyclomatic_complexity', 0)
        error_handling = health_metrics.get('has_error_handling', 0)
        security_usage = health_metrics.get('eval_usage', 0) + health_metrics.get('exec_usage', 0)
        
        # Error likelihood prediction
        error_likelihood = 0.1  # Base likelihood
        error_likelihood += complexity * 0.02  # Higher complexity = more errors
        error_likelihood -= error_handling * 0.3  # Error handling reduces likelihood
        error_likelihood += security_usage * 0.5  # Security issues increase likelihood
        predictions['error_likelihood'] = min(error_likelihood, 1.0)
        
        # Maintenance difficulty
        maintenance_difficulty = 0.2  # Base difficulty
        maintenance_difficulty += complexity * 0.03
        maintenance_difficulty += health_metrics.get('nesting_depth', 0) * 0.05
        maintenance_difficulty -= health_metrics.get('comment_ratio', 0) * 0.5
        predictions['maintenance_difficulty'] = min(maintenance_difficulty, 1.0)
        
        # Performance issues
        performance_risk = 0.1
        performance_risk += health_metrics.get('nesting_depth', 0) * 0.02
        performance_risk += health_metrics.get('lines_of_code', 0) * 0.0001
        predictions['performance_issues'] = min(performance_risk, 1.0)
        
        # Security vulnerability
        security_risk = 0.05
        security_risk += security_usage * 0.8
        security_risk += health_metrics.get('shell_usage', 0) * 0.6
        predictions['security_vulnerability'] = min(security_risk, 1.0)
        
        predictions['confidence'] = 0.6  # Moderate confidence for rule-based predictions
        
        return predictions
    
    def _generate_health_recommendations(self, health_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on health analysis"""
        recommendations = []
        
        # Based on anomalies
        for anomaly in health_analysis.get('anomalies', []):
            if anomaly['type'] == 'threshold_violation':
                metric = anomaly['metric']
                if metric == 'cyclomatic_complexity':
                    recommendations.append("Reduce cyclomatic complexity by breaking down complex functions")
                elif metric == 'nesting_depth':
                    recommendations.append("Reduce nesting depth using early returns and guard clauses")
                elif metric == 'print_statements':
                    recommendations.append("Replace print statements with proper logging")
        
        # Based on risk factors
        for risk in health_analysis.get('risk_factors', []):
            if 'mitigation' in risk:
                recommendations.append(risk['mitigation'])
        
        # Based on predictions
        predictions = health_analysis.get('predictions', {})
        if predictions.get('error_likelihood', 0) > 0.5:
            recommendations.append("High error likelihood detected - add comprehensive error handling")
        if predictions.get('security_vulnerability', 0) > 0.3:
            recommendations.append("Security vulnerability risk - review code for unsafe operations")
        if predictions.get('maintenance_difficulty', 0) > 0.6:
            recommendations.append("Code may be difficult to maintain - consider refactoring")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Limit to top 10 recommendations
    
    def _calculate_health_score(self, health_analysis: Dict[str, Any]) -> float:
        """Calculate overall health score (0-1 scale)"""
        score = 1.0  # Start with perfect health
        
        # Penalize based on anomalies
        for anomaly in health_analysis.get('anomalies', []):
            severity = anomaly.get('severity', 'low')
            if severity == 'critical':
                score -= 0.3
            elif severity == 'high':
                score -= 0.2
            elif severity == 'medium':
                score -= 0.1
            else:  # low
                score -= 0.05
        
        # Penalize based on risk factors
        for risk in health_analysis.get('risk_factors', []):
            risk_level = risk.get('risk_level', 'low')
            if risk_level == 'critical':
                score -= 0.25
            elif risk_level == 'high':
                score -= 0.15
            elif risk_level == 'medium':
                score -= 0.1
            else:  # low
                score -= 0.05
        
        # Penalize based on predictions
        predictions = health_analysis.get('predictions', {})
        score -= predictions.get('error_likelihood', 0) * 0.2
        score -= predictions.get('security_vulnerability', 0) * 0.3
        score -= predictions.get('maintenance_difficulty', 0) * 0.15
        
        return max(0.0, score)


class AutoLearningSystem:
    """System that continuously learns and improves from debugging sessions"""
    
    def __init__(self):
        self.learning_database = {}
        self.pattern_evolution = {}
        self.user_feedback = {}
        self.model_performance = {}
        self.learning_enabled = True
        self.adaptation_threshold = 0.1  # Minimum improvement to update models
        
    def record_debugging_session(self, session_data: Dict[str, Any]):
        """Record a debugging session for learning"""
        if not self.learning_enabled:
            return
        
        session_id = session_data.get('session_id', f"session_{len(self.learning_database)}")
        
        learning_record = {
            'session_id': session_id,
            'timestamp': datetime.now(),
            'errors_encountered': session_data.get('errors', []),
            'fixes_applied': session_data.get('fixes_applied', []),
            'user_actions': session_data.get('user_actions', []),
            'success_metrics': session_data.get('success_metrics', {}),
            'duration': session_data.get('duration', 0),
            'user_satisfaction': session_data.get('user_satisfaction', 0.0)
        }
        
        self.learning_database[session_id] = learning_record
        
        # Extract patterns for learning
        self._extract_learning_patterns(learning_record)
        
        # Update model performance metrics
        self._update_model_performance(learning_record)
    
    def _extract_learning_patterns(self, session_record: Dict[str, Any]):
        """Extract patterns from debugging session"""
        patterns = []
        
        # Error-fix patterns
        for error in session_record.get('errors_encountered', []):
            for fix in session_record.get('fixes_applied', []):
                if fix.get('error_id') == error.get('error_id'):
                    pattern = {
                        'pattern_type': 'error_fix',
                        'error_signature': self._create_error_signature(error),
                        'fix_signature': self._create_fix_signature(fix),
                        'success': fix.get('success', False),
                        'user_rating': fix.get('user_rating', 0.0),
                        'context': error.get('context', {}),
                        'timestamp': datetime.now()
                    }
                    patterns.append(pattern)
        
        # User behavior patterns
        user_actions = session_record.get('user_actions', [])
        if user_actions:
            behavior_pattern = {
                'pattern_type': 'user_behavior',
                'action_sequence': [action.get('type') for action in user_actions],
                'success_rate': session_record.get('success_metrics', {}).get('success_rate', 0.0),
                'efficiency': len(user_actions) / max(session_record.get('duration', 1), 1),
                'satisfaction': session_record.get('user_satisfaction', 0.0)
            }
            patterns.append(behavior_pattern)
        
        # Store patterns for future learning
        for pattern in patterns:
            pattern_key = self._generate_pattern_key(pattern)
            if pattern_key not in self.pattern_evolution:
                self.pattern_evolution[pattern_key] = {
                    'pattern': pattern,
                    'occurrences': 0,
                    'success_count': 0,
                    'total_rating': 0.0,
                    'evolution_history': []
                }
            
            pattern_data = self.pattern_evolution[pattern_key]
            pattern_data['occurrences'] += 1
            
            if pattern.get('success', False):
                pattern_data['success_count'] += 1
            
            if 'user_rating' in pattern:
                pattern_data['total_rating'] += pattern['user_rating']
            
            # Track evolution
            pattern_data['evolution_history'].append({
                'timestamp': pattern['timestamp'],
                'success': pattern.get('success', False),
                'rating': pattern.get('user_rating', 0.0)
            })
    
    def _create_error_signature(self, error: Dict[str, Any]) -> str:
        """Create a signature for error pattern recognition"""
        error_type = error.get('error_type', 'unknown')
        error_message = error.get('message', '')
        context_keys = sorted(error.get('context', {}).keys())
        
        # Create a hash-based signature
        signature_text = f"{error_type}:{error_message}:{':'.join(context_keys)}"
        return hashlib.md5(signature_text.encode()).hexdigest()[:16]
    
    def _create_fix_signature(self, fix: Dict[str, Any]) -> str:
        """Create a signature for fix pattern recognition"""
        fix_type = fix.get('fix_type', 'unknown')
        fix_description = fix.get('description', '')
        
        signature_text = f"{fix_type}:{fix_description}"
        return hashlib.md5(signature_text.encode()).hexdigest()[:16]
    
    def _generate_pattern_key(self, pattern: Dict[str, Any]) -> str:
        """Generate a unique key for pattern storage"""
        pattern_type = pattern.get('pattern_type', 'unknown')
        
        if pattern_type == 'error_fix':
            return f"ef_{pattern.get('error_signature', 'unknown')}"
        elif pattern_type == 'user_behavior':
            action_hash = hashlib.md5(str(pattern.get('action_sequence', [])).encode()).hexdigest()[:8]
            return f"ub_{action_hash}"
        else:
            return f"unknown_{hash(str(pattern)) % 10000}"
    
    def _update_model_performance(self, session_record: Dict[str, Any]):
        """Update model performance tracking"""
        session_metrics = session_record.get('success_metrics', {})
        
        # Track various performance metrics
        performance_update = {
            'timestamp': session_record['timestamp'],
            'error_detection_accuracy': session_metrics.get('error_detection_accuracy', 0.0),
            'fix_suggestion_quality': session_metrics.get('fix_suggestion_quality', 0.0),
            'prediction_accuracy': session_metrics.get('prediction_accuracy', 0.0),
            'user_satisfaction': session_record.get('user_satisfaction', 0.0),
            'session_duration': session_record.get('duration', 0)
        }
        
        # Store in performance history
        if 'performance_history' not in self.model_performance:
            self.model_performance['performance_history'] = []
        
        self.model_performance['performance_history'].append(performance_update)
        
        # Calculate rolling averages
        recent_sessions = self.model_performance['performance_history'][-50:]  # Last 50 sessions
        
        self.model_performance['current_metrics'] = {
            'avg_error_detection': np.mean([s['error_detection_accuracy'] for s in recent_sessions]),
            'avg_fix_quality': np.mean([s['fix_suggestion_quality'] for s in recent_sessions]),
            'avg_prediction_accuracy': np.mean([s['prediction_accuracy'] for s in recent_sessions]),
            'avg_user_satisfaction': np.mean([s['user_satisfaction'] for s in recent_sessions]),
            'avg_session_duration': np.mean([s['session_duration'] for s in recent_sessions])
        }
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning data"""
        insights = {
            'total_sessions': len(self.learning_database),
            'total_patterns': len(self.pattern_evolution),
            'top_error_patterns': [],
            'most_effective_fixes': [],
            'user_behavior_insights': [],
            'model_improvements': {},
            'learning_trends': {}
        }
        
        if not self.pattern_evolution:
            return insights
        
        # Top error patterns
        error_patterns = [p for p in self.pattern_evolution.values() 
                         if p['pattern']['pattern_type'] == 'error_fix']
        error_patterns.sort(key=lambda x: x['occurrences'], reverse=True)
        
        insights['top_error_patterns'] = [
            {
                'error_signature': p['pattern']['error_signature'],
                'occurrences': p['occurrences'],
                'success_rate': p['success_count'] / max(p['occurrences'], 1),
                'avg_rating': p['total_rating'] / max(p['occurrences'], 1)
            }
            for p in error_patterns[:10]
        ]
        
        # Most effective fixes
        effective_fixes = [p for p in error_patterns if p['success_count'] > 0]
        effective_fixes.sort(key=lambda x: x['success_count'] / max(x['occurrences'], 1), reverse=True)
        
        insights['most_effective_fixes'] = [
            {
                'fix_signature': p['pattern']['fix_signature'],
                'success_rate': p['success_count'] / max(p['occurrences'], 1),
                'usage_count': p['occurrences'],
                'avg_rating': p['total_rating'] / max(p['occurrences'], 1)
            }
            for p in effective_fixes[:10]
        ]
        
        # User behavior insights
        behavior_patterns = [p for p in self.pattern_evolution.values() 
                           if p['pattern']['pattern_type'] == 'user_behavior']
        
        if behavior_patterns:
            avg_satisfaction = np.mean([p['pattern'].get('satisfaction', 0) for p in behavior_patterns])
            avg_efficiency = np.mean([p['pattern'].get('efficiency', 0) for p in behavior_patterns])
            
            insights['user_behavior_insights'] = {
                'average_satisfaction': avg_satisfaction,
                'average_efficiency': avg_efficiency,
                'common_action_sequences': self._analyze_action_sequences(behavior_patterns)
            }
        
        # Model performance trends
        if 'performance_history' in self.model_performance:
            history = self.model_performance['performance_history']
            if len(history) > 10:
                recent_10 = history[-10:]
                older_10 = history[-20:-10] if len(history) > 20 else history[:-10]
                
                improvements = {}
                metrics = ['error_detection_accuracy', 'fix_suggestion_quality', 'prediction_accuracy', 'user_satisfaction']
                
                for metric in metrics:
                    recent_avg = np.mean([s[metric] for s in recent_10])
                    older_avg = np.mean([s[metric] for s in older_10]) if older_10 else recent_avg
                    improvement = recent_avg - older_avg
                    improvements[metric] = improvement
                
                insights['model_improvements'] = improvements
        
        return insights
    
    def _analyze_action_sequences(self, behavior_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze common action sequences"""
        sequence_counts = defaultdict(int)
        sequence_success = defaultdict(list)
        
        for pattern in behavior_patterns:
            sequence = tuple(pattern['pattern'].get('action_sequence', []))
            if len(sequence) > 1:  # Only consider sequences with multiple actions
                sequence_counts[sequence] += 1
                sequence_success[sequence].append(pattern['pattern'].get('success_rate', 0))
        
        # Find most common sequences
        common_sequences = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        result = []
        for sequence, count in common_sequences:
            avg_success = np.mean(sequence_success[sequence]) if sequence_success[sequence] else 0
            result.append({
                'action_sequence': list(sequence),
                'frequency': count,
                'average_success_rate': avg_success
            })
        
        return result
    
    def suggest_model_updates(self) -> List[Dict[str, Any]]:
        """Suggest updates to ML models based on learning"""
        suggestions = []
        
        if not self.pattern_evolution:
            return suggestions
        
        # Analyze pattern effectiveness
        error_fix_patterns = [p for p in self.pattern_evolution.values() 
                            if p['pattern']['pattern_type'] == 'error_fix']
        
        # Suggest new error patterns to train on
        high_success_patterns = [p for p in error_fix_patterns 
                               if p['success_count'] / max(p['occurrences'], 1) > 0.8 and p['occurrences'] >= 5]
        
        if high_success_patterns:
            suggestions.append({
                'type': 'new_training_patterns',
                'description': f'Found {len(high_success_patterns)} highly successful error-fix patterns',
                'action': 'Add these patterns to training data for error detection model',
                'priority': 'high',
                'patterns': [p['pattern']['error_signature'] for p in high_success_patterns[:5]]
            })
        
        # Suggest model retraining based on performance trends
        if 'current_metrics' in self.model_performance:
            current = self.model_performance['current_metrics']
            
            if current['avg_error_detection'] < 0.7:
                suggestions.append({
                    'type': 'model_retraining',
                    'description': 'Error detection accuracy is below threshold',
                    'action': 'Retrain error detection model with recent data',
                    'priority': 'high',
                    'current_accuracy': current['avg_error_detection']
                })
            
            if current['avg_fix_quality'] < 0.6:
                suggestions.append({
                    'type': 'fix_generation_improvement',
                    'description': 'Fix suggestion quality is low',
                    'action': 'Update fix generation templates and ML models',
                    'priority': 'medium',
                    'current_quality': current['avg_fix_quality']
                })
        
        # Suggest feature improvements based on user behavior
        behavior_patterns = [p for p in self.pattern_evolution.values() 
                           if p['pattern']['pattern_type'] == 'user_behavior']
        
        if behavior_patterns:
            low_satisfaction_patterns = [p for p in behavior_patterns 
                                       if p['pattern'].get('satisfaction', 0) < 0.5]
            
            if len(low_satisfaction_patterns) > len(behavior_patterns) * 0.3:
                suggestions.append({
                    'type': 'user_experience',
                    'description': 'High percentage of low satisfaction sessions',
                    'action': 'Review and improve user interface and interaction patterns',
                    'priority': 'high',
                    'affected_sessions': len(low_satisfaction_patterns)
                })
        
        return suggestions
    
    def apply_learning_updates(self, suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply suggested learning updates"""
        results = {
            'applied_updates': [],
            'skipped_updates': [],
            'errors': []
        }
        
        for suggestion in suggestions:
            try:
                update_type = suggestion.get('type')
                
                if update_type == 'new_training_patterns':
                    # Extract and prepare new training data
                    self._extract_training_data_from_patterns(suggestion['patterns'])
                    results['applied_updates'].append(f"Added {len(suggestion['patterns'])} new training patterns")
                
                elif update_type == 'model_retraining':
                    # Trigger model retraining (placeholder)
                    self._schedule_model_retraining('error_detection')
                    results['applied_updates'].append("Scheduled error detection model retraining")
                
                elif update_type == 'fix_generation_improvement':
                    # Update fix generation logic
                    self._update_fix_generation()
                    results['applied_updates'].append("Updated fix generation system")
                
                elif update_type == 'user_experience':
                    # Log UX improvements needed
                    results['applied_updates'].append("Logged user experience improvements for development team")
                
                else:
                    results['skipped_updates'].append(f"Unknown update type: {update_type}")
            
            except Exception as e:
                results['errors'].append(f"Failed to apply {suggestion.get('type', 'unknown')}: {str(e)}")
        
        return results
    
    def _extract_training_data_from_patterns(self, pattern_signatures: List[str]):
        """Extract training data from successful patterns"""
        # This would extract the actual error and fix data for retraining
        # Implementation depends on the specific ML models being used
        pass
    
    def _schedule_model_retraining(self, model_type: str):
        """Schedule model retraining"""
        # This would trigger actual model retraining
        # Implementation depends on the ML infrastructure
        pass
    
    def _update_fix_generation(self):
        """Update fix generation based on learned patterns"""
        # This would update the fix generation templates and logic
        # Implementation depends on the fix generation system
        pass
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        stats = {
            'total_sessions_learned': len(self.learning_database),
            'total_patterns_discovered': len(self.pattern_evolution),
            'learning_enabled': self.learning_enabled,
            'database_size_mb': sys.getsizeof(self.learning_database) / 1024 / 1024,
            'oldest_session': None,
            'newest_session': None,
            'pattern_categories': defaultdict(int),
            'learning_effectiveness': 0.0
        }
        
        if self.learning_database:
            sessions = list(self.learning_database.values())
            timestamps = [s['timestamp'] for s in sessions]
            stats['oldest_session'] = min(timestamps)
            stats['newest_session'] = max(timestamps)
        
        # Categorize patterns
        for pattern_data in self.pattern_evolution.values():
            pattern_type = pattern_data['pattern']['pattern_type']
            stats['pattern_categories'][pattern_type] += 1
        
        # Calculate learning effectiveness
        if 'current_metrics' in self.model_performance:
            current = self.model_performance['current_metrics']
            effectiveness_components = [
                current.get('avg_error_detection', 0),
                current.get('avg_fix_quality', 0),
                current.get('avg_prediction_accuracy', 0),
                current.get('avg_user_satisfaction', 0)
            ]
            stats['learning_effectiveness'] = np.mean(effectiveness_components)
        
        return stats


# Export main classes
__all__ = [
    'TransformerErrorAnalyzer',
    'SemanticCodeAnalyzer', 
    'IntelligentFixGenerator',
    'PredictiveDebugger',
    'AutoLearningSystem',
    'MLDebugLevel'
]