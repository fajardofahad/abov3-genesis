"""
ABOV3 Genesis - Sensitive Data Classifier and Redaction System
Enterprise-grade sensitive data detection, classification, and redaction for debug operations
"""

import re
import logging
import hashlib
import secrets
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import base64

from .audit_logger import SecurityAuditLogger
from .crypto_manager import CryptographyManager


class DataSensitivityLevel(Enum):
    """Data sensitivity levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class DataType(Enum):
    """Types of sensitive data"""
    PERSONAL_INFO = "personal_info"
    FINANCIAL = "financial"
    HEALTHCARE = "healthcare"
    AUTHENTICATION = "authentication"
    CRYPTOGRAPHIC = "cryptographic"
    SYSTEM_CONFIG = "system_config"
    BUSINESS_DATA = "business_data"
    TECHNICAL = "technical"
    LEGAL = "legal"
    PROPRIETARY = "proprietary"


class RedactionMethod(Enum):
    """Methods for data redaction"""
    MASK = "mask"                    # Replace with ****
    PARTIAL_MASK = "partial_mask"    # Show first/last chars
    HASH = "hash"                    # Replace with hash
    REMOVE = "remove"                # Remove completely
    ENCRYPT = "encrypt"              # Encrypt in place
    TOKENIZE = "tokenize"            # Replace with token


@dataclass
class DataPattern:
    """Pattern for detecting sensitive data"""
    pattern_id: str
    name: str
    description: str
    regex_pattern: str
    data_type: DataType
    sensitivity_level: DataSensitivityLevel
    redaction_method: RedactionMethod
    enabled: bool = True
    confidence_threshold: float = 0.8
    context_keywords: List[str] = field(default_factory=list)
    exclusion_patterns: List[str] = field(default_factory=list)


@dataclass
class DetectionResult:
    """Result of sensitive data detection"""
    pattern_id: str
    data_type: DataType
    sensitivity_level: DataSensitivityLevel
    start_position: int
    end_position: int
    matched_text: str
    confidence_score: float
    context: str
    redacted_text: str
    redaction_method: RedactionMethod


@dataclass
class ClassificationResult:
    """Result of data classification"""
    original_data: str
    overall_sensitivity: DataSensitivityLevel
    detections: List[DetectionResult]
    redacted_data: str
    metadata: Dict[str, Any]
    processing_time: float
    confidence_score: float


class SensitiveDataClassifier:
    """
    Enterprise-grade sensitive data classifier with comprehensive pattern detection
    Provides automatic detection, classification, and redaction of sensitive information
    """
    
    def __init__(
        self,
        audit_logger: Optional[SecurityAuditLogger] = None,
        crypto_manager: Optional[CryptographyManager] = None
    ):
        self.audit_logger = audit_logger
        self.crypto_manager = crypto_manager
        
        # Pattern storage
        self.patterns: Dict[str, DataPattern] = {}
        self.compiled_patterns: Dict[str, re.Pattern] = {}
        
        # Redaction tokens and mappings
        self.redaction_tokens: Dict[str, str] = {}
        self.token_mappings: Dict[str, str] = {}
        
        # Classification cache
        self.classification_cache: Dict[str, ClassificationResult] = {}
        self.cache_max_size = 1000
        
        # Statistics
        self.stats = {
            'total_classifications': 0,
            'sensitive_data_found': 0,
            'redactions_performed': 0,
            'data_types_detected': {},
            'sensitivity_levels_found': {}
        }
        
        # Setup logging
        self.logger = logging.getLogger('abov3.security.data_classifier')
        
        # Initialize built-in patterns
        self._initialize_builtin_patterns()
    
    def _initialize_builtin_patterns(self):
        """Initialize built-in sensitive data patterns"""
        
        patterns = [
            # Personal Information
            DataPattern(
                pattern_id="ssn_us",
                name="US Social Security Number",
                description="US Social Security Number (XXX-XX-XXXX)",
                regex_pattern=r'\b\d{3}-\d{2}-\d{4}\b',
                data_type=DataType.PERSONAL_INFO,
                sensitivity_level=DataSensitivityLevel.RESTRICTED,
                redaction_method=RedactionMethod.MASK,
                context_keywords=["ssn", "social security", "social_security_number"]
            ),
            DataPattern(
                pattern_id="email",
                name="Email Address",
                description="Email address pattern",
                regex_pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                data_type=DataType.PERSONAL_INFO,
                sensitivity_level=DataSensitivityLevel.CONFIDENTIAL,
                redaction_method=RedactionMethod.PARTIAL_MASK,
                context_keywords=["email", "e-mail", "mail"]
            ),
            DataPattern(
                pattern_id="phone_us",
                name="US Phone Number",
                description="US phone number in various formats",
                regex_pattern=r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                data_type=DataType.PERSONAL_INFO,
                sensitivity_level=DataSensitivityLevel.CONFIDENTIAL,
                redaction_method=RedactionMethod.PARTIAL_MASK,
                context_keywords=["phone", "telephone", "mobile", "cell"]
            ),
            
            # Financial Information
            DataPattern(
                pattern_id="credit_card",
                name="Credit Card Number",
                description="Credit card number (various formats)",
                regex_pattern=r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
                data_type=DataType.FINANCIAL,
                sensitivity_level=DataSensitivityLevel.RESTRICTED,
                redaction_method=RedactionMethod.PARTIAL_MASK,
                context_keywords=["credit", "card", "visa", "mastercard", "amex"]
            ),
            DataPattern(
                pattern_id="iban",
                name="International Bank Account Number",
                description="IBAN format",
                regex_pattern=r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b',
                data_type=DataType.FINANCIAL,
                sensitivity_level=DataSensitivityLevel.RESTRICTED,
                redaction_method=RedactionMethod.MASK,
                context_keywords=["iban", "bank", "account"]
            ),
            
            # Authentication & Security
            DataPattern(
                pattern_id="password_field",
                name="Password Fields",
                description="Password-like strings in various contexts",
                regex_pattern=r'(?i)(?:password|pwd|pass|secret|key)\s*[:=]\s*["\']?([^"\'\s\n]{4,})["\']?',
                data_type=DataType.AUTHENTICATION,
                sensitivity_level=DataSensitivityLevel.TOP_SECRET,
                redaction_method=RedactionMethod.REMOVE,
                context_keywords=["password", "pwd", "pass", "secret", "auth"]
            ),
            DataPattern(
                pattern_id="api_key",
                name="API Keys",
                description="API key patterns",
                regex_pattern=r'(?i)(?:api[-_]?key|apikey|access[-_]?key|secret[-_]?key)\s*[:=]\s*["\']?([A-Za-z0-9+/]{20,})["\']?',
                data_type=DataType.AUTHENTICATION,
                sensitivity_level=DataSensitivityLevel.TOP_SECRET,
                redaction_method=RedactionMethod.REMOVE,
                context_keywords=["api", "key", "token", "secret"]
            ),
            DataPattern(
                pattern_id="jwt_token",
                name="JWT Tokens",
                description="JSON Web Token format",
                regex_pattern=r'\beyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\b',
                data_type=DataType.AUTHENTICATION,
                sensitivity_level=DataSensitivityLevel.RESTRICTED,
                redaction_method=RedactionMethod.HASH,
                context_keywords=["jwt", "token", "bearer", "authorization"]
            ),
            
            # Cryptographic Data
            DataPattern(
                pattern_id="private_key",
                name="Private Key",
                description="Private key in PEM format",
                regex_pattern=r'-----BEGIN [A-Z\s]*PRIVATE KEY-----[\s\S]*?-----END [A-Z\s]*PRIVATE KEY-----',
                data_type=DataType.CRYPTOGRAPHIC,
                sensitivity_level=DataSensitivityLevel.TOP_SECRET,
                redaction_method=RedactionMethod.REMOVE,
                context_keywords=["private", "key", "rsa", "ecdsa"]
            ),
            DataPattern(
                pattern_id="hash_values",
                name="Hash Values",
                description="Hash values (MD5, SHA)",
                regex_pattern=r'\b[a-fA-F0-9]{32,128}\b',
                data_type=DataType.CRYPTOGRAPHIC,
                sensitivity_level=DataSensitivityLevel.CONFIDENTIAL,
                redaction_method=RedactionMethod.PARTIAL_MASK,
                context_keywords=["hash", "md5", "sha", "checksum"]
            ),
            
            # Healthcare (HIPAA)
            DataPattern(
                pattern_id="medical_record",
                name="Medical Record Number",
                description="Medical record number patterns",
                regex_pattern=r'\b(?:MRN|MR|MEDICAL[-_]?RECORD)[-_\s]?#?[-_\s]?([A-Z0-9]{6,15})\b',
                data_type=DataType.HEALTHCARE,
                sensitivity_level=DataSensitivityLevel.RESTRICTED,
                redaction_method=RedactionMethod.HASH,
                context_keywords=["medical", "record", "patient", "mrn"]
            ),
            
            # System Configuration
            DataPattern(
                pattern_id="database_url",
                name="Database Connection String",
                description="Database connection URLs",
                regex_pattern=r'(?i)(?:jdbc:|mongodb://|mysql://|postgresql://|sqlite://)[^\s\'"]+',
                data_type=DataType.SYSTEM_CONFIG,
                sensitivity_level=DataSensitivityLevel.RESTRICTED,
                redaction_method=RedactionMethod.PARTIAL_MASK,
                context_keywords=["database", "db", "connection", "url"]
            ),
            DataPattern(
                pattern_id="ip_address",
                name="IP Address",
                description="IPv4 and IPv6 addresses",
                regex_pattern=r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b|\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
                data_type=DataType.TECHNICAL,
                sensitivity_level=DataSensitivityLevel.INTERNAL,
                redaction_method=RedactionMethod.PARTIAL_MASK,
                context_keywords=["ip", "address", "server", "host"]
            ),
            
            # Business Data
            DataPattern(
                pattern_id="employee_id",
                name="Employee ID",
                description="Employee identification numbers",
                regex_pattern=r'\b(?:EMP|EMPLOYEE)[-_]?ID[-_\s]?#?[-_\s]?([A-Z0-9]{4,12})\b',
                data_type=DataType.BUSINESS_DATA,
                sensitivity_level=DataSensitivityLevel.CONFIDENTIAL,
                redaction_method=RedactionMethod.HASH,
                context_keywords=["employee", "emp", "staff", "worker"]
            ),
            
            # Generic sensitive patterns
            DataPattern(
                pattern_id="generic_secrets",
                name="Generic Secrets",
                description="Generic secret-like strings",
                regex_pattern=r'(?i)(?:secret|token|credential|auth)[-_]?[a-zA-Z0-9+/]{16,}',
                data_type=DataType.AUTHENTICATION,
                sensitivity_level=DataSensitivityLevel.RESTRICTED,
                redaction_method=RedactionMethod.HASH,
                context_keywords=["secret", "token", "credential", "auth"]
            )
        ]
        
        # Register all patterns
        for pattern in patterns:
            self.add_pattern(pattern)
    
    def add_pattern(self, pattern: DataPattern) -> bool:
        """
        Add a new sensitive data pattern
        
        Args:
            pattern: DataPattern to add
            
        Returns:
            bool indicating success
        """
        try:
            # Validate regex pattern
            compiled_regex = re.compile(pattern.regex_pattern, re.IGNORECASE | re.MULTILINE)
            
            # Store pattern
            self.patterns[pattern.pattern_id] = pattern
            self.compiled_patterns[pattern.pattern_id] = compiled_regex
            
            self.logger.info(f"Added data pattern: {pattern.pattern_id} ({pattern.name})")
            return True
            
        except re.error as e:
            self.logger.error(f"Invalid regex pattern for {pattern.pattern_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to add pattern {pattern.pattern_id}: {e}")
            return False
    
    def remove_pattern(self, pattern_id: str) -> bool:
        """Remove a data pattern"""
        try:
            if pattern_id in self.patterns:
                del self.patterns[pattern_id]
            if pattern_id in self.compiled_patterns:
                del self.compiled_patterns[pattern_id]
            
            self.logger.info(f"Removed data pattern: {pattern_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove pattern {pattern_id}: {e}")
            return False
    
    async def classify_data(
        self,
        data: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        cache_results: bool = True
    ) -> ClassificationResult:
        """
        Classify data for sensitive information
        
        Args:
            data: Data to classify
            context: Optional context information
            user_id: User requesting classification
            session_id: Session identifier
            cache_results: Whether to cache results
            
        Returns:
            ClassificationResult with detection and redaction information
        """
        start_time = datetime.now()
        
        try:
            # Check cache first
            data_hash = hashlib.sha256(data.encode()).hexdigest()
            if cache_results and data_hash in self.classification_cache:
                cached_result = self.classification_cache[data_hash]
                self.logger.debug(f"Using cached classification result for data hash: {data_hash[:16]}...")
                return cached_result
            
            # Perform detection
            detections = await self._detect_sensitive_data(data, context)
            
            # Calculate overall sensitivity
            overall_sensitivity = self._calculate_overall_sensitivity(detections)
            
            # Perform redaction
            redacted_data = await self._redact_data(data, detections)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(detections)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = ClassificationResult(
                original_data=data,
                overall_sensitivity=overall_sensitivity,
                detections=detections,
                redacted_data=redacted_data,
                metadata={
                    'data_length': len(data),
                    'patterns_matched': len(detections),
                    'unique_data_types': len(set(d.data_type for d in detections)),
                    'context': context or {},
                    'processing_timestamp': start_time.isoformat()
                },
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            
            # Cache result
            if cache_results:
                self._cache_result(data_hash, result)
            
            # Update statistics
            self._update_statistics(result)
            
            # Audit classification
            if self.audit_logger:
                await self._audit_classification(result, user_id, session_id)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Data classification failed: {e}")
            
            # Return safe fallback result
            return ClassificationResult(
                original_data=data,
                overall_sensitivity=DataSensitivityLevel.RESTRICTED,  # Safe default
                detections=[],
                redacted_data="[CLASSIFICATION ERROR]",
                metadata={'error': str(e)},
                processing_time=(datetime.now() - start_time).total_seconds(),
                confidence_score=0.0
            )
    
    async def _detect_sensitive_data(
        self,
        data: str,
        context: Optional[Dict[str, Any]]
    ) -> List[DetectionResult]:
        """Detect sensitive data patterns in text"""
        detections = []
        
        for pattern_id, pattern in self.patterns.items():
            if not pattern.enabled:
                continue
            
            compiled_pattern = self.compiled_patterns.get(pattern_id)
            if not compiled_pattern:
                continue
            
            # Find all matches
            matches = compiled_pattern.finditer(data)
            
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                matched_text = match.group()
                
                # Get context around match
                context_start = max(0, start_pos - 50)
                context_end = min(len(data), end_pos + 50)
                match_context = data[context_start:context_end]
                
                # Calculate confidence score
                confidence = await self._calculate_match_confidence(
                    pattern, matched_text, match_context, context
                )
                
                # Skip if confidence is too low
                if confidence < pattern.confidence_threshold:
                    continue
                
                # Check exclusion patterns
                if await self._is_excluded(pattern, matched_text, match_context):
                    continue
                
                # Create redacted version
                redacted_text = await self._apply_redaction(
                    matched_text, pattern.redaction_method
                )
                
                detection = DetectionResult(
                    pattern_id=pattern_id,
                    data_type=pattern.data_type,
                    sensitivity_level=pattern.sensitivity_level,
                    start_position=start_pos,
                    end_position=end_pos,
                    matched_text=matched_text,
                    confidence_score=confidence,
                    context=match_context,
                    redacted_text=redacted_text,
                    redaction_method=pattern.redaction_method
                )
                
                detections.append(detection)
        
        # Sort detections by position
        detections.sort(key=lambda d: d.start_position)
        
        # Remove overlapping detections (keep highest confidence)
        detections = self._remove_overlapping_detections(detections)
        
        return detections
    
    async def _calculate_match_confidence(
        self,
        pattern: DataPattern,
        matched_text: str,
        context: str,
        additional_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for a pattern match"""
        base_confidence = 0.7  # Base confidence for regex match
        
        # Boost confidence based on context keywords
        context_lower = context.lower()
        keyword_boost = 0.0
        
        for keyword in pattern.context_keywords:
            if keyword.lower() in context_lower:
                keyword_boost += 0.1
        
        keyword_boost = min(keyword_boost, 0.3)  # Cap at 0.3
        
        # Additional context boost
        additional_boost = 0.0
        if additional_context:
            context_str = json.dumps(additional_context).lower()
            for keyword in pattern.context_keywords:
                if keyword.lower() in context_str:
                    additional_boost += 0.05
        
        additional_boost = min(additional_boost, 0.2)  # Cap at 0.2
        
        # Pattern-specific adjustments
        pattern_adjustment = 0.0
        
        # For credit cards, validate checksum
        if pattern.pattern_id == "credit_card":
            if self._validate_luhn_checksum(matched_text):
                pattern_adjustment = 0.2
            else:
                pattern_adjustment = -0.3
        
        # For emails, check for common patterns
        elif pattern.pattern_id == "email":
            if any(domain in matched_text.lower() for domain in ['test', 'example', 'dummy']):
                pattern_adjustment = -0.4
        
        total_confidence = base_confidence + keyword_boost + additional_boost + pattern_adjustment
        return max(0.0, min(1.0, total_confidence))
    
    def _validate_luhn_checksum(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm"""
        try:
            # Remove non-digits
            digits = [int(d) for d in card_number if d.isdigit()]
            
            if len(digits) < 13 or len(digits) > 19:
                return False
            
            # Luhn algorithm
            checksum = 0
            is_even = False
            
            for digit in reversed(digits):
                if is_even:
                    digit *= 2
                    if digit > 9:
                        digit -= 9
                checksum += digit
                is_even = not is_even
            
            return checksum % 10 == 0
            
        except:
            return False
    
    async def _is_excluded(
        self,
        pattern: DataPattern,
        matched_text: str,
        context: str
    ) -> bool:
        """Check if match should be excluded based on exclusion patterns"""
        for exclusion_pattern in pattern.exclusion_patterns:
            if re.search(exclusion_pattern, context, re.IGNORECASE):
                return True
        
        # Common exclusions
        exclusion_keywords = ['test', 'example', 'dummy', 'fake', 'sample', 'placeholder']
        context_lower = context.lower()
        
        for keyword in exclusion_keywords:
            if keyword in context_lower:
                return True
        
        return False
    
    def _remove_overlapping_detections(
        self,
        detections: List[DetectionResult]
    ) -> List[DetectionResult]:
        """Remove overlapping detections, keeping highest confidence"""
        if not detections:
            return detections
        
        non_overlapping = []
        
        for detection in detections:
            overlapping = False
            
            for existing in non_overlapping:
                # Check if ranges overlap
                if not (detection.end_position <= existing.start_position or 
                       detection.start_position >= existing.end_position):
                    overlapping = True
                    
                    # Keep the one with higher confidence
                    if detection.confidence_score > existing.confidence_score:
                        non_overlapping.remove(existing)
                        non_overlapping.append(detection)
                    
                    break
            
            if not overlapping:
                non_overlapping.append(detection)
        
        return sorted(non_overlapping, key=lambda d: d.start_position)
    
    async def _redact_data(
        self,
        data: str,
        detections: List[DetectionResult]
    ) -> str:
        """Redact sensitive data based on detections"""
        if not detections:
            return data
        
        redacted_data = data
        offset = 0
        
        for detection in detections:
            # Adjust positions for previous redactions
            start_pos = detection.start_position + offset
            end_pos = detection.end_position + offset
            
            original_text = redacted_data[start_pos:end_pos]
            redacted_text = detection.redacted_text
            
            # Replace in the string
            redacted_data = (
                redacted_data[:start_pos] + 
                redacted_text + 
                redacted_data[end_pos:]
            )
            
            # Update offset for subsequent replacements
            offset += len(redacted_text) - len(original_text)
        
        return redacted_data
    
    async def _apply_redaction(
        self,
        text: str,
        method: RedactionMethod
    ) -> str:
        """Apply specific redaction method to text"""
        if method == RedactionMethod.MASK:
            return "*" * len(text)
        
        elif method == RedactionMethod.PARTIAL_MASK:
            if len(text) <= 4:
                return "*" * len(text)
            elif len(text) <= 8:
                return text[0] + "*" * (len(text) - 2) + text[-1]
            else:
                return text[:2] + "*" * (len(text) - 4) + text[-2:]
        
        elif method == RedactionMethod.HASH:
            hash_value = hashlib.sha256(text.encode()).hexdigest()[:16]
            return f"[HASH:{hash_value}]"
        
        elif method == RedactionMethod.REMOVE:
            return "[REDACTED]"
        
        elif method == RedactionMethod.ENCRYPT:
            if self.crypto_manager:
                try:
                    encrypted = await self.crypto_manager.encrypt_data(text.encode())
                    return f"[ENCRYPTED:{base64.b64encode(encrypted).decode()[:32]}...]"
                except:
                    return "[ENCRYPTION_FAILED]"
            return "[REDACTED]"
        
        elif method == RedactionMethod.TOKENIZE:
            token = self._generate_token(text)
            return f"[TOKEN:{token}]"
        
        else:
            return "[REDACTED]"
    
    def _generate_token(self, text: str) -> str:
        """Generate a token for the given text"""
        # Check if we already have a token for this text
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        if text_hash in self.token_mappings:
            return self.token_mappings[text_hash]
        
        # Generate new token
        token = secrets.token_hex(8).upper()
        
        # Store bidirectional mapping
        self.redaction_tokens[token] = text
        self.token_mappings[text_hash] = token
        
        return token
    
    def detokenize(self, token: str) -> Optional[str]:
        """Retrieve original text for a token"""
        return self.redaction_tokens.get(token)
    
    def _calculate_overall_sensitivity(
        self,
        detections: List[DetectionResult]
    ) -> DataSensitivityLevel:
        """Calculate overall sensitivity level from detections"""
        if not detections:
            return DataSensitivityLevel.PUBLIC
        
        # Get highest sensitivity level
        sensitivity_levels = [d.sensitivity_level for d in detections]
        
        # Order by sensitivity (most sensitive first)
        level_order = [
            DataSensitivityLevel.TOP_SECRET,
            DataSensitivityLevel.RESTRICTED,
            DataSensitivityLevel.CONFIDENTIAL,
            DataSensitivityLevel.INTERNAL,
            DataSensitivityLevel.PUBLIC
        ]
        
        for level in level_order:
            if level in sensitivity_levels:
                return level
        
        return DataSensitivityLevel.PUBLIC
    
    def _calculate_confidence_score(self, detections: List[DetectionResult]) -> float:
        """Calculate overall confidence score"""
        if not detections:
            return 0.0
        
        # Weighted average based on sensitivity level
        total_weighted_confidence = 0.0
        total_weights = 0.0
        
        weight_map = {
            DataSensitivityLevel.TOP_SECRET: 5.0,
            DataSensitivityLevel.RESTRICTED: 4.0,
            DataSensitivityLevel.CONFIDENTIAL: 3.0,
            DataSensitivityLevel.INTERNAL: 2.0,
            DataSensitivityLevel.PUBLIC: 1.0
        }
        
        for detection in detections:
            weight = weight_map.get(detection.sensitivity_level, 1.0)
            total_weighted_confidence += detection.confidence_score * weight
            total_weights += weight
        
        return total_weighted_confidence / total_weights if total_weights > 0 else 0.0
    
    def _cache_result(self, data_hash: str, result: ClassificationResult):
        """Cache classification result"""
        if len(self.classification_cache) >= self.cache_max_size:
            # Remove oldest entries
            oldest_keys = list(self.classification_cache.keys())[:100]
            for key in oldest_keys:
                del self.classification_cache[key]
        
        self.classification_cache[data_hash] = result
    
    def _update_statistics(self, result: ClassificationResult):
        """Update classification statistics"""
        self.stats['total_classifications'] += 1
        
        if result.detections:
            self.stats['sensitive_data_found'] += 1
            self.stats['redactions_performed'] += len(result.detections)
            
            for detection in result.detections:
                data_type_key = detection.data_type.value
                if data_type_key not in self.stats['data_types_detected']:
                    self.stats['data_types_detected'][data_type_key] = 0
                self.stats['data_types_detected'][data_type_key] += 1
                
                sensitivity_key = detection.sensitivity_level.value
                if sensitivity_key not in self.stats['sensitivity_levels_found']:
                    self.stats['sensitivity_levels_found'][sensitivity_key] = 0
                self.stats['sensitivity_levels_found'][sensitivity_key] += 1
    
    async def _audit_classification(
        self,
        result: ClassificationResult,
        user_id: Optional[str],
        session_id: Optional[str]
    ):
        """Audit data classification operation"""
        if not self.audit_logger:
            return
        
        await self.audit_logger.log_event(
            event_type="data_classification",
            user_id=user_id,
            session_id=session_id,
            context={
                'data_length': len(result.original_data),
                'overall_sensitivity': result.overall_sensitivity.value,
                'detections_count': len(result.detections),
                'data_types_found': [d.data_type.value for d in result.detections],
                'confidence_score': result.confidence_score,
                'processing_time': result.processing_time,
                'redactions_applied': len([d for d in result.detections if d.redacted_text != d.matched_text])
            },
            sensitive_data=result.overall_sensitivity in [
                DataSensitivityLevel.RESTRICTED,
                DataSensitivityLevel.TOP_SECRET
            ]
        )
        
        # Audit individual high-sensitivity detections
        for detection in result.detections:
            if detection.sensitivity_level in [DataSensitivityLevel.RESTRICTED, DataSensitivityLevel.TOP_SECRET]:
                await self.audit_logger.log_event(
                    event_type="sensitive_data_detected",
                    user_id=user_id,
                    session_id=session_id,
                    context={
                        'pattern_id': detection.pattern_id,
                        'data_type': detection.data_type.value,
                        'sensitivity_level': detection.sensitivity_level.value,
                        'confidence_score': detection.confidence_score,
                        'redaction_method': detection.redaction_method.value,
                        'position': f"{detection.start_position}-{detection.end_position}"
                    },
                    sensitive_data=True
                )
    
    async def bulk_classify(
        self,
        data_items: List[str],
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[ClassificationResult]:
        """Classify multiple data items efficiently"""
        results = []
        
        for data in data_items:
            try:
                result = await self.classify_data(data, context, user_id, session_id)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Bulk classification error for item: {e}")
                # Add error result
                results.append(ClassificationResult(
                    original_data=data,
                    overall_sensitivity=DataSensitivityLevel.RESTRICTED,
                    detections=[],
                    redacted_data="[CLASSIFICATION ERROR]",
                    metadata={'error': str(e)},
                    processing_time=0.0,
                    confidence_score=0.0
                ))
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics"""
        return {
            **self.stats,
            'patterns_loaded': len(self.patterns),
            'cache_size': len(self.classification_cache),
            'tokens_generated': len(self.redaction_tokens)
        }
    
    def get_patterns(self) -> List[Dict[str, Any]]:
        """Get list of all patterns"""
        return [
            {
                'pattern_id': pattern.pattern_id,
                'name': pattern.name,
                'description': pattern.description,
                'data_type': pattern.data_type.value,
                'sensitivity_level': pattern.sensitivity_level.value,
                'redaction_method': pattern.redaction_method.value,
                'enabled': pattern.enabled
            }
            for pattern in self.patterns.values()
        ]
    
    def enable_pattern(self, pattern_id: str) -> bool:
        """Enable a data pattern"""
        if pattern_id in self.patterns:
            self.patterns[pattern_id].enabled = True
            return True
        return False
    
    def disable_pattern(self, pattern_id: str) -> bool:
        """Disable a data pattern"""
        if pattern_id in self.patterns:
            self.patterns[pattern_id].enabled = False
            return True
        return False
    
    def clear_cache(self):
        """Clear classification cache"""
        self.classification_cache.clear()
        self.logger.info("Classification cache cleared")
    
    def clear_tokens(self):
        """Clear redaction tokens"""
        self.redaction_tokens.clear()
        self.token_mappings.clear()
        self.logger.info("Redaction tokens cleared")
    
    async def export_patterns(self, file_path: str) -> bool:
        """Export patterns to file"""
        try:
            patterns_data = {
                'exported_at': datetime.now().isoformat(),
                'patterns': [
                    {
                        'pattern_id': p.pattern_id,
                        'name': p.name,
                        'description': p.description,
                        'regex_pattern': p.regex_pattern,
                        'data_type': p.data_type.value,
                        'sensitivity_level': p.sensitivity_level.value,
                        'redaction_method': p.redaction_method.value,
                        'enabled': p.enabled,
                        'confidence_threshold': p.confidence_threshold,
                        'context_keywords': p.context_keywords,
                        'exclusion_patterns': p.exclusion_patterns
                    }
                    for p in self.patterns.values()
                ]
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(patterns_data, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export patterns: {e}")
            return False
    
    async def import_patterns(self, file_path: str) -> bool:
        """Import patterns from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                patterns_data = json.load(f)
            
            imported_count = 0
            
            for pattern_data in patterns_data.get('patterns', []):
                try:
                    pattern = DataPattern(
                        pattern_id=pattern_data['pattern_id'],
                        name=pattern_data['name'],
                        description=pattern_data['description'],
                        regex_pattern=pattern_data['regex_pattern'],
                        data_type=DataType(pattern_data['data_type']),
                        sensitivity_level=DataSensitivityLevel(pattern_data['sensitivity_level']),
                        redaction_method=RedactionMethod(pattern_data['redaction_method']),
                        enabled=pattern_data.get('enabled', True),
                        confidence_threshold=pattern_data.get('confidence_threshold', 0.8),
                        context_keywords=pattern_data.get('context_keywords', []),
                        exclusion_patterns=pattern_data.get('exclusion_patterns', [])
                    )
                    
                    if self.add_pattern(pattern):
                        imported_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed to import pattern {pattern_data.get('pattern_id')}: {e}")
                    continue
            
            self.logger.info(f"Imported {imported_count} patterns from {file_path}")
            return imported_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to import patterns from {file_path}: {e}")
            return False