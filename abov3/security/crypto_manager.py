"""
ABOV3 Genesis - Cryptography Manager
Enterprise-grade encryption and data protection system
"""

import os
import hashlib
import secrets
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import json


class CryptographyManager:
    """
    Enterprise Cryptography Manager
    Handles encryption, decryption, hashing, and key management
    """
    
    def __init__(self, security_dir: Path):
        self.security_dir = security_dir
        
        # Key storage
        self.keys_dir = security_dir / 'keys'
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize master key
        self.master_key_file = self.keys_dir / 'master.key'
        self.master_key = self._load_or_generate_master_key()
        
        # Fernet cipher for symmetric encryption
        self.fernet = Fernet(self.master_key)
        
        # RSA keys for asymmetric encryption
        self.rsa_private_key = None
        self.rsa_public_key = None
        self._load_or_generate_rsa_keys()
        
        # Encryption statistics
        self.crypto_stats = {
            'files_encrypted': 0,
            'files_decrypted': 0,
            'data_encrypted_bytes': 0,
            'data_decrypted_bytes': 0,
            'hashes_computed': 0
        }
    
    def _load_or_generate_master_key(self) -> bytes:
        """Load existing master key or generate new one"""
        if self.master_key_file.exists():
            with open(self.master_key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.master_key_file, 'wb') as f:
                f.write(key)
            os.chmod(self.master_key_file, 0o600)
            return key
    
    def _load_or_generate_rsa_keys(self):
        """Load or generate RSA key pair"""
        private_key_file = self.keys_dir / 'rsa_private.pem'
        public_key_file = self.keys_dir / 'rsa_public.pem'
        
        if private_key_file.exists() and public_key_file.exists():
            # Load existing keys
            with open(private_key_file, 'rb') as f:
                self.rsa_private_key = serialization.load_pem_private_key(f.read(), password=None)
            
            with open(public_key_file, 'rb') as f:
                self.rsa_public_key = serialization.load_pem_public_key(f.read())
        else:
            # Generate new RSA key pair
            self.rsa_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
            )
            self.rsa_public_key = self.rsa_private_key.public_key()
            
            # Save private key
            private_pem = self.rsa_private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            with open(private_key_file, 'wb') as f:
                f.write(private_pem)
            os.chmod(private_key_file, 0o600)
            
            # Save public key
            public_pem = self.rsa_public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            with open(public_key_file, 'wb') as f:
                f.write(public_pem)
    
    async def initialize(self) -> bool:
        """Initialize cryptography manager"""
        try:
            # Test encryption/decryption
            test_data = b"ABOV3 Genesis Crypto Test"
            encrypted = await self.encrypt_data(test_data)
            decrypted = await self.decrypt_data(encrypted['encrypted_data'])
            
            if decrypted['data'] != test_data:
                return False
            
            return True
        except Exception as e:
            print(f"Crypto manager initialization failed: {e}")
            return False
    
    async def encrypt_data(self, data: Union[str, bytes], algorithm: str = "fernet") -> Dict[str, Any]:
        """Encrypt data using specified algorithm"""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            if algorithm == "fernet":
                encrypted_data = self.fernet.encrypt(data)
                result = {
                    'success': True,
                    'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8'),
                    'algorithm': 'fernet',
                    'timestamp': datetime.now().isoformat()
                }
            
            elif algorithm == "rsa":
                # RSA can only encrypt small amounts of data
                if len(data) > 446:  # 4096-bit RSA with OAEP padding
                    return {'success': False, 'error': 'Data too large for RSA encryption'}
                
                encrypted_data = self.rsa_public_key.encrypt(
                    data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                result = {
                    'success': True,
                    'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8'),
                    'algorithm': 'rsa',
                    'timestamp': datetime.now().isoformat()
                }
            
            else:
                return {'success': False, 'error': f'Unsupported algorithm: {algorithm}'}
            
            # Update statistics
            self.crypto_stats['data_encrypted_bytes'] += len(data)
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Encryption failed: {str(e)}'}
    
    async def decrypt_data(self, encrypted_data: str, algorithm: str = "fernet") -> Dict[str, Any]:
        """Decrypt data"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            
            if algorithm == "fernet":
                decrypted_data = self.fernet.decrypt(encrypted_bytes)
            
            elif algorithm == "rsa":
                decrypted_data = self.rsa_private_key.decrypt(
                    encrypted_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            
            else:
                return {'success': False, 'error': f'Unsupported algorithm: {algorithm}'}
            
            # Update statistics
            self.crypto_stats['data_decrypted_bytes'] += len(decrypted_data)
            
            return {
                'success': True,
                'data': decrypted_data,
                'algorithm': algorithm
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Decryption failed: {str(e)}'}
    
    async def encrypt_file(self, file_path: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Encrypt entire file"""
        try:
            if not file_path.exists():
                return {'success': False, 'error': 'File not found'}
            
            # Read file
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Encrypt data
            encrypted_result = await self.encrypt_data(file_data)
            if not encrypted_result['success']:
                return encrypted_result
            
            # Write encrypted file
            if output_path is None:
                output_path = file_path.with_suffix(file_path.suffix + '.encrypted')
            
            encrypted_file_data = {
                'original_filename': file_path.name,
                'encrypted_data': encrypted_result['encrypted_data'],
                'algorithm': encrypted_result['algorithm'],
                'timestamp': encrypted_result['timestamp'],
                'file_hash': hashlib.sha256(file_data).hexdigest()
            }
            
            with open(output_path, 'w') as f:
                json.dump(encrypted_file_data, f, indent=2)
            
            self.crypto_stats['files_encrypted'] += 1
            
            return {
                'success': True,
                'encrypted_file': str(output_path),
                'original_size': len(file_data),
                'encrypted_size': output_path.stat().st_size
            }
            
        except Exception as e:
            return {'success': False, 'error': f'File encryption failed: {str(e)}'}
    
    async def decrypt_file(self, encrypted_file_path: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Decrypt entire file"""
        try:
            if not encrypted_file_path.exists():
                return {'success': False, 'error': 'Encrypted file not found'}
            
            # Read encrypted file
            with open(encrypted_file_path, 'r') as f:
                encrypted_file_data = json.load(f)
            
            # Decrypt data
            decrypted_result = await self.decrypt_data(
                encrypted_file_data['encrypted_data'],
                encrypted_file_data['algorithm']
            )
            
            if not decrypted_result['success']:
                return decrypted_result
            
            # Verify hash
            file_hash = hashlib.sha256(decrypted_result['data']).hexdigest()
            if file_hash != encrypted_file_data.get('file_hash'):
                return {'success': False, 'error': 'File integrity check failed'}
            
            # Write decrypted file
            if output_path is None:
                original_name = encrypted_file_data['original_filename']
                output_path = encrypted_file_path.parent / original_name
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_result['data'])
            
            self.crypto_stats['files_decrypted'] += 1
            
            return {
                'success': True,
                'decrypted_file': str(output_path),
                'original_filename': encrypted_file_data['original_filename']
            }
            
        except Exception as e:
            return {'success': False, 'error': f'File decryption failed: {str(e)}'}
    
    def compute_hash(self, data: Union[str, bytes], algorithm: str = "sha256") -> str:
        """Compute hash of data"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        self.crypto_stats['hashes_computed'] += 1
        
        if algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(data).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    def compute_file_hash(self, file_path: Path, algorithm: str = "sha256") -> Optional[str]:
        """Compute hash of file"""
        try:
            hash_func = getattr(hashlib, algorithm)()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            
            self.crypto_stats['hashes_computed'] += 1
            return hash_func.hexdigest()
            
        except Exception:
            return None
    
    def generate_secure_random(self, length: int = 32) -> str:
        """Generate secure random string"""
        return secrets.token_urlsafe(length)
    
    def generate_api_key(self, prefix: str = "abov3") -> str:
        """Generate API key with prefix"""
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}_{random_part}"
    
    def derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Derive encryption key from password using PBKDF2"""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password.encode('utf-8'))
        return key, salt
    
    def get_public_key_pem(self) -> str:
        """Get RSA public key in PEM format"""
        public_pem = self.rsa_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return public_pem.decode('utf-8')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cryptography statistics"""
        return self.crypto_stats.copy()
    
    def reset_statistics(self):
        """Reset statistics"""
        self.crypto_stats = {
            'files_encrypted': 0,
            'files_decrypted': 0,
            'data_encrypted_bytes': 0,
            'data_decrypted_bytes': 0,
            'hashes_computed': 0
        }