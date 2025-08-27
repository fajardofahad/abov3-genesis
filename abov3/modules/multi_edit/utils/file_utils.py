"""
File Utility Functions

Helper functions for file operations, encoding detection, and content manipulation.
"""

import os
import chardet
import hashlib
import mimetypes
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FileUtils:
    """Utility class for file operations"""
    
    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """
        Detect file encoding
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected encoding or 'utf-8' as default
        """
        
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            
            # Fallback to utf-8 for common cases
            if encoding is None or encoding.lower() in ['ascii']:
                encoding = 'utf-8'
            
            return encoding
            
        except Exception as e:
            logger.warning(f"Failed to detect encoding for {file_path}: {e}")
            return 'utf-8'
    
    @staticmethod
    def read_file_safe(file_path: str, encoding: str = None) -> Tuple[str, str]:
        """
        Safely read file with encoding detection
        
        Args:
            file_path: Path to the file
            encoding: Specific encoding to use (auto-detect if None)
            
        Returns:
            Tuple of (content, encoding_used)
        """
        
        if not encoding:
            encoding = FileUtils.detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return content, encoding
            
        except UnicodeDecodeError:
            # Try with different encodings
            fallback_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for fallback_encoding in fallback_encodings:
                if fallback_encoding == encoding:
                    continue
                
                try:
                    with open(file_path, 'r', encoding=fallback_encoding) as f:
                        content = f.read()
                    logger.info(f"Used fallback encoding {fallback_encoding} for {file_path}")
                    return content, fallback_encoding
                except UnicodeDecodeError:
                    continue
            
            # Last resort: read as binary and decode with errors ignored
            with open(file_path, 'rb') as f:
                raw_content = f.read()
            content = raw_content.decode('utf-8', errors='replace')
            logger.warning(f"Used error-tolerant decoding for {file_path}")
            return content, 'utf-8'
            
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return "", encoding
    
    @staticmethod
    def write_file_safe(
        file_path: str, 
        content: str, 
        encoding: str = 'utf-8',
        create_dirs: bool = True
    ) -> bool:
        """
        Safely write file with proper error handling
        
        Args:
            file_path: Path to the file
            content: Content to write
            encoding: File encoding
            create_dirs: Create parent directories if needed
            
        Returns:
            Success status
        """
        
        try:
            file_path = Path(file_path)
            
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return False
    
    @staticmethod
    def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
        """
        Calculate hash of a file
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
            
        Returns:
            Hex digest of the file hash
        """
        
        try:
            hash_obj = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """
        Get comprehensive file information
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {}
        
        try:
            stat = file_path.stat()
            
            info = {
                'path': str(file_path),
                'name': file_path.name,
                'extension': file_path.suffix,
                'size_bytes': stat.st_size,
                'created_time': stat.st_ctime,
                'modified_time': stat.st_mtime,
                'is_file': file_path.is_file(),
                'is_directory': file_path.is_dir(),
                'is_symlink': file_path.is_symlink(),
                'permissions': oct(stat.st_mode)[-3:],
                'mime_type': mimetypes.guess_type(str(file_path))[0],
                'encoding': None,
                'line_count': 0,
                'hash_sha256': ""
            }
            
            if file_path.is_file():
                # Get encoding and line count for text files
                try:
                    content, encoding = FileUtils.read_file_safe(str(file_path))
                    info['encoding'] = encoding
                    info['line_count'] = len(content.split('\n'))
                except Exception:
                    pass
                
                # Calculate hash
                info['hash_sha256'] = FileUtils.calculate_file_hash(str(file_path))
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            return {}
    
    @staticmethod
    def is_text_file(file_path: str) -> bool:
        """
        Check if file is likely a text file
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file appears to be text
        """
        
        # Check by extension first
        text_extensions = {
            '.txt', '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.htm', 
            '.css', '.scss', '.sass', '.less', '.json', '.xml', '.yml', 
            '.yaml', '.md', '.rst', '.java', '.c', '.cpp', '.h', '.hpp',
            '.cs', '.php', '.rb', '.go', '.rs', '.sql', '.sh', '.bat',
            '.ps1', '.r', '.m', '.scala', '.kt', '.swift', '.dart',
            '.vue', '.svelte', '.jsx', '.tsx'
        }
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext in text_extensions:
            return True
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type.startswith('text/'):
            return True
        
        # Check file content (sample first 1024 bytes)
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(1024)
            
            # Check for null bytes (binary indicator)
            if b'\x00' in sample:
                return False
            
            # Try to decode as text
            try:
                sample.decode('utf-8')
                return True
            except UnicodeDecodeError:
                try:
                    sample.decode('latin-1')
                    return True
                except UnicodeDecodeError:
                    return False
                    
        except Exception:
            return False
    
    @staticmethod
    def backup_file(file_path: str, backup_dir: str = None) -> Optional[str]:
        """
        Create a backup of a file
        
        Args:
            file_path: Path to the file to backup
            backup_dir: Directory for backup (same dir if None)
            
        Returns:
            Path to backup file if successful
        """
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return None
            
            if backup_dir:
                backup_dir = Path(backup_dir)
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_dir / f"{file_path.name}.backup"
            else:
                backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            
            # Add timestamp if backup already exists
            counter = 1
            original_backup_path = backup_path
            while backup_path.exists():
                backup_path = original_backup_path.with_suffix(
                    f"{original_backup_path.suffix}.{counter}"
                )
                counter += 1
            
            # Copy file
            import shutil
            shutil.copy2(file_path, backup_path)
            
            logger.info(f"Created backup: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to backup file {file_path}: {e}")
            return None
    
    @staticmethod
    def restore_from_backup(backup_path: str, target_path: str = None) -> bool:
        """
        Restore file from backup
        
        Args:
            backup_path: Path to backup file
            target_path: Target restore path (derive from backup if None)
            
        Returns:
            Success status
        """
        
        try:
            backup_path = Path(backup_path)
            
            if not backup_path.exists():
                logger.error(f"Backup file does not exist: {backup_path}")
                return False
            
            if not target_path:
                # Try to derive original path
                target_path = backup_path
                if backup_path.name.endswith('.backup'):
                    target_path = backup_path.with_name(
                        backup_path.name[:-7]  # Remove .backup
                    )
            else:
                target_path = Path(target_path)
            
            # Copy backup to target
            import shutil
            shutil.copy2(backup_path, target_path)
            
            logger.info(f"Restored from backup: {backup_path} -> {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from backup {backup_path}: {e}")
            return False
    
    @staticmethod
    def find_files(
        directory: str,
        patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        recursive: bool = True
    ) -> List[str]:
        """
        Find files matching patterns
        
        Args:
            directory: Directory to search
            patterns: List of glob patterns to match
            exclude_patterns: List of glob patterns to exclude
            recursive: Search recursively
            
        Returns:
            List of matching file paths
        """
        
        import fnmatch
        
        directory = Path(directory)
        
        if not directory.exists():
            return []
        
        files = []
        
        try:
            if recursive:
                for root, dirs, filenames in os.walk(directory):
                    for filename in filenames:
                        file_path = Path(root) / filename
                        files.append(str(file_path.resolve()))
            else:
                for file_path in directory.iterdir():
                    if file_path.is_file():
                        files.append(str(file_path.resolve()))
            
            # Apply include patterns
            if patterns:
                matched_files = []
                for pattern in patterns:
                    for file_path in files:
                        if fnmatch.fnmatch(Path(file_path).name, pattern):
                            matched_files.append(file_path)
                files = list(set(matched_files))  # Remove duplicates
            
            # Apply exclude patterns
            if exclude_patterns:
                filtered_files = []
                for file_path in files:
                    exclude = False
                    for pattern in exclude_patterns:
                        if fnmatch.fnmatch(Path(file_path).name, pattern):
                            exclude = True
                            break
                    if not exclude:
                        filtered_files.append(file_path)
                files = filtered_files
            
            return sorted(files)
            
        except Exception as e:
            logger.error(f"Failed to find files in {directory}: {e}")
            return []
    
    @staticmethod
    def get_line_ending_style(content: str) -> str:
        """
        Detect line ending style in content
        
        Args:
            content: Text content
            
        Returns:
            Line ending style ('unix', 'windows', 'mac', 'mixed')
        """
        
        crlf_count = content.count('\r\n')
        lf_count = content.count('\n') - crlf_count
        cr_count = content.count('\r') - crlf_count
        
        if crlf_count > 0 and lf_count == 0 and cr_count == 0:
            return 'windows'
        elif lf_count > 0 and crlf_count == 0 and cr_count == 0:
            return 'unix'
        elif cr_count > 0 and crlf_count == 0 and lf_count == 0:
            return 'mac'
        elif crlf_count > 0 or lf_count > 0 or cr_count > 0:
            return 'mixed'
        else:
            return 'unix'  # Default for empty content
    
    @staticmethod
    def normalize_line_endings(content: str, style: str = 'unix') -> str:
        """
        Normalize line endings in content
        
        Args:
            content: Text content
            style: Target style ('unix', 'windows', 'mac')
            
        Returns:
            Content with normalized line endings
        """
        
        # First normalize to unix style
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Then convert to target style
        if style == 'windows':
            content = content.replace('\n', '\r\n')
        elif style == 'mac':
            content = content.replace('\n', '\r')
        # unix style is already normalized
        
        return content