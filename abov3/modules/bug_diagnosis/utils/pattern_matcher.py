"""
Pattern matching utilities for finding similar code patterns
"""

import re
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import difflib

class PatternMatcher:
    """
    Find and match code patterns across the codebase
    """
    
    def __init__(self):
        self.pattern_cache = {}
        self.similarity_threshold = 0.6
        
    def find_similar_patterns(
        self,
        pattern: str,
        search_path: str,
        language: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar patterns in the codebase
        """
        results = []
        
        # Get file extensions to search
        extensions = self._get_extensions(language)
        
        # Search for files
        for file_path in self._find_files(search_path, extensions):
            matches = self._search_file(file_path, pattern)
            results.extend(matches)
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:max_results]
    
    def _get_extensions(self, language: Optional[str]) -> List[str]:
        """Get file extensions for a language"""
        language_extensions = {
            "python": [".py"],
            "javascript": [".js", ".jsx", ".mjs"],
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "csharp": [".cs"],
            "go": [".go"],
            "rust": [".rs"],
            "ruby": [".rb"],
            "php": [".php"],
            "cpp": [".cpp", ".cc", ".hpp", ".h"]
        }
        
        if language and language in language_extensions:
            return language_extensions[language]
        
        # Return all common extensions if no language specified
        all_extensions = []
        for exts in language_extensions.values():
            all_extensions.extend(exts)
        return list(set(all_extensions))
    
    def _find_files(self, search_path: str, extensions: List[str]) -> List[str]:
        """Find all files with given extensions"""
        files = []
        
        try:
            path = Path(search_path)
            
            # Skip common directories
            skip_dirs = {
                "node_modules", ".git", "__pycache__", ".venv", "venv",
                "build", "dist", ".next", "target", "vendor"
            }
            
            for ext in extensions:
                for file_path in path.rglob(f"*{ext}"):
                    # Skip if in excluded directory
                    if any(skip_dir in str(file_path) for skip_dir in skip_dirs):
                        continue
                    files.append(str(file_path))
        
        except Exception as e:
            # Log error but don't fail
            pass
        
        return files
    
    def _search_file(self, file_path: str, pattern: str) -> List[Dict[str, Any]]:
        """Search for pattern in a single file"""
        matches = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Search for exact matches first
            for i, line in enumerate(lines):
                if pattern.lower() in line.lower():
                    matches.append({
                        "file": file_path,
                        "line": i + 1,
                        "pattern": line.strip(),
                        "similarity": 1.0,
                        "match_type": "exact"
                    })
            
            # Search for similar patterns using fuzzy matching
            pattern_words = set(re.findall(r'\w+', pattern.lower()))
            
            for i, line in enumerate(lines):
                line_words = set(re.findall(r'\w+', line.lower()))
                
                # Calculate Jaccard similarity
                if pattern_words and line_words:
                    intersection = pattern_words & line_words
                    union = pattern_words | line_words
                    similarity = len(intersection) / len(union)
                    
                    if similarity >= self.similarity_threshold:
                        matches.append({
                            "file": file_path,
                            "line": i + 1,
                            "pattern": line.strip(),
                            "similarity": similarity,
                            "match_type": "fuzzy"
                        })
            
            # Use difflib for sequence matching
            matcher = difflib.SequenceMatcher(None, pattern.lower(), "")
            
            for i, line in enumerate(lines):
                matcher.set_seq2(line.lower())
                ratio = matcher.ratio()
                
                if ratio >= self.similarity_threshold:
                    # Check if not already added
                    if not any(m["line"] == i + 1 for m in matches):
                        matches.append({
                            "file": file_path,
                            "line": i + 1,
                            "pattern": line.strip(),
                            "similarity": ratio,
                            "match_type": "sequence"
                        })
        
        except Exception as e:
            # Log error but don't fail
            pass
        
        return matches
    
    def find_error_patterns(
        self,
        error_type: str,
        search_path: str,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find common error patterns in the codebase
        """
        # Define common error-prone patterns
        error_patterns = {
            "null_reference": [
                r"(\w+)\.(\w+)",  # Direct property access
                r"(\w+)\[(\w+)\]",  # Array/dict access
                r"(\w+)\(\)",  # Method call
            ],
            "index_error": [
                r"(\w+)\[(\d+)\]",  # Hard-coded index
                r"(\w+)\[len\((\w+)\)\]",  # Length-based access
                r"for .+ in range\(len\(",  # Range-based loops
            ],
            "type_error": [
                r"int\((\w+)\)",  # Type conversion
                r"str\((\w+)\)",
                r"float\((\w+)\)",
                r"(\w+) \+ (\w+)",  # Type mixing in operations
            ],
            "async_error": [
                r"async\s+def",  # Async functions
                r"await\s+",  # Await calls
                r"asyncio\.",  # Asyncio usage
            ]
        }
        
        patterns = error_patterns.get(error_type, [])
        results = []
        
        extensions = self._get_extensions(language)
        
        for file_path in self._find_files(search_path, extensions):
            for pattern in patterns:
                matches = self._find_regex_matches(file_path, pattern)
                results.extend(matches)
        
        return results
    
    def _find_regex_matches(self, file_path: str, pattern: str) -> List[Dict[str, Any]]:
        """Find regex matches in a file"""
        matches = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            regex = re.compile(pattern)
            
            for i, line in enumerate(lines):
                if regex.search(line):
                    matches.append({
                        "file": file_path,
                        "line": i + 1,
                        "pattern": line.strip(),
                        "regex": pattern,
                        "match_type": "regex"
                    })
        
        except Exception:
            pass
        
        return matches
    
    def extract_context(
        self,
        file_path: str,
        line_number: int,
        context_lines: int = 5
    ) -> Dict[str, Any]:
        """
        Extract code context around a specific line
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)
            
            context = {
                "before": lines[start:line_number-1],
                "target": lines[line_number-1] if line_number <= len(lines) else "",
                "after": lines[line_number:end],
                "function": self._find_containing_function(lines, line_number),
                "class": self._find_containing_class(lines, line_number)
            }
            
            return context
        
        except Exception:
            return {
                "before": [],
                "target": "",
                "after": [],
                "function": None,
                "class": None
            }
    
    def _find_containing_function(self, lines: List[str], target_line: int) -> Optional[str]:
        """Find the function containing the target line"""
        # Simple heuristic - look backwards for function definition
        for i in range(target_line - 1, -1, -1):
            line = lines[i]
            
            # Check for common function patterns
            if re.match(r'\s*(def|function|func|fn)\s+(\w+)', line):
                match = re.match(r'\s*(?:def|function|func|fn)\s+(\w+)', line)
                if match:
                    return match.group(1)
            
            # Stop if we hit a class definition (went too far)
            if re.match(r'\s*class\s+\w+', line):
                break
        
        return None
    
    def _find_containing_class(self, lines: List[str], target_line: int) -> Optional[str]:
        """Find the class containing the target line"""
        # Simple heuristic - look backwards for class definition
        for i in range(target_line - 1, -1, -1):
            line = lines[i]
            
            # Check for class definition
            if re.match(r'\s*class\s+(\w+)', line):
                match = re.match(r'\s*class\s+(\w+)', line)
                if match:
                    return match.group(1)
        
        return None