"""
Language and framework detection utilities
"""

import re
from typing import Optional, Dict, List, Tuple
import os

class LanguageDetector:
    """
    Detect programming language and framework from code and context
    """
    
    def __init__(self):
        self.language_signatures = self._init_language_signatures()
        self.framework_signatures = self._init_framework_signatures()
        
    def _init_language_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Initialize language detection signatures"""
        return {
            "python": {
                "extensions": [".py", ".pyw", ".pyx"],
                "keywords": ["def", "class", "import", "from", "self", "lambda", "yield"],
                "patterns": [
                    r"^#!/usr/bin/env python",
                    r"^# -\*- coding:",
                    r"def \w+\([^)]*\):",
                    r"class \w+(\([^)]*\))?:",
                    r"if __name__ == ['\"]__main__['\"]:"
                ],
                "imports": ["import", "from.*import"],
                "confidence_boost": ["pip", "requirements.txt", "setup.py", "pyproject.toml"]
            },
            "javascript": {
                "extensions": [".js", ".jsx", ".mjs"],
                "keywords": ["const", "let", "var", "function", "return", "async", "await"],
                "patterns": [
                    r"const \w+ = ",
                    r"let \w+ = ",
                    r"function \w+\([^)]*\)",
                    r"=>\s*{",
                    r"async\s+function",
                    r"class \w+ extends"
                ],
                "imports": ["import.*from", "require\\(", "export"],
                "confidence_boost": ["package.json", "node_modules", ".npmrc"]
            },
            "typescript": {
                "extensions": [".ts", ".tsx"],
                "keywords": ["interface", "type", "enum", "namespace", "declare"],
                "patterns": [
                    r"interface \w+",
                    r"type \w+ = ",
                    r": (string|number|boolean|any|void)",
                    r"<\w+>",
                    r"as \w+",
                    r"enum \w+"
                ],
                "imports": ["import.*from", "export"],
                "confidence_boost": ["tsconfig.json", "tslint.json", ".d.ts"]
            },
            "java": {
                "extensions": [".java"],
                "keywords": ["public", "private", "protected", "class", "interface", "extends", "implements"],
                "patterns": [
                    r"public class \w+",
                    r"private \w+ \w+;",
                    r"@\w+",
                    r"package \w+",
                    r"import \w+\.\w+",
                    r"public static void main"
                ],
                "imports": ["import java", "import javax", "package"],
                "confidence_boost": ["pom.xml", "build.gradle", "gradlew"]
            },
            "csharp": {
                "extensions": [".cs"],
                "keywords": ["namespace", "using", "public", "private", "class", "interface"],
                "patterns": [
                    r"namespace \w+",
                    r"using System",
                    r"public class \w+",
                    r"private \w+ \w+;",
                    r"\[.*\]",
                    r"async Task"
                ],
                "imports": ["using System", "using Microsoft"],
                "confidence_boost": [".csproj", ".sln", "NuGet.config"]
            },
            "go": {
                "extensions": [".go"],
                "keywords": ["package", "import", "func", "type", "struct", "interface", "defer"],
                "patterns": [
                    r"package \w+",
                    r"func \w+\([^)]*\)",
                    r"type \w+ struct",
                    r"if err != nil",
                    r":= ",
                    r"go func"
                ],
                "imports": ["import \\(", "import \""],
                "confidence_boost": ["go.mod", "go.sum"]
            },
            "rust": {
                "extensions": [".rs"],
                "keywords": ["fn", "let", "mut", "impl", "pub", "use", "struct", "enum", "trait"],
                "patterns": [
                    r"fn \w+\([^)]*\)",
                    r"let mut \w+",
                    r"impl \w+ for",
                    r"pub fn",
                    r"use std::",
                    r"struct \w+"
                ],
                "imports": ["use std", "use crate"],
                "confidence_boost": ["Cargo.toml", "Cargo.lock"]
            },
            "ruby": {
                "extensions": [".rb"],
                "keywords": ["def", "end", "class", "module", "require", "include"],
                "patterns": [
                    r"def \w+",
                    r"class \w+",
                    r"module \w+",
                    r"require ['\"]",
                    r"attr_(reader|writer|accessor)",
                    r"end$"
                ],
                "imports": ["require", "require_relative"],
                "confidence_boost": ["Gemfile", "Gemfile.lock", "Rakefile"]
            },
            "php": {
                "extensions": [".php"],
                "keywords": ["<?php", "function", "class", "public", "private", "echo", "$"],
                "patterns": [
                    r"<\?php",
                    r"function \w+\([^)]*\)",
                    r"class \w+",
                    r"\$\w+ = ",
                    r"namespace \w+",
                    r"use \w+"
                ],
                "imports": ["use ", "require", "include"],
                "confidence_boost": ["composer.json", "composer.lock"]
            },
            "cpp": {
                "extensions": [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".h"],
                "keywords": ["#include", "namespace", "class", "template", "std::", "void"],
                "patterns": [
                    r"#include [<\"]",
                    r"using namespace",
                    r"class \w+",
                    r"template<",
                    r"std::",
                    r"int main\("
                ],
                "imports": ["#include"],
                "confidence_boost": ["CMakeLists.txt", "Makefile"]
            }
        }
    
    def _init_framework_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Initialize framework detection signatures"""
        return {
            "django": {
                "language": "python",
                "indicators": ["django", "models.Model", "views.py", "urls.py", "settings.py"],
                "imports": ["from django", "import django"],
                "files": ["manage.py", "wsgi.py", "asgi.py"]
            },
            "flask": {
                "language": "python",
                "indicators": ["flask", "Flask", "@app.route", "render_template"],
                "imports": ["from flask", "import flask"],
                "files": ["app.py", "application.py"]
            },
            "fastapi": {
                "language": "python",
                "indicators": ["fastapi", "FastAPI", "@app.get", "@app.post", "pydantic"],
                "imports": ["from fastapi", "import fastapi"],
                "files": ["main.py"]
            },
            "react": {
                "language": "javascript",
                "indicators": ["react", "React", "useState", "useEffect", "jsx", "Component"],
                "imports": ["from 'react'", "from \"react\""],
                "files": ["App.js", "App.jsx", "index.js"]
            },
            "angular": {
                "language": "typescript",
                "indicators": ["@angular", "@Component", "@Injectable", "NgModule"],
                "imports": ["from '@angular"],
                "files": ["angular.json", "app.component.ts"]
            },
            "vue": {
                "language": "javascript",
                "indicators": ["vue", "Vue", "<template>", "<script>", "v-if", "v-for"],
                "imports": ["from 'vue'", "from \"vue\""],
                "files": [".vue", "vue.config.js"]
            },
            "express": {
                "language": "javascript",
                "indicators": ["express", "app.get", "app.post", "router", "middleware"],
                "imports": ["require('express')", "from 'express'"],
                "files": ["server.js", "app.js"]
            },
            "spring": {
                "language": "java",
                "indicators": ["@SpringBootApplication", "@RestController", "@Service", "@Component"],
                "imports": ["org.springframework"],
                "files": ["application.properties", "application.yml"]
            },
            "rails": {
                "language": "ruby",
                "indicators": ["Rails", "ActiveRecord", "ActionController"],
                "imports": ["require 'rails'"],
                "files": ["Gemfile", "config.ru", "Rakefile"]
            },
            "laravel": {
                "language": "php",
                "indicators": ["Laravel", "Illuminate", "Artisan", "Eloquent"],
                "imports": ["use Illuminate", "use App\\"],
                "files": ["artisan", "composer.json"]
            }
        }
    
    def detect_language(self, code: str, file_path: Optional[str] = None) -> Tuple[Optional[str], float]:
        """
        Detect programming language from code
        Returns: (language, confidence)
        """
        scores = {}
        
        # Check file extension first
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            for lang, info in self.language_signatures.items():
                if ext in info["extensions"]:
                    scores[lang] = scores.get(lang, 0) + 0.5
        
        # Analyze code patterns
        for lang, info in self.language_signatures.items():
            score = 0.0
            
            # Check keywords
            for keyword in info["keywords"]:
                if re.search(r'\b' + keyword + r'\b', code):
                    score += 0.1
            
            # Check patterns
            for pattern in info["patterns"]:
                if re.search(pattern, code, re.MULTILINE):
                    score += 0.15
            
            # Check imports
            for import_pattern in info["imports"]:
                if re.search(import_pattern, code, re.MULTILINE):
                    score += 0.2
            
            scores[lang] = scores.get(lang, 0) + score
        
        # Find best match
        if scores:
            best_lang = max(scores, key=scores.get)
            confidence = min(1.0, scores[best_lang])
            
            # Boost confidence if we have supporting files
            if file_path and os.path.dirname(file_path):
                dir_path = os.path.dirname(file_path)
                for boost_file in self.language_signatures[best_lang].get("confidence_boost", []):
                    if os.path.exists(os.path.join(dir_path, boost_file)):
                        confidence = min(1.0, confidence + 0.1)
            
            return best_lang, confidence
        
        return None, 0.0
    
    def detect_framework(self, code: str, language: Optional[str] = None) -> Tuple[Optional[str], float]:
        """
        Detect framework from code
        Returns: (framework, confidence)
        """
        scores = {}
        
        for framework, info in self.framework_signatures.items():
            # Skip if language doesn't match
            if language and info["language"] != language:
                continue
            
            score = 0.0
            
            # Check indicators
            for indicator in info["indicators"]:
                if indicator in code:
                    score += 0.2
            
            # Check imports
            for import_pattern in info["imports"]:
                if import_pattern in code:
                    score += 0.3
            
            if score > 0:
                scores[framework] = score
        
        # Find best match
        if scores:
            best_framework = max(scores, key=scores.get)
            confidence = min(1.0, scores[best_framework])
            return best_framework, confidence
        
        return None, 0.0
    
    def detect_from_error(self, error_message: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect language and framework from error message
        Returns: (language, framework)
        """
        language = None
        framework = None
        
        # Language-specific error patterns
        language_errors = {
            "python": ["Traceback", "SyntaxError", "IndentationError", "ImportError"],
            "javascript": ["ReferenceError", "TypeError", "SyntaxError", "at Object"],
            "java": ["java.lang", "Exception in thread", ".java:"],
            "csharp": ["System.", "at System.", ".cs:line"],
            "go": ["panic:", "goroutine", ".go:"],
            "rust": ["error[E", "rustc", ".rs:"],
            "ruby": [".rb:", "from", "in `"],
            "php": ["Fatal error:", "Parse error:", "Warning:", ".php on line"]
        }
        
        for lang, patterns in language_errors.items():
            if any(pattern in error_message for pattern in patterns):
                language = lang
                break
        
        # Framework-specific error patterns
        framework_errors = {
            "django": ["django.", "DoesNotExist", "models.py"],
            "flask": ["werkzeug", "flask."],
            "react": ["ReactDOM", "Component", "useState"],
            "angular": ["@angular", "NgModule"],
            "spring": ["springframework", "org.springframework"],
            "rails": ["ActiveRecord", "ActionController"]
        }
        
        for fw, patterns in framework_errors.items():
            if any(pattern in error_message for pattern in patterns):
                framework = fw
                break
        
        return language, framework