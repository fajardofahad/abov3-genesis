"""
ABOV3 Genesis - Intelligent Issue Tracking System
Enterprise-grade issue tracking with ML-powered insights and automated workflows
"""

import uuid
import json
import sqlite3
import threading
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import logging
import hashlib
import re
from functools import lru_cache

# ML imports with fallbacks
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    HAS_ML = True
except ImportError:
    HAS_ML = False


class IssueStatus(Enum):
    """Issue status states"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    TESTING = "testing"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REOPENED = "reopened"
    BLOCKED = "blocked"
    DUPLICATE = "duplicate"
    WONT_FIX = "wont_fix"


class IssuePriority(Enum):
    """Issue priority levels"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    TRIVIAL = 1


class IssueType(Enum):
    """Issue types"""
    BUG = "bug"
    FEATURE = "feature"
    ENHANCEMENT = "enhancement"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    CONFIGURATION = "configuration"


@dataclass
class IssueMetadata:
    """Metadata for an issue"""
    created_by: str
    assigned_to: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    components: List[str] = field(default_factory=list)
    affected_versions: List[str] = field(default_factory=list)
    fixed_versions: List[str] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IssueAttachment:
    """Attachment for an issue"""
    attachment_id: str
    filename: str
    content_type: str
    size: int
    uploaded_at: datetime
    uploaded_by: str
    description: Optional[str] = None
    content: Optional[bytes] = None  # For small files


@dataclass
class IssueComment:
    """Comment on an issue"""
    comment_id: str
    issue_id: str
    author: str
    content: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_internal: bool = False
    attachments: List[IssueAttachment] = field(default_factory=list)


@dataclass
class IssueLink:
    """Link between issues"""
    link_id: str
    source_issue_id: str
    target_issue_id: str
    link_type: str  # blocks, duplicates, relates_to, causes, etc.
    created_at: datetime
    created_by: str


@dataclass
class IssueTransition:
    """State transition for an issue"""
    transition_id: str
    issue_id: str
    from_status: IssueStatus
    to_status: IssueStatus
    transitioned_at: datetime
    transitioned_by: str
    comment: Optional[str] = None
    duration_in_status: Optional[float] = None  # Time spent in previous status


@dataclass
class Issue:
    """Comprehensive issue representation"""
    issue_id: str
    title: str
    description: str
    issue_type: IssueType
    status: IssueStatus
    priority: IssuePriority
    created_at: datetime
    updated_at: datetime
    metadata: IssueMetadata
    
    # Tracking
    error_ids: List[str] = field(default_factory=list)  # Links to error resolution engine
    stack_traces: List[str] = field(default_factory=list)
    code_references: List[Dict[str, Any]] = field(default_factory=list)
    
    # Relationships
    parent_issue_id: Optional[str] = None
    sub_issues: List[str] = field(default_factory=list)
    linked_issues: List[IssueLink] = field(default_factory=list)
    
    # History
    comments: List[IssueComment] = field(default_factory=list)
    transitions: List[IssueTransition] = field(default_factory=list)
    attachments: List[IssueAttachment] = field(default_factory=list)
    
    # Metrics
    resolution_time: Optional[float] = None
    reopen_count: int = 0
    comment_count: int = 0
    watchers: List[str] = field(default_factory=list)
    
    # ML features
    embedding: Optional[List[float]] = None
    similar_issues: List[Tuple[str, float]] = field(default_factory=list)  # (issue_id, similarity)
    predicted_resolution_time: Optional[float] = None
    auto_assigned: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert enums
        data['issue_type'] = self.issue_type.value
        data['status'] = self.status.value
        data['priority'] = self.priority.value
        # Convert datetime
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data


class IssueDatabase:
    """SQLite database for issue storage"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / '.abov3' / 'issues.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database schema"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Issues table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS issues (
                    issue_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    issue_type TEXT,
                    status TEXT,
                    priority INTEGER,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT,
                    error_ids TEXT,
                    parent_issue_id TEXT,
                    resolution_time REAL,
                    reopen_count INTEGER DEFAULT 0,
                    embedding TEXT
                )
            ''')
            
            # Comments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS comments (
                    comment_id TEXT PRIMARY KEY,
                    issue_id TEXT,
                    author TEXT,
                    content TEXT,
                    created_at TEXT,
                    is_internal INTEGER DEFAULT 0,
                    FOREIGN KEY (issue_id) REFERENCES issues (issue_id)
                )
            ''')
            
            # Transitions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transitions (
                    transition_id TEXT PRIMARY KEY,
                    issue_id TEXT,
                    from_status TEXT,
                    to_status TEXT,
                    transitioned_at TEXT,
                    transitioned_by TEXT,
                    duration_in_status REAL,
                    FOREIGN KEY (issue_id) REFERENCES issues (issue_id)
                )
            ''')
            
            # Links table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS issue_links (
                    link_id TEXT PRIMARY KEY,
                    source_issue_id TEXT,
                    target_issue_id TEXT,
                    link_type TEXT,
                    created_at TEXT,
                    created_by TEXT,
                    FOREIGN KEY (source_issue_id) REFERENCES issues (issue_id),
                    FOREIGN KEY (target_issue_id) REFERENCES issues (issue_id)
                )
            ''')
            
            # Create indices
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_issues_status ON issues (status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_issues_priority ON issues (priority)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_issues_type ON issues (issue_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_issues_created ON issues (created_at)')
            
            conn.commit()
    
    def save_issue(self, issue: Issue):
        """Save issue to database"""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Serialize complex fields
                metadata_json = json.dumps(asdict(issue.metadata))
                error_ids_json = json.dumps(issue.error_ids)
                embedding_json = json.dumps(issue.embedding) if issue.embedding else None
                
                cursor.execute('''
                    INSERT OR REPLACE INTO issues 
                    (issue_id, title, description, issue_type, status, priority,
                     created_at, updated_at, metadata, error_ids, parent_issue_id,
                     resolution_time, reopen_count, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    issue.issue_id, issue.title, issue.description,
                    issue.issue_type.value, issue.status.value, issue.priority.value,
                    issue.created_at.isoformat(), issue.updated_at.isoformat(),
                    metadata_json, error_ids_json, issue.parent_issue_id,
                    issue.resolution_time, issue.reopen_count, embedding_json
                ))
                
                # Save comments
                for comment in issue.comments:
                    cursor.execute('''
                        INSERT OR REPLACE INTO comments
                        (comment_id, issue_id, author, content, created_at, is_internal)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        comment.comment_id, issue.issue_id, comment.author,
                        comment.content, comment.created_at.isoformat(),
                        1 if comment.is_internal else 0
                    ))
                
                # Save transitions
                for transition in issue.transitions:
                    cursor.execute('''
                        INSERT OR REPLACE INTO transitions
                        (transition_id, issue_id, from_status, to_status,
                         transitioned_at, transitioned_by, duration_in_status)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        transition.transition_id, issue.issue_id,
                        transition.from_status.value, transition.to_status.value,
                        transition.transitioned_at.isoformat(),
                        transition.transitioned_by, transition.duration_in_status
                    ))
                
                conn.commit()
    
    def get_issue(self, issue_id: str) -> Optional[Issue]:
        """Retrieve issue from database"""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Get issue
                cursor.execute('SELECT * FROM issues WHERE issue_id = ?', (issue_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                # Reconstruct issue
                issue = self._row_to_issue(row)
                
                # Load comments
                cursor.execute('SELECT * FROM comments WHERE issue_id = ?', (issue_id,))
                for comment_row in cursor.fetchall():
                    comment = IssueComment(
                        comment_id=comment_row[0],
                        issue_id=comment_row[1],
                        author=comment_row[2],
                        content=comment_row[3],
                        created_at=datetime.fromisoformat(comment_row[4]),
                        is_internal=bool(comment_row[5])
                    )
                    issue.comments.append(comment)
                
                # Load transitions
                cursor.execute('SELECT * FROM transitions WHERE issue_id = ?', (issue_id,))
                for trans_row in cursor.fetchall():
                    transition = IssueTransition(
                        transition_id=trans_row[0],
                        issue_id=trans_row[1],
                        from_status=IssueStatus(trans_row[2]),
                        to_status=IssueStatus(trans_row[3]),
                        transitioned_at=datetime.fromisoformat(trans_row[4]),
                        transitioned_by=trans_row[5],
                        duration_in_status=trans_row[6]
                    )
                    issue.transitions.append(transition)
                
                return issue
    
    def _row_to_issue(self, row) -> Issue:
        """Convert database row to Issue object"""
        metadata_dict = json.loads(row[8])
        metadata = IssueMetadata(**metadata_dict)
        
        issue = Issue(
            issue_id=row[0],
            title=row[1],
            description=row[2],
            issue_type=IssueType(row[3]),
            status=IssueStatus(row[4]),
            priority=IssuePriority(row[5]),
            created_at=datetime.fromisoformat(row[6]),
            updated_at=datetime.fromisoformat(row[7]),
            metadata=metadata,
            error_ids=json.loads(row[9]),
            parent_issue_id=row[10],
            resolution_time=row[11],
            reopen_count=row[12]
        )
        
        if row[13]:  # embedding
            issue.embedding = json.loads(row[13])
        
        return issue
    
    def search_issues(self, **criteria) -> List[Issue]:
        """Search issues by criteria"""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                query = 'SELECT * FROM issues WHERE 1=1'
                params = []
                
                if 'status' in criteria:
                    query += ' AND status = ?'
                    params.append(criteria['status'].value if isinstance(criteria['status'], IssueStatus) else criteria['status'])
                
                if 'priority' in criteria:
                    query += ' AND priority >= ?'
                    params.append(criteria['priority'].value if isinstance(criteria['priority'], IssuePriority) else criteria['priority'])
                
                if 'issue_type' in criteria:
                    query += ' AND issue_type = ?'
                    params.append(criteria['issue_type'].value if isinstance(criteria['issue_type'], IssueType) else criteria['issue_type'])
                
                if 'created_after' in criteria:
                    query += ' AND created_at >= ?'
                    params.append(criteria['created_after'].isoformat())
                
                cursor.execute(query, params)
                
                issues = []
                for row in cursor.fetchall():
                    issues.append(self._row_to_issue(row))
                
                return issues


class IssueAnalyzer:
    """ML-powered issue analyzer"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000) if HAS_ML else None
        self.clustering_model = None
        self.similarity_threshold = 0.7
        self.issue_embeddings = {}
    
    def analyze_issue(self, issue: Issue) -> Dict[str, Any]:
        """Analyze issue with ML"""
        analysis = {
            'complexity_score': self._calculate_complexity(issue),
            'urgency_score': self._calculate_urgency(issue),
            'impact_score': self._estimate_impact(issue),
            'resolution_prediction': self._predict_resolution_time(issue),
            'suggested_assignee': self._suggest_assignee(issue),
            'related_components': self._identify_components(issue),
            'risk_assessment': self._assess_risk(issue)
        }
        
        # Generate embedding
        if HAS_ML:
            issue.embedding = self._generate_embedding(issue)
            analysis['similar_issues'] = self._find_similar_issues(issue)
        
        return analysis
    
    def _calculate_complexity(self, issue: Issue) -> float:
        """Calculate issue complexity"""
        score = 0.5  # Base score
        
        # Factor in description length
        desc_length = len(issue.description)
        if desc_length > 1000:
            score += 0.2
        elif desc_length > 500:
            score += 0.1
        
        # Factor in stack traces
        if issue.stack_traces:
            score += 0.1 * min(len(issue.stack_traces), 3)
        
        # Factor in linked issues
        if issue.linked_issues:
            score += 0.1 * min(len(issue.linked_issues), 3)
        
        # Factor in code references
        if issue.code_references:
            score += 0.1 * min(len(issue.code_references), 3)
        
        return min(score, 1.0)
    
    def _calculate_urgency(self, issue: Issue) -> float:
        """Calculate urgency score"""
        urgency = issue.priority.value / 5.0  # Normalize priority
        
        # Factor in issue type
        if issue.issue_type == IssueType.SECURITY:
            urgency = min(urgency + 0.3, 1.0)
        elif issue.issue_type == IssueType.BUG:
            urgency = min(urgency + 0.1, 1.0)
        
        # Factor in time since creation
        age = (datetime.now() - issue.created_at).days
        if age > 7:
            urgency = min(urgency + 0.1, 1.0)
        elif age > 30:
            urgency = min(urgency + 0.2, 1.0)
        
        return urgency
    
    def _estimate_impact(self, issue: Issue) -> float:
        """Estimate issue impact"""
        impact = 0.5  # Base impact
        
        # Factor in affected versions
        if issue.metadata.affected_versions:
            impact += 0.1 * min(len(issue.metadata.affected_versions), 3)
        
        # Factor in watchers
        if issue.watchers:
            impact += 0.05 * min(len(issue.watchers), 5)
        
        # Factor in sub-issues
        if issue.sub_issues:
            impact += 0.1 * min(len(issue.sub_issues), 3)
        
        return min(impact, 1.0)
    
    def _predict_resolution_time(self, issue: Issue) -> float:
        """Predict resolution time in hours"""
        # Simple heuristic-based prediction
        base_time = 8.0  # Base 8 hours
        
        # Adjust based on priority
        priority_multiplier = {
            IssuePriority.CRITICAL: 0.5,
            IssuePriority.HIGH: 0.75,
            IssuePriority.MEDIUM: 1.0,
            IssuePriority.LOW: 1.5,
            IssuePriority.TRIVIAL: 2.0
        }
        base_time *= priority_multiplier.get(issue.priority, 1.0)
        
        # Adjust based on complexity
        complexity = self._calculate_complexity(issue)
        base_time *= (1 + complexity)
        
        # Adjust based on type
        type_multiplier = {
            IssueType.BUG: 1.0,
            IssueType.FEATURE: 2.0,
            IssueType.ENHANCEMENT: 1.5,
            IssueType.DOCUMENTATION: 0.5,
            IssueType.CONFIGURATION: 0.75
        }
        base_time *= type_multiplier.get(issue.issue_type, 1.0)
        
        return base_time
    
    def _suggest_assignee(self, issue: Issue) -> Optional[str]:
        """Suggest best assignee for issue"""
        # This would use historical data and expertise matching
        # For now, return None
        return None
    
    def _identify_components(self, issue: Issue) -> List[str]:
        """Identify affected components"""
        components = set(issue.metadata.components)
        
        # Extract from description
        description_lower = issue.description.lower()
        
        component_keywords = {
            'ui': ['interface', 'gui', 'frontend', 'display'],
            'backend': ['server', 'api', 'database', 'backend'],
            'auth': ['authentication', 'authorization', 'login', 'security'],
            'data': ['data', 'storage', 'database', 'cache'],
            'network': ['network', 'connection', 'http', 'socket']
        }
        
        for component, keywords in component_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                components.add(component)
        
        return list(components)
    
    def _assess_risk(self, issue: Issue) -> Dict[str, Any]:
        """Assess risk associated with issue"""
        risk = {
            'level': 'low',
            'factors': [],
            'score': 0.0
        }
        
        score = 0.0
        
        # Security issues are high risk
        if issue.issue_type == IssueType.SECURITY:
            risk['factors'].append('Security vulnerability')
            score += 0.5
        
        # Critical priority is high risk
        if issue.priority == IssuePriority.CRITICAL:
            risk['factors'].append('Critical priority')
            score += 0.3
        
        # Many linked issues indicate higher risk
        if len(issue.linked_issues) > 5:
            risk['factors'].append('Multiple dependencies')
            score += 0.2
        
        # Reopened issues are higher risk
        if issue.reopen_count > 0:
            risk['factors'].append(f'Reopened {issue.reopen_count} times')
            score += 0.1 * issue.reopen_count
        
        risk['score'] = min(score, 1.0)
        
        if score >= 0.7:
            risk['level'] = 'high'
        elif score >= 0.4:
            risk['level'] = 'medium'
        else:
            risk['level'] = 'low'
        
        return risk
    
    def _generate_embedding(self, issue: Issue) -> List[float]:
        """Generate embedding for issue"""
        if not HAS_ML:
            return []
        
        # Combine text fields
        text = f"{issue.title} {issue.description}"
        
        # Add metadata
        text += f" {issue.issue_type.value} {issue.priority.name}"
        
        try:
            # Generate TF-IDF embedding
            embedding = self.vectorizer.fit_transform([text]).toarray()[0]
            return embedding.tolist()
        except:
            return []
    
    def _find_similar_issues(self, issue: Issue, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar issues"""
        if not HAS_ML or not issue.embedding:
            return []
        
        similar_issues = []
        
        for other_id, other_embedding in self.issue_embeddings.items():
            if other_id != issue.issue_id:
                similarity = cosine_similarity(
                    [issue.embedding],
                    [other_embedding]
                )[0][0]
                
                if similarity >= self.similarity_threshold:
                    similar_issues.append((other_id, similarity))
        
        # Sort by similarity
        similar_issues.sort(key=lambda x: x[1], reverse=True)
        
        return similar_issues[:top_k]


class IssueWorkflowEngine:
    """Workflow engine for issue management"""
    
    def __init__(self):
        self.workflows = self._define_workflows()
        self.automation_rules = []
        self.notification_handlers = []
    
    def _define_workflows(self) -> Dict[IssueType, Dict[IssueStatus, List[IssueStatus]]]:
        """Define allowed status transitions"""
        return {
            IssueType.BUG: {
                IssueStatus.OPEN: [IssueStatus.IN_PROGRESS, IssueStatus.CLOSED, IssueStatus.DUPLICATE],
                IssueStatus.IN_PROGRESS: [IssueStatus.TESTING, IssueStatus.BLOCKED, IssueStatus.OPEN],
                IssueStatus.TESTING: [IssueStatus.RESOLVED, IssueStatus.IN_PROGRESS],
                IssueStatus.RESOLVED: [IssueStatus.CLOSED, IssueStatus.REOPENED],
                IssueStatus.CLOSED: [IssueStatus.REOPENED],
                IssueStatus.REOPENED: [IssueStatus.IN_PROGRESS, IssueStatus.CLOSED],
                IssueStatus.BLOCKED: [IssueStatus.IN_PROGRESS, IssueStatus.CLOSED],
                IssueStatus.DUPLICATE: [],
                IssueStatus.WONT_FIX: []
            },
            IssueType.FEATURE: {
                IssueStatus.OPEN: [IssueStatus.IN_PROGRESS, IssueStatus.CLOSED, IssueStatus.WONT_FIX],
                IssueStatus.IN_PROGRESS: [IssueStatus.TESTING, IssueStatus.BLOCKED, IssueStatus.OPEN],
                IssueStatus.TESTING: [IssueStatus.RESOLVED, IssueStatus.IN_PROGRESS],
                IssueStatus.RESOLVED: [IssueStatus.CLOSED, IssueStatus.REOPENED],
                IssueStatus.CLOSED: [IssueStatus.REOPENED],
                IssueStatus.REOPENED: [IssueStatus.IN_PROGRESS],
                IssueStatus.BLOCKED: [IssueStatus.IN_PROGRESS, IssueStatus.CLOSED],
                IssueStatus.WONT_FIX: []
            }
        }
    
    def can_transition(self, issue: Issue, to_status: IssueStatus) -> bool:
        """Check if status transition is allowed"""
        workflow = self.workflows.get(issue.issue_type, {})
        allowed_transitions = workflow.get(issue.status, [])
        return to_status in allowed_transitions
    
    def transition_issue(self, issue: Issue, to_status: IssueStatus, 
                        user: str, comment: Optional[str] = None) -> bool:
        """Transition issue to new status"""
        if not self.can_transition(issue, to_status):
            return False
        
        # Calculate time in previous status
        duration = None
        if issue.transitions:
            last_transition = issue.transitions[-1]
            duration = (datetime.now() - last_transition.transitioned_at).total_seconds()
        
        # Create transition record
        transition = IssueTransition(
            transition_id=str(uuid.uuid4()),
            issue_id=issue.issue_id,
            from_status=issue.status,
            to_status=to_status,
            transitioned_at=datetime.now(),
            transitioned_by=user,
            comment=comment,
            duration_in_status=duration
        )
        
        # Update issue
        issue.status = to_status
        issue.updated_at = datetime.now()
        issue.transitions.append(transition)
        
        # Handle special transitions
        if to_status == IssueStatus.RESOLVED:
            issue.resolution_time = (datetime.now() - issue.created_at).total_seconds() / 3600
        elif to_status == IssueStatus.REOPENED:
            issue.reopen_count += 1
        
        # Run automation rules
        self._run_automation_rules(issue, transition)
        
        return True
    
    def _run_automation_rules(self, issue: Issue, transition: IssueTransition):
        """Run automation rules on transition"""
        for rule in self.automation_rules:
            if rule['trigger'] == 'status_change':
                if transition.to_status.value in rule['conditions'].get('to_status', []):
                    self._execute_automation_action(rule['action'], issue)
    
    def _execute_automation_action(self, action: Dict[str, Any], issue: Issue):
        """Execute automation action"""
        action_type = action.get('type')
        
        if action_type == 'assign':
            issue.metadata.assigned_to = action.get('assignee')
        elif action_type == 'add_label':
            issue.metadata.labels.append(action.get('label'))
        elif action_type == 'notify':
            self._send_notification(issue, action.get('recipients', []))
    
    def _send_notification(self, issue: Issue, recipients: List[str]):
        """Send notification about issue"""
        for handler in self.notification_handlers:
            handler(issue, recipients)
    
    def add_automation_rule(self, rule: Dict[str, Any]):
        """Add automation rule"""
        self.automation_rules.append(rule)
    
    def add_notification_handler(self, handler: Callable):
        """Add notification handler"""
        self.notification_handlers.append(handler)


class IssueTrackingSystem:
    """Main issue tracking system"""
    
    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path or Path.cwd()
        self.database = IssueDatabase()
        self.analyzer = IssueAnalyzer()
        self.workflow_engine = IssueWorkflowEngine()
        
        # Caches
        self._issue_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Setup logging
        self.logger = logging.getLogger('abov3.issue_tracking')
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def create_issue(self, title: str, description: str, issue_type: IssueType,
                    priority: IssuePriority, created_by: str, **kwargs) -> Issue:
        """Create new issue"""
        issue_id = str(uuid.uuid4())
        
        metadata = IssueMetadata(
            created_by=created_by,
            assigned_to=kwargs.get('assigned_to'),
            labels=kwargs.get('labels', []),
            components=kwargs.get('components', []),
            affected_versions=kwargs.get('affected_versions', []),
            environment=kwargs.get('environment', {})
        )
        
        issue = Issue(
            issue_id=issue_id,
            title=title,
            description=description,
            issue_type=issue_type,
            status=IssueStatus.OPEN,
            priority=priority,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata,
            error_ids=kwargs.get('error_ids', []),
            parent_issue_id=kwargs.get('parent_issue_id')
        )
        
        # Analyze issue
        analysis = self.analyzer.analyze_issue(issue)
        
        # Auto-assign if suggested
        if analysis.get('suggested_assignee') and not issue.metadata.assigned_to:
            issue.metadata.assigned_to = analysis['suggested_assignee']
            issue.auto_assigned = True
        
        # Set predicted resolution time
        issue.predicted_resolution_time = analysis.get('resolution_prediction')
        
        # Find similar issues
        if 'similar_issues' in analysis:
            issue.similar_issues = analysis['similar_issues']
        
        # Save to database
        self.database.save_issue(issue)
        
        # Cache
        self._issue_cache[issue_id] = (issue, datetime.now())
        
        self.logger.info(f"Created issue {issue_id}: {title}")
        
        return issue
    
    def get_issue(self, issue_id: str) -> Optional[Issue]:
        """Get issue by ID"""
        # Check cache
        if issue_id in self._issue_cache:
            issue, cached_at = self._issue_cache[issue_id]
            if (datetime.now() - cached_at).seconds < self._cache_ttl:
                return issue
        
        # Load from database
        issue = self.database.get_issue(issue_id)
        
        if issue:
            self._issue_cache[issue_id] = (issue, datetime.now())
        
        return issue
    
    def update_issue(self, issue_id: str, **updates) -> bool:
        """Update issue"""
        issue = self.get_issue(issue_id)
        if not issue:
            return False
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(issue, key):
                setattr(issue, key, value)
            elif hasattr(issue.metadata, key):
                setattr(issue.metadata, key, value)
        
        issue.updated_at = datetime.now()
        
        # Save to database
        self.database.save_issue(issue)
        
        # Update cache
        self._issue_cache[issue_id] = (issue, datetime.now())
        
        self.logger.info(f"Updated issue {issue_id}")
        
        return True
    
    def add_comment(self, issue_id: str, author: str, content: str, 
                   is_internal: bool = False) -> Optional[IssueComment]:
        """Add comment to issue"""
        issue = self.get_issue(issue_id)
        if not issue:
            return None
        
        comment = IssueComment(
            comment_id=str(uuid.uuid4()),
            issue_id=issue_id,
            author=author,
            content=content,
            created_at=datetime.now(),
            is_internal=is_internal
        )
        
        issue.comments.append(comment)
        issue.comment_count += 1
        issue.updated_at = datetime.now()
        
        # Save to database
        self.database.save_issue(issue)
        
        # Update cache
        self._issue_cache[issue_id] = (issue, datetime.now())
        
        self.logger.info(f"Added comment to issue {issue_id}")
        
        return comment
    
    def link_issues(self, source_id: str, target_id: str, link_type: str, 
                   created_by: str) -> Optional[IssueLink]:
        """Link two issues"""
        source = self.get_issue(source_id)
        target = self.get_issue(target_id)
        
        if not source or not target:
            return None
        
        link = IssueLink(
            link_id=str(uuid.uuid4()),
            source_issue_id=source_id,
            target_issue_id=target_id,
            link_type=link_type,
            created_at=datetime.now(),
            created_by=created_by
        )
        
        source.linked_issues.append(link)
        source.updated_at = datetime.now()
        
        # Save to database
        self.database.save_issue(source)
        
        # Update cache
        self._issue_cache[source_id] = (source, datetime.now())
        
        self.logger.info(f"Linked issues {source_id} -> {target_id} ({link_type})")
        
        return link
    
    def transition_status(self, issue_id: str, to_status: IssueStatus, 
                         user: str, comment: Optional[str] = None) -> bool:
        """Transition issue status"""
        issue = self.get_issue(issue_id)
        if not issue:
            return False
        
        success = self.workflow_engine.transition_issue(issue, to_status, user, comment)
        
        if success:
            # Save to database
            self.database.save_issue(issue)
            
            # Update cache
            self._issue_cache[issue_id] = (issue, datetime.now())
            
            self.logger.info(f"Transitioned issue {issue_id} to {to_status.value}")
        
        return success
    
    def search_issues(self, **criteria) -> List[Issue]:
        """Search for issues"""
        return self.database.search_issues(**criteria)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        all_issues = self.search_issues()
        
        stats = {
            'total_issues': len(all_issues),
            'open_issues': len([i for i in all_issues if i.status == IssueStatus.OPEN]),
            'in_progress': len([i for i in all_issues if i.status == IssueStatus.IN_PROGRESS]),
            'resolved': len([i for i in all_issues if i.status == IssueStatus.RESOLVED]),
            'by_type': Counter(i.issue_type.value for i in all_issues),
            'by_priority': Counter(i.priority.name for i in all_issues),
            'average_resolution_time': None,
            'reopen_rate': 0.0
        }
        
        # Calculate average resolution time
        resolved_times = [i.resolution_time for i in all_issues if i.resolution_time]
        if resolved_times:
            stats['average_resolution_time'] = sum(resolved_times) / len(resolved_times)
        
        # Calculate reopen rate
        reopened = [i for i in all_issues if i.reopen_count > 0]
        if all_issues:
            stats['reopen_rate'] = len(reopened) / len(all_issues)
        
        return stats
    
    def generate_report(self, report_type: str = 'summary', 
                       **filters) -> Dict[str, Any]:
        """Generate report"""
        issues = self.search_issues(**filters)
        
        if report_type == 'summary':
            return self._generate_summary_report(issues)
        elif report_type == 'detailed':
            return self._generate_detailed_report(issues)
        elif report_type == 'trends':
            return self._generate_trends_report(issues)
        else:
            return {}
    
    def _generate_summary_report(self, issues: List[Issue]) -> Dict[str, Any]:
        """Generate summary report"""
        return {
            'total': len(issues),
            'by_status': Counter(i.status.value for i in issues),
            'by_type': Counter(i.issue_type.value for i in issues),
            'by_priority': Counter(i.priority.name for i in issues),
            'critical_issues': [i.to_dict() for i in issues if i.priority == IssuePriority.CRITICAL][:10]
        }
    
    def _generate_detailed_report(self, issues: List[Issue]) -> Dict[str, Any]:
        """Generate detailed report"""
        return {
            'issues': [i.to_dict() for i in issues],
            'statistics': self.get_statistics(),
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'issue_count': len(issues)
            }
        }
    
    def _generate_trends_report(self, issues: List[Issue]) -> Dict[str, Any]:
        """Generate trends report"""
        # Group by creation date
        by_date = defaultdict(list)
        for issue in issues:
            date_key = issue.created_at.date().isoformat()
            by_date[date_key].append(issue)
        
        trends = {
            'issues_over_time': {
                date: len(issues_list) 
                for date, issues_list in sorted(by_date.items())
            },
            'resolution_time_trend': [],
            'reopen_trend': []
        }
        
        return trends


# Global instance
_issue_tracker = None

def get_issue_tracker(project_path: Optional[Path] = None) -> IssueTrackingSystem:
    """Get global issue tracker instance"""
    global _issue_tracker
    if _issue_tracker is None:
        _issue_tracker = IssueTrackingSystem(project_path)
    return _issue_tracker


# Convenience functions
def create_issue(title: str, description: str, issue_type: IssueType = IssueType.BUG,
                priority: IssuePriority = IssuePriority.MEDIUM, **kwargs) -> Issue:
    """Create an issue"""
    tracker = get_issue_tracker()
    return tracker.create_issue(title, description, issue_type, priority, 
                               created_by=kwargs.get('created_by', 'system'), **kwargs)


def get_issue(issue_id: str) -> Optional[Issue]:
    """Get issue by ID"""
    tracker = get_issue_tracker()
    return tracker.get_issue(issue_id)