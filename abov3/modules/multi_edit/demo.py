"""
Multi-file Edits & Patch Sets Demo

Comprehensive demonstration of the multi-file edit capabilities with real-world scenarios
including complex refactoring, conflict resolution, and enterprise workflows.
"""

import asyncio
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
import logging

from . import MultiEditOrchestrator
from .core.patch_manager import PatchSetManager, FileChange
from .diff.engine import DiffEngine
from .review.interface import ReviewInterface
from .conflict.resolver import ConflictResolver
from .transaction.manager import TransactionManager
from .history.tracker import HistoryTracker
from .git_integration.manager import GitIntegrationManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiEditDemo:
    """Comprehensive demonstration of multi-file edit capabilities"""
    
    def __init__(self, demo_root: str = None):
        self.demo_root = Path(demo_root) if demo_root else Path(tempfile.mkdtemp(prefix="abov3_multi_edit_demo_"))
        self.demo_root.mkdir(exist_ok=True)
        
        # Initialize orchestrator
        self.orchestrator = MultiEditOrchestrator(str(self.demo_root))
        
        # Demo data
        self.sample_files = {}
        self.patch_sets = []
        
        print(f"Demo initialized in: {self.demo_root}")
    
    async def run_complete_demo(self):
        """Run complete demonstration of all features"""
        
        print("\n" + "="*80)
        print("ABOV3 GENESIS - MULTI-FILE EDITS & PATCH SETS DEMO")
        print("="*80)
        
        try:
            # Setup demo environment
            await self.setup_demo_environment()
            
            # Scenario 1: Basic multi-file patch creation and application
            await self.demo_basic_patch_operations()
            
            # Scenario 2: Complex refactoring across multiple files
            await self.demo_complex_refactoring()
            
            # Scenario 3: Conflict detection and resolution
            await self.demo_conflict_resolution()
            
            # Scenario 4: Interactive review workflow
            await self.demo_interactive_review()
            
            # Scenario 5: Transaction management with rollback
            await self.demo_transaction_management()
            
            # Scenario 6: Version tracking and history
            await self.demo_version_tracking()
            
            # Scenario 7: Git integration
            await self.demo_git_integration()
            
            # Scenario 8: Enterprise workflow
            await self.demo_enterprise_workflow()
            
            print("\n" + "="*80)
            print("DEMO COMPLETED SUCCESSFULLY!")
            print("="*80)
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"\nDemo failed with error: {e}")
            raise
    
    async def setup_demo_environment(self):
        """Setup demo environment with sample files"""
        
        print("\n--- Setting up Demo Environment ---")
        
        # Create sample Python project structure
        project_structure = {
            "main.py": '''"""
Main application entry point
"""
import sys
from utils.helpers import process_data, validate_input
from models.user import User
from services.auth import authenticate_user

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    
    if not validate_input(username):
        print("Invalid username")
        sys.exit(1)
    
    user = authenticate_user(username)
    if user:
        print(f"Welcome, {user.name}!")
        process_data(user.data)
    else:
        print("Authentication failed")

if __name__ == "__main__":
    main()
''',
            
            "utils/__init__.py": "",
            
            "utils/helpers.py": '''"""
Utility helper functions
"""
import re
from typing import Any, Dict

def validate_input(data: str) -> bool:
    """Validate user input"""
    if not data or len(data) < 3:
        return False
    
    # Check for valid characters
    pattern = r'^[a-zA-Z0-9_]+$'
    return bool(re.match(pattern, data))

def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process user data"""
    processed = {}
    
    for key, value in data.items():
        if isinstance(value, str):
            processed[key] = value.strip().lower()
        else:
            processed[key] = value
    
    return processed

def format_output(data: Any) -> str:
    """Format data for output"""
    if isinstance(data, dict):
        return "\\n".join(f"{k}: {v}" for k, v in data.items())
    return str(data)
''',
            
            "models/__init__.py": "",
            
            "models/user.py": '''"""
User model
"""
from typing import Dict, Any
from datetime import datetime

class User:
    """User class"""
    
    def __init__(self, username: str, email: str, data: Dict[str, Any] = None):
        self.username = username
        self.email = email
        self.name = username.capitalize()
        self.data = data or {}
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
        return {
            'username': self.username,
            'email': self.email,
            'name': self.name,
            'data': self.data,
            'created_at': self.created_at.isoformat()
        }
    
    def update_data(self, new_data: Dict[str, Any]):
        """Update user data"""
        self.data.update(new_data)
''',
            
            "services/__init__.py": "",
            
            "services/auth.py": '''"""
Authentication service
"""
from typing import Optional
from models.user import User

# Mock user database
USERS_DB = {
    "admin": {"email": "admin@example.com", "data": {"role": "admin"}},
    "user1": {"email": "user1@example.com", "data": {"role": "user"}},
    "guest": {"email": "guest@example.com", "data": {"role": "guest"}}
}

def authenticate_user(username: str) -> Optional[User]:
    """Authenticate user by username"""
    if username in USERS_DB:
        user_data = USERS_DB[username]
        return User(
            username=username,
            email=user_data["email"], 
            data=user_data["data"]
        )
    return None

def is_admin(user: User) -> bool:
    """Check if user is admin"""
    return user.data.get("role") == "admin"
''',
            
            "config.json": '''
{
    "app_name": "Demo Application",
    "version": "1.0.0",
    "debug": false,
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "demo_db"
    },
    "logging": {
        "level": "INFO",
        "file": "app.log"
    }
}
''',
            
            "requirements.txt": '''
requests>=2.25.0
pytest>=6.0.0
black>=21.0.0
mypy>=0.812
'''
        }
        
        # Create files
        for file_path, content in project_structure.items():
            full_path = self.demo_root / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        self.sample_files = project_structure
        print(f"Created {len(project_structure)} sample files")
    
    async def demo_basic_patch_operations(self):
        """Demonstrate basic patch set operations"""
        
        print("\n--- Demo 1: Basic Patch Set Operations ---")
        
        # Create a simple patch set
        changes = {
            "main.py": {
                "type": "modify",
                "content": self.sample_files["main.py"].replace("python main.py", "python3 main.py")
            },
            "README.md": {
                "type": "create",
                "content": "# Demo Application\n\nA simple demonstration application for ABOV3 multi-file edits.\n"
            }
        }
        
        # Create patch set
        patch_id = await self.orchestrator.create_patch_set(
            changes, 
            "Add README and update main.py usage"
        )
        
        print(f"Created patch set: {patch_id}")
        
        # Get patch statistics
        stats = await self.orchestrator.patch_manager.get_patch_statistics(patch_id)
        print(f"Patch statistics: {json.dumps(stats, indent=2)}")
        
        # Apply patch set
        await self.orchestrator.patch_manager.update_patch_status(patch_id, 'approved')
        result = await self.orchestrator.apply_changes(patch_id)
        
        print(f"Applied patch set with results: {json.dumps(result['statistics'], indent=2)}")
        self.patch_sets.append(patch_id)
    
    async def demo_complex_refactoring(self):
        """Demonstrate complex refactoring across multiple files"""
        
        print("\n--- Demo 2: Complex Refactoring ---")
        
        # Refactor: Rename function and update all references
        changes = {
            "utils/helpers.py": {
                "type": "modify",
                "content": self.sample_files["utils/helpers.py"].replace(
                    "def validate_input", "def validate_user_input"
                ).replace(
                    "def process_data", "def process_user_data"
                ).replace(
                    "def format_output", "def format_display_output"
                )
            },
            "main.py": {
                "type": "modify", 
                "content": self.sample_files["main.py"].replace(
                    "from utils.helpers import process_data, validate_input",
                    "from utils.helpers import process_user_data, validate_user_input"
                ).replace(
                    "validate_input(username)", "validate_user_input(username)"
                ).replace(
                    "process_data(user.data)", "process_user_data(user.data)"
                )
            },
            "models/user.py": {
                "type": "modify",
                "content": self.sample_files["models/user.py"] + '''
    
    def __str__(self) -> str:
        """String representation of user"""
        return f"User({self.username}, {self.email})"
    
    def __repr__(self) -> str:
        """Detailed representation of user"""
        return f"User(username='{self.username}', email='{self.email}', data={self.data})"
'''
            }
        }
        
        # Create refactoring patch
        patch_id = await self.orchestrator.create_patch_set(
            changes,
            "Refactor function names and add User string representations"
        )
        
        print(f"Created refactoring patch: {patch_id}")
        
        # Generate diff for review
        patch_set = await self.orchestrator.patch_manager.get_patch_set(patch_id)
        
        for file_change in patch_set.files:
            diff = await self.orchestrator.diff_engine.generate_file_diff(
                file_change.old_content or "",
                file_change.new_content or "",
                file_change.file_path
            )
            
            unified_diff = await self.orchestrator.diff_engine.format_unified_diff(diff)
            print(f"\\nDiff for {file_change.file_path}:")
            print(unified_diff[:500] + "..." if len(unified_diff) > 500 else unified_diff)
        
        self.patch_sets.append(patch_id)
    
    async def demo_conflict_resolution(self):
        """Demonstrate conflict detection and resolution"""
        
        print("\n--- Demo 3: Conflict Resolution ---")
        
        # Create two conflicting patch sets
        patch_set1_changes = {
            "config.json": {
                "type": "modify",
                "content": '''
{
    "app_name": "Enhanced Demo Application",
    "version": "1.1.0",
    "debug": true,
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "demo_db"
    },
    "logging": {
        "level": "DEBUG",
        "file": "app.log"
    }
}
'''
            }
        }
        
        patch_set2_changes = {
            "config.json": {
                "type": "modify",
                "content": '''
{
    "app_name": "Demo Application Pro",
    "version": "1.0.1",
    "debug": false,
    "database": {
        "host": "prod-db.example.com",
        "port": 5432,
        "name": "production_db"
    },
    "logging": {
        "level": "ERROR",
        "file": "prod.log"
    },
    "features": {
        "analytics": true,
        "monitoring": true
    }
}
'''
            }
        }
        
        # Create patch sets
        patch_id1 = await self.orchestrator.create_patch_set(
            patch_set1_changes,
            "Enable debug mode and update version"
        )
        
        patch_id2 = await self.orchestrator.create_patch_set(
            patch_set2_changes,
            "Production configuration with new features"
        )
        
        print(f"Created conflicting patches: {patch_id1}, {patch_id2}")
        
        # Detect conflicts
        patch_sets = [
            await self.orchestrator.patch_manager.get_patch_set(patch_id1),
            await self.orchestrator.patch_manager.get_patch_set(patch_id2)
        ]
        
        conflicts = await self.orchestrator.conflict_resolver.detect_conflicts(patch_sets)
        print(f"Detected {len(conflicts)} conflicts")
        
        for i, conflict in enumerate(conflicts):
            print(f"Conflict {i+1}: {conflict.conflict_type.value} in {conflict.file_path}")
        
        # Resolve conflicts using smart merge
        if conflicts:
            resolutions = await self.orchestrator.conflict_resolver.resolve_conflicts(
                conflicts, strategy=self.orchestrator.conflict_resolver.ResolutionStrategy.SMART_MERGE
            )
            
            for resolution in resolutions:
                print(f"Resolution: {resolution.resolution_strategy.value} with confidence {resolution.confidence}")
    
    async def demo_interactive_review(self):
        """Demonstrate interactive review workflow"""
        
        print("\n--- Demo 4: Interactive Review Workflow ---")
        
        if not self.patch_sets:
            print("No patch sets available for review")
            return
        
        # Get first patch set for review
        patch_id = self.patch_sets[0]
        patch_set = await self.orchestrator.patch_manager.get_patch_set(patch_id)
        
        print(f"Starting automated review for patch set: {patch_id}")
        
        # Start automated review (simulates human review)
        session_id = await self.orchestrator.review_interface.start_review(
            patch_set,
            reviewer="Demo Reviewer",
            output_format="json",
            auto_mode=True
        )
        
        # Get review session results
        session = await self.orchestrator.review_interface.get_review_session(session_id)
        if session:
            print(f"Review session {session_id} completed with status: {session.status}")
            print(f"Decisions made: {len(session.decisions)}")
            
            # Get approved changes
            approved = await self.orchestrator.review_interface.get_approved_changes(session_id)
            print(f"Approved files: {approved}")
    
    async def demo_transaction_management(self):
        """Demonstrate transaction management with rollback"""
        
        print("\n--- Demo 5: Transaction Management ---")
        
        # Create a transaction that will be rolled back
        patch_id = None
        
        try:
            async with self.orchestrator.transaction_manager.transaction("demo_transaction") as tx:
                # Add some operations to the transaction
                await tx.add_operation(
                    self.orchestrator.transaction_manager.OperationType.CREATE,
                    "temp_file.txt",
                    new_content="This is a temporary file that will be rolled back"
                )
                
                await tx.add_operation(
                    self.orchestrator.transaction_manager.OperationType.MODIFY,
                    "main.py",
                    old_content=self.sample_files["main.py"],
                    new_content=self.sample_files["main.py"] + "\\n# Temporary modification"
                )
                
                # Create a savepoint
                savepoint_id = await tx.create_savepoint("before_error")
                print(f"Created savepoint: {savepoint_id}")
                
                # Simulate an error
                raise Exception("Simulated transaction error")
                
        except Exception as e:
            print(f"Transaction failed as expected: {e}")
            print("All changes should be rolled back automatically")
        
        # Verify rollback
        temp_file = self.demo_root / "temp_file.txt"
        if not temp_file.exists():
            print("✓ Transaction rollback successful - temporary file was not created")
        else:
            print("✗ Transaction rollback failed - temporary file still exists")
        
        # Check transaction statistics
        stats = await self.orchestrator.transaction_manager.get_transaction_stats()
        print(f"Transaction statistics: {json.dumps(stats, indent=2)}")
    
    async def demo_version_tracking(self):
        """Demonstrate version tracking and history"""
        
        print("\n--- Demo 6: Version Tracking and History ---")
        
        # Record some changes
        change_id = await self.orchestrator.history_tracker.record_change(
            patch_set_id="demo_patch_1",
            transaction_id="demo_tx_1",
            operation_type="modify",
            file_path="main.py",
            author="Demo User",
            content_before=self.sample_files["main.py"],
            content_after=self.sample_files["main.py"] + "\\n# Version tracking demo"
        )
        
        print(f"Recorded change: {change_id}")
        
        # Create a version
        version_id = await self.orchestrator.history_tracker.create_version(
            patch_set_id="demo_patch_1",
            version_number="1.0.0-demo",
            author="Demo User",
            description="Demo version for showcasing version tracking",
            tags=["demo", "showcase"]
        )
        
        print(f"Created version: {version_id}")
        
        # Get file history
        history = await self.orchestrator.history_tracker.get_file_history("main.py", max_entries=5)
        print(f"File history entries: {len(history)}")
        
        # Get change statistics
        stats = await self.orchestrator.history_tracker.get_change_statistics(
            by_author=True,
            by_file_type=True
        )
        print(f"Change statistics: {json.dumps(stats, indent=2, default=str)}")
        
        # Create a branch
        branch_created = await self.orchestrator.history_tracker.create_branch(
            "demo_branch",
            "main",
            version_id,
            "Demo User",
            "Demo branch for testing"
        )
        
        if branch_created:
            print("✓ Demo branch created successfully")
        
        # List branches
        branches = await self.orchestrator.history_tracker.list_branches()
        print(f"Available branches: {[b.name for b in branches]}")
    
    async def demo_git_integration(self):
        """Demonstrate git integration"""
        
        print("\n--- Demo 7: Git Integration ---")
        
        # Initialize git repository
        git_initialized = await self.orchestrator.git_manager.initialize_repository()
        
        if git_initialized:
            print("✓ Git repository initialized")
            
            # Get git status
            status = await self.orchestrator.git_manager.get_status()
            print(f"Git status - Clean: {status.is_clean}, Modified: {len(status.modified_files)}, Untracked: {len(status.untracked_files)}")
            
            # Create a commit
            commit_hash = await self.orchestrator.git_manager.create_commit_for_patch_set(
                "demo_patch_git",
                "Demo commit from multi-edit system",
                "ABOV3 Demo"
            )
            
            if commit_hash:
                print(f"✓ Created git commit: {commit_hash[:8]}")
            
            # Get commit history
            history = await self.orchestrator.git_manager.get_commit_history(max_count=5)
            print(f"Git history: {len(history)} commits")
            
            for commit in history[:3]:
                print(f"  {commit.hash[:8]} - {commit.message}")
        
        else:
            print("Git integration demo skipped (git not available)")
    
    async def demo_enterprise_workflow(self):
        """Demonstrate complete enterprise workflow"""
        
        print("\n--- Demo 8: Complete Enterprise Workflow ---")
        
        # Simulate a complete enterprise workflow
        workflow_steps = [
            "1. Developer creates feature branch",
            "2. Multiple files are modified for feature",
            "3. Patch set is created and reviewed",
            "4. Conflicts are detected and resolved", 
            "5. Changes are approved and applied",
            "6. Version is tagged and history is recorded",
            "7. Changes are committed to git",
            "8. Metrics and reports are generated"
        ]
        
        print("Enterprise Workflow Steps:")
        for step in workflow_steps:
            print(f"  {step}")
        
        # Execute workflow
        try:
            # Step 1-2: Create feature branch and modify files
            feature_changes = {
                "services/auth.py": {
                    "type": "modify",
                    "content": self.sample_files["services/auth.py"] + '''
def get_user_permissions(user: User) -> List[str]:
    """Get user permissions based on role"""
    role = user.data.get("role", "guest")
    
    permissions = {
        "admin": ["read", "write", "delete", "manage"],
        "user": ["read", "write"],
        "guest": ["read"]
    }
    
    return permissions.get(role, [])

def check_permission(user: User, permission: str) -> bool:
    """Check if user has specific permission"""
    user_permissions = get_user_permissions(user)
    return permission in user_permissions
'''
                },
                "utils/helpers.py": {
                    "type": "modify",
                    "content": self.sample_files["utils/helpers.py"] + '''

def log_user_action(username: str, action: str, details: Dict[str, Any] = None):
    """Log user action for audit trail"""
    from datetime import datetime
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "username": username,
        "action": action,
        "details": details or {}
    }
    
    # In real implementation, this would write to a log file or database
    print(f"AUDIT: {log_entry}")
'''
                },
                "tests/test_auth.py": {
                    "type": "create",
                    "content": '''"""
Unit tests for authentication service
"""
import pytest
from services.auth import authenticate_user, get_user_permissions, check_permission
from models.user import User

def test_authenticate_valid_user():
    """Test authentication with valid user"""
    user = authenticate_user("admin")
    assert user is not None
    assert user.username == "admin"
    assert user.data["role"] == "admin"

def test_authenticate_invalid_user():
    """Test authentication with invalid user"""
    user = authenticate_user("nonexistent")
    assert user is None

def test_user_permissions():
    """Test user permission system"""
    admin = User("admin", "admin@test.com", {"role": "admin"})
    regular_user = User("user", "user@test.com", {"role": "user"})
    
    admin_perms = get_user_permissions(admin)
    user_perms = get_user_permissions(regular_user)
    
    assert "manage" in admin_perms
    assert "manage" not in user_perms
    assert "read" in user_perms

def test_permission_check():
    """Test permission checking"""
    admin = User("admin", "admin@test.com", {"role": "admin"})
    
    assert check_permission(admin, "read") == True
    assert check_permission(admin, "manage") == True
'''
                }
            }
            
            # Step 3: Create and review patch set
            patch_id = await self.orchestrator.create_patch_set(
                feature_changes,
                "Add user permissions system and audit logging"
            )
            
            print(f"✓ Created feature patch set: {patch_id}")
            
            # Step 4: Automated conflict check (no conflicts expected for new feature)
            patch_set = await self.orchestrator.patch_manager.get_patch_set(patch_id)
            conflicts = await self.orchestrator.conflict_resolver.detect_conflicts([patch_set])
            print(f"✓ Conflict check completed: {len(conflicts)} conflicts found")
            
            # Step 5: Automated review and approval
            session_id = await self.orchestrator.review_interface.start_review(
                patch_set,
                reviewer="Enterprise Reviewer",
                auto_mode=True
            )
            
            await self.orchestrator.patch_manager.update_patch_status(patch_id, 'approved')
            result = await self.orchestrator.apply_changes(patch_id)
            print(f"✓ Applied changes: {result['statistics']['files_created'] + result['statistics']['files_modified']} files affected")
            
            # Step 6: Version and history tracking
            version_id = await self.orchestrator.history_tracker.create_version(
                patch_set_id=patch_id,
                version_number="1.1.0-enterprise",
                author="Enterprise System",
                description="User permissions and audit logging feature",
                tags=["feature", "security", "enterprise"]
            )
            print(f"✓ Created version: {version_id}")
            
            # Step 7: Git commit (if available)
            if self.orchestrator.git_manager.is_git_repo:
                commit_hash = await self.orchestrator.git_manager.create_commit_for_patch_set(
                    patch_id,
                    "feat: Add user permissions system and audit logging\\n\\nImplements role-based permissions and audit trail for enterprise security requirements.",
                    "Enterprise System"
                )
                if commit_hash:
                    print(f"✓ Git commit created: {commit_hash[:8]}")
            
            # Step 8: Generate metrics and reports
            await self.generate_enterprise_reports()
            
            print("✓ Enterprise workflow completed successfully!")
            
        except Exception as e:
            logger.error(f"Enterprise workflow failed: {e}")
            print(f"✗ Enterprise workflow failed: {e}")
    
    async def generate_enterprise_reports(self):
        """Generate enterprise reports and metrics"""
        
        print("\\n--- Enterprise Reports ---")
        
        # Patch set statistics
        all_patches = await self.orchestrator.patch_manager.list_patch_sets()
        print(f"Total patch sets created: {len(all_patches)}")
        
        applied_patches = [p for p in all_patches if p.status == 'applied']
        print(f"Successfully applied patches: {len(applied_patches)}")
        
        # Transaction statistics
        tx_stats = await self.orchestrator.transaction_manager.get_transaction_stats()
        print(f"Transaction success rate: {tx_stats['successful_commits']}/{tx_stats['total_transactions']}")
        
        # File change statistics
        change_stats = await self.orchestrator.history_tracker.get_change_statistics(
            by_file_type=True
        )
        print(f"Total file changes: {change_stats['total_changes']}")
        print(f"File types modified: {list(change_stats.get('changes_by_file_type', {}).keys())}")
        
        # Review statistics
        print(f"Review sessions conducted: {len(self.orchestrator.review_interface.active_sessions)}")
        
        print("✓ Enterprise reports generated")
    
    async def cleanup(self):
        """Cleanup demo environment"""
        
        print(f"\\nDemo completed. Files created in: {self.demo_root}")
        print("You can explore the generated files and .abov3 directory to see the results.")


async def main():
    """Run the comprehensive multi-edit demo"""
    
    demo = MultiEditDemo()
    
    try:
        await demo.run_complete_demo()
        await demo.cleanup()
    except KeyboardInterrupt:
        print("\\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())