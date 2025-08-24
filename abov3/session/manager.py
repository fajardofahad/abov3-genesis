"""
ABOV3 Genesis Session Manager
Handles session persistence, recovery, and history management
"""

import asyncio
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import gzip
import shutil

class SessionManager:
    """
    Session Manager for ABOV3 Genesis
    Handles conversation persistence and recovery
    """
    
    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.abov3_dir = self.project_path / '.abov3'
        self.sessions_dir = self.abov3_dir / 'sessions'
        self.history_dir = self.abov3_dir / 'history'
        
        # Session files
        self.current_session_file = self.sessions_dir / 'current.session'
        self.backup_session_file = self.sessions_dir / 'backup.session'
        self.session_index_file = self.sessions_dir / 'index.json'
        
        # Ensure directories exist
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        # Session data
        self.current_session = {
            'session_id': None,
            'created_at': None,
            'last_updated': None,
            'messages': [],
            'context': {},
            'current_agent': None,
            'genesis_state': {},
            'tasks': [],
            'metadata': {}
        }
        
        # Session configuration
        self.max_messages = 1000  # Maximum messages to keep in session
        self.auto_save_interval = 30  # Auto-save every 30 seconds
        self.max_history_files = 100  # Maximum history files to keep
    
    def create_new_session(self) -> str:
        """Create a new session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'messages': [],
            'context': {},
            'current_agent': None,
            'genesis_state': {},
            'tasks': [],
            'metadata': {
                'project_path': str(self.project_path),
                'abov3_version': '1.0.0'
            }
        }
        
        return session_id
    
    async def save_session(self, compress: bool = True):
        """Save the current session"""
        if not self.current_session.get('session_id'):
            return False
        
        try:
            # Update timestamp
            self.current_session['last_updated'] = datetime.now().isoformat()
            
            # Create backup of current session
            if self.current_session_file.exists():
                shutil.copy2(self.current_session_file, self.backup_session_file)
            
            # Save session data
            session_data = json.dumps(self.current_session, indent=2)
            
            if compress:
                # Save compressed
                with gzip.open(self.current_session_file, 'wt', encoding='utf-8') as f:
                    f.write(session_data)
            else:
                # Save uncompressed
                with open(self.current_session_file, 'w', encoding='utf-8') as f:
                    f.write(session_data)
            
            # Update session index
            await self._update_session_index()
            
            return True
            
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    async def load_session(self, session_file: Path = None) -> Optional[Dict[str, Any]]:
        """Load a session"""
        if session_file is None:
            session_file = self.current_session_file
        
        if not session_file.exists():
            return None
        
        try:
            # Try to load as compressed first
            try:
                with gzip.open(session_file, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            except (gzip.BadGzipFile, json.JSONDecodeError):
                # Try to load as uncompressed
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            self.current_session = data
            return data
            
        except Exception as e:
            print(f"Error loading session: {e}")
            return None
    
    def has_previous_session(self) -> bool:
        """Check if there's a previous session to restore"""
        return self.current_session_file.exists()
    
    async def restore_session(self) -> Optional[Dict[str, Any]]:
        """Restore the most recent session"""
        session_data = await self.load_session()
        if session_data:
            return session_data
        
        # Try backup if current session fails
        if self.backup_session_file.exists():
            return await self.load_session(self.backup_session_file)
        
        return None
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the current session"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.current_session['messages'].append(message)
        
        # Trim messages if too many
        if len(self.current_session['messages']) > self.max_messages:
            # Keep first 10 messages (context) and last max_messages - 10
            keep_count = self.max_messages - 10
            self.current_session['messages'] = (
                self.current_session['messages'][:10] + 
                self.current_session['messages'][-keep_count:]
            )
    
    def get_messages(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get conversation messages"""
        messages = self.current_session.get('messages', [])
        if limit:
            return messages[-limit:]
        return messages
    
    def clear_messages(self):
        """Clear conversation messages"""
        self.current_session['messages'] = []
    
    def update_context(self, key: str, value: Any):
        """Update session context"""
        if 'context' not in self.current_session:
            self.current_session['context'] = {}
        self.current_session['context'][key] = value
    
    def get_context(self, key: str = None) -> Any:
        """Get session context"""
        if key:
            return self.current_session.get('context', {}).get(key)
        return self.current_session.get('context', {})
    
    def set_current_agent(self, agent_name: str):
        """Set the current agent"""
        self.current_session['current_agent'] = agent_name
    
    def get_current_agent(self) -> Optional[str]:
        """Get the current agent"""
        return self.current_session.get('current_agent')
    
    def update_genesis_state(self, state: Dict[str, Any]):
        """Update Genesis workflow state"""
        if 'genesis_state' not in self.current_session:
            self.current_session['genesis_state'] = {}
        self.current_session['genesis_state'].update(state)
    
    def get_genesis_state(self) -> Dict[str, Any]:
        """Get Genesis workflow state"""
        return self.current_session.get('genesis_state', {})
    
    def add_task(self, task_data: Dict[str, Any]):
        """Add a task to the session"""
        if 'tasks' not in self.current_session:
            self.current_session['tasks'] = []
        self.current_session['tasks'].append(task_data)
    
    def get_tasks(self) -> List[Dict[str, Any]]:
        """Get session tasks"""
        return self.current_session.get('tasks', [])
    
    async def archive_session(self) -> bool:
        """Archive the current session to history"""
        if not self.current_session.get('session_id'):
            return False
        
        try:
            # Create history filename
            session_id = self.current_session['session_id']
            history_file = self.history_dir / f"{session_id}.json.gz"
            
            # Save compressed session to history
            session_data = json.dumps(self.current_session, indent=2)
            with gzip.open(history_file, 'wt', encoding='utf-8') as f:
                f.write(session_data)
            
            # Update history index
            await self._update_history_index()
            
            # Clean up old history files
            await self._cleanup_old_history()
            
            return True
            
        except Exception as e:
            print(f"Error archiving session: {e}")
            return False
    
    async def list_history_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List historical sessions"""
        history_files = sorted(
            self.history_dir.glob("session_*.json.gz"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        sessions = []
        for history_file in history_files[:limit]:
            try:
                with gzip.open(history_file, 'rt', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                sessions.append({
                    'session_id': session_data.get('session_id'),
                    'created_at': session_data.get('created_at'),
                    'last_updated': session_data.get('last_updated'),
                    'message_count': len(session_data.get('messages', [])),
                    'agent': session_data.get('current_agent'),
                    'file_path': str(history_file)
                })
            except Exception as e:
                print(f"Error reading history file {history_file}: {e}")
                continue
        
        return sessions
    
    async def load_history_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific historical session"""
        history_file = self.history_dir / f"{session_id}.json.gz"
        
        if not history_file.exists():
            return None
        
        try:
            with gzip.open(history_file, 'rt', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading history session {session_id}: {e}")
            return None
    
    async def export_session(self, session_id: str = None, include_history: bool = False) -> Dict[str, Any]:
        """Export session data"""
        if session_id:
            session_data = await self.load_history_session(session_id)
        else:
            session_data = self.current_session
        
        export_data = {
            'session': session_data,
            'exported_at': datetime.now().isoformat(),
            'abov3_version': '1.0.0'
        }
        
        if include_history:
            export_data['history'] = await self.list_history_sessions(100)
        
        return export_data
    
    async def import_session(self, session_data: Dict[str, Any]) -> bool:
        """Import session data"""
        try:
            if 'session' in session_data:
                self.current_session = session_data['session']
                await self.save_session()
                return True
            return False
        except Exception as e:
            print(f"Error importing session: {e}")
            return False
    
    async def _update_session_index(self):
        """Update session index"""
        try:
            index_data = {
                'current_session': self.current_session.get('session_id'),
                'last_updated': datetime.now().isoformat(),
                'message_count': len(self.current_session.get('messages', [])),
                'created_at': self.current_session.get('created_at')
            }
            
            with open(self.session_index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
                
        except Exception as e:
            print(f"Error updating session index: {e}")
    
    async def _update_history_index(self):
        """Update history index"""
        try:
            history_index_file = self.history_dir / 'index.json'
            sessions = await self.list_history_sessions(1000)  # Get all sessions
            
            index_data = {
                'total_sessions': len(sessions),
                'updated_at': datetime.now().isoformat(),
                'sessions': sessions
            }
            
            with open(history_index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
                
        except Exception as e:
            print(f"Error updating history index: {e}")
    
    async def _cleanup_old_history(self):
        """Clean up old history files"""
        try:
            history_files = sorted(
                self.history_dir.glob("session_*.json.gz"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            
            # Remove old files beyond max_history_files
            for old_file in history_files[self.max_history_files:]:
                old_file.unlink()
                
        except Exception as e:
            print(f"Error cleaning up history: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        messages = self.current_session.get('messages', [])
        
        # Count messages by role
        role_counts = {}
        for message in messages:
            role = message.get('role', 'unknown')
            role_counts[role] = role_counts.get(role, 0) + 1
        
        # Calculate session duration
        created_at = self.current_session.get('created_at')
        duration = None
        if created_at:
            try:
                start_time = datetime.fromisoformat(created_at)
                duration = (datetime.now() - start_time).total_seconds()
            except:
                pass
        
        return {
            'session_id': self.current_session.get('session_id'),
            'created_at': created_at,
            'last_updated': self.current_session.get('last_updated'),
            'duration_seconds': duration,
            'total_messages': len(messages),
            'role_counts': role_counts,
            'current_agent': self.current_session.get('current_agent'),
            'has_genesis_state': bool(self.current_session.get('genesis_state')),
            'task_count': len(self.current_session.get('tasks', []))
        }
    
    async def delete_history_session(self, session_id: str) -> bool:
        """Delete a historical session"""
        history_file = self.history_dir / f"{session_id}.json.gz"
        
        if history_file.exists():
            try:
                history_file.unlink()
                await self._update_history_index()
                return True
            except Exception as e:
                print(f"Error deleting history session: {e}")
        
        return False
    
    async def clear_all_history(self) -> int:
        """Clear all history sessions"""
        history_files = list(self.history_dir.glob("session_*.json.gz"))
        count = 0
        
        for history_file in history_files:
            try:
                history_file.unlink()
                count += 1
            except Exception as e:
                print(f"Error deleting {history_file}: {e}")
        
        # Update index
        if count > 0:
            await self._update_history_index()
        
        return count
    
    def set_auto_save_interval(self, seconds: int):
        """Set auto-save interval"""
        self.auto_save_interval = max(10, seconds)  # Minimum 10 seconds
    
    def set_max_messages(self, count: int):
        """Set maximum messages to keep in session"""
        self.max_messages = max(100, count)  # Minimum 100 messages