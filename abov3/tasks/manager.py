"""
ABOV3 Genesis Task Manager
Manages tasks and progress tracking
"""

import asyncio
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class Task:
    id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: str = None
    updated_at: str = None
    completed_at: str = None
    assigned_agent: str = None
    genesis_phase: str = None
    dependencies: List[str] = None
    output_files: List[str] = None
    progress: int = 0  # 0-100
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.dependencies is None:
            self.dependencies = []
        if self.output_files is None:
            self.output_files = []

class TaskManager:
    """
    Task Manager for ABOV3 Genesis
    Handles task creation, tracking, and management
    """
    
    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.abov3_dir = self.project_path / '.abov3'
        self.tasks_dir = self.abov3_dir / 'tasks'
        
        # Task files
        self.active_tasks_file = self.tasks_dir / 'active.yaml'
        self.completed_tasks_file = self.tasks_dir / 'completed.yaml'
        self.task_queue_file = self.tasks_dir / 'queue.yaml'
        
        # Ensure directories exist
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        
        # Task storage
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []
        
        # Load existing tasks
        self.load_tasks()
    
    def load_tasks(self):
        """Load existing tasks"""
        # Load active tasks
        if self.active_tasks_file.exists():
            try:
                with open(self.active_tasks_file, 'r') as f:
                    data = yaml.safe_load(f) or {}
                
                for task_data in data.get('tasks', []):
                    task = Task(**task_data)
                    task.status = TaskStatus(task.status) if isinstance(task.status, str) else task.status
                    task.priority = TaskPriority(task.priority) if isinstance(task.priority, str) else task.priority
                    self.active_tasks[task.id] = task
                    
            except Exception as e:
                print(f"Error loading active tasks: {e}")
        
        # Load completed tasks
        if self.completed_tasks_file.exists():
            try:
                with open(self.completed_tasks_file, 'r') as f:
                    data = yaml.safe_load(f) or {}
                
                for task_data in data.get('tasks', []):
                    task = Task(**task_data)
                    task.status = TaskStatus(task.status) if isinstance(task.status, str) else task.status
                    task.priority = TaskPriority(task.priority) if isinstance(task.priority, str) else task.priority
                    self.completed_tasks[task.id] = task
                    
            except Exception as e:
                print(f"Error loading completed tasks: {e}")
        
        # Load task queue
        if self.task_queue_file.exists():
            try:
                with open(self.task_queue_file, 'r') as f:
                    data = yaml.safe_load(f) or {}
                self.task_queue = data.get('queue', [])
            except Exception as e:
                print(f"Error loading task queue: {e}")
    
    def save_tasks(self):
        """Save all tasks"""
        try:
            # Save active tasks
            active_data = {
                'tasks': [self._task_to_dict(task) for task in self.active_tasks.values()],
                'updated': datetime.now().isoformat()
            }
            with open(self.active_tasks_file, 'w') as f:
                yaml.dump(active_data, f, default_flow_style=False)
            
            # Save completed tasks
            completed_data = {
                'tasks': [self._task_to_dict(task) for task in self.completed_tasks.values()],
                'updated': datetime.now().isoformat()
            }
            with open(self.completed_tasks_file, 'w') as f:
                yaml.dump(completed_data, f, default_flow_style=False)
            
            # Save task queue
            queue_data = {
                'queue': self.task_queue,
                'updated': datetime.now().isoformat()
            }
            with open(self.task_queue_file, 'w') as f:
                yaml.dump(queue_data, f, default_flow_style=False)
                
        except Exception as e:
            print(f"Error saving tasks: {e}")
    
    def _task_to_dict(self, task: Task) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        task_dict = asdict(task)
        task_dict['status'] = task.status.value
        task_dict['priority'] = task.priority.value
        return task_dict
    
    def create_task(
        self,
        title: str,
        description: str = "",
        priority: TaskPriority = TaskPriority.MEDIUM,
        assigned_agent: str = None,
        genesis_phase: str = None,
        dependencies: List[str] = None
    ) -> str:
        """Create a new task"""
        # Generate unique ID
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        task = Task(
            id=task_id,
            title=title,
            description=description,
            priority=priority,
            assigned_agent=assigned_agent,
            genesis_phase=genesis_phase,
            dependencies=dependencies or []
        )
        
        self.active_tasks[task_id] = task
        self.save_tasks()
        
        return task_id
    
    def update_task(
        self,
        task_id: str,
        title: str = None,
        description: str = None,
        status: TaskStatus = None,
        priority: TaskPriority = None,
        assigned_agent: str = None,
        progress: int = None
    ) -> bool:
        """Update an existing task"""
        task = self.get_task(task_id)
        if not task:
            return False
        
        # Update fields
        if title is not None:
            task.title = title
        if description is not None:
            task.description = description
        if status is not None:
            task.status = status
        if priority is not None:
            task.priority = priority
        if assigned_agent is not None:
            task.assigned_agent = assigned_agent
        if progress is not None:
            task.progress = max(0, min(100, progress))
        
        task.updated_at = datetime.now().isoformat()
        
        # Handle status changes
        if status == TaskStatus.COMPLETED:
            task.completed_at = datetime.now().isoformat()
            task.progress = 100
            # Move to completed tasks
            self.completed_tasks[task_id] = task
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
        
        self.save_tasks()
        return True
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        return None
    
    def get_active_tasks(self) -> List[Task]:
        """Get all active tasks"""
        return list(self.active_tasks.values())
    
    def get_completed_tasks(self) -> List[Task]:
        """Get all completed tasks"""
        return list(self.completed_tasks.values())
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get tasks by status"""
        return [task for task in self.active_tasks.values() if task.status == status]
    
    def get_tasks_by_priority(self, priority: TaskPriority) -> List[Task]:
        """Get tasks by priority"""
        return [task for task in self.active_tasks.values() if task.priority == priority]
    
    def get_tasks_by_agent(self, agent_name: str) -> List[Task]:
        """Get tasks assigned to an agent"""
        return [task for task in self.active_tasks.values() if task.assigned_agent == agent_name]
    
    def get_tasks_by_phase(self, phase: str) -> List[Task]:
        """Get tasks for a Genesis phase"""
        return [task for task in self.active_tasks.values() if task.genesis_phase == phase]
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task"""
        deleted = False
        
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
            deleted = True
        
        if task_id in self.completed_tasks:
            del self.completed_tasks[task_id]
            deleted = True
        
        if task_id in self.task_queue:
            self.task_queue.remove(task_id)
        
        if deleted:
            self.save_tasks()
        
        return deleted
    
    def complete_task(self, task_id: str, output_files: List[str] = None) -> bool:
        """Mark a task as completed"""
        task = self.get_task(task_id)
        if not task or task.status == TaskStatus.COMPLETED:
            return False
        
        if output_files:
            task.output_files.extend(output_files)
        
        return self.update_task(task_id, status=TaskStatus.COMPLETED)
    
    def add_to_queue(self, task_id: str) -> bool:
        """Add task to processing queue"""
        if task_id not in self.active_tasks:
            return False
        
        if task_id not in self.task_queue:
            self.task_queue.append(task_id)
            self.save_tasks()
        
        return True
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next task from queue"""
        # First check queue
        while self.task_queue:
            task_id = self.task_queue[0]
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                if task.status == TaskStatus.PENDING and self._can_start_task(task):
                    return task
            # Remove invalid task from queue
            self.task_queue.pop(0)
        
        # Find highest priority task that can be started
        available_tasks = [
            task for task in self.active_tasks.values()
            if task.status == TaskStatus.PENDING and self._can_start_task(task)
        ]
        
        if available_tasks:
            # Sort by priority (urgent > high > medium > low)
            priority_order = {
                TaskPriority.URGENT: 4,
                TaskPriority.HIGH: 3,
                TaskPriority.MEDIUM: 2,
                TaskPriority.LOW: 1
            }
            available_tasks.sort(key=lambda t: priority_order[t.priority], reverse=True)
            return available_tasks[0]
        
        return None
    
    def _can_start_task(self, task: Task) -> bool:
        """Check if a task can be started (dependencies met)"""
        for dep_id in task.dependencies:
            dep_task = self.get_task(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    def get_summary(self) -> str:
        """Get a summary of task status"""
        active_count = len(self.active_tasks)
        completed_count = len(self.completed_tasks)
        
        if active_count == 0 and completed_count == 0:
            return "No tasks"
        
        in_progress = len(self.get_tasks_by_status(TaskStatus.IN_PROGRESS))
        pending = len(self.get_tasks_by_status(TaskStatus.PENDING))
        
        parts = []
        if completed_count > 0:
            parts.append(f"{completed_count} done")
        if in_progress > 0:
            parts.append(f"{in_progress} active")
        if pending > 0:
            parts.append(f"{pending} pending")
        
        return ", ".join(parts)
    
    def get_progress_stats(self) -> Dict[str, Any]:
        """Get detailed progress statistics"""
        active_tasks = self.get_active_tasks()
        completed_tasks = self.get_completed_tasks()
        
        # Calculate overall progress
        total_tasks = len(active_tasks) + len(completed_tasks)
        if total_tasks == 0:
            overall_progress = 0
        else:
            completed_progress = len(completed_tasks) * 100
            active_progress = sum(task.progress for task in active_tasks)
            overall_progress = (completed_progress + active_progress) / total_tasks
        
        # Status counts
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = len(self.get_tasks_by_status(status))
        
        # Priority counts
        priority_counts = {}
        for priority in TaskPriority:
            priority_counts[priority.value] = len(self.get_tasks_by_priority(priority))
        
        return {
            'total_tasks': total_tasks,
            'active_tasks': len(active_tasks),
            'completed_tasks': len(completed_tasks),
            'overall_progress': overall_progress,
            'status_counts': status_counts,
            'priority_counts': priority_counts,
            'queued_tasks': len(self.task_queue)
        }
    
    def create_genesis_tasks(self, idea: str, phase: str = "design") -> List[str]:
        """Create tasks for Genesis workflow"""
        tasks = []
        
        genesis_task_templates = {
            "design": [
                ("Analyze requirements", f"Analyze and break down the idea: {idea}"),
                ("Design architecture", "Create system architecture and component design"),
                ("Choose technology stack", "Select appropriate technologies and frameworks"),
                ("Create implementation plan", "Plan development phases and milestones")
            ],
            "build": [
                ("Set up project structure", "Create project directories and configuration files"),
                ("Implement core functionality", "Build the main application features"),
                ("Add user interface", "Create and implement the user interface"),
                ("Integrate components", "Connect all system components together")
            ],
            "test": [
                ("Create unit tests", "Write comprehensive unit tests"),
                ("Implement integration tests", "Create integration and system tests"),
                ("Perform manual testing", "Conduct manual testing and user acceptance"),
                ("Fix bugs and issues", "Address any discovered issues")
            ],
            "deploy": [
                ("Prepare for deployment", "Set up deployment configuration"),
                ("Deploy to staging", "Deploy application to staging environment"),
                ("Deploy to production", "Deploy application to production"),
                ("Set up monitoring", "Configure monitoring and logging")
            ]
        }
        
        if phase in genesis_task_templates:
            for title, description in genesis_task_templates[phase]:
                task_id = self.create_task(
                    title=title,
                    description=description,
                    priority=TaskPriority.HIGH,
                    genesis_phase=phase
                )
                tasks.append(task_id)
        
        return tasks
    
    def export_tasks(self) -> Dict[str, Any]:
        """Export all tasks for backup"""
        return {
            'active_tasks': [self._task_to_dict(task) for task in self.active_tasks.values()],
            'completed_tasks': [self._task_to_dict(task) for task in self.completed_tasks.values()],
            'task_queue': self.task_queue,
            'exported': datetime.now().isoformat()
        }
    
    def import_tasks(self, data: Dict[str, Any]) -> bool:
        """Import tasks from backup"""
        try:
            if 'active_tasks' in data:
                for task_data in data['active_tasks']:
                    task = Task(**task_data)
                    task.status = TaskStatus(task.status) if isinstance(task.status, str) else task.status
                    task.priority = TaskPriority(task.priority) if isinstance(task.priority, str) else task.priority
                    self.active_tasks[task.id] = task
            
            if 'completed_tasks' in data:
                for task_data in data['completed_tasks']:
                    task = Task(**task_data)
                    task.status = TaskStatus(task.status) if isinstance(task.status, str) else task.status
                    task.priority = TaskPriority(task.priority) if isinstance(task.priority, str) else task.priority
                    self.completed_tasks[task.id] = task
            
            if 'task_queue' in data:
                self.task_queue = data['task_queue']
            
            self.save_tasks()
            return True
        except Exception as e:
            print(f"Error importing tasks: {e}")
            return False
    
    def clear_completed_tasks(self) -> int:
        """Clear all completed tasks"""
        count = len(self.completed_tasks)
        self.completed_tasks.clear()
        self.save_tasks()
        return count
    
    def get_task_dependencies(self, task_id: str) -> List[Task]:
        """Get tasks that this task depends on"""
        task = self.get_task(task_id)
        if not task:
            return []
        
        dependencies = []
        for dep_id in task.dependencies:
            dep_task = self.get_task(dep_id)
            if dep_task:
                dependencies.append(dep_task)
        
        return dependencies
    
    def add_task_dependency(self, task_id: str, dependency_id: str) -> bool:
        """Add a dependency to a task"""
        task = self.get_task(task_id)
        if not task or dependency_id not in self.active_tasks:
            return False
        
        if dependency_id not in task.dependencies:
            task.dependencies.append(dependency_id)
            task.updated_at = datetime.now().isoformat()
            self.save_tasks()
        
        return True