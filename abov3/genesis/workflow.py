"""
ABOV3 Genesis Workflow - Manages the Idea → Reality transformation flow
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class PhaseStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowPhase:
    name: str
    description: str
    icon: str
    status: PhaseStatus = PhaseStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.output is None:
            self.output = {}

class GenesisWorkflow:
    """
    Genesis Workflow Manager - orchestrates the transformation from idea to reality
    Manages the five phases: Idea → Design → Build → Test → Deploy
    """
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.abov3_dir = project_path / '.abov3'
        
        # Define the Genesis phases
        self.phases = [
            WorkflowPhase(
                name="idea",
                description="Capture and analyze the core idea",
                icon="💡"
            ),
            WorkflowPhase(
                name="design", 
                description="Create system architecture and design",
                icon="📐"
            ),
            WorkflowPhase(
                name="build",
                description="Generate code and build the application",
                icon="🔨"
            ),
            WorkflowPhase(
                name="test",
                description="Create tests and ensure quality",
                icon="🧪"
            ),
            WorkflowPhase(
                name="deploy",
                description="Deploy and make the application live",
                icon="🚀"
            )
        ]
        
        # Phase callbacks for custom processing
        self.phase_callbacks: Dict[str, List[Callable]] = {}
        
        # Current phase index
        self.current_phase_index = 0
    
    def get_phase(self, name: str) -> Optional[WorkflowPhase]:
        """Get a phase by name"""
        for phase in self.phases:
            if phase.name == name:
                return phase
        return None
    
    def get_current_phase(self) -> WorkflowPhase:
        """Get the current active phase"""
        return self.phases[self.current_phase_index]
    
    def get_all_phases(self) -> List[WorkflowPhase]:
        """Get all workflow phases"""
        return self.phases.copy()
    
    def get_phase_progress(self) -> Dict[str, Any]:
        """Get progress information for all phases"""
        completed_count = sum(1 for phase in self.phases if phase.status == PhaseStatus.COMPLETE)
        total_count = len(self.phases)
        
        return {
            'completed_phases': completed_count,
            'total_phases': total_count,
            'progress_percentage': (completed_count / max(1, total_count)) * 100,
            'current_phase': self.get_current_phase().name,
            'phases': [
                {
                    'name': phase.name,
                    'description': phase.description,
                    'icon': phase.icon,
                    'status': phase.status.value,
                    'started_at': phase.started_at.isoformat() if phase.started_at else None,
                    'completed_at': phase.completed_at.isoformat() if phase.completed_at else None
                }
                for phase in self.phases
            ]
        }
    
    async def start_phase(self, phase_name: str) -> bool:
        """Start a specific phase"""
        phase = self.get_phase(phase_name)
        if not phase:
            return False
        
        phase.status = PhaseStatus.IN_PROGRESS
        phase.started_at = datetime.now()
        
        # Update current phase index
        for i, p in enumerate(self.phases):
            if p.name == phase_name:
                self.current_phase_index = i
                break
        
        # Execute phase callbacks
        await self._execute_phase_callbacks(phase_name, 'start')
        
        return True
    
    async def complete_phase(self, phase_name: str, output: Dict[str, Any] = None) -> bool:
        """Complete a specific phase"""
        phase = self.get_phase(phase_name)
        if not phase:
            return False
        
        phase.status = PhaseStatus.COMPLETE
        phase.completed_at = datetime.now()
        if output:
            phase.output.update(output)
        
        # Move to next phase if not at the end
        if self.current_phase_index < len(self.phases) - 1:
            self.current_phase_index += 1
        
        # Execute phase callbacks
        await self._execute_phase_callbacks(phase_name, 'complete')
        
        return True
    
    async def fail_phase(self, phase_name: str, error: str = None) -> bool:
        """Mark a phase as failed"""
        phase = self.get_phase(phase_name)
        if not phase:
            return False
        
        phase.status = PhaseStatus.FAILED
        if error:
            phase.output['error'] = error
        
        # Execute phase callbacks
        await self._execute_phase_callbacks(phase_name, 'fail')
        
        return True
    
    async def skip_phase(self, phase_name: str, reason: str = None) -> bool:
        """Skip a specific phase"""
        phase = self.get_phase(phase_name)
        if not phase:
            return False
        
        phase.status = PhaseStatus.SKIPPED
        if reason:
            phase.output['skip_reason'] = reason
        
        # Move to next phase
        if self.current_phase_index < len(self.phases) - 1:
            self.current_phase_index += 1
        
        # Execute phase callbacks  
        await self._execute_phase_callbacks(phase_name, 'skip')
        
        return True
    
    def register_phase_callback(self, phase_name: str, callback: Callable) -> None:
        """Register a callback for phase events"""
        if phase_name not in self.phase_callbacks:
            self.phase_callbacks[phase_name] = []
        self.phase_callbacks[phase_name].append(callback)
    
    async def _execute_phase_callbacks(self, phase_name: str, event: str) -> None:
        """Execute registered callbacks for a phase event"""
        if phase_name in self.phase_callbacks:
            for callback in self.phase_callbacks[phase_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(phase_name, event)
                    else:
                        callback(phase_name, event)
                except Exception as e:
                    print(f"Error in phase callback: {e}")
    
    def is_workflow_complete(self) -> bool:
        """Check if the entire workflow is complete"""
        return all(phase.status == PhaseStatus.COMPLETE for phase in self.phases)
    
    def get_next_phase(self) -> Optional[WorkflowPhase]:
        """Get the next phase to execute"""
        current_index = self.current_phase_index
        
        # Look for next pending phase
        for i in range(current_index, len(self.phases)):
            if self.phases[i].status == PhaseStatus.PENDING:
                return self.phases[i]
        
        return None
    
    def reset_workflow(self) -> None:
        """Reset the workflow to start from the beginning"""
        for phase in self.phases:
            phase.status = PhaseStatus.PENDING
            phase.started_at = None
            phase.completed_at = None
            phase.output = {}
        
        self.current_phase_index = 0
    
    def reset_from_phase(self, phase_name: str) -> bool:
        """Reset workflow from a specific phase onwards"""
        phase_index = None
        for i, phase in enumerate(self.phases):
            if phase.name == phase_name:
                phase_index = i
                break
        
        if phase_index is None:
            return False
        
        # Reset from this phase onwards
        for i in range(phase_index, len(self.phases)):
            phase = self.phases[i]
            phase.status = PhaseStatus.PENDING
            phase.started_at = None
            phase.completed_at = None
            phase.output = {}
        
        self.current_phase_index = phase_index
        return True
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a comprehensive workflow summary"""
        progress = self.get_phase_progress()
        
        # Calculate timing
        start_time = None
        end_time = None
        
        for phase in self.phases:
            if phase.started_at and (not start_time or phase.started_at < start_time):
                start_time = phase.started_at
            if phase.completed_at and (not end_time or phase.completed_at > end_time):
                end_time = phase.completed_at
        
        duration = None
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()
        elif start_time:
            duration = (datetime.now() - start_time).total_seconds()
        
        return {
            'workflow_status': 'complete' if self.is_workflow_complete() else 'in_progress',
            'current_phase': self.get_current_phase().name,
            'progress': progress,
            'start_time': start_time.isoformat() if start_time else None,
            'end_time': end_time.isoformat() if end_time else None,
            'duration_seconds': duration,
            'failed_phases': [phase.name for phase in self.phases if phase.status == PhaseStatus.FAILED],
            'skipped_phases': [phase.name for phase in self.phases if phase.status == PhaseStatus.SKIPPED]
        }
    
    def export_workflow_data(self) -> Dict[str, Any]:
        """Export workflow data for persistence"""
        return {
            'current_phase_index': self.current_phase_index,
            'phases': [
                {
                    'name': phase.name,
                    'description': phase.description,
                    'icon': phase.icon,
                    'status': phase.status.value,
                    'started_at': phase.started_at.isoformat() if phase.started_at else None,
                    'completed_at': phase.completed_at.isoformat() if phase.completed_at else None,
                    'output': phase.output
                }
                for phase in self.phases
            ]
        }
    
    def import_workflow_data(self, data: Dict[str, Any]) -> None:
        """Import workflow data from persistence"""
        if 'current_phase_index' in data:
            self.current_phase_index = data['current_phase_index']
        
        if 'phases' in data:
            for i, phase_data in enumerate(data['phases']):
                if i < len(self.phases):
                    phase = self.phases[i]
                    phase.status = PhaseStatus(phase_data.get('status', 'pending'))
                    
                    if phase_data.get('started_at'):
                        phase.started_at = datetime.fromisoformat(phase_data['started_at'])
                    
                    if phase_data.get('completed_at'):
                        phase.completed_at = datetime.fromisoformat(phase_data['completed_at'])
                    
                    if phase_data.get('output'):
                        phase.output = phase_data['output']
    
    async def auto_advance_workflow(self) -> Optional[str]:
        """Automatically advance to the next ready phase"""
        next_phase = self.get_next_phase()
        if next_phase:
            await self.start_phase(next_phase.name)
            return next_phase.name
        return None
    
    def get_phase_dependencies(self, phase_name: str) -> List[str]:
        """Get the phases that must be completed before this phase"""
        phase_order = [phase.name for phase in self.phases]
        if phase_name not in phase_order:
            return []
        
        phase_index = phase_order.index(phase_name)
        return phase_order[:phase_index]
    
    def can_start_phase(self, phase_name: str) -> bool:
        """Check if a phase can be started (dependencies met)"""
        dependencies = self.get_phase_dependencies(phase_name)
        
        for dep_phase_name in dependencies:
            dep_phase = self.get_phase(dep_phase_name)
            if not dep_phase or dep_phase.status != PhaseStatus.COMPLETE:
                return False
        
        return True