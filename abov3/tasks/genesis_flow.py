"""
ABOV3 Genesis Flow Tracker
Tracks the Genesis workflow phases and progress
"""

import asyncio
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class PhaseStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETE = "complete"
    FAILED = "failed"

@dataclass
class GenesisPhase:
    name: str
    description: str
    icon: str
    status: PhaseStatus = PhaseStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_files: List[str] = None
    
    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []

class GenesisFlow:
    """
    Genesis Flow Tracker
    Manages the transformation workflow from idea to reality
    """
    
    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.abov3_dir = self.project_path / '.abov3'
        self.flow_dir = self.abov3_dir / 'genesis_flow'
        self.flow_file = self.abov3_dir / 'genesis.yaml'
        
        # Ensure directories exist
        self.flow_dir.mkdir(parents=True, exist_ok=True)
        
        # Genesis phases
        self.phases = [
            GenesisPhase(
                name="idea",
                description="Capture and analyze the core concept",
                icon="💡"
            ),
            GenesisPhase(
                name="design", 
                description="Create system architecture and design",
                icon="📐"
            ),
            GenesisPhase(
                name="build",
                description="Generate code and build the application",
                icon="🔨"
            ),
            GenesisPhase(
                name="test",
                description="Create tests and ensure quality",
                icon="🧪"
            ),
            GenesisPhase(
                name="deploy",
                description="Deploy and launch the application",
                icon="🚀"
            )
        ]
        
        # Load existing flow data
        self.flow_data = {}
        self.load_flow_data()
        
    def load_flow_data(self):
        """Load existing Genesis flow data"""
        if self.flow_file.exists():
            try:
                with open(self.flow_file, 'r') as f:
                    self.flow_data = yaml.safe_load(f) or {}
                
                # Update phase statuses from loaded data
                phases_data = self.flow_data.get('phases', {})
                for phase in self.phases:
                    if phase.name in phases_data:
                        phase_data = phases_data[phase.name]
                        phase.status = PhaseStatus(phase_data.get('status', 'pending'))
                        
                        if phase_data.get('started_at'):
                            phase.started_at = datetime.fromisoformat(phase_data['started_at'])
                        
                        if phase_data.get('completed_at'):
                            phase.completed_at = datetime.fromisoformat(phase_data['completed_at'])
                        
                        phase.output_files = phase_data.get('output_files', [])
                        
            except Exception as e:
                print(f"Error loading Genesis flow data: {e}")
                self.flow_data = {}
    
    def save_flow_data(self):
        """Save Genesis flow data"""
        # Update flow data with current phase info
        if 'phases' not in self.flow_data:
            self.flow_data['phases'] = {}
        
        for phase in self.phases:
            self.flow_data['phases'][phase.name] = {
                'status': phase.status.value,
                'started_at': phase.started_at.isoformat() if phase.started_at else None,
                'completed_at': phase.completed_at.isoformat() if phase.completed_at else None,
                'output_files': phase.output_files
            }
        
        self.flow_data['updated'] = datetime.now().isoformat()
        
        try:
            with open(self.flow_file, 'w') as f:
                yaml.dump(self.flow_data, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving Genesis flow data: {e}")
    
    def get_current_phase(self) -> str:
        """Get the current Genesis phase"""
        return self.flow_data.get('current_phase', 'idea')
    
    def get_phase(self, name: str) -> Optional[GenesisPhase]:
        """Get a phase by name"""
        for phase in self.phases:
            if phase.name == name:
                return phase
        return None
    
    def get_all_phases(self) -> List[GenesisPhase]:
        """Get all Genesis phases"""
        return self.phases.copy()
    
    async def start_phase(self, phase_name: str) -> bool:
        """Start a Genesis phase"""
        phase = self.get_phase(phase_name)
        if not phase:
            return False
        
        phase.status = PhaseStatus.IN_PROGRESS
        phase.started_at = datetime.now()
        
        # Update current phase
        self.flow_data['current_phase'] = phase_name
        
        self.save_flow_data()
        return True
    
    async def complete_phase(self, phase_name: str, output_files: List[str] = None) -> bool:
        """Complete a Genesis phase"""
        phase = self.get_phase(phase_name)
        if not phase:
            return False
        
        phase.status = PhaseStatus.COMPLETE
        phase.completed_at = datetime.now()
        
        if output_files:
            phase.output_files.extend(output_files)
        
        # Move to next phase
        phase_names = [p.name for p in self.phases]
        current_index = phase_names.index(phase_name)
        
        if current_index < len(phase_names) - 1:
            next_phase = phase_names[current_index + 1]
            self.flow_data['current_phase'] = next_phase
        else:
            # All phases complete
            self.flow_data['current_phase'] = 'complete'
            self.flow_data['completed_at'] = datetime.now().isoformat()
        
        self.save_flow_data()
        return True
    
    async def fail_phase(self, phase_name: str, error_message: str = None) -> bool:
        """Mark a phase as failed"""
        phase = self.get_phase(phase_name)
        if not phase:
            return False
        
        phase.status = PhaseStatus.FAILED
        
        # Store error information
        if 'errors' not in self.flow_data:
            self.flow_data['errors'] = {}
        
        self.flow_data['errors'][phase_name] = {
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        }
        
        self.save_flow_data()
        return True
    
    async def update_phase(self, phase_name: str, status: str) -> bool:
        """Update phase status"""
        phase = self.get_phase(phase_name)
        if not phase:
            return False
        
        try:
            new_status = PhaseStatus(status)
            phase.status = new_status
            
            if new_status == PhaseStatus.IN_PROGRESS:
                await self.start_phase(phase_name)
            elif new_status == PhaseStatus.COMPLETE:
                await self.complete_phase(phase_name)
            
            return True
        except ValueError:
            return False
    
    def get_progress(self) -> Dict[str, Any]:
        """Get Genesis progress information"""
        completed_phases = sum(1 for phase in self.phases if phase.status == PhaseStatus.COMPLETE)
        total_phases = len(self.phases)
        progress_percentage = (completed_phases / total_phases) * 100 if total_phases > 0 else 0
        
        current_phase = self.get_current_phase()
        
        return {
            'current_phase': current_phase,
            'completed_phases': completed_phases,
            'total_phases': total_phases,
            'progress_percentage': progress_percentage,
            'is_complete': current_phase == 'complete',
            'phases': [
                {
                    'name': phase.name,
                    'description': phase.description,
                    'icon': phase.icon,
                    'status': phase.status.value,
                    'started_at': phase.started_at.isoformat() if phase.started_at else None,
                    'completed_at': phase.completed_at.isoformat() if phase.completed_at else None,
                    'output_files': phase.output_files
                }
                for phase in self.phases
            ]
        }
    
    def get_next_phase(self) -> Optional[str]:
        """Get the next phase to work on"""
        current_phase = self.get_current_phase()
        
        if current_phase == 'complete':
            return None
        
        phase_names = [p.name for p in self.phases]
        
        try:
            current_index = phase_names.index(current_phase)
            if current_index < len(phase_names) - 1:
                return phase_names[current_index + 1]
        except ValueError:
            # Current phase not found, return first pending phase
            pass
        
        # Find first pending phase
        for phase in self.phases:
            if phase.status == PhaseStatus.PENDING:
                return phase.name
        
        return None
    
    def get_phase_files(self, phase_name: str) -> List[str]:
        """Get output files for a phase"""
        phase = self.get_phase(phase_name)
        if phase:
            return phase.output_files.copy()
        return []
    
    def add_phase_output(self, phase_name: str, file_path: str):
        """Add an output file to a phase"""
        phase = self.get_phase(phase_name)
        if phase and file_path not in phase.output_files:
            phase.output_files.append(file_path)
            self.save_flow_data()
    
    def get_idea(self) -> str:
        """Get the original idea"""
        return self.flow_data.get('idea', 'No idea recorded')
    
    def set_idea(self, idea: str):
        """Set the original idea"""
        self.flow_data['idea'] = idea
        self.save_flow_data()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Genesis statistics"""
        progress = self.get_progress()
        
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
        
        # Count outputs
        total_files = sum(len(phase.output_files) for phase in self.phases)
        
        stats = {
            'ideas_completed': 1 if progress['is_complete'] else 0,
            'lines_generated': 0,  # Would need to count actual lines
            'files_created': total_files,
            'phases_completed': progress['completed_phases'],
            'total_time_seconds': duration,
            'start_time': start_time.isoformat() if start_time else None,
            'end_time': end_time.isoformat() if end_time else None,
            'current_phase': progress['current_phase'],
            'progress_percentage': progress['progress_percentage']
        }
        
        return stats
    
    def reset_flow(self):
        """Reset the Genesis flow"""
        for phase in self.phases:
            phase.status = PhaseStatus.PENDING
            phase.started_at = None
            phase.completed_at = None
            phase.output_files = []
        
        self.flow_data = {
            'current_phase': 'idea',
            'reset_at': datetime.now().isoformat()
        }
        
        self.save_flow_data()
    
    def export_flow_data(self) -> Dict[str, Any]:
        """Export flow data for backup/transfer"""
        return {
            'flow_data': self.flow_data,
            'phases': [
                {
                    'name': phase.name,
                    'description': phase.description,
                    'icon': phase.icon,
                    'status': phase.status.value,
                    'started_at': phase.started_at.isoformat() if phase.started_at else None,
                    'completed_at': phase.completed_at.isoformat() if phase.completed_at else None,
                    'output_files': phase.output_files
                }
                for phase in self.phases
            ],
            'exported': datetime.now().isoformat()
        }
    
    def import_flow_data(self, data: Dict[str, Any]) -> bool:
        """Import flow data from backup"""
        try:
            if 'flow_data' in data:
                self.flow_data = data['flow_data']
            
            if 'phases' in data:
                for phase_data in data['phases']:
                    phase = self.get_phase(phase_data['name'])
                    if phase:
                        phase.status = PhaseStatus(phase_data.get('status', 'pending'))
                        
                        if phase_data.get('started_at'):
                            phase.started_at = datetime.fromisoformat(phase_data['started_at'])
                        
                        if phase_data.get('completed_at'):
                            phase.completed_at = datetime.fromisoformat(phase_data['completed_at'])
                        
                        phase.output_files = phase_data.get('output_files', [])
            
            self.save_flow_data()
            return True
        except Exception as e:
            print(f"Error importing flow data: {e}")
            return False
    
    def get_flow_summary(self) -> str:
        """Get a human-readable flow summary"""
        progress = self.get_progress()
        
        summary = f"**Genesis Flow Summary**\n\n"
        summary += f"💡 **Idea**: {self.get_idea()}\n"
        summary += f"📍 **Current Phase**: {progress['current_phase'].capitalize()}\n"
        summary += f"📊 **Progress**: {progress['progress_percentage']:.0f}% Complete\n\n"
        
        summary += f"**Phase Status**:\n"
        for phase_info in progress['phases']:
            icon = phase_info['icon']
            name = phase_info['name'].capitalize()
            status = phase_info['status']
            
            if status == 'complete':
                status_icon = '✅'
            elif status == 'in_progress':
                status_icon = '⏳'
            elif status == 'failed':
                status_icon = '❌'
            else:
                status_icon = '⏸️'
            
            summary += f"  {icon} {name}: {status_icon}\n"
        
        if progress['is_complete']:
            summary += f"\n🎉 **Genesis Complete!** From idea to reality!"
        
        return summary