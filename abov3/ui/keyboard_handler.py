"""
ABOV3 Genesis - Advanced Keyboard Controls
Real-time keyboard event handling for enterprise-grade user interaction
"""

import sys
import os
import asyncio
import threading
import time
from typing import Dict, Callable, Optional, Any, Set, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import signal

# Platform-specific imports
try:
    import termios
    import tty
    import select
    UNIX_KEYBOARD_AVAILABLE = True
except ImportError:
    # Windows or other platforms without termios
    termios = None
    tty = None
    select = None
    UNIX_KEYBOARD_AVAILABLE = False

# Windows-specific imports
if os.name == 'nt':
    try:
        import msvcrt
        WINDOWS_KEYBOARD_AVAILABLE = True
    except ImportError:
        msvcrt = None
        WINDOWS_KEYBOARD_AVAILABLE = False
else:
    msvcrt = None
    WINDOWS_KEYBOARD_AVAILABLE = False

class KeyEvent(Enum):
    """Keyboard event types"""
    ESC_INTERRUPT = "esc_interrupt"
    CTRL_T_TODO_TOGGLE = "ctrl_t_todo_toggle"
    CTRL_C_INTERRUPT = "ctrl_c_interrupt"
    CTRL_D_EXIT = "ctrl_d_exit"
    CTRL_L_CLEAR = "ctrl_l_clear"
    CTRL_R_RESTART = "ctrl_r_restart"
    CTRL_S_SAVE = "ctrl_s_save"
    CTRL_Z_UNDO = "ctrl_z_undo"
    F1_HELP = "f1_help"
    F5_REFRESH = "f5_refresh"

@dataclass
class KeyBinding:
    """Keyboard binding configuration"""
    event: KeyEvent
    key_sequence: bytes
    description: str
    handler: Optional[Callable] = None
    enabled: bool = True
    global_scope: bool = True
    
@dataclass
class KeyboardState:
    """Current keyboard handling state"""
    active: bool = False
    handlers_registered: Dict[KeyEvent, List[Callable]] = field(default_factory=dict)
    interrupt_requested: bool = False
    last_key_time: float = 0.0
    todo_visible: bool = True
    current_operation: Optional[str] = None
    operation_start_time: float = 0.0

class KeyboardHandler:
    """
    Advanced keyboard event handler with Claude Coder-level responsiveness.
    Handles real-time keyboard input without blocking main operations.
    """
    
    def __init__(self):
        self.state = KeyboardState()
        self.bindings: Dict[KeyEvent, KeyBinding] = {}
        self.original_termios = None
        self.input_thread = None
        self.running = False
        self.loop = None
        
        # Key sequence mappings (platform-specific)
        self.key_sequences = {
            b'\x1b': KeyEvent.ESC_INTERRUPT,          # ESC key
            b'\x14': KeyEvent.CTRL_T_TODO_TOGGLE,     # Ctrl+T
            b'\x03': KeyEvent.CTRL_C_INTERRUPT,       # Ctrl+C
            b'\x04': KeyEvent.CTRL_D_EXIT,            # Ctrl+D
            b'\x0c': KeyEvent.CTRL_L_CLEAR,           # Ctrl+L
            b'\x12': KeyEvent.CTRL_R_RESTART,         # Ctrl+R
            b'\x13': KeyEvent.CTRL_S_SAVE,            # Ctrl+S
            b'\x1a': KeyEvent.CTRL_Z_UNDO,            # Ctrl+Z
            b'\x1bOP': KeyEvent.F1_HELP,              # F1
            b'\x1b[15~': KeyEvent.F5_REFRESH,         # F5
        }
        
        # Default key bindings
        self._setup_default_bindings()
        
        # Platform detection
        self.is_windows = os.name == 'nt'
        self.is_unix = not self.is_windows
        
        # Windows-specific imports
        if self.is_windows:
            try:
                import msvcrt
                self.msvcrt = msvcrt
            except ImportError:
                self.msvcrt = None
                print("Warning: Windows keyboard handling not available")
        
        # Performance metrics
        self.metrics = {
            'keys_processed': 0,
            'interrupt_count': 0,
            'response_times': [],
            'last_performance_check': time.time()
        }
        
        # Interrupt callbacks
        self.interrupt_callbacks: Set[Callable] = set()
        
    def _setup_default_bindings(self):
        """Setup default keyboard bindings"""
        bindings = [
            KeyBinding(
                KeyEvent.ESC_INTERRUPT,
                b'\x1b',
                "Emergency interrupt - stops any running operation immediately",
                self._handle_esc_interrupt,
                True,
                True
            ),
            KeyBinding(
                KeyEvent.CTRL_T_TODO_TOGGLE,
                b'\x14',
                "Toggle todo list visibility",
                self._handle_ctrl_t_toggle,
                True,
                True
            ),
            KeyBinding(
                KeyEvent.CTRL_C_INTERRUPT,
                b'\x03',
                "Graceful interrupt",
                self._handle_ctrl_c_interrupt,
                True,
                True
            ),
            KeyBinding(
                KeyEvent.CTRL_L_CLEAR,
                b'\x0c',
                "Clear screen",
                self._handle_ctrl_l_clear,
                True,
                False
            ),
            KeyBinding(
                KeyEvent.F1_HELP,
                b'\x1bOP',
                "Show keyboard shortcuts help",
                self._handle_f1_help,
                True,
                False
            )
        ]
        
        for binding in bindings:
            self.bindings[binding.event] = binding
    
    async def initialize(self, loop: asyncio.AbstractEventLoop = None):
        """Initialize keyboard handler with async event loop"""
        if self.running:
            return
            
        self.loop = loop or asyncio.get_event_loop()
        self.running = True
        self.state.active = True
        
        # Setup terminal for raw mode (Unix only)
        if self.is_unix:
            self._setup_raw_mode()
        
        # Start input monitoring thread
        self.input_thread = threading.Thread(
            target=self._input_monitor_thread,
            daemon=True
        )
        self.input_thread.start()
        
        print("\n🎯 ABOV3 Keyboard Controls Active:")
        print("   ESC     → Emergency interrupt")
        print("   Ctrl+T  → Toggle todo list")
        print("   Ctrl+C  → Graceful stop")
        print("   F1      → Show help\n")
    
    def _setup_raw_mode(self):
        """Setup terminal raw mode for Unix systems"""
        if not self.is_unix:
            return
            
        try:
            self.original_termios = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        except Exception as e:
            print(f"Warning: Could not setup raw mode: {e}")
    
    def _restore_terminal(self):
        """Restore original terminal settings"""
        if self.is_unix and self.original_termios:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_termios)
            except Exception:
                pass
    
    def _input_monitor_thread(self):
        """Background thread for monitoring keyboard input"""
        buffer = b''
        
        while self.running:
            try:
                if self.is_windows and self.msvcrt:
                    # Windows input handling
                    if self.msvcrt.kbhit():
                        char = self.msvcrt.getch()
                        if char:
                            buffer += char
                            self._process_input_buffer(buffer)
                            buffer = b''
                    time.sleep(0.01)  # Small delay to prevent high CPU usage
                
                elif self.is_unix:
                    # Unix input handling with select
                    if select.select([sys.stdin], [], [], 0.01)[0]:
                        char = sys.stdin.read(1).encode()
                        if char:
                            buffer += char
                            
                            # Check for complete sequences
                            if self._is_complete_sequence(buffer):
                                self._process_input_buffer(buffer)
                                buffer = b''
                            elif len(buffer) > 10:  # Prevent buffer overflow
                                buffer = b''
                
            except Exception as e:
                if self.running:  # Only log if we're supposed to be running
                    print(f"\nKeyboard handler error: {e}")
                break
    
    def _is_complete_sequence(self, buffer: bytes) -> bool:
        """Check if buffer contains a complete key sequence"""
        # Single character sequences
        if len(buffer) == 1:
            return buffer in [b'\x1b', b'\x03', b'\x04', b'\x0c', b'\x12', b'\x13', b'\x14', b'\x1a']
        
        # Multi-character sequences (escape sequences)
        if buffer.startswith(b'\x1b'):
            # Common escape sequences
            escape_sequences = [b'\x1bOP', b'\x1b[15~', b'\x1b[1~', b'\x1b[4~']
            return buffer in escape_sequences
        
        return True  # Default to complete for other sequences
    
    def _process_input_buffer(self, buffer: bytes):
        """Process input buffer and trigger appropriate handlers"""
        if not self.running or not buffer:
            return
        
        start_time = time.time()
        
        # Find matching key event
        key_event = None
        for sequence, event in self.key_sequences.items():
            if buffer == sequence or buffer.startswith(sequence):
                key_event = event
                break
        
        if key_event and key_event in self.bindings:
            binding = self.bindings[key_event]
            if binding.enabled:
                try:
                    # Execute handler
                    if binding.handler:
                        if asyncio.iscoroutinefunction(binding.handler):
                            # Schedule coroutine
                            if self.loop:
                                asyncio.run_coroutine_threadsafe(
                                    binding.handler(), self.loop
                                )
                        else:
                            # Execute sync handler
                            binding.handler()
                    
                    # Update metrics
                    response_time = time.time() - start_time
                    self.metrics['keys_processed'] += 1
                    self.metrics['response_times'].append(response_time)
                    
                    # Keep only recent response times
                    if len(self.metrics['response_times']) > 100:
                        self.metrics['response_times'] = self.metrics['response_times'][-50:]
                
                except Exception as e:
                    print(f"\nError handling key event {key_event}: {e}")
    
    async def _handle_esc_interrupt(self):
        """Handle ESC key - emergency interrupt"""
        self.state.interrupt_requested = True
        self.metrics['interrupt_count'] += 1
        
        print("\n🚨 EMERGENCY INTERRUPT REQUESTED (ESC)")
        print("⏹️  Stopping current operation...")
        
        # Trigger all interrupt callbacks
        for callback in self.interrupt_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                print(f"Error in interrupt callback: {e}")
        
        # Set global interrupt flag
        if hasattr(self, '_global_interrupt_event'):
            self._global_interrupt_event.set()
    
    async def _handle_ctrl_t_toggle(self):
        """Handle Ctrl+T - toggle todo list"""
        self.state.todo_visible = not self.state.todo_visible
        status = "visible" if self.state.todo_visible else "hidden"
        print(f"\n📋 Todo list is now {status}")
        
        # Emit todo toggle event
        if hasattr(self, '_todo_toggle_callback'):
            try:
                await self._todo_toggle_callback(self.state.todo_visible)
            except Exception as e:
                print(f"Error in todo toggle callback: {e}")
    
    async def _handle_ctrl_c_interrupt(self):
        """Handle Ctrl+C - graceful interrupt"""
        print("\n⚠️  Graceful interrupt requested (Ctrl+C)")
        print("🔄 Finishing current step then stopping...")
        
        # Set graceful interrupt flag
        self.state.interrupt_requested = True
    
    def _handle_ctrl_l_clear(self):
        """Handle Ctrl+L - clear screen"""
        os.system('cls' if self.is_windows else 'clear')
        print("🎯 ABOV3 Genesis - Screen Cleared")
    
    def _handle_f1_help(self):
        """Handle F1 - show help"""
        self._show_help()
    
    def _show_help(self):
        """Display keyboard shortcuts help"""
        print("\n" + "="*60)
        print("🎯 ABOV3 GENESIS - KEYBOARD SHORTCUTS")
        print("="*60)
        
        for event, binding in self.bindings.items():
            if binding.enabled:
                key_name = self._get_key_name(binding.key_sequence)
                status = "✅" if binding.enabled else "❌"
                print(f"{status} {key_name:12} → {binding.description}")
        
        print("\n📊 Performance Metrics:")
        avg_response = sum(self.metrics['response_times']) / max(len(self.metrics['response_times']), 1)
        print(f"   Keys processed: {self.metrics['keys_processed']}")
        print(f"   Interrupts: {self.metrics['interrupt_count']}")
        print(f"   Avg response: {avg_response:.3f}ms")
        print("="*60 + "\n")
    
    def _get_key_name(self, sequence: bytes) -> str:
        """Convert key sequence to human-readable name"""
        key_names = {
            b'\x1b': 'ESC',
            b'\x14': 'Ctrl+T',
            b'\x03': 'Ctrl+C',
            b'\x04': 'Ctrl+D',
            b'\x0c': 'Ctrl+L',
            b'\x12': 'Ctrl+R',
            b'\x13': 'Ctrl+S',
            b'\x1a': 'Ctrl+Z',
            b'\x1bOP': 'F1',
            b'\x1b[15~': 'F5',
        }
        return key_names.get(sequence, f"0x{sequence.hex()}")
    
    def register_handler(self, event: KeyEvent, handler: Callable) -> bool:
        """Register a custom handler for a key event"""
        if event not in self.bindings:
            return False
        
        if event not in self.state.handlers_registered:
            self.state.handlers_registered[event] = []
        
        self.state.handlers_registered[event].append(handler)
        return True
    
    def unregister_handler(self, event: KeyEvent, handler: Callable) -> bool:
        """Unregister a custom handler"""
        if event in self.state.handlers_registered:
            try:
                self.state.handlers_registered[event].remove(handler)
                return True
            except ValueError:
                pass
        return False
    
    def add_interrupt_callback(self, callback: Callable):
        """Add callback to be called on interrupt"""
        self.interrupt_callbacks.add(callback)
    
    def remove_interrupt_callback(self, callback: Callable):
        """Remove interrupt callback"""
        self.interrupt_callbacks.discard(callback)
    
    def set_todo_toggle_callback(self, callback: Callable):
        """Set callback for todo toggle events"""
        self._todo_toggle_callback = callback
    
    def is_interrupt_requested(self) -> bool:
        """Check if an interrupt has been requested"""
        return self.state.interrupt_requested
    
    def clear_interrupt(self):
        """Clear the interrupt flag"""
        self.state.interrupt_requested = False
    
    def set_current_operation(self, operation: str):
        """Set the current operation name for context"""
        self.state.current_operation = operation
        self.state.operation_start_time = time.time()
    
    def clear_current_operation(self):
        """Clear the current operation"""
        self.state.current_operation = None
        self.state.operation_start_time = 0.0
    
    def get_operation_duration(self) -> float:
        """Get current operation duration in seconds"""
        if self.state.operation_start_time > 0:
            return time.time() - self.state.operation_start_time
        return 0.0
    
    @contextmanager
    def operation_context(self, operation_name: str):
        """Context manager for tracking operations"""
        self.set_current_operation(operation_name)
        self.clear_interrupt()
        try:
            yield self
        finally:
            self.clear_current_operation()
    
    async def wait_for_key(self, timeout: float = None) -> Optional[KeyEvent]:
        """Wait for a specific key press"""
        start_time = time.time()
        
        while self.running:
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            # Check if any key was pressed recently
            if self.state.last_key_time > start_time:
                # Return the last key event (simplified)
                return KeyEvent.ESC_INTERRUPT  # Placeholder
            
            await asyncio.sleep(0.01)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        now = time.time()
        duration = now - self.metrics['last_performance_check']
        
        avg_response = 0.0
        if self.metrics['response_times']:
            avg_response = sum(self.metrics['response_times']) / len(self.metrics['response_times'])
        
        metrics = {
            'keys_processed': self.metrics['keys_processed'],
            'interrupt_count': self.metrics['interrupt_count'],
            'average_response_time_ms': avg_response * 1000,
            'keys_per_second': self.metrics['keys_processed'] / max(duration, 1),
            'uptime_seconds': duration,
            'todo_visible': self.state.todo_visible,
            'current_operation': self.state.current_operation,
            'operation_duration': self.get_operation_duration(),
            'interrupt_requested': self.state.interrupt_requested
        }
        
        self.metrics['last_performance_check'] = now
        return metrics
    
    def enable_binding(self, event: KeyEvent, enabled: bool = True):
        """Enable or disable a key binding"""
        if event in self.bindings:
            self.bindings[event].enabled = enabled
    
    def shutdown(self):
        """Gracefully shutdown the keyboard handler"""
        if not self.running:
            return
        
        self.running = False
        self.state.active = False
        
        # Restore terminal
        self._restore_terminal()
        
        # Wait for input thread to finish
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        
        print("\n🎯 Keyboard controls deactivated")

# Global keyboard handler instance
_global_keyboard_handler: Optional[KeyboardHandler] = None

def get_keyboard_handler() -> KeyboardHandler:
    """Get the global keyboard handler instance"""
    global _global_keyboard_handler
    if _global_keyboard_handler is None:
        _global_keyboard_handler = KeyboardHandler()
    return _global_keyboard_handler

async def initialize_keyboard_controls(loop: asyncio.AbstractEventLoop = None) -> KeyboardHandler:
    """Initialize global keyboard controls"""
    handler = get_keyboard_handler()
    await handler.initialize(loop)
    return handler

def shutdown_keyboard_controls():
    """Shutdown global keyboard controls"""
    global _global_keyboard_handler
    if _global_keyboard_handler:
        _global_keyboard_handler.shutdown()
        _global_keyboard_handler = None