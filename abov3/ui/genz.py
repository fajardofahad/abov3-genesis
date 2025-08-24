"""
ABOV3 Genesis - GenZ Status Messages
Fun, engaging status messages that keep users entertained during processing
"""

import random
import asyncio
import time
from typing import Dict, List, Optional, Callable
from enum import Enum

class StatusCategory(Enum):
    THINKING = "thinking"
    BUILDING = "building"
    WORKING = "working"
    INSTALLING = "installing"
    ERROR_RECOVERY = "error_recovery"
    SUCCESS = "success"

class GenZStatus:
    """
    GenZ Status Messages - provides fun, engaging status updates
    with GenZ-style language and vibes
    """
    
    def __init__(self):
        self.status_messages = {
            StatusCategory.THINKING: [
                "🧠 Big brain time fr fr...",
                "💭 Living in my head rent-free rn...",
                "🤔 No cap, this is bussin...",
                "⚡ Vibing with the algorithm...",
                "🔥 Cooking something fire...",
                "✨ Manifesting code excellence...",
                "🚀 About to drop some heat...",
                "💅 Slay mode activated...",
                "🎯 Lock in, we're so back...",
                "📱 Texting the matrix real quick...",
                "☕ Sips tea while computing...",
                "🎮 Speedrunning this solution...",
                "👾 Glitching into the mainframe...",
                "🌟 Main character energy loading...",
                "🎪 It's giving circus but make it code...",
                "🧪 Mixing potions in the think tank...",
                "🎨 Channeling my inner Picasso...",
                "⚡ Brain cells are absolutely vibing...",
                "🔮 Consulting the algorithm gods...",
                "💫 Entering the flow state bestie..."
            ],
            
            StatusCategory.BUILDING: [
                "🏗️ From idea to reality, watch this...",
                "⚙️ Genesis mode: creating something from nothing...",
                "🎨 Painting your masterpiece in code...",
                "🧪 Cooking in the lab, Gordon Ramsay style...",
                "🏭 Factory settings but make it innovation...",
                "🔨 Bob the Builder but for code...",
                "🌱 Planting seeds of genius...",
                "🎬 Behind the scenes of greatness...",
                "🏰 Building your digital empire...",
                "🎯 Manifesting your vision into reality...",
                "⚡ Genesis energy is absolutely unmatched...",
                "🔥 Your idea said 'build me' and we said bet...",
                "✨ Transforming dreams into deployable code...",
                "🚀 Houston, we're creating something iconic...",
                "💎 Crafting diamonds in the code mine..."
            ],
            
            StatusCategory.WORKING: [
                "⚙️ Grinding harder than coffee beans...",
                "🏗️ Building your empire, bestie...",
                "🎨 Picasso mode: activated...",
                "🧪 Mixing potions in the code lab...",
                "🎯 360 no-scope coding rn...",
                "🏃 Speedrun any% world record attempt...",
                "🎮 Button mashing but professionally...",
                "💫 Channeling my inner code wizard...",
                "🌊 Riding the wave of innovation...",
                "🎪 Juggling bits and bytes like a pro...",
                "⚡ Work mode: absolutely unhinged...",
                "🔥 Your project said 'make me iconic'...",
                "✨ Serving looks while serving code...",
                "💅 Perfectionist era is in full swing...",
                "🎭 Method acting as your personal developer..."
            ],
            
            StatusCategory.INSTALLING: [
                "📦 Unboxing fresh packages...",
                "🛒 Shopping spree in the package store...",
                "🎁 Christmas morning but it's dependencies...",
                "💳 Swiping the company card on npm...",
                "🏪 Package manager said 'say less'...",
                "📮 Your packages are on the way bestie...",
                "🚚 Dependency delivery service at your door...",
                "📋 Adding to cart: literally everything you need...",
                "💎 Collecting infinity stones but they're packages...",
                "🎪 Package party and everyone's invited...",
                "⚡ One-day shipping but for code dependencies...",
                "🎯 Precision targeting the exact packages you need...",
                "🏃 Express lane checkout in the npm store...",
                "🔥 Your dependency list said 'make me complete'...",
                "✨ Summoning packages from the digital realm..."
            ],
            
            StatusCategory.ERROR_RECOVERY: [
                "😅 Oop, let me fix that real quick...",
                "🔧 It's giving broken but we'll fix it...",
                "🩹 Slapping a band-aid on this bug...",
                "🐛 Bug caught in 4K, fixing now...",
                "💀 Code said 'nah' but we persist...",
                "🔄 Ctrl+Z energy but better...",
                "🚑 Code medic reporting for duty...",
                "🛠️ Time for some digital surgery...",
                "🎭 Plot twist: we're debugging now...",
                "⚡ Error said 'catch me if you can'...",
                "🔍 Detective mode: hunting down that bug...",
                "💊 Prescribing some fixes for this code...",
                "🎪 Debugging circus, main event starting...",
                "🧩 Puzzle piece was just upside down...",
                "✨ Turning this oopsie into a feature..."
            ],
            
            StatusCategory.SUCCESS: [
                "✨ From idea to reality - absolutely slayed! 💅",
                "🔥 Built and shipped! No cap!",
                "💯 Genesis complete! That's on period!",
                "🎯 We manifested and delivered!",
                "⭐ From concept to creation - understood the assignment!",
                "🚀 NASA called, they want our genesis code!",
                "👑 Created a masterpiece, crown yourself!",
                "🏆 W in the chat! Idea = Reality!",
                "💪 Built different, from scratch!",
                "🎪 The genesis is complete, crowd goes wild!",
                "⚡ Your vision said 'make me real' and we said bet!",
                "🌟 Main character moment achieved!",
                "💎 Diamond hands created diamond code!",
                "🎨 Michelangelo could never! This is ART!",
                "🔥 Your idea was fire, now it's an entire wildfire!",
                "✨ Plot twist: you're now a successful developer!",
                "💅 Served absolute excellence on a silver platter!",
                "🎯 Bullseye! Direct hit on perfection!",
                "🌈 Manifested a whole rainbow of functionality!",
                "🚀 Houston, we've successfully landed in Reality!"
            ]
        }
        
        # Track recently used messages to avoid repetition
        self.recent_messages: Dict[StatusCategory, List[str]] = {
            category: [] for category in StatusCategory
        }
        
        # Maximum recent messages to track per category
        self.max_recent = 5
    
    def get_status(self, category: str) -> str:
        """Get a random status message from the specified category"""
        try:
            status_category = StatusCategory(category.lower())
        except ValueError:
            # Default to thinking if invalid category
            status_category = StatusCategory.THINKING
        
        return self._get_message(status_category)
    
    def _get_message(self, category: StatusCategory) -> str:
        """Get a message from a category, avoiding recent duplicates"""
        available_messages = self.status_messages[category]
        recent = self.recent_messages[category]
        
        # Filter out recently used messages if we have enough options
        if len(available_messages) > self.max_recent:
            filtered_messages = [msg for msg in available_messages if msg not in recent]
            if filtered_messages:
                available_messages = filtered_messages
        
        # Select random message
        message = random.choice(available_messages)
        
        # Update recent messages
        recent.append(message)
        if len(recent) > self.max_recent:
            recent.pop(0)
        
        return message
    
    def get_thinking_status(self) -> str:
        """Get a thinking/processing status message"""
        return self._get_message(StatusCategory.THINKING)
    
    def get_building_status(self) -> str:
        """Get a building/creating status message"""
        return self._get_message(StatusCategory.BUILDING)
    
    def get_working_status(self) -> str:
        """Get a working/processing status message"""
        return self._get_message(StatusCategory.WORKING)
    
    def get_installing_status(self) -> str:
        """Get an installing/downloading status message"""
        return self._get_message(StatusCategory.INSTALLING)
    
    def get_error_recovery_status(self) -> str:
        """Get an error recovery status message"""
        return self._get_message(StatusCategory.ERROR_RECOVERY)
    
    def get_success_status(self) -> str:
        """Get a success/completion status message"""
        return self._get_message(StatusCategory.SUCCESS)
    
    def get_random_motivation(self) -> str:
        """Get a random motivational message"""
        motivations = [
            "🌟 Every masterpiece starts with a single idea",
            "🚀 From zero to hero, one line at a time", 
            "💫 Where imagination meets implementation",
            "🔥 Let's turn that spark into a wildfire",
            "✨ Your idea + ABOV3 = Absolute Reality",
            "💎 Diamonds are formed under pressure, so is great code",
            "🎯 Aim high, build higher, ship highest",
            "⚡ You're not just coding, you're crafting the future",
            "🌈 Every color in your vision will become reality",
            "👑 You're the architect of your digital empire",
            "🎪 Welcome to the greatest show on earth: your genesis",
            "🎨 Van Gogh painted Starry Night, you're painting apps",
            "🚀 Apollo went to the moon, your app will go further",
            "💅 Serving innovation with a side of excellence",
            "🔥 Your potential is literally infinite, no cap"
        ]
        return random.choice(motivations)
    
    def get_phase_transition_message(self, from_phase: str, to_phase: str) -> str:
        """Get a message for transitioning between Genesis phases"""
        phase_transitions = {
            ('idea', 'design'): [
                "💡➡️📐 From spark to blueprint - let's architect this!",
                "🌟 Idea captured, now let's design the masterpiece!",
                "⚡ Brain dump complete, time to build the foundation!",
                "✨ Concept locked in, switching to architect mode!"
            ],
            ('design', 'build'): [
                "📐➡️🔨 Blueprint ready, time to build the dream!",
                "🏗️ Design approved, let's make it rain code!",
                "⚙️ Architecture complete, construction time!",
                "🎯 Plans are perfect, let's bring them to life!"
            ],
            ('build', 'test'): [
                "🔨➡️🧪 Code complete, time for quality control!",
                "✨ Built and beautiful, let's make sure it's bulletproof!",
                "🎮 Game built, now let's test all the levels!",
                "🔍 Creation complete, detective mode activated!"
            ],
            ('test', 'deploy'): [
                "🧪➡️🚀 Tests passed, ready for lift off!",
                "✅ Quality assured, let's ship this masterpiece!",
                "🌟 Everything perfect, time to share with the world!",
                "🎉 All systems go, deploying your genesis!"
            ],
            ('deploy', 'complete'): [
                "🚀➡️✨ Deployed successfully, genesis complete!",
                "🎯 Mission accomplished, from idea to reality!",
                "👑 Empire built, reality achieved, crown yourself!",
                "💎 Diamond app deployed, absolutely iconic!"
            ]
        }
        
        key = (from_phase.lower(), to_phase.lower())
        if key in phase_transitions:
            return random.choice(phase_transitions[key])
        else:
            return f"⚡ Transitioning from {from_phase} to {to_phase} - let's keep the momentum!"
    
    def get_category_icon(self, category: str) -> str:
        """Get the icon for a status category"""
        icons = {
            'thinking': '🧠',
            'building': '🏗️', 
            'working': '⚙️',
            'installing': '📦',
            'error_recovery': '🔧',
            'success': '✨'
        }
        return icons.get(category.lower(), '⚡')
    
    def add_custom_status(self, category: str, message: str) -> bool:
        """Add a custom status message to a category"""
        try:
            status_category = StatusCategory(category.lower())
            if status_category in self.status_messages:
                self.status_messages[status_category].append(message)
                return True
        except ValueError:
            pass
        return False
    
    def get_all_categories(self) -> List[str]:
        """Get all available status categories"""
        return [category.value for category in StatusCategory]
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about status messages"""
        return {
            category.value: len(messages) 
            for category, messages in self.status_messages.items()
        }

class AnimatedStatus:
    """
    Animated status display for GenZ messages with fun animations
    """
    
    def __init__(self, console=None):
        self.console = console
        self.genz = GenZStatus()
        self.is_animating = False
        self.current_task = None
        
        # Animation frames for different states with flickering effects
        self.loading_frames = [
            "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"
        ]
        
        # Flickering thinking frames with intensity variations
        self.thinking_frames = [
            ("🧠", "bright"), ("💭", "dim"), ("🧠", "bright"), ("⚡", "flash"), 
            ("✨", "sparkle"), ("💫", "dim"), ("🌟", "bright"), ("💭", "dim"),
            ("🧠", "bright"), ("⚡", "flash"), ("💭", "dim"), ("✨", "sparkle")
        ]
        
        # Building frames with construction rhythm
        self.building_frames = [
            ("🏗️", "bright"), ("⚙️", "dim"), ("🔨", "impact"), ("🏗️", "bright"),
            ("🎨", "creative"), ("⚙️", "dim"), ("🧪", "experiment"), ("🔨", "impact"),
            ("⚡", "flash"), ("🏗️", "bright"), ("🎨", "creative"), ("⚙️", "dim")
        ]
        
        # Success frames with celebration intensity
        self.success_frames = [
            ("✨", "sparkle"), ("🌟", "bright"), ("⭐", "twinkle"), ("💫", "dim"),
            ("🎉", "celebrate"), ("🔥", "intense"), ("💎", "brilliant"), ("👑", "royal"),
            ("✨", "sparkle"), ("🚀", "launch"), ("💎", "brilliant"), ("🎉", "celebrate")
        ]
        
        # Color/intensity mappings for different effect types
        self.effect_styles = {
            "bright": "\033[1;97m",      # Bright white
            "dim": "\033[2;90m",         # Dim gray
            "flash": "\033[5;93m",       # Flashing yellow
            "sparkle": "\033[1;95m",     # Bright magenta
            "impact": "\033[1;91m",      # Bright red
            "creative": "\033[1;96m",    # Bright cyan
            "experiment": "\033[1;92m",  # Bright green
            "celebrate": "\033[1;94m",   # Bright blue
            "intense": "\033[1;31m",     # Intense red
            "brilliant": "\033[1;33m",   # Bright yellow
            "royal": "\033[1;35m",       # Bright magenta
            "launch": "\033[1;32m",      # Bright green
            "twinkle": "\033[1;37m",     # Bright white
            "reset": "\033[0m"           # Reset
        }
        
        self.frame_index = 0
        
    def _get_animation_frames(self, category: str):
        """Get animation frames for a category"""
        if category == "thinking":
            return self.thinking_frames
        elif category == "building":
            return self.building_frames  
        elif category == "success":
            return self.success_frames
        else:
            # Convert loading frames to tuple format for consistency
            return [(frame, "bright") for frame in self.loading_frames]
    
    async def animate_status(
        self, 
        category: str, 
        duration: float = 3.0,
        message: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Display an animated status message using Rich Live for proper in-place updates
        """
        if not message:
            message = self.genz.get_status(category)
        
        frames = self._get_animation_frames(category)
        start_time = time.time()
        self.is_animating = True
        
        try:
            if self.console:
                from rich.live import Live
                from rich.text import Text
                
                with Live(console=self.console, refresh_per_second=8, transient=True) as live:
                    while time.time() - start_time < duration and self.is_animating:
                        # Get current frame and effect
                        frame_data = frames[self.frame_index % len(frames)]
                        
                        # Create animated text with Rich styling
                        animated_text = Text()
                        if isinstance(frame_data, tuple):
                            frame_icon, effect_type = frame_data
                            # Map effects to Rich styles
                            style_map = {
                                "bright": "bold bright_white",
                                "dim": "dim white", 
                                "flash": "bold bright_yellow",
                                "sparkle": "bold bright_magenta",
                                "impact": "bold bright_red",
                                "creative": "bold bright_cyan",
                                "experiment": "bold bright_green",
                                "celebrate": "bold bright_blue",
                                "intense": "bold red",
                                "brilliant": "bold bright_yellow",
                                "royal": "bold bright_magenta",
                                "launch": "bold bright_green",
                                "twinkle": "bold white"
                            }
                            style = style_map.get(effect_type, "bold white")
                            animated_text.append(frame_icon, style=style)
                            animated_text.append(f" {message}")
                        else:
                            animated_text.append(f"{frame_data} {message}")
                        
                        # Update the live display
                        live.update(animated_text)
                        
                        # Update frame and wait
                        self.frame_index = (self.frame_index + 1) % len(frames)
                        
                        # Variable animation speed
                        if isinstance(frame_data, tuple):
                            _, effect_type = frame_data
                            speed_map = {
                                "flash": 0.15,
                                "sparkle": 0.2,
                                "impact": 0.25,
                                "bright": 0.3,
                                "dim": 0.4,
                                "twinkle": 0.2
                            }
                            await asyncio.sleep(speed_map.get(effect_type, 0.3))
                        else:
                            await asyncio.sleep(0.3)
                
                # Final static display after animation
                icon = self.genz.get_category_icon(category)
                final_text = Text()
                final_text.append(icon, style="bold")
                final_text.append(f" {message}")
                self.console.print(final_text)
                
            else:
                # Fallback for non-Rich console
                while time.time() - start_time < duration and self.is_animating:
                    frame_data = frames[self.frame_index % len(frames)]
                    frame_icon = frame_data[0] if isinstance(frame_data, tuple) else frame_data
                    print(f"\r{frame_icon} {message}", end="", flush=True)
                    self.frame_index = (self.frame_index + 1) % len(frames)
                    await asyncio.sleep(0.3)
                
                print(f"\r{self.genz.get_category_icon(category)} {message}")
            
            # Call callback if provided
            if callback:
                await callback()
                
            return f"{self.genz.get_category_icon(category)} {message}"
            
        finally:
            self.is_animating = False
    
    async def animate_thinking(self, duration: float = 2.0, message: Optional[str] = None) -> str:
        """Animate a thinking status"""
        return await self.animate_status("thinking", duration, message)
    
    async def animate_building(self, duration: float = 3.0, message: Optional[str] = None) -> str:
        """Animate a building status"""
        return await self.animate_status("building", duration, message)
    
    async def animate_success(self, duration: float = 2.0, message: Optional[str] = None) -> str:
        """Animate a success status"""
        return await self.animate_status("success", duration, message)
    
    async def animate_phase_transition(self, from_phase: str, to_phase: str, duration: float = 3.0) -> str:
        """Animate a phase transition with special flickering effects"""
        message = self.genz.get_phase_transition_message(from_phase, to_phase)
        
        # Special transition animation with flickering intensity
        transition_frames = [
            ("⚡", "flash"), ("✨", "sparkle"), ("🌟", "bright"), ("💫", "twinkle"),
            ("🚀", "launch"), ("⚡", "flash"), ("🎯", "impact"), ("✨", "sparkle"),
            ("🌟", "bright"), ("💫", "dim"), ("🚀", "launch"), ("⚡", "flash")
        ]
        start_time = time.time()
        frame_index = 0
        
        while time.time() - start_time < duration:
            frame_icon, effect_type = transition_frames[frame_index % len(transition_frames)]
            
            # Apply intensive flickering for transitions
            style = self.effect_styles.get(effect_type, self.effect_styles["bright"])
            reset = self.effect_styles["reset"]
            frame = f"{style}{frame_icon}{reset}"
            
            if self.console:
                self.console.print(f"\r{frame} {message}", end="", highlight=False)
            else:
                print(f"\r{frame} {message}", end="", flush=True)
                
            frame_index = (frame_index + 1) % len(transition_frames)
            
            # Fast transition animation
            speed_map = {
                "flash": 0.1,
                "sparkle": 0.15,
                "impact": 0.2,
                "launch": 0.12,
                "twinkle": 0.18
            }
            await asyncio.sleep(speed_map.get(effect_type, 0.15))
        
        # Final display with celebration effect
        final_style = self.effect_styles["brilliant"]
        reset = self.effect_styles["reset"]
        final_message = f"{final_style}🎯{reset} {message}"
        
        if self.console:
            self.console.print(f"\r{final_message}")
        else:
            print(f"\r{final_message}")
            
        return final_message
    
    async def animate_progress(
        self, 
        steps: List[str], 
        category: str = "working",
        step_duration: float = 1.5
    ) -> List[str]:
        """
        Animate through multiple progress steps
        
        Args:
            steps: List of step descriptions
            category: Animation category
            step_duration: Duration for each step
            
        Returns:
            List of final messages for each step
        """
        results = []
        frames = self._get_animation_frames(category)
        
        for i, step in enumerate(steps):
            if self.console:
                self.console.print(f"\n[{i+1}/{len(steps)}] Starting: {step}")
            else:
                print(f"\n[{i+1}/{len(steps)}] Starting: {step}")
            
            # Animate this step
            result = await self.animate_status(category, step_duration, step)
            results.append(result)
            
            # Brief pause between steps
            await asyncio.sleep(0.5)
        
        return results
    
    def stop_animation(self):
        """Stop current animation"""
        self.is_animating = False
    
    async def show_completion_celebration(self, success_message: Optional[str] = None) -> str:
        """Show an intense flickering celebration animation for completion"""
        if not success_message:
            success_message = self.genz.get_success_status()
        
        # Intense celebration frames with flickering effects
        celebration_frames = [
            ("🎉", "celebrate"), ("✨", "sparkle"), ("🔥", "intense"), ("💎", "brilliant"), 
            ("👑", "royal"), ("🌟", "bright"), ("⭐", "twinkle"), ("💫", "dim"),
            ("🚀", "launch"), ("🎯", "impact"), ("💅", "sparkle"), ("👾", "flash"),
            ("🎪", "celebrate"), ("🎨", "creative"), ("⚡", "flash"), ("✨", "sparkle"),
            ("🔥", "intense"), ("💎", "brilliant"), ("🎉", "celebrate"), ("🌟", "bright")
        ]
        
        if self.console:
            from rich.live import Live
            from rich.text import Text
            
            with Live(console=self.console, refresh_per_second=12, transient=True) as live:
                # Intense celebration sequence with rapid flickering
                for i in range(30):  # 30 frames of intense celebration
                    frame_icon, effect_type = random.choice(celebration_frames)
                    
                    # Map to Rich styles
                    style_map = {
                        "celebrate": "bold bright_blue",
                        "sparkle": "bold bright_magenta", 
                        "intense": "bold red",
                        "brilliant": "bold bright_yellow",
                        "royal": "bold bright_magenta",
                        "bright": "bold bright_white",
                        "twinkle": "bold white",
                        "dim": "dim white",
                        "launch": "bold bright_green",
                        "impact": "bold bright_red",
                        "flash": "bold bright_yellow",
                        "creative": "bold bright_cyan"
                    }
                    
                    celebration_text = Text()
                    style = style_map.get(effect_type, "bold white")
                    celebration_text.append(frame_icon, style=style)
                    celebration_text.append(f" {success_message}")
                    
                    live.update(celebration_text)
                    
                    # Variable speed for dramatic effect
                    if i < 10:  # Start fast
                        await asyncio.sleep(0.08)
                    elif i < 20:  # Medium speed
                        await asyncio.sleep(0.12)
                    else:  # Slow down for finale
                        await asyncio.sleep(0.18)
            
            # Final success display with permanent glow
            final_text = Text()
            final_text.append("🎉", style="bold bright_yellow")
            final_text.append(f" {success_message}")
            self.console.print(final_text)
            
        else:
            # Fallback for non-Rich console
            for i in range(15):  # Shorter for fallback
                frame_icon, _ = random.choice(celebration_frames)
                print(f"\r{frame_icon} {success_message}", end="", flush=True)
                await asyncio.sleep(0.15)
            
            print(f"\r🎉 {success_message}")
            
        return f"🎉 {success_message}"