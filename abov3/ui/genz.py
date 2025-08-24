"""
ABOV3 Genesis - GenZ Status Messages
Fun, engaging status messages that keep users entertained during processing
"""

import random
from typing import Dict, List
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