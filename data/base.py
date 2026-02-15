from dataclasses import dataclass, field
from typing import List

@dataclass
class Turn:
    turn: int = 0
    user_message: str = ""
    candidates: List[str] = field(default_factory=list)
    chosen: str = ""
    
    def __repr__(self):
        return f"Turn {self.turn}:\nUser Message:\n {self.user_message[:50]}\n\nCandidates:\n{'\n'.join([c[:50] for c in self.candidates])}\n\nChosen:\n{self.chosen[:50]}"
    
@dataclass
class Conversation:
    conversation_id: str = ""
    turns: List[Turn] = field(default_factory=list)
    
    def __repr__(self):
        return f"Conversation {self.conversation_id}:\n{"\n\n".join([repr(turn) for turn in self.turns])}"
    
@dataclass
class UserData:
    user_id: str = ""
    conversations: List[Conversation] = field(default_factory=list)
    
    def __repr__(self):
        return f"User {self.user_id}:\n{"\n\n".join([repr(conv) for conv in self.conversations])}"