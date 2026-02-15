from typing import Dict, List
from datasets import load_dataset
from .base import Turn, Conversation, UserData
import random

def group_by_turns(conversation_history: List[Dict]) -> List[Turn]:
    turns = {}
    for msg in conversation_history:
        turn = msg.get('turn', 0)
        if turn not in turns:
            turns[turn] = Turn()
        role = msg.get('role')
        if role == 'user':
            turns[turn].user_message = msg.get('content')
        else:
            turns[turn].candidates.append(msg.get('content'))
            if msg.get('if_chosen', False):
                turns[turn].chosen = msg.get('content')
    return [turns[i] for i in sorted(turns.keys())]


def load_prism(n_users: int = None) -> List[UserData]:
    """
    Load PRISM conversations and return per-user bundles with normalized conversations and turns.
    If n_users is provided, limit to the first n unique users (by dataset order).
    """
    dataset = load_dataset("HannahRoseKirk/prism-alignment", "conversations")
    train_data = dataset['train']

    user_conversations: Dict[List[Dict]] = {}
    user_order: List[str] = []
    for rec in train_data:
        uid = rec['user_id']
        cid = rec.get('conversation_id', 'unknown')
        conv_hist = rec.get('conversation_history', None)
        if uid not in user_conversations:
            user_conversations[uid] = []
            user_order.append(uid)
        user_conversations[uid].append({
            'conversation_id': cid,
            'conversation_history': conv_hist
        })

    if n_users is not None:
        user_order = random.sample(user_order, n_users)

    users: List[UserData] = []
    for uid in user_order:
        convs = []
        for conv in user_conversations.get(uid, []):
            cid = conv['conversation_id']
            history = conv['conversation_history']
            turns = group_by_turns(history)
            convs.append(Conversation(
                conversation_id=cid,
                turns=turns
            ))
        users.append(UserData(
            user_id=uid,
            conversations=convs
        ))
    return users


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick test for PRISM adapter output")
    parser.add_argument("--n-users", type=int, default=2, help="Number of users to load for preview")
    parser.add_argument("--preview", action="store_true", help="Print a short preview of the first user's first conversation")
    parser.add_argument("--dump-json", action="store_true", help="Dump the first user's bundle as JSON (truncated)")
    args = parser.parse_args()

    users = load_prism(n_users=args.n_users)
    print(users)