"""
PRISM dataset adapter: load and normalize data for PreferenceTracer
- Input: HuggingFace dataset 'HannahRoseKirk/prism-alignment', config 'conversations'
- Output per user: {
    'user_id': str,
    'conversations': [
        {
            'conversation_id': str,
            'turns': [
                {
                    'turn': int,
                    'user_message': {...} | None,
                    'candidates': [ {...}, ... ],
                    'chosen': {...} | None
                }, ...
            ]
        }, ...
    ]
}
"""
from typing import Dict, List, Any
import os
from datasets import load_dataset


def group_by_turns(conversation_history: List[Dict]) -> List[Dict]:
    """Normalize a PRISM conversation history into turns format expected by tracer."""
    turns = {}
    for msg in conversation_history:
        turn = msg.get('turn', 0)
        if turn not in turns:
            turns[turn] = {'turn': turn, 'user_message': None, 'candidates': [], 'chosen': None}
        
        role = msg.get('role')
        if role == 'user':
            turns[turn]['user_message'] = msg.get('content')
        else:
            turns[turn]['candidates'].append(msg.get('content'))
            if msg.get('if_chosen', False):
                turns[turn]['chosen'] = msg.get('content')
    
    # sort by turn index
    return [turns[i] for i in sorted(turns.keys())]


def load_prism_users(n_users: int = None) -> List[Dict[str, Any]]:
    """
    Load PRISM conversations and return per-user bundles with normalized conversations and turns.
    If n_users is provided, limit to the first n unique users (by dataset order).

    IMPORTANT: The PRISM 'conversations' config stores the entire conversation under the key
    'conversation_history'. We must consume that list to build turns.
    """
    dataset = load_dataset("HannahRoseKirk/prism-alignment", "conversations")
    train_data = dataset['train']

    # Build nested mapping: user_id -> ordered list of (conversation_id, conversation_history)
    user_conversations: Dict[str, List[Dict[str, Any]]] = {}
    user_order: List[str] = []
    for rec in train_data:
        uid = rec['user_id']
        cid = rec.get('conversation_id', 'unknown')
        conv_hist = rec.get('conversation_history', None)

        if uid not in user_conversations:
            user_conversations[uid] = []
            user_order.append(uid)

        # Append this conversation as it appears in the dataset order
        user_conversations[uid].append({
            'conversation_id': cid,
            'conversation_history': conv_hist if conv_hist is not None else []
        })

    # Limit users if requested
    if n_users is not None:
        user_order = user_order[:n_users]

    # Build normalized user bundles
    users: List[Dict[str, Any]] = []
    for uid in user_order:
        convs = []
        for conv in user_conversations.get(uid, []):
            cid = conv['conversation_id']
            history = conv['conversation_history'] or []
            # Ensure stable within-conversation ordering
            def _hist_key(x):
                t = x.get('turn', 0)
                if t is None:
                    t = 0
                w = x.get('within_turn_id', -1)
                if w is None:
                    w = -1
                return (t, w)
            history.sort(key=_hist_key)
            turns = group_by_turns(history)
            convs.append({
                'conversation_id': cid,
                'turns': turns
            })

        users.append({
            'user_id': uid,
            'conversations': convs
        })

    return users


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Quick test for PRISM adapter output")
    parser.add_argument("--n-users", type=int, default=2, help="Number of users to load for preview")
    parser.add_argument("--preview", action="store_true", help="Print a short preview of the first user's first conversation")
    parser.add_argument("--dump-json", action="store_true", help="Dump the first user's bundle as JSON (truncated)")
    args = parser.parse_args()

    users = load_prism_users(n_users=args.n_users)
    print(json.dumps(users, indent=2))