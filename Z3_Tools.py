import numpy as np
import torch
from z3 import *
import random
from collections import deque
from utils import * 


# Global symbol table
SYMBOLS = {
    "A": Bool("A"),
    "B": Bool("B"),
    "C": Bool("C"),
<<<<<<< HEAD
    "D": Bool("D")
=======
    "D": Bool("D"),
    "E": Bool("E")
>>>>>>> 5af86c3 (Added transformer (encoder) to models.py)
}

def parse(s):
    s = s.strip()

    # Implication
    if "->" in s:
        left, right = s.split("->")
        return Implies(parse(left), parse(right))

    # AND
    if "&" in s:
        left, right = s.split("&")
        return And(parse(left), parse(right))

    # OR
    if "|" in s:
        left, right = s.split("|")
        return Or(parse(left), parse(right))
    
    # NOT 
    if s.startswith("~"):
        return Not(parse(s[1:]))

    # Atomic symbol
    return SYMBOLS[s]


def verify_step(known_facts, proposed_fact):
    """ known_facts: list of strings, e.g. ["A", "A->B"] proposed_fact: string, e.g. "B" """ 
    solver = Solver() 
    # Add all known facts 
    for fact in known_facts: 
        solver.add(parse(fact)) 
    # Add the negation of what we want to prove 
    solver.add(Not(parse(proposed_fact))) 
    result = solver.check() # If UNSAT, the negation is impossible => fact is implied 
    return result == unsat




def nn_score(state, prop, model, ALL_FORMULAS):
    x = np.concatenate([encode_state(state, ALL_FORMULAS), encode_state([prop], ALL_FORMULAS)]).astype(np.float32)
    x = torch.tensor([x])
    with torch.no_grad():
        return model(x).item()
    
        
def rank_candidates(state, model, ALL_FORMULAS):
    scores = []
    for f in ALL_FORMULAS:
        s = nn_score(state, f, model, ALL_FORMULAS)
        scores.append((f, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def pick_valid_step(state, model, ALL_FORMULAS, top_k=5):
    ranked = rank_candidates(state, model, ALL_FORMULAS)
    for f, score in ranked[:top_k]:
        if verify_step(state, f) and f not in state:
            return f, score   # ACCEPTED by Z3

    return None, None   # NN failed to find a valid step



### Shorter version to find the smallest possible step (for theorem proving...) 
def verify_step_small(state, f):
    state = set(state)

    # Modus Ponens:  A, A->B  =>  B
    for impl in state:
        if "->" in impl:
            A, B = impl.split("->")
            if A in state and f == B:
                return True
            
    # And-Elimination
    for impl in state:
        if "&" in impl:
            A, B = impl.split("&")
            if f == A or f == B: 
                return True

    # And-introduction: A, B  =>  A&B
    for A in state:
        for B in state:
            if f == f"{A}&{B}" or f == f"{B}&{A}":
                return True
            
    # Or-Elimination
    for s in state: 
        if "|" in s:
            A, B = s.split("|")
            if f"~{A}" in state and f==B: 
                return True
            if f"~{B}" in state and f==A: 
                return True

    # Or-introduction: A => A|B or B|A
    if "|" in f:
        L, R = f.split("|")
        if L in state or R in state:
            return True

    return False

    
    
    
   
    
# ## BFS Method

def find_shortest_proof(initial_state, goal, ALL_FORMULAS, max_depth=5):
    """
    Returns the sequence of inference steps (formulas) needed to reach `goal`.
    Steps are guaranteed shortest because BFS is used.
    """
    start = frozenset(initial_state)
    queue = deque([(start, [])])
    visited = {start}

    while queue:
        
        state, path = queue.popleft()
        
        if len(path) > max_depth: continue

        # If the goal is already implied, return the path+goal .. ? no 
#         if verify_step_small(list(state), goal):
#             return path + [goal]
        
        # Only stop if the goal is literally in the path ... 
        if goal in state: 
            return path


        # Try adding each possible formula
        for f in ALL_FORMULAS:
            if f in state:
                continue
            if verify_step_small(list(state), f):
                new_state = frozenset(list(state) + [f])
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [f]))

    return None  # no proof found

    
    

def find_all_shortest_proofs(initial_state, ALL_FORMULAS, max_depth=5):
    """
    Runs ONE BFS and returns:
    dict: goal -> shortest proof path
    """
    start = frozenset(initial_state)
    queue = deque([(start, [])])
    visited = {start}
    results = {}   # goal -> path

    while queue:
        state, path = queue.popleft()

        if len(path) > max_depth:
            continue

        # Record shortest path to every formula reached
        for f in state:
            if f not in results:
                results[f] = path

        # Expand BFS normally
        for f in ALL_FORMULAS:
            if f in state:
                continue
            if verify_step_small(list(state), f):
                new_state = frozenset(list(state) + [f])
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [f]))

    return results


    
    
