import numpy as np
import json
import logging
from collections import deque
from scipy.optimize import linear_sum_assignment

# Setup logging
logging.basicConfig(filename='assignment_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

class AdaptiveAssignmentSystem:
    def add_participant(self, name, experience_level):
        """
        Add a new participant if under max limit.
        """
        if name in self.participants:
            return False
        if len(self.participants) >= 18:
            return False
        self.participants.append(name)
        self.experience[name] = experience_level
        self.selection_count[name] = 0
        self.weights[name] = 1.0
        self.overrides[name] = 0.0
        self.log_event('add', {'participant': name, 'experience': experience_level})
        return True

    def random_remove_participant(self):
        """
        Remove a random participant if above min limit.
        """
        if len(self.participants) <= 6:
            return False
        import random
        to_remove = random.choice(self.participants)
        self.remove_participant(to_remove)
        return to_remove

    def random_add_participant(self):
        """
        Add a random participant (with random experience) if below max limit.
        """
        if len(self.participants) >= 18:
            return False
        import random
        # Generate a unique name
        base = 'User'
        i = 1
        while f'{base}{i}' in self.participants:
            i += 1
        name = f'{base}{i}'
        exp = random.choice(['beginner', 'experienced'])
        self.add_participant(name, exp)
        return name
    def __init__(self, participants, experience, window_size=10, alpha=1.0, beta=0.8, gini_threshold=0.2):
        self.participants = participants  # List of participant names
        self.experience = experience      # Dict: name -> "beginner"/"experienced"
        self.selection_count = {p: 0 for p in participants}
        self.weights = {p: 1.0 for p in participants}
        self.window = deque(maxlen=window_size)
        self.alpha = alpha
        self.beta = beta
        self.gini_threshold = gini_threshold
        self.overrides = {p: 0.0 for p in participants}
        self.assignment_history = []

    def softmax_probs(self):
        w = np.array([1/(self.selection_count[p]+1) for p in self.participants])
        w = w * np.exp([self.overrides[p] for p in self.participants])
        exp_w = np.exp(self.alpha * w)
        probs = exp_w / np.sum(exp_w)
        return dict(zip(self.participants, probs))

    def select_participant(self):
        probs = self.softmax_probs()
        chosen = np.random.choice(self.participants, p=list(probs.values()))
        self.log_event('selection', {'probs': probs, 'chosen': chosen})
        return chosen

    def update_weights(self, chosen):
        for p in self.participants:
            w_new = 1/(self.selection_count[p]+1)
            self.weights[p] = self.beta * self.weights[p] + (1-self.beta) * w_new
        self.log_event('weight_update', {'weights': self.weights.copy()})

    def apply_override(self, participant, delta):
        self.overrides[participant] += delta
        self.log_event('override', {'participant': participant, 'delta': delta})

    def assign_pairings(self):
        beginners = [p for p in self.participants if self.experience[p] == 'beginner']
        experienced = [p for p in self.participants if self.experience[p] == 'experienced']
        cost_matrix = np.zeros((len(beginners), len(experienced)))
        for i, b in enumerate(beginners):
            for j, e in enumerate(experienced):
                # Lower cost for less frequent pairings
                prev_pairs = sum(1 for h in self.assignment_history if set(h) == {b, e})
                cost_matrix[i, j] = prev_pairs
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        pairs = [(beginners[i], experienced[j]) for i, j in zip(row_ind, col_ind)]
        self.log_event('pairing', {'pairs': pairs})
        return pairs

    def fairness_metrics(self):
        counts = np.array(list(self.selection_count.values()))
        gini = np.sum(np.abs(counts[:, None] - counts)) / (2 * len(counts) * np.sum(counts)) if np.sum(counts) > 0 else 0
        variance = np.var(counts)
        self.log_event('fairness', {'gini': gini, 'variance': variance})
        return gini, variance

    def log_event(self, event_type, data):
        logging.info(json.dumps({'event': event_type, 'data': data}))

    def self_improve(self):
        """
        Modular self-improvement: optimize parameters, detect anomalies, decay overrides.
        """
        gini, variance = self.fairness_metrics()
        self._parameter_optimization(gini, variance)
        self._anomaly_detection()
        self._decay_overrides()

    def _parameter_optimization(self, gini, variance):
        """
        Adjust alpha, beta, and gini_threshold based on fairness and variance.
        """
        # Adjust alpha for fairness
        if gini > self.gini_threshold:
            old_alpha = self.alpha
            self.alpha = max(0.1, self.alpha * 0.9)
            self.log_event('self_improve', {'param': 'alpha', 'old': old_alpha, 'new': self.alpha, 'reason': 'gini_above_threshold'})
        # Adjust beta for high variance (faster adaptation)
        if variance > 2.0:  # Threshold can be tuned
            old_beta = self.beta
            self.beta = max(0.5, self.beta * 0.95)
            self.log_event('self_improve', {'param': 'beta', 'old': old_beta, 'new': self.beta, 'reason': 'variance_high'})
        # Optionally, adjust gini_threshold if system is too strict/lenient
        if gini < 0.05:
            old_gini = self.gini_threshold
            self.gini_threshold = max(0.05, self.gini_threshold * 0.95)
            self.log_event('self_improve', {'param': 'gini_threshold', 'old': old_gini, 'new': self.gini_threshold, 'reason': 'gini_low'})

    def _anomaly_detection(self):
        """
        Detect under/over-selected participants and auto-apply overrides.
        """
        counts = np.array(list(self.selection_count.values()))
        mean = np.mean(counts)
        std = np.std(counts)
        for p, c in self.selection_count.items():
            # Under-selected: boost
            if c < mean - 2 * std:
                self.apply_override(p, 0.1)
                self.log_event('auto_override', {'participant': p, 'delta': 0.1, 'reason': 'under_selected'})
            # Over-selected: penalize
            elif c > mean + 2 * std:
                self.apply_override(p, -0.1)
                self.log_event('auto_override', {'participant': p, 'delta': -0.1, 'reason': 'over_selected'})

    def _decay_overrides(self, decay=0.95):
        """
        Decay all overrides gradually toward zero.
        """
        for p in self.overrides:
            old = self.overrides[p]
            self.overrides[p] *= decay
            if abs(self.overrides[p]) < 1e-3:
                self.overrides[p] = 0.0
            if old != self.overrides[p]:
                self.log_event('override_decay', {'participant': p, 'old': old, 'new': self.overrides[p]})

    def assign(self):
        chosen = self.select_participant()
        self.selection_count[chosen] += 1
        self.window.append(chosen)
        self.assignment_history.append([chosen])
        self.update_weights(chosen)
        self.self_improve()
        return chosen

    def remove_participant(self, participant):
        if participant in self.participants:
            self.participants.remove(participant)
            del self.selection_count[participant]
            del self.weights[participant]
            del self.overrides[participant]
            self.log_event('remove', {'participant': participant})
            # Rebalance: redistribute their tasks (simple version)
            for h in self.assignment_history:
                if participant in h:
                    h.remove(participant)
                    if h:  # If still someone left in the assignment
                        self.selection_count[h[0]] += 1
            self.self_improve()

# Example usage:
if __name__ == "__main__":
    import random
    # Start with 8 participants
    names = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank', 'Grace', 'Heidi']
    experience = {n: random.choice(['beginner', 'experienced']) for n in names}
    system = AdaptiveAssignmentSystem(list(names), experience)
    for step in range(40):
        # Randomly add or remove participants
        action = random.choice(['add', 'remove', 'none'])
        if action == 'add':
            added = system.random_add_participant()
            if added:
                print(f"Added participant: {added}")
        elif action == 'remove':
            removed = system.random_remove_participant()
            if removed:
                print(f"Removed participant: {removed}")
        # Always assign
        chosen = system.assign()
        print(f"Assigned: {chosen}")
        # Show current count
        print(f"Current participants: {len(system.participants)}")
    pairs = system.assign_pairings()
    print("Pairings:", pairs)
