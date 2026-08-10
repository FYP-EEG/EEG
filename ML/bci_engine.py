"""
Author: Brian
Created on: 10/08/2026
Purpose: serves as key middleware between the UDP/OpenBCI data stream and the game UI controller
"""
import numpy as np
from collections import deque, Counter
from ML.hybrid_classifier import HybridSSVEPClassifier, MotorImageryClassifier

class BCIEngine:
    def __init__(self, mi_model_path=None, mode='HYBRID', debounce_window=5, threshold=0.6):
        """
        :param mode: 'MI', 'SSVEP', or 'HYBRID'
        """
        self.mode = mode
        self.debounce_window = debounce_window
        self.threshold = threshold
        self.history = deque(maxlen=debounce_window)

        self.ssvep_clf = HybridSSVEPClassifier()
        self.mi_clf = MotorImageryClassifier(model_path=mi_model_path)

    def process_frame(self, raw_trial_data):
        """
        :return: (smoothed_decision, is_triggered)
        """
        if self.ssvep_clf.artifact_detection(raw_trial_data):
            raw_pred = "BLINK"
        
        # single mode choosing
        elif self.mode == 'SSVEP':
            raw_pred = f"SSVEP_{self.ssvep_clf.cca_ssvep_detection(raw_trial_data)}"
        elif self.mode == 'MI':
            raw_pred = f"MI_{self.mi_clf.predict(raw_trial_data)}"
            
        elif self.mode == 'HYBRID':
            ssvep_res = self.ssvep_clf.cca_ssvep_detection(raw_trial_data)
            mi_res = self.mi_clf.predict(raw_trial_data)
            raw_pred = f"MI_{mi_res}|SSVEP_{ssvep_res}"

        # Debouncing
        self.history.append(raw_pred)
        
        if len(self.history) < self.debounce_window:
            return None, False 

        counts = Counter(self.history)
        most_common_class, count = counts.most_common(1)[0]
        
        if (count / self.debounce_window) >= self.threshold:
            return most_common_class, True
        
        return None, False