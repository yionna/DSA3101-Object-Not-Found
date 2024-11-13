import numpy as np
import pandas as pd
import scipy.stats as stats

#### Part I: Timing, Frequency, Channel Optimizer (Multi-armed Bandit)
class MainOptimizer:
    def __init__(self, n_arms, learning_rate=0.01, alpha=0.5, beta=0.7):
        self.n_arms = n_arms
        self.preferences = np.zeros(n_arms)
        self.context_weights = np.random.rand(n_arms, 10)  # We initialize it with weight of 10 features
        self.learning_rate = learning_rate
        self.alpha = alpha  # control between campaign history and RFM
        self.beta = beta    # control between ctr and conversion rate

    def select_arm(self, customer_features):
        customer_features = np.asarray(customer_features, dtype=np.float64)
        context_scores = np.dot(self.context_weights, customer_features)
        exp_scores = np.exp(context_scores - np.max(context_scores))  
        probabilities = exp_scores / np.sum(exp_scores)

        if np.any(np.isnan(probabilities)) or np.sum(probabilities) == 0:
            probabilities = np.ones(self.n_arms) / self.n_arms  

        return np.random.choice(self.n_arms, p=probabilities)

    def update(self, chosen_arm, customer_features, conversion_rate, clickthrough_rate, rfm_score):
        customer_features = np.asarray(customer_features, dtype=np.float64)

        # Calculate the adjusted engagement score
        adjusted_engagement_score = (
            self.alpha * (self.beta * conversion_rate + (1 - self.beta) * clickthrough_rate) +
            (1 - self.alpha) * rfm_score
        )

        # Update the model using the adjusted engagement score
        gradient = adjusted_engagement_score - np.dot(self.context_weights[chosen_arm], customer_features)
        self.context_weights[chosen_arm] += self.learning_rate * gradient * customer_features
    
        # Clipping the weights to ensure they stay within [0, 1]
        np.clip(self.context_weights[chosen_arm], 0, 1, out=self.context_weights[chosen_arm])

#### Part II: Optimizers for cashback
class SegmentedCashbackOptimizer:
    def __init__(self, initial_thresholds, min_threshold=1000, max_threshold=5000, adjustment_step=100):
        # Independent thresholds for different financial statuses
        self.thresholds = initial_thresholds  # e.g., {'High': 4000, 'Moderate': 3000, 'Low': 2000}
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adjustment_step = adjustment_step

    def calculate_roi(self, total_revenue, total_cost):
        """
        Calculate the ROI given total revenue and total cost.
        """
        return total_revenue / total_cost if total_cost > 0 else 0

    def adjust_threshold(self, financial_status, conversion_rate, current_roi, target_roi=1.2):
        """
        Adjust the cashback threshold for a specific financial status based on conversion rate and ROI.
        """
        current_threshold = self.thresholds[financial_status]
        if current_roi < target_roi and conversion_rate > 0.1:  # High conversion but low ROI
            # Increase the threshold to reduce cost
            new_threshold = min(current_threshold + self.adjustment_step, self.max_threshold)
        elif current_roi >= target_roi and conversion_rate < 0.05:  # Low conversion but good ROI
            # Decrease the threshold to attract more customers
            new_threshold = max(current_threshold - self.adjustment_step, self.min_threshold)
        else:
            new_threshold = current_threshold  # No change if within acceptable ranges

        self.thresholds[financial_status] = new_threshold
        return new_threshold