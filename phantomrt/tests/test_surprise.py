"""Tests for the Surprise Detection system."""
import pytest
import torch
from atlas.core.surprise import SurpriseDetector


class TestSurpriseDetector:
    def test_initial_state(self):
        detector = SurpriseDetector()
        
        assert detector.threshold == 0.1
        assert detector.total_surprises == 0
        assert detector.total_predictions == 0
    
    def test_compute_surprise(self):
        detector = SurpriseDetector(initial_threshold=0.01)
        
        # Perfect prediction → no surprise
        real = torch.ones(4, 32)
        predicted = torch.ones(4, 32)
        
        score, surprising = detector.compute_surprise(real, predicted)
        assert score.item() < 0.01
        assert not surprising
    
    def test_surprising_error(self):
        detector = SurpriseDetector(initial_threshold=0.001)
        
        # Bad prediction → surprise!
        real = torch.ones(4, 32)
        predicted = torch.zeros(4, 32)
        
        score, surprising = detector.compute_surprise(real, predicted)
        assert score.item() > 0.5
        assert surprising
    
    def test_adaptive_threshold(self):
        detector = SurpriseDetector(initial_threshold=0.5, adaptation_rate=0.1)
        
        real = torch.ones(10, 32)
        
        # Consistently low error → threshold should decrease
        for _ in range(100):
            predicted = real + torch.randn_like(real) * 0.01
            detector.compute_surprise(real, predicted)
        
        assert detector.threshold < 0.5
    
    def test_history_tracking(self):
        detector = SurpriseDetector()
        
        real = torch.ones(4, 32)
        predicted = torch.zeros(4, 32)
        
        for _ in range(10):
            detector.compute_surprise(real, predicted)
        
        stats = detector.get_stats()
        assert stats["total_predictions"] == 40
        assert len(detector.history) == 10
    
    def test_novelty_score(self):
        detector = SurpriseDetector()
        
        real = torch.ones(4, 32)
        
        # Baseline surprise
        predicted = real + torch.randn_like(real) * 0.01
        detector.compute_surprise(real, predicted)
        
        # Novel observation (should be higher novelty)
        novel_real = torch.ones(4, 32) * 10
        novelty = detector.compute_novelty(novel_real, predicted)
        
        assert novelty.mean().item() > 1.0  # Novelty > 1 means surprising
    
    def test_reset(self):
        detector = SurpriseDetector()
        
        real = torch.ones(4, 32)
        predicted = torch.zeros(4, 32)
        
        for _ in range(10):
            detector.compute_surprise(real, predicted)
        
        detector.reset()
        
        assert detector.total_surprises == 0
        assert detector.total_predictions == 0
        assert len(detector.history) == 0
