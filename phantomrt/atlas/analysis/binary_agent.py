"""
Binary Analysis Agent — The Vulnerability Hunter

Ties together:
  1. World Model (understands execution patterns)
  2. Execution Monitor (sees everything the binary does)
  3. Knowledge Base (knows what instructions mean)
  4. Planner (figures out what inputs to try next)

This is the complete AI that:
  - Reads binary execution
  - Builds understanding of how it works
  - Predicts where vulnerabilities are
  - Plans inputs to test its predictions
  - Learns from every binary it analyzes
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
from pathlib import Path
from typing import Optional
from collections import defaultdict

from ..core.world_model import WorldModel
from ..monitor.execution_monitor import ExecutionMonitor, ExecutionTrace, analyze_trace
from ..knowledge.instruction_vocab import BinaryKnowledgeBase, InstructionEncoder


class BinaryAnalysisAgent:
    """
    Complete binary analysis agent.
    
    Workflow:
      1. Load binary into sandbox
      2. Feed it inputs
      3. Watch execution via monitor
      4. Encode execution into world model
      5. Build understanding of binary behavior
      6. Predict vulnerabilities
      7. Plan inputs to test predictions
      8. Learn from results
    """
    
    def __init__(
        self,
        world_model: Optional[WorldModel] = None,
        latent_dim: int = 256,
        device: str = "cpu",
    ):
        self.device = device
        self.latent_dim = latent_dim
        
        # Components
        self.monitor = ExecutionMonitor()
        self.knowledge = BinaryKnowledgeBase()
        self.instruction_encoder = InstructionEncoder(output_dim=128)
        
        # World model for understanding execution
        if world_model is None:
            # Execution trace features → latent state
            # Input: encoded instruction sequence + execution stats
            trace_feature_dim = 128 + 20  # instruction embedding + stats
            self.world_model = WorldModel(
                obs_dim=trace_feature_dim,
                action_dim=8,  # input modification actions
                latent_dim=latent_dim,
                hidden_dim=256,
            ).to(device)
        else:
            self.world_model = world_model.to(device)
        
        # Learning state
        self.experience_count = 0
        self.known_patterns = defaultdict(int)
        self.confirmed_vulnerabilities = []
        self.false_positives = []
        self.trace_history = []
        
        # Confidence tracking
        self.pattern_confidence = defaultdict(float)
        
        # Binary state tracking
        self.current_binary = None
        self.binary_understanding = {}
    
    def load_binary(self, binary_path: str) -> dict:
        """
        Load and analyze a binary file.
        
        Returns initial understanding of the binary.
        """
        self.current_binary = binary_path
        path = Path(binary_path)
        
        if not path.exists():
            return {"error": f"Binary not found: {binary_path}"}
        
        # Read binary
        data = path.read_bytes()
        
        # Basic analysis
        info = {
            "path": str(path),
            "size": len(data),
            "format": self._detect_format(data),
            "architecture": self._detect_architecture(data),
        }
        
        # Disassemble entry point for initial understanding
        info["entry_analysis"] = self._analyze_entry_point(data)
        
        # Store understanding
        self.binary_understanding = {
            "info": info,
            "traces": [],
            "vulnerabilities": [],
            "risk_assessment": 0.0,
        }
        
        return info
    
    def analyze_with_input(self, input_data: bytes, num_variations: int = 5) -> dict:
        """
        Analyze binary with given input and variations.
        
        This is the core learning loop:
          1. Execute with input
          2. Monitor execution
          3. Feed to world model
          4. Detect anomalies
          5. Learn patterns
        """
        results = {
            "input": input_data.hex(),
            "traces": [],
            "anomalies": [],
            "new_patterns": [],
            "confidence_updates": [],
        }
        
        # Execute with original input
        trace = self.monitor.trace_execution(
            self.current_binary, input_data
        )
        results["traces"].append(trace.summary())
        
        # Analyze the trace
        analysis = analyze_trace(trace)
        results["analysis"] = analysis
        
        # Feed to world model
        world_model_output = self._process_trace(trace)
        results["world_model"] = {
            "surprise_score": world_model_output["surprise_score"],
            "is_surprising": world_model_output["is_surprising"],
            "latent_state_norm": world_model_output["latent_state_norm"],
        }
        
        # Track patterns
        self._update_patterns(trace, analysis)
        
        # Generate variations and test
        variations = self._generate_variations(input_data, num_variations)
        for var_input in variations:
            var_trace = self.monitor.trace_execution(
                self.current_binary, var_input
            )
            var_analysis = analyze_trace(var_trace)
            
            if var_analysis["vulnerability_indicators"]:
                results["anomalies"].extend(var_analysis["vulnerability_indicators"])
            
            # Check if this variation confirms or denies a pattern
            self._check_pattern_consistency(trace, var_trace, var_analysis)
        
        # Update confidence scores
        results["confidence_updates"] = self._update_confidence()
        
        # Store experience
        self.trace_history.append({
            "input": input_data,
            "trace": trace,
            "analysis": analysis,
        })
        self.experience_count += 1
        
        return results
    
    def predict_vulnerabilities(self) -> list:
        """
        Based on accumulated understanding, predict where
        vulnerabilities might exist in the current binary.
        
        Uses world model's latent state to reason about patterns.
        """
        if not self.trace_history:
            return [{"error": "No analysis history. Run analyze_with_input first."}]
        
        predictions = []
        
        # Get recent traces for pattern analysis
        recent_traces = self.trace_history[-100:]
        
        # Analyze patterns across traces
        pattern_freq = defaultdict(int)
        risk_areas = defaultdict(float)
        
        for entry in recent_traces:
            trace = entry["trace"]
            analysis = entry["analysis"]
            
            # Track instruction patterns
            for inst in trace.instructions:
                pattern = f"{inst.mnemonic}_{inst.operands[:20]}"
                pattern_freq[pattern] += 1
                
                # Risk assessment
                risk = self.knowledge.assess_risk(inst.mnemonic, inst.operands)
                risk_areas[inst.address] += risk
        
        # Find high-risk areas
        for addr, risk in sorted(risk_areas.items(), key=lambda x: -x[1]):
            if risk > 1.0:  # threshold
                # Check confidence
                pattern_key = f"addr_{addr}"
                confidence = self.pattern_confidence.get(pattern_key, 0.0)
                
                predictions.append({
                    "address": hex(addr),
                    "risk_score": min(risk / 10, 1.0),
                    "confidence": confidence,
                    "pattern": "high_risk_instruction_cluster",
                    "reasoning": f"Address {hex(addr)} has accumulated risk score {risk:.2f} "
                               f"across {len(recent_traces)} traces",
                })
        
        # Check knowledge base patterns
        for entry in recent_traces[-10:]:  # last 10 traces
            pattern_matches = self.knowledge.get_pattern_matches(entry["trace"].instructions)
            for match in pattern_matches:
                predictions.append({
                    "pattern": match["pattern"],
                    "severity": match["severity"],
                    "signature": match["signature"],
                    "confidence": self.pattern_confidence.get(match["pattern"], 0.0),
                })
        
        return predictions
    
    def generate_smart_input(self) -> bytes:
        """
        Generate an input designed to test predictions
        or explore unknown areas of the binary.
        
        Uses the world model to imagine which inputs
        would be most informative.
        """
        if not self.trace_history:
            # Cold start: generate random inputs
            return self._random_input()
        
        # Get predictions
        predictions = self.predict_vulnerabilities()
        
        # Focus on high-risk, low-confidence areas
        uncertain = [p for p in predictions if p.get("confidence", 0) < 0.7]
        
        if uncertain:
            # Generate input targeting uncertain areas
            return self._target_input(uncertain)
        
        # Explore new areas
        return self._exploration_input()
    
    def _process_trace(self, trace: ExecutionTrace) -> dict:
        """Process an execution trace through the world model."""
        # Encode instructions
        if trace.instructions:
            encoded = self.instruction_encoder.encode_trace([
                {"mnemonic": inst.mnemonic, "operands": inst.operands}
                for inst in trace.instructions[:100]
            ])
        else:
            encoded = torch.zeros(1, 128)
        
        # Add summary features
        summary = trace.summary()
        stats = torch.tensor([
            summary["total_instructions"] / 10000.0,
            summary["total_memory_reads"] / 1000.0,
            summary["total_memory_writes"] / 1000.0,
            summary["stack_depth_max"] / 10000.0,
            float(summary["crash"]),
            summary["branch_count"] / 1000.0,
            summary["syscall_count"] / 100.0,
            float(summary.get("unique_mnemonics", {}).get("call", 0)) / 10.0,
            float(summary.get("unique_mnemonics", {}).get("ret", 0)) / 10.0,
            float(summary.get("unique_mnemonics", {}).get("jmp", 0)) / 10.0,
        ])
        
        # Combine: take mean of instruction embeddings + stats
        inst_embedding = encoded.mean(dim=0)  # [128]
        features = torch.cat([inst_embedding, stats])  # [138]
        
        # Pad to match world model obs_dim
        target_dim = self.world_model.obs_dim
        if features.shape[0] < target_dim:
            features = torch.cat([features, torch.zeros(target_dim - features.shape[0])])
        elif features.shape[0] > target_dim:
            features = features[:target_dim]
        
        features = features.unsqueeze(0).to(self.device)
        
        # Forward through world model
        with torch.no_grad():
            output = self.world_model(features)
        
        return {
            "surprise_score": output.surprise_score,
            "is_surprising": output.is_surprising,
            "latent_state_norm": output.latent_state.norm().item(),
            "reconstruction_error": output.reconstruction_loss.item(),
        }
    
    def _update_patterns(self, trace: ExecutionTrace, analysis: dict):
        """Track and update pattern knowledge."""
        # Track instruction sequences
        for i in range(len(trace.instructions) - 2):
            pattern = (
                trace.instructions[i].mnemonic,
                trace.instructions[i+1].mnemonic,
                trace.instructions[i+2].mnemonic,
            )
            self.known_patterns[pattern] += 1
        
        # Track vulnerability indicators
        for indicator in analysis.get("vulnerability_indicators", []):
            key = f"{indicator['type']}_{indicator.get('address', 0)}"
            self.known_patterns[key] += 1
    
    def _check_pattern_consistency(self, original_trace, variation_trace, variation_analysis):
        """Check if patterns are consistent across variations."""
        # If original had a pattern and variation has it too, it's likely real
        # If only one had it, might be a fluke
        
        for indicator in variation_analysis.get("vulnerability_indicators", []):
            key = indicator["type"]
            
            # Check if this pattern appeared in original trace
            original_analysis = analyze_trace(original_trace)
            original_has = any(i["type"] == key for i in original_analysis.get("vulnerability_indicators", []))
            
            if original_has:
                # Confirmed pattern! Increase confidence
                self.pattern_confidence[key] = min(
                    self.pattern_confidence.get(key, 0.0) + 0.1,
                    1.0
                )
            else:
                # Might be variation-specific, don't increase confidence
                pass
    
    def _update_confidence(self) -> list:
        """Update confidence scores based on accumulated evidence."""
        updates = []
        
        for pattern, count in self.known_patterns.items():
            old_conf = self.pattern_confidence.get(pattern, 0.0)
            
            # Confidence increases with repeated observations
            new_conf = min(1.0 - (1.0 - old_conf) * 0.9, 1.0)
            
            self.pattern_confidence[pattern] = new_conf
            
            if abs(new_conf - old_conf) > 0.01:
                updates.append({
                    "pattern": str(pattern),
                    "old_confidence": old_conf,
                    "new_confidence": new_conf,
                    "observation_count": count,
                })
        
        return updates
    
    def _generate_variations(self, input_data: bytes, num_variations: int) -> list:
        """Generate input variations for testing."""
        variations = []
        
        for _ in range(num_variations):
            var = bytearray(input_data)
            
            # Random mutation
            if len(var) > 0:
                idx = random.randint(0, len(var) - 1)
                var[idx] = random.randint(0, 255)
            
            variations.append(bytes(var))
        
        return variations
    
    def _random_input(self) -> bytes:
        """Generate a random input."""
        size = random.randint(1, 256)
        return bytes(random.randint(0, 255) for _ in range(size))
    
    def _target_input(self, predictions: list) -> bytes:
        """Generate input targeting predicted vulnerability areas."""
        # Start with common vulnerability-triggering patterns
        patterns = [
            b"A" * 64,           # buffer overflow attempt
            b"%s%s%s%s%s",       # format string attempt
            b"\x00" * 32,        # null bytes
            b"\xff" * 64,        # max values
            b"A" * 32 + b"B" * 32,  # boundary test
            b"A" * 256,          # large buffer
        ]
        
        return random.choice(patterns)
    
    def _exploration_input(self) -> bytes:
        """Generate input to explore unknown behavior."""
        size = random.randint(1, 128)
        data = bytes(random.randint(0, 255) for _ in range(size))
        return data
    
    def _detect_format(self, data: bytes) -> str:
        """Detect binary format."""
        if data[:2] == b"MZ":
            return "PE (Windows)"
        elif data[:4] == b"\x7fELF":
            return "ELF (Linux)"
        elif data[:4] == b"\xfe\xed\xfa\xce" or data[:4] == b"\xfe\xed\xfa\xcf":
            return "Mach-O (macOS)"
        elif data[:2] == b"#!":
            return "Script"
        else:
            return "Unknown"
    
    def _detect_architecture(self, data: bytes) -> str:
        """Detect architecture."""
        if data[:4] == b"\x7fELF":
            if len(data) > 4 and data[4] == 1:
                return "x86 (32-bit)"
            elif len(data) > 4 and data[4] == 2:
                return "x86-64 (64-bit)"
        return "Unknown"
    
    def _analyze_entry_point(self, data: bytes) -> dict:
        """Basic entry point analysis."""
        return {
            "size": len(data),
            "first_bytes": data[:16].hex(),
            "printable_ratio": sum(32 <= b <= 126 for b in data[:1000]) / min(len(data), 1000),
        }
    
    def get_stats(self) -> dict:
        """Get analysis statistics."""
        return {
            "total_analyses": self.experience_count,
            "known_patterns": len(self.known_patterns),
            "confirmed_vulnerabilities": len(self.confirmed_vulnerabilities),
            "false_positives": len(self.false_positives),
            "avg_confidence": (
                sum(self.pattern_confidence.values()) / len(self.pattern_confidence)
                if self.pattern_confidence else 0.0
            ),
        }
    
    def save_state(self, path: str):
        """Save agent state for continuation."""
        state = {
            "known_patterns": dict(self.known_patterns),
            "pattern_confidence": dict(self.pattern_confidence),
            "confirmed_vulnerabilities": self.confirmed_vulnerabilities,
            "experience_count": self.experience_count,
        }
        Path(path).write_text(json.dumps(state, indent=2, default=str))
    
    def load_state(self, path: str):
        """Load agent state."""
        state = json.loads(Path(path).read_text())
        self.known_patterns = defaultdict(int, state["known_patterns"])
        self.pattern_confidence = defaultdict(float, state["pattern_confidence"])
        self.confirmed_vulnerabilities = state["confirmed_vulnerabilities"]
        self.experience_count = state["experience_count"]
