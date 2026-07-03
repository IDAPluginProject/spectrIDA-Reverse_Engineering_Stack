"""
Performance Optimization Suite for Project Atlas

Makes the world model run FAST for 24/7 continuous operation.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Optional
from pathlib import Path
import threading
import queue
from concurrent.futures import ThreadPoolExecutor


class ModelOptimizer:
    """
    Optimizes the world model for maximum inference speed.
    """
    
    @staticmethod
    def quantize_int8(model: nn.Module) -> nn.Module:
        """
        Quantize model to INT8 for 2-4x speedup.
        """
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},  # quantize linear layers
            dtype=torch.qint8
        )
        return quantized
    
    @staticmethod
    def quantize_int4(model: nn.Module) -> dict:
        """
        Aggressive INT4 quantization for 4-8x speedup.
        Returns state dict (can't run INT4 directly, need to dequant at runtime).
        """
        state_dict = model.state_dict()
        quantized = {}
        
        for name, param in state_dict.items():
            if param.dtype == torch.float32:
                # INT4 quantization: group quantization
                param_flat = param.reshape(-1)
                num_groups = max(1, len(param_flat) // 32)
                group_size = len(param_flat) // num_groups
                
                groups = param_flat[:num_groups * group_size].reshape(num_groups, group_size)
                scales = groups.abs().max(dim=1).values / 7.0
                
                quantized_groups = torch.clamp(
                    torch.round(groups / scales.unsqueeze(1)),
                    -8, 7
                ).to(torch.int8)
                
                quantized[f"{name}_quantized"] = quantized_groups.to(torch.int8)
                quantized[f"{name}_scales"] = scales
                quantized[f"{name}_shape"] = torch.tensor(param.shape)
            else:
                quantized[name] = param
        
        return quantized
    
    @staticmethod
    def export_onnx(model: nn.Module, input_shape: tuple, save_path: str):
        """Export model to ONNX for optimized inference."""
        dummy_input = torch.randn(*input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )
    
    @staticmethod
    def benchmark(model: nn.Module, input_shape: tuple, num_runs: int = 1000) -> dict:
        """Benchmark model speed."""
        dummy = torch.randn(*input_shape)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                model(dummy)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(num_runs):
            with torch.no_grad():
                model(dummy)
        elapsed = time.perf_counter() - start
        
        return {
            "total_time": elapsed,
            "per_run_ms": (elapsed / num_runs) * 1000,
            "throughput": num_runs / elapsed,
            "num_runs": num_runs,
        }


class AsyncPipeline:
    """
    Asynchronous processing pipeline.
    
    Never waits — always processing.
    Like a factory assembly line for analysis.
    """
    
    def __init__(self, agent, max_queue_size: int = 256):
        self.agent = agent
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.workers = []
        self.stats = {
            "processed": 0,
            "errors": 0,
            "avg_latency_ms": 0,
        }
    
    def start(self, num_workers: int = 4):
        """Start the async pipeline."""
        self.running = True
        
        # Input generation worker
        t = threading.Thread(target=self._input_worker, daemon=True)
        t.start()
        self.workers.append(t)
        
        # Processing workers
        for _ in range(num_workers):
            t = threading.Thread(target=self._process_worker, daemon=True)
            t.start()
            self.workers.append(t)
        
        # Stats worker
        t = threading.Thread(target=self._stats_worker, daemon=True)
        t.start()
        self.workers.append(t)
        
        print(f"Pipeline started with {num_workers} workers")
    
    def stop(self):
        """Stop the pipeline."""
        self.running = False
        for w in self.workers:
            w.join(timeout=5)
        print("Pipeline stopped")
    
    def submit(self, binary_path: str, input_data: bytes):
        """Submit work to the pipeline."""
        self.input_queue.put((binary_path, input_data))
    
    def get_result(self, timeout: float = 1.0) -> Optional[dict]:
        """Get a result from the pipeline."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _input_worker(self):
        """Generates inputs continuously."""
        while self.running:
            try:
                # Generate smart input
                input_data = self.agent.generate_smart_input()
                binary_path = self.agent.current_binary
                
                if binary_path:
                    self.input_queue.put(
                        (binary_path, input_data),
                        timeout=1
                    )
            except queue.Full:
                time.sleep(0.01)
            except Exception as e:
                self.stats["errors"] += 1
    
    def _process_worker(self):
        """Processes analyses continuously."""
        while self.running:
            try:
                binary_path, input_data = self.input_queue.get(timeout=1)
                
                start = time.perf_counter()
                
                # Analyze
                self.agent.current_binary = binary_path
                result = self.agent.analyze_with_input(input_data, num_variations=3)
                
                latency = (time.perf_counter() - start) * 1000
                
                result["latency_ms"] = latency
                self.result_queue.put(result, timeout=1)
                
                self.stats["processed"] += 1
                self._update_latency(latency)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.stats["errors"] += 1
    
    def _stats_worker(self):
        """Prints stats periodically."""
        while self.running:
            time.sleep(10)
            print(f"[STATS] Processed: {self.stats['processed']} | "
                  f"Errors: {self.stats['errors']} | "
                  f"Avg latency: {self.stats['avg_latency_ms']:.1f}ms")
    
    def _update_latency(self, latency: float):
        """Update running average latency."""
        n = self.stats["processed"]
        old_avg = self.stats["avg_latency_ms"]
        self.stats["avg_latency_ms"] = old_avg + (latency - old_avg) / n


class BatchProcessor:
    """
    Process multiple inputs simultaneously.
    Maximizes GPU/CPU utilization.
    """
    
    def __init__(self, agent, batch_size: int = 32):
        self.agent = agent
        self.batch_size = batch_size
    
    def process_batch(self, binary_path: str, inputs: list) -> list:
        """Process a batch of inputs simultaneously."""
        self.agent.current_binary = binary_path
        
        results = []
        
        # Execute all inputs (can be parallelized)
        traces = []
        for inp in inputs:
            trace = self.agent.monitor.trace_execution(binary_path, inp)
            traces.append(trace)
        
        # Batch process through world model
        batch_features = self._prepare_batch(traces)
        
        # Single forward pass for entire batch
        with torch.no_grad():
            output = self.agent.world_model(batch_features)
        
        # Unpack results
        for i in range(len(traces)):
            result = {
                "input": inputs[i].hex()[:40],
                "trace_summary": traces[i].summary(),
                "surprise_score": output.surprise_score.item(),
                "is_surprising": output.is_surprising,
            }
            results.append(result)
        
        return results
    
    def _prepare_batch(self, traces) -> torch.Tensor:
        """Prepare a batch of traces for the world model."""
        features = []
        
        for trace in traces:
            # Encode trace summary
            summary = trace.summary()
            stats = [
                summary["total_instructions"] / 10000.0,
                summary["total_memory_reads"] / 1000.0,
                summary["total_memory_writes"] / 1000.0,
                summary["stack_depth_max"] / 10000.0,
                float(summary["crash"]),
                summary["branch_count"] / 1000.0,
                summary["syscall_count"] / 100.0,
            ]
            
            # Pad to match model input
            while len(stats) < self.agent.world_model.obs_dim:
                stats.append(0.0)
            
            features.append(stats[:self.agent.world_model.obs_dim])
        
        return torch.tensor(features, dtype=torch.float32)


class ContinuousLearner:
    """
    Runs the agent 24/7, learning continuously.
    
    Modes:
      - analysis: analyze binaries and learn
      - consolidation: replay and consolidate knowledge
      - exploration: explore new input space
    """
    
    def __init__(self, agent, mode: str = "auto"):
        self.agent = agent
        self.mode = mode
        self.cycle_count = 0
        self.running = False
        
        # Phase tracking
        self.phases = {
            "analysis": 0.6,      # 60% of time analyzing
            "consolidation": 0.2, # 20% consolidating
            "exploration": 0.2,   # 20% exploring
        }
    
    def run_forever(self, binary_path: str, checkpoint_every: int = 1000):
        """Run continuous learning forever."""
        self.running = True
        self.agent.load_binary(binary_path)
        
        print(f"Continuous learning started on {binary_path}")
        print(f"Press Ctrl+C to stop")
        
        try:
            while self.running:
                self.cycle_count += 1
                
                # Determine phase
                phase = self._get_phase()
                
                if phase == "analysis":
                    self._analysis_cycle()
                elif phase == "consolidation":
                    self._consolidation_cycle()
                else:
                    self._exploration_cycle()
                
                # Checkpoint
                if self.cycle_count % checkpoint_every == 0:
                    self.agent.save_state(
                        f"experiments/continuous_state_{self.cycle_count}.json"
                    )
                    print(f"[CYCLE {self.cycle_count}] State saved")
                
                # Print progress every 100 cycles
                if self.cycle_count % 100 == 0:
                    self._print_progress()
        
        except KeyboardInterrupt:
            print("\nStopping...")
        
        self.agent.save_state("experiments/continuous_final.json")
        print(f"Stopped after {self.cycle_count} cycles")
    
    def _get_phase(self) -> str:
        """Determine which phase to run."""
        if self.mode == "auto":
            # Auto mode: rotate through phases
            if self.cycle_count < 100:
                return "analysis"  # start with heavy analysis
            elif self.cycle_count % 10 == 0:
                return "consolidation"
            elif self.cycle_count % 7 == 0:
                return "exploration"
            else:
                return "analysis"
        return self.mode
    
    def _analysis_cycle(self):
        """Run one analysis cycle."""
        input_data = self.agent.generate_smart_input()
        result = self.agent.analyze_with_input(input_data, num_variations=3)
        return result
    
    def _consolidation_cycle(self):
        """Consolidate knowledge (replay recent experiences)."""
        if len(self.agent.trace_history) > 10:
            # Replay recent traces
            recent = self.agent.trace_history[-10:]
            for entry in recent:
                self.agent._process_trace(entry["trace"])
    
    def _exploration_cycle(self):
        """Explore new input space."""
        # Generate very different inputs
        for _ in range(5):
            input_data = bytes(np.random.randint(0, 256, size=np.random.randint(1, 512)))
            self.agent.analyze_with_input(input_data, num_variations=1)
    
    def _print_progress(self):
        """Print current progress."""
        stats = self.agent.get_stats()
        print(
            f"[CYCLE {self.cycle_count}] "
            f"Analyses: {stats['total_analyses']} | "
            f"Patterns: {stats['known_patterns']} | "
            f"Confidence: {stats['avg_confidence']:.3f} | "
            f"Vulns: {stats['confirmed_vulnerabilities']}"
        )


class SpeedBenchmark:
    """Benchmark and compare different optimization levels."""
    
    @staticmethod
    def full_benchmark(agent, input_shapes: dict = None):
        """Run complete speed benchmark."""
        print("=" * 60)
        print("  SPEED BENCHMARK")
        print("=" * 60)
        
        results = {}
        
        # Benchmark world model
        obs_dim = agent.world_model.obs_dim
        print(f"\nBenchmarking world model (obs_dim={obs_dim})...")
        
        # PyTorch default
        pytorch_stats = ModelOptimizer.benchmark(
            agent.world_model,
            input_shape=(1, obs_dim)
        )
        results["pytorch"] = pytorch_stats
        print(f"  PyTorch: {pytorch_stats['per_run_ms']:.2f} ms/run")
        
        # INT8 quantized
        try:
            quantized = ModelOptimizer.quantize_int8(agent.world_model)
            quant_stats = ModelOptimizer.benchmark(
                quantized,
                input_shape=(1, obs_dim)
            )
            results["int8"] = quant_stats
            print(f"  INT8:    {quant_stats['per_run_ms']:.2f} ms/run "
                  f"({pytorch_stats['per_run_ms'] / quant_stats['per_run_ms']:.1f}x faster)")
        except Exception as e:
            print(f"  INT8: Failed ({e})")
        
        # Benchmark instruction encoder
        print(f"\nBenchmarking instruction encoder...")
        enc_stats = ModelOptimizer.benchmark(
            agent.instruction_encoder.feature_proj,
            input_shape=(1, 64)  # approximate input size
        )
        results["encoder"] = enc_stats
        print(f"  Encoder: {enc_stats['per_run_ms']:.2f} ms/run")
        
        # Overall throughput estimate
        print(f"\nEstimated throughput:")
        print(f"  Single analysis: ~{pytorch_stats['per_run_ms'] * 3:.1f} ms")
        print(f"  Batch of 32:     ~{pytorch_stats['per_run_ms'] * 3:.1f} ms total")
        print(f"  Per-analysis in batch: ~{pytorch_stats['per_run_ms'] * 3 / 32:.2f} ms")
        print(f"  Analyses per second:   ~{32 / (pytorch_stats['per_run_ms'] * 3 / 1000):.0f}")
        print(f"  Analyses per hour:     ~{32 / (pytorch_stats['per_run_ms'] * 3 / 1000) * 3600:.0f}")
        
        print()
        return results
