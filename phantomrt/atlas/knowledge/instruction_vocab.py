"""
Instruction Vocabulary — The Model's Dictionary

Gives the world model a BASE understanding of:
  - What each CPU instruction does
  - What registers are for
  - What memory regions mean
  - Common code patterns
  - Vulnerability signatures

This is the "learning to read" foundation.
Without this, the model stares at raw bytes.
With this, it understands what it's seeing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════
# INSTRUCTION SEMANTICS — What each instruction DOES
# ═══════════════════════════════════════════════════════════════

INSTRUCTION_SEMANTICS = {
    # Data Movement
    "mov": {"category": "data_move", "risk": 0.1, "description": "Copy data from source to destination"},
    "push": {"category": "data_move", "risk": 0.1, "description": "Push value onto stack"},
    "pop": {"category": "data_move", "risk": 0.1, "description": "Pop value from stack"},
    "lea": {"category": "data_move", "risk": 0.05, "description": "Load effective address (compute, don't access)"},
    "xchg": {"category": "data_move", "risk": 0.05, "description": "Swap two values"},
    "cmov": {"category": "data_move", "risk": 0.05, "description": "Conditional move"},
    
    # Arithmetic
    "add": {"category": "arithmetic", "risk": 0.15, "description": "Add source to destination"},
    "sub": {"category": "arithmetic", "risk": 0.15, "description": "Subtract source from destination"},
    "inc": {"category": "arithmetic", "risk": 0.05, "description": "Increment by 1"},
    "dec": {"category": "arithmetic", "risk": 0.05, "description": "Decrement by 1"},
    "imul": {"category": "arithmetic", "risk": 0.25, "description": "Signed multiply (can overflow)"},
    "mul": {"category": "arithmetic", "risk": 0.25, "description": "Unsigned multiply (can overflow)"},
    "idiv": {"category": "arithmetic", "risk": 0.3, "description": "Signed divide (can crash on zero)"},
    "div": {"category": "arithmetic", "risk": 0.3, "description": "Unsigned divide (can crash on zero)"},
    "neg": {"category": "arithmetic", "risk": 0.1, "description": "Negate value (two's complement)"},
    
    # Logic
    "and": {"category": "logic", "risk": 0.05, "description": "Bitwise AND"},
    "or": {"category": "logic", "risk": 0.05, "description": "Bitwise OR"},
    "xor": {"category": "logic", "risk": 0.05, "description": "Bitwise XOR"},
    "not": {"category": "logic", "risk": 0.05, "description": "Bitwise NOT"},
    "shl": {"category": "logic", "risk": 0.1, "description": "Shift left (multiply by 2^n)"},
    "shr": {"category": "logic", "risk": 0.1, "description": "Shift right (divide by 2^n)"},
    "sar": {"category": "logic", "risk": 0.1, "description": "Arithmetic shift right (preserves sign)"},
    "rol": {"category": "logic", "risk": 0.05, "description": "Rotate left"},
    "ror": {"category": "logic", "risk": 0.05, "description": "Rotate right"},
    
    # Comparison
    "cmp": {"category": "compare", "risk": 0.0, "description": "Compare two values (sets flags)"},
    "test": {"category": "compare", "risk": 0.0, "description": "Test two values (AND, sets flags)"},
    
    # Control Flow
    "jmp": {"category": "control_flow", "risk": 0.05, "description": "Unconditional jump"},
    "je": {"category": "control_flow", "risk": 0.05, "description": "Jump if equal"},
    "jne": {"category": "control_flow", "risk": 0.05, "description": "Jump if not equal"},
    "jg": {"category": "control_flow", "risk": 0.05, "description": "Jump if greater"},
    "jge": {"category": "control_flow", "risk": 0.05, "description": "Jump if greater or equal"},
    "jl": {"category": "control_flow", "risk": 0.05, "description": "Jump if less"},
    "jle": {"category": "control_flow", "risk": 0.05, "description": "Jump if less or equal"},
    "ja": {"category": "control_flow", "risk": 0.05, "description": "Jump if above (unsigned)"},
    "jb": {"category": "control_flow", "risk": 0.05, "description": "Jump if below (unsigned)"},
    "call": {"category": "control_flow", "risk": 0.1, "description": "Call function (pushes return address)"},
    "ret": {"category": "control_flow", "risk": 0.15, "description": "Return from function (pops return address)"},
    "loop": {"category": "control_flow", "risk": 0.05, "description": "Loop (dec ECX, jump if not zero)"},
    
    # String Operations (HIGH RISK for buffer overflows)
    "rep": {"category": "string", "risk": 0.2, "description": "Repeat next instruction ECX times"},
    "movsb": {"category": "string", "risk": 0.2, "description": "Move byte string (DS:RSI -> ES:RDI)"},
    "movsw": {"category": "string", "risk": 0.2, "description": "Move word string"},
    "movsd": {"category": "string", "risk": 0.2, "description": "Move dword string"},
    "movsq": {"category": "string", "risk": 0.2, "description": "Move qword string"},
    "stosb": {"category": "string", "risk": 0.15, "description": "Store byte to string"},
    "stosw": {"category": "string", "risk": 0.15, "description": "Store word to string"},
    "stosd": {"category": "string", "risk": 0.15, "description": "Store dword to string"},
    "stosq": {"category": "string", "risk": 0.15, "description": "Store qword to string"},
    "cmpsb": {"category": "string", "risk": 0.05, "description": "Compare byte strings"},
    "scasb": {"category": "string", "risk": 0.05, "description": "Scan byte string"},
    
    # Stack Operations
    "enter": {"category": "stack", "risk": 0.05, "description": "Create stack frame"},
    "leave": {"category": "stack", "risk": 0.05, "description": "Destroy stack frame"},
    
    # System
    "syscall": {"category": "system", "risk": 0.3, "description": "System call (kernel transition)"},
    "int": {"category": "system", "risk": 0.3, "description": "Software interrupt"},
    "sysenter": {"category": "system", "risk": 0.3, "description": "Fast system call"},
    "hlt": {"category": "system", "risk": 0.0, "description": "Halt processor"},
    
    # SIMD (can be used for overflows)
    "movdqu": {"category": "simd", "risk": 0.2, "description": "Move unaligned 128-bit data"},
    "movdqa": {"category": "simd", "risk": 0.15, "description": "Move aligned 128-bit data"},
    "paddd": {"category": "simd", "risk": 0.1, "description": "Add packed 32-bit integers"},
}


# ═══════════════════════════════════════════════════════════════
# REGISTER SEMANTICS — What each register is USED FOR
# ═══════════════════════════════════════════════════════════════

REGISTER_SEMANTICS = {
    # General Purpose
    "rax": {"role": "accumulator", "usage": "Return value, arithmetic", "volatility": "caller_saved"},
    "rbx": {"role": "base", "usage": "General purpose, often preserved", "volatility": "callee_saved"},
    "rcx": {"role": "counter", "usage": "Loop counter, 4th argument", "volatility": "caller_saved"},
    "rdx": {"role": "data", "usage": "I/O operations, 3rd argument", "volatility": "caller_saved"},
    "rsi": {"role": "source_index", "usage": "Source for string ops, 2nd argument", "volatility": "caller_saved"},
    "rdi": {"role": "dest_index", "usage": "Destination for string ops, 1st argument", "volatility": "caller_saved"},
    "rbp": {"role": "base_pointer", "usage": "Stack frame base", "volatility": "callee_saved"},
    "rsp": {"role": "stack_pointer", "usage": "Top of stack", "volatility": "callee_saved"},
    "r8": {"role": "general", "usage": "5th argument", "volatility": "caller_saved"},
    "r9": {"role": "general", "usage": "6th argument", "volatility": "caller_saved"},
    "r10": {"role": "general", "usage": "Temporary", "volatility": "caller_saved"},
    "r11": {"role": "general", "usage": "Temporary, trashed by syscall", "volatility": "caller_saved"},
    "r12": {"role": "general", "usage": "General purpose", "volatility": "callee_saved"},
    "r13": {"role": "general", "usage": "General purpose", "volatility": "callee_saved"},
    "r14": {"role": "general", "usage": "General purpose", "volatility": "callee_saved"},
    "r15": {"role": "general", "usage": "General purpose", "volatility": "callee_saved"},
    "rip": {"role": "instruction_pointer", "usage": "Next instruction to execute", "volatility": "special"},
    "rflags": {"role": "flags", "usage": "Condition codes, control flags", "volatility": "special"},
}


# ═══════════════════════════════════════════════════════════════
# MEMORY REGIONS — What different addresses MEAN
# ═══════════════════════════════════════════════════════════════

MEMORY_REGIONS = {
    "stack": {
        "address_range": (0x7fff0000, 0x7fffffff),
        "properties": ["grows_down", "local_variables", "function_returns"],
        "risk": "buffer_overflow, stack_smash, return_oriented_programming",
    },
    "heap": {
        "address_range": (0x60000000, 0x6fffffff),
        "properties": ["grows_up", "dynamic_allocation", "free_list"],
        "risk": "heap_overflow, use_after_free, double_free, heap_spray",
    },
    "code": {
        "address_range": (0x00400000, 0x00600000),
        "properties": ["read_only", "executable", "instructions"],
        "risk": "code_injection, shellcode",
    },
    "data": {
        "address_range": (0x00600000, 0x00700000),
        "properties": ["read_write", "global_variables", "constants"],
        "risk": "data_corruption",
    },
    "mmap": {
        "address_range": (0x7f000000, 0x7fffffff),
        "properties": ["dynamic", "libraries", "shared_memory"],
        "risk": "mmap_exploitation",
    },
}


# ═══════════════════════════════════════════════════════════════
# VULNERABILITY PATTERNS — Known dangerous patterns
# ═══════════════════════════════════════════════════════════════

VULNERABILITY_PATTERNS = {
    "stack_buffer_overflow": {
        "signatures": [
            "rep movsb with large ECX and stack destination",
            "mov with stack write past frame size",
            "gets() call (no bounds checking)",
            "strcpy() call (no bounds checking)",
            "sprintf() call (no bounds checking)",
        ],
        "indicators": [
            "excessive stack growth",
            "return address modification",
            "saved RBP modification",
        ],
        "severity": "critical",
    },
    "heap_buffer_overflow": {
        "signatures": [
            "write past allocated heap chunk",
            "heap metadata corruption",
        ],
        "indicators": [
            "heap chunk header modification",
            "adjacent chunk corruption",
        ],
        "severity": "critical",
    },
    "use_after_free": {
        "signatures": [
            "accessing freed pointer",
            "use after free() call",
        ],
        "indicators": [
            "pointer used after deallocation",
            "double free detection",
        ],
        "severity": "high",
    },
    "format_string": {
        "signatures": [
            "printf with user-controlled format",
            "sprintf with user-controlled format",
            "fprintf with user-controlled format",
        ],
        "indicators": [
            "format string without format specifier",
            "user input directly in format position",
        ],
        "severity": "high",
    },
    "integer_overflow": {
        "signatures": [
            "multiply without overflow check",
            "add without bounds check before allocation",
            "signed/unsigned confusion",
        ],
        "indicators": [
            "arithmetic before memory allocation",
            "size calculation overflow",
        ],
        "severity": "medium",
    },
    "race_condition": {
        "signatures": [
            "TOCTOU (time-of-check-to-time-of-use)",
            "unsynchronized shared access",
        ],
        "indicators": [
            "check-then-act pattern without lock",
            "shared memory access without synchronization",
        ],
        "severity": "medium",
    },
}


# ═══════════════════════════════════════════════════════════════
# INSTRUCTION ENCODER — Convert instructions to vectors
# ═══════════════════════════════════════════════════════════════

class InstructionEncoder(nn.Module):
    """
    Encodes x86 instructions into dense vectors that the world model can process.
    
    Converts:
      "mov eax, dword [rbp-0x10]" → 128-dimensional vector
    
    The vector captures:
      - What the instruction does (mnemonic semantics)
      - What it operates on (operand types)
      - Risk level
      - Memory access pattern
    """
    
    VOCAB_SIZE = 200  # number of known mnemonics
    EMBED_DIM = 32    # dimension per mnemonic
    FEATURE_DIM = 16  # additional feature dimensions
    
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        
        # Mnemonic embedding
        self.mnemonic_embed = nn.Embedding(self.VOCAB_SIZE, self.EMBED_DIM)
        
        # Category embedding
        self.category_embed = nn.Embedding(10, 16)  # ~10 categories
        
        # Register embeddings
        self.num_registers = 20  # common registers
        self.register_embed = nn.Embedding(self.num_registers, 8)
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(self.EMBED_DIM + 16 + 16 + self.FEATURE_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # Build vocabulary
        self.mnemonic_to_idx = {}
        self.register_to_idx = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build lookup tables."""
        for i, mnemonic in enumerate(INSTRUCTION_SEMANTICS.keys()):
            if i < self.VOCAB_SIZE:
                self.mnemonic_to_idx[mnemonic] = i
        
        for i, reg in enumerate(REGISTER_SEMANTICS.keys()):
            if i < self.num_registers:
                self.register_to_idx[reg] = i
    
    def encode_instruction(self, mnemonic: str, operands: str) -> torch.Tensor:
        """Encode a single instruction."""
        # Mnemonic index
        m_idx = self.mnemonic_to_idx.get(mnemonic, 0)
        m_emb = self.mnemonic_embed(torch.tensor(m_idx))
        
        # Category
        sem = INSTRUCTION_SEMANTICS.get(mnemonic, {"category": "unknown", "risk": 0.0})
        categories = list(set(s["category"] for s in INSTRUCTION_SEMANTICS.values()))
        cat_idx = categories.index(sem["category"]) if sem["category"] in categories else 0
        c_emb = self.category_embed(torch.tensor(cat_idx))
        
        # Features
        features = torch.zeros(self.FEATURE_DIM)
        features[0] = sem.get("risk", 0.0)  # risk level
        features[1] = 1.0 if "rsp" in operands or "rbp" in operands else 0.0  # stack related
        features[2] = 1.0 if "[" in operands else 0.0  # memory access
        features[3] = 1.0 if "call" in mnemonic else 0.0  # function call
        features[4] = 1.0 if "ret" in mnemonic else 0.0  # function return
        features[5] = 1.0 if mnemonic in ("jmp", "je", "jne", "jg", "jl") else 0.0  # branch
        features[6] = 1.0 if mnemonic in ("syscall", "int") else 0.0  # syscall
        features[7] = 1.0 if mnemonic in ("rep", "movsb", "stosb") else 0.0  # string op
        features[8] = 1.0 if "dword" in operands else 0.0  # 32-bit access
        features[9] = 1.0 if "qword" in operands else 0.0  # 64-bit access
        
        # Concatenate and project
        combined = torch.cat([m_emb, c_emb, torch.zeros(16), features])  # register emb placeholder
        return self.feature_proj(combined)
    
    def encode_trace(self, trace: list) -> torch.Tensor:
        """Encode an execution trace into a sequence of vectors."""
        encoded = []
        for inst in trace:
            vec = self.encode_instruction(inst.mnemonic, inst.operands)
            encoded.append(vec)
        
        if encoded:
            return torch.stack(encoded)  # [seq_len, output_dim]
        return torch.zeros(1, self.output_dim)


# ═══════════════════════════════════════════════════════════════
# KNOWLEDGE BASE — Pre-trained understanding
# ═══════════════════════════════════════════════════════════════

class BinaryKnowledgeBase:
    """
    Pre-loaded knowledge about binary analysis.
    
    This is the "education" the model receives before
    it starts analyzing real binaries.
    
    Like teaching a child ABCs before they read books.
    """
    
    def __init__(self):
        self.instruction_encoder = InstructionEncoder()
        self.vulnerability_patterns = VULNERABILITY_PATTERNS
        self.memory_regions = MEMORY_REGIONS
        self.register_info = REGISTER_SEMANTICS
    
    def get_instruction_info(self, mnemonic: str) -> dict:
        """Get full information about an instruction."""
        return INSTRUCTION_SEMANTICS.get(mnemonic, {
            "category": "unknown",
            "risk": 0.5,
            "description": f"Unknown instruction: {mnemonic}"
        })
    
    def assess_risk(self, mnemonic: str, operands: str) -> float:
        """Assess the risk level of an instruction in context."""
        base_risk = INSTRUCTION_SEMANTICS.get(mnemonic, {}).get("risk", 0.5)
        
        # Increase risk for memory operations near stack
        if "rsp" in operands or "rbp" in operands:
            base_risk *= 1.5
        
        # Increase risk for string operations
        if mnemonic in ("rep", "movsb", "stosb", "movsd"):
            base_risk *= 2.0
        
        # Decrease risk for comparison/branch
        if mnemonic in ("cmp", "test", "je", "jne"):
            base_risk *= 0.3
        
        return min(base_risk, 1.0)
    
    def get_pattern_matches(self, trace: list) -> list:
        """Check execution trace against known vulnerability patterns."""
        matches = []
        
        for pattern_name, pattern_info in self.vulnerability_patterns.items():
            for signature in pattern_info["signatures"]:
                if self._check_signature(trace, signature):
                    matches.append({
                        "pattern": pattern_name,
                        "signature": signature,
                        "severity": pattern_info["severity"],
                    })
        
        return matches
    
    def _check_signature(self, trace: list, signature: str) -> bool:
        """Check if a trace matches a vulnerability signature."""
        # Simplified pattern matching
        sig_lower = signature.lower()
        
        for inst in trace:
            if "rep movsb" in sig_lower and inst.mnemonic in ("rep", "movsb"):
                if "rsp" in inst.operands or "rbp" in inst.operands:
                    return True
            
            if "gets()" in sig_lower and inst.mnemonic == "call" and "gets" in inst.operands:
                return True
            
            if "strcpy()" in sig_lower and inst.mnemonic == "call" and "strcpy" in inst.operands:
                return True
            
            if "free()" in sig_lower and inst.mnemonic == "call" and "free" in inst.operands:
                return True
        
        return False
    
    def generate_training_examples(self, num_examples: int = 1000) -> list:
        """
        Generate synthetic training examples for pre-training.
        
        Creates labeled examples of:
        - Normal execution patterns
        - Vulnerable execution patterns
        
        This gives the model a head start before real analysis.
        """
        import random
        
        examples = []
        
        for _ in range(num_examples):
            # Decide if this example is vulnerable
            is_vulnerable = random.random() < 0.3  # 30% vulnerable
            
            if is_vulnerable:
                trace = self._generate_vulnerable_trace()
                label = 1
            else:
                trace = self._generate_normal_trace()
                label = 0
            
            examples.append({
                "trace": trace,
                "label": label,
                "vulnerability_type": trace.get("vuln_type", "none"),
            })
        
        return examples
    
    def _generate_normal_trace(self) -> dict:
        """Generate a normal (non-vulnerable) execution trace."""
        return {
            "instructions": [
                {"mnemonic": "push", "operands": "rbp", "risk": 0.1},
                {"mnemonic": "mov", "operands": "rbp, rsp", "risk": 0.1},
                {"mnemonic": "sub", "operands": "rsp, 0x20", "risk": 0.1},
                {"mnemonic": "mov", "operands": "dword [rbp-0x14], edi", "risk": 0.15},
                {"mnemonic": "mov", "operands": "dword [rbp-0x8], 0", "risk": 0.1},
                {"mnemonic": "cmp", "operands": "dword [rbp-0x8], 10", "risk": 0.0},
                {"mnemonic": "jge", "operands": ".end", "risk": 0.05},
                {"mnemonic": "add", "operands": "dword [rbp-0x8], 1", "risk": 0.1},
                {"mnemonic": "jmp", "operands": ".loop", "risk": 0.05},
                {"mnemonic": ".end:", "operands": "", "risk": 0.0},
                {"mnemonic": "mov", "operands": "eax, dword [rbp-0x8]", "risk": 0.1},
                {"mnemonic": "add", "operands": "rsp, 0x20", "risk": 0.1},
                {"mnemonic": "pop", "operands": "rbp", "risk": 0.1},
                {"mnemonic": "ret", "operands": "", "risk": 0.15},
            ],
            "vuln_type": "none",
        }
    
    def _generate_vulnerable_trace(self) -> dict:
        """Generate a vulnerable execution trace."""
        vulns = ["stack_overflow", "format_string", "integer_overflow", "use_after_free"]
        vuln_type = random.choice(vulns)
        
        if vuln_type == "stack_overflow":
            return {
                "instructions": [
                    {"mnemonic": "push", "operands": "rbp", "risk": 0.1},
                    {"mnemonic": "mov", "operands": "rbp, rsp", "risk": 0.1},
                    {"mnemonic": "sub", "operands": "rsp, 0x40", "risk": 0.1},
                    {"mnemonic": "mov", "operands": "edi, dword [rbp-0x34]", "risk": 0.15},
                    {"mnemonic": "lea", "operands": "rax, [rbp-0x30]", "risk": 0.05},
                    {"mnemonic": "mov", "operands": "esi, eax", "risk": 0.1},
                    {"mnemonic": "call", "operands": "gets", "risk": 0.9},  # DANGEROUS
                    {"mnemonic": "nop", "operands": "", "risk": 0.0},
                    {"mnemonic": "leave", "operands": "", "risk": 0.05},
                    {"mnemonic": "ret", "operands": "", "risk": 0.15},
                ],
                "vuln_type": "stack_overflow",
            }
        elif vuln_type == "format_string":
            return {
                "instructions": [
                    {"mnemonic": "push", "operands": "rbp", "risk": 0.1},
                    {"mnemonic": "mov", "operands": "rbp, rsp", "risk": 0.1},
                    {"mnemonic": "sub", "operands": "rsp, 0x10", "risk": 0.1},
                    {"mnemonic": "mov", "operands": "dword [rbp-0x8], edi", "risk": 0.15},
                    {"mnemonic": "mov", "operands": "eax, dword [rbp-0x8]", "risk": 0.1},
                    {"mnemonic": "mov", "operands": "esi, eax", "risk": 0.1},
                    {"mnemonic": "lea", "operands": "rdi, [rbp-0x4]", "risk": 0.05},
                    {"mnemonic": "call", "operands": "printf", "risk": 0.8},  # DANGEROUS
                    {"mnemonic": "nop", "operands": "", "risk": 0.0},
                    {"mnemonic": "leave", "operands": "", "risk": 0.05},
                    {"mnemonic": "ret", "operands": "", "risk": 0.15},
                ],
                "vuln_type": "format_string",
            }
        else:
            return {
                "instructions": [
                    {"mnemonic": "push", "operands": "rbp", "risk": 0.1},
                    {"mnemonic": "mov", "operands": "rbp, rsp", "risk": 0.1},
                    {"mnemonic": "imul", "operands": "edi, esi", "risk": 0.4},  # DANGEROUS
                    {"mnemonic": "cdqe", "operands": "", "risk": 0.15},
                    {"mnemonic": "mov", "operands": "edi, eax", "risk": 0.1},
                    {"mnemonic": "call", "operands": "malloc", "risk": 0.15},
                    {"mnemonic": "leave", "operands": "", "risk": 0.05},
                    {"mnemonic": "ret", "operands": "", "risk": 0.15},
                ],
                "vuln_type": "integer_overflow",
            }
