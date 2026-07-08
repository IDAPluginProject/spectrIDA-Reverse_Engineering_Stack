"""Self-verifying decompilation loop."""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field

from spectrida.verify.oracle import (
    OracleVerdict, compile_c_to_shared,
    emulate_function, compare_emulations, extract_function_bytes,
)

NL = chr(10)
_LT = "<"
_GT = ">"


@dataclass
class LiftAttempt:
    attempt_num: int = 0
    c_code: str = ""
    compiled: bool = False
    compile_error: str = ""
    verified: bool = False
    verdict: OracleVerdict | None = None


@dataclass
class LiftResult:
    func_name: str = ""
    verified_c: str = ""
    attempts: list = field(default_factory=list)
    total_attempts: int = 0
    success: bool = False


def _extract_struct_context(pseudocode: str) -> str:
    type_offsets = {}
    for m in re.finditer(r"\*\(\((\w+)\s*\*\)\s*\((\w+)\s*\+\s*(\d+)\)", pseudocode):
        field_type = m.group(1)
        offset = int(m.group(3))
        if field_type not in type_offsets:
            type_offsets[field_type] = []
        type_offsets[field_type].append(offset)
    structs = []
    for ft, offsets in type_offsets.items():
        offsets.sort()
        fields = [f"    {ft} field_{o};" for o in offsets]
        structs.append("typedef struct {" + NL + NL.join(fields) + NL + "}")
    return NL.join(structs) if structs else ""


def _normalize_types(code: str) -> str:
    code = re.sub(r"\w+::", "", code)
    code = code.replace("this", "self")
    code = re.sub(r"\(void\* (\w+)\)", r"(ThisStruct* \1)", code)
    reps = [
        ("__int64", "long long"), ("__int32", "int"), ("__int16", "short"),
        ("__int8", "char"), ("_BYTE", "unsigned char"), ("_WORD", "unsigned short"),
        ("_DWORD", "unsigned int"), ("_QWORD", "unsigned long long"),
        ("__fastcall", ""), ("__cdecl", ""),
        ("int64_t", "long long"), ("uint64_t", "unsigned long long"),
        ("int32_t", "int"), ("uint32_t", "unsigned int"),
        ("int16_t", "short"), ("uint16_t", "unsigned short"),
        ("int8_t", "char"), ("uint8_t", "unsigned char"),
        ("size_t", "unsigned long long"),
    ]
    for old, new in reps:
        code = code.replace(old, new)
    known = {"int", "long", "char", "void", "float", "double", "unsigned",
             "signed", "short", "struct", "union", "enum", "const", "static",
             "extern", "volatile", "inline", "register", "auto", "typedef"}
    def fix(m):
        tn = m.group(1)
        if tn.lower() in known or tn.startswith("uint") or tn.startswith("int"):
            return m.group(0)
        return "ThisStruct* " + m.group(2)
    code = re.sub(r"(\w+)\s*\*\s*(\w+)", fix, code)
    code = re.sub(r"&?off_[0-9a-fA-F]+", "0", code)
    code = re.sub(r"&?unk_[0-9a-fA-F]+", "0", code)
    sd = "typedef struct { unsigned long long vtable, m_archive, m_name, m_fileSize, m_dataSize, m_offset, m_entryIndex, m_childCount, m_childList0, m_childList1, m_childList2; unsigned int m_flags, m_entryType; } ThisStruct;" + NL + NL
    code = sd + code
    return code


LIFT_SYSTEM = "You are an expert C programmer. Given Hex-Rays pseudocode, produce compilable C. Use GCC-compatible types. No placeholders. No external calls. Just the function."

LIFT_PROMPT_TEMPLATE = "Convert this pseudocode to compilable C.\n\nCallee prototypes:\n{callee_prototypes}\n\n{struct_context}\n```c\n{pseudocode}\n```\n\nReturn ONLY the C code."


async def _query_model(http, ollama_url, system, user_msg, *, ollama_model=""):
    prompt = _LT + "im" + _GT + "system" + _LT + "/im" + _GT + NL + system + _LT + "/im" + _GT + NL + _LT + "im" + _GT + "user" + _LT + "/im" + _GT + NL + user_msg + _LT + "/im" + _GT + NL + _LT + "im" + _GT + "assistant" + _LT + "/im" + _GT + NL
    if ollama_model:
        url = ollama_url.rstrip("/") + "/api/generate"
        payload = {"model": ollama_model, "prompt": prompt, "stream": False, "think": False,
                   "options": {"temperature": 0.3, "num_predict": 1024}}
        resp = await http.post(url, json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    else:
        payload = {"prompt": prompt, "temperature": 0.3, "n_predict": 1024}
        resp = await http.post(ollama_url, json=payload)
        resp.raise_for_status()
        return resp.json().get("content", "").strip()


def _extract_c_code(text: str) -> str:
    text = re.sub(r"```c?\s*\n?", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    lines = text.split(NL)
    start = 0
    for i, line in enumerate(lines):
        if re.match(r"^\s*(typedef|struct|static|void|int|long|char|unsigned|float|double|__int|const|inline)", line):
            start = i
            break
    return NL.join(lines[start:]).strip()


async def lift_function(
    pseudocode,
    original_bytes,
    func_name="",
    *,
    http=None,
    ollama_url="",
    ollama_model="",
    max_attempts=3,
    args=None,
):
    import httpx as _httpx
    if http is None:
        http = _httpx.AsyncClient(timeout=120)
        own_client = True
    else:
        own_client = False

    result = LiftResult(func_name=func_name)

    try:
        for attempt_num in range(1, max_attempts + 1):
            attempt = LiftAttempt(attempt_num=attempt_num)
            struct_ctx = _extract_struct_context(pseudocode)

            callees = set()
            for m in re.finditer(r"([A-Z]\w*(?:::[A-Z]\w*)*)\s*\(", pseudocode):
                callees.add(m.group(1))
            callee_protos = []
            for c in sorted(callees):
                callee_protos.append(f"long long {c.replace('::', '_')}(long long* args);")
            callee_protos_str = NL.join(callee_protos)

            user_msg = LIFT_PROMPT_TEMPLATE.format(
                callee_prototypes=callee_protos_str,
                struct_context=struct_ctx,
                pseudocode=pseudocode,
            )

            try:
                raw_response = await _query_model(http, ollama_url, LIFT_SYSTEM, user_msg, ollama_model=ollama_model)
                c_code = _normalize_types(_extract_c_code(raw_response))
                attempt.c_code = c_code
            except Exception as e:
                attempt.compile_error = f"model query failed: {e}"
                result.attempts.append(attempt)
                continue

            if not c_code:
                attempt.compile_error = "model returned empty C code"
                result.attempts.append(attempt)
                continue

            import tempfile as _tf
            fd, _dll = _tf.mkstemp(suffix=".dll")
            os.close(fd)
            compile_result = compile_c_to_shared(c_code, _dll)
            if not compile_result["ok"]:
                attempt.compile_error = compile_result["error"][:300]
                result.attempts.append(attempt)
                continue

            attempt.compiled = True

            extract_result = extract_function_bytes(_dll, func_name)
            if not extract_result["ok"]:
                attempt.compile_error = f"extract failed: {extract_result['error']}"
                result.attempts.append(attempt)
                continue

            recompiled_bytes = bytes.fromhex(extract_result["bytes"])
            orig_result = emulate_function(original_bytes, args=args or [0, 0, 0, 0])
            recomp_result = emulate_function(recompiled_bytes, args=args or [0, 0, 0, 0])
            verdict = compare_emulations(orig_result, recomp_result)
            attempt.verdict = verdict

            if verdict.equivalent:
                attempt.verified = True
                result.verified_c = c_code
                result.final_verdict = verdict
                result.success = True
                result.attempts.append(attempt)
                break
            else:
                result.attempts.append(attempt)

    finally:
        if own_client:
            await http.aclose()

    result.total_attempts = len(result.attempts)
    return result
