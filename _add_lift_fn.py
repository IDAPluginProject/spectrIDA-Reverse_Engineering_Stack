"""Add lift_function to lift.py."""
import pathlib

p = pathlib.Path(r"C:\Users\Administrator\Desktop\scrape\mini-mythos\spectrIDA\spectrida\verify\lift.py")
content = p.read_text(encoding="utf-8")

NL = chr(10)
LT = "<"
GT = ">"

part3 = '''
LIFT_SYSTEM = "You are an expert C programmer. Given Hex-Rays pseudocode, produce compilable C. Use GCC-compatible types. No placeholders. No external calls. Just the function."

LIFT_PROMPT_TEMPLATE = "Convert this pseudocode to compilable C.\\n\\nCallee prototypes:\\n{callee_prototypes}\\n\\n{struct_context}\\n```c\\n{pseudocode}\\n```\\n\\nReturn ONLY the C code."

async def _query_model(http, ollama_url, system, user_msg, *, ollama_model=""):
    prompt = "''' + LT + 'im' + GT + '''system\\n''' + system + '''