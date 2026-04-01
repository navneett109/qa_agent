from state.state import State
from models.questions import QuestionSet
from langchain_core.messages import HumanMessage, SystemMessage
from prompts.question_generation_prompt import SYSTEM_PROMPT, USER_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI


import re

def _normalize_bloom_level(raw: str | None) -> str:
    if not raw:
        return ""
    
    # Extract L1-L7 from formats like "L1 - Recall"
    match = re.search(r"l\s*([1-7])", str(raw).lower())
    if match:
        return f"L{match.group(1)}"

    value = str(raw).strip().upper()
    aliases = {
        "REMEMBER": "L1", "REMEMBERING": "L1", "RECALL": "L1", "L1": "L1",
        "UNDERSTAND": "L2", "UNDERSTANDING": "L2", "L2": "L2",
        "APPLY": "L3", "APPLICATION": "L3", "L3": "L3",
        "ANALYZE": "L4", "ANALYSIS": "L4", "L4": "L4",
        "EVALUATE": "L5", "EVALUATION": "L5", "L5": "L5",
        "CREATE": "L6", "CREATION": "L6", "L6": "L6",
        "INNOVATE": "L7", "INNOVATION": "L7", "L7": "L7",
        "MIXED": "",
    }
    return aliases.get(value, "")

def generate_question(state: State):
    print(f"Generating questions for run_id {state.get('run_id', 'unknown')}...")

    selected_bloom = _normalize_bloom_level(state.get("bloom_level"))
    bloom_for_prompt = selected_bloom or "mixed"
    mode = "single_bloom" if selected_bloom else "mixed_bloom"

    api_key = state.get("api_keys", {}).get("gemini", "")
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", api_key=api_key)
    llm_set = llm.with_structured_output(QuestionSet)

    result = llm_set.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(N=state["N"])),
            HumanMessage(
                content=USER_PROMPT.format(
                    subject=state["subject"],
                    subject_description=state.get("subject_description", "N/A"),
                    difficulty=state["difficulty"],
                    mode=mode,
                    bloom_level=bloom_for_prompt,
                    true_or_false="true",
                    N=state["N"],
                    run_id=state.get("run_id", 0),
                )
            ),
        ]
    )

    run_id = int(state.get("run_id") or 0)
    per_run_n = int(state.get("N") or len(result.questions) or 1)
    base_id = run_id * per_run_n

    for i, q in enumerate(result.questions):
        # Keep IDs globally unique across parallel runs to prevent collisions.
        q.id = base_id + i + 1
        if selected_bloom:
            q.bloom_level = state.get("bloom_level") or selected_bloom

    return {"questions": result.questions, "duplicate_results": []}
