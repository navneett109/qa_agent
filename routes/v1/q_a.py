import os
import json
import time
import re
from threading import Lock

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from schemas.initial_state import InitialState

# graph
from graph.graph import graph

# DB
from db.db import Session as DbSession
from models.models import UserSettings

router = APIRouter(
    prefix="/v1/generate",
    tags=["generate"],
)

# Thread-safe rate limit store
LAST_COMPLETED_TIME = {}
lock = Lock()
active_streams_lock = Lock()
ACTIVE_STREAMS = 0
PEAK_ACTIVE_STREAMS = 0
TOTAL_STREAM_REQUESTS = 0
TOTAL_QUESTIONS_GENERATED = 0
REJECTED_BUSY = 0
REJECTED_RATE_LIMIT = 0
REJECTED_WORKLOAD = 0
ACTIVE_USER_IDS = set()

# Configurable runtime controls
COOLDOWN = int(os.getenv("COOLDOWN_SECONDS", "30"))
MAX_PARALLEL_WORKFLOWS = int(os.getenv("MAX_PARALLEL_WORKFLOWS", "10"))
MAX_TOTAL_QUESTIONS = int(os.getenv("MAX_TOTAL_QUESTIONS", "100"))
MAX_CONCURRENT_STREAMS = int(os.getenv("MAX_CONCURRENT_STREAMS", "5"))


@router.get("/metrics")
def stream_metrics():
    return {
        "active_streams": ACTIVE_STREAMS,
        "current_active_users": len(ACTIVE_USER_IDS),
        "peak_active_streams": PEAK_ACTIVE_STREAMS,
        "total_stream_requests": TOTAL_STREAM_REQUESTS,
        "total_questions_generated": TOTAL_QUESTIONS_GENERATED,
        "rejected_busy": REJECTED_BUSY,
        "rejected_rate_limit": REJECTED_RATE_LIMIT,
        "rejected_workload": REJECTED_WORKLOAD,
        "config": {
            "cooldown_seconds": COOLDOWN,
            "max_parallel_workflows": MAX_PARALLEL_WORKFLOWS,
            "max_total_questions": MAX_TOTAL_QUESTIONS,
            "max_concurrent_streams": MAX_CONCURRENT_STREAMS,
        },
    }


@router.post("/stream")
def generate_stream(state: InitialState, request: Request):
    global ACTIVE_STREAMS, PEAK_ACTIVE_STREAMS, TOTAL_STREAM_REQUESTS
    global REJECTED_BUSY, REJECTED_RATE_LIMIT, REJECTED_WORKLOAD
    global TOTAL_QUESTIONS_GENERATED

    with lock:
        TOTAL_STREAM_REQUESTS += 1

    # ✅ Use user_id instead of IP
    user_id = state.user_id
    selected_bloom = (state.bloom_level or "").strip()

    def bloom_code(value: str) -> str:
        match = re.search(r"l\s*([1-7])", str(value or "").lower())
        return f"l{match.group(1)}" if match else ""

    selected_bloom_code = bloom_code(selected_bloom)

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    if state.parallel_workflows > MAX_PARALLEL_WORKFLOWS:
        with lock:
            REJECTED_WORKLOAD += 1
        raise HTTPException(
            status_code=400,
            detail=(
                f"parallel_workflows exceeds limit. Requested={state.parallel_workflows}, "
                f"max_allowed={MAX_PARALLEL_WORKFLOWS}."
            ),
        )

    if state.parallel_workflows % 3 != 0:
        with lock:
            REJECTED_WORKLOAD += 1
        raise HTTPException(
            status_code=400,
            detail=(
                "parallel_workflows must be a multiple of 3 "
                "for balanced easy/medium/hard generation."
            ),
        )

    requested_questions = state.N * state.parallel_workflows
    if requested_questions > MAX_TOTAL_QUESTIONS:
        with lock:
            REJECTED_WORKLOAD += 1
        raise HTTPException(
            status_code=400,
            detail=(
                f"Request too large for stable concurrency. Requested total questions={requested_questions}, "
                f"max_allowed={MAX_TOTAL_QUESTIONS}."
            ),
        )

    # ✅ Thread-safe rate limiting
    with lock:
        if user_id in LAST_COMPLETED_TIME:
            elapsed = time.time() - LAST_COMPLETED_TIME[user_id]
            if elapsed < COOLDOWN:
                REJECTED_RATE_LIMIT += 1
                raise HTTPException(
                    status_code=429,
                    detail=f"Please wait {int(COOLDOWN - elapsed)} seconds before your next request.",
                )

    # ✅ Per-worker admission control for predictable concurrency
    with active_streams_lock:
        if ACTIVE_STREAMS >= MAX_CONCURRENT_STREAMS:
            REJECTED_BUSY += 1
            raise HTTPException(
                status_code=503,
                detail="Server is busy. Please retry in a few seconds.",
            )
        ACTIVE_STREAMS += 1
        ACTIVE_USER_IDS.add(user_id)
        PEAK_ACTIVE_STREAMS = max(PEAK_ACTIVE_STREAMS, ACTIVE_STREAMS)

    def event_stream():
        global TOTAL_QUESTIONS_GENERATED
        try:
            input_state = state.model_dump()

            # ✅ DB session
            db_session = DbSession()
            try:
                settings = db_session.query(UserSettings).filter(
                    UserSettings.user_id == user_id
                ).first()

                if not settings or not settings.gemini_api_key or not settings.groq_api_key or not settings.huggingfacehub_api_token:
                    raise HTTPException(
                        status_code=400,
                        detail="API keys are not configured. Please set them in Settings.",
                    )

                input_state["api_keys"] = {
                    "gemini": settings.gemini_api_key or "",
                    "gemini_2": settings.gemini_api_key_2 or settings.gemini_api_key or "",
                    "groq": settings.groq_api_key or "",
                    "groq_2": settings.groq_api_key_2 or settings.groq_api_key or "",
                    "hf": settings.huggingfacehub_api_token or "",
                    "db_uri": settings.database_uri or "",
                }

            finally:
                db_session.close()

            yield json.dumps({"step": "started"}) + "\n"

            final_questions = []
            final_answers = []
            duplicate_results = []
            discarded_db_qids = set()

            # 🚀 MAIN PIPELINE
            for chunk in graph.stream(input_state, stream_mode="updates"):

                # QUESTIONS
                gen_q = chunk.get("generate_question")
                if gen_q and isinstance(gen_q, dict) and "questions" in gen_q:
                    # In LangGraph stream updates mode, each parallel branch yields its own update.
                    # ONLY extend if this branch actually returned generated questions.
                    new_qs = gen_q["questions"] or []
                    # Filter out any questions we already added (e.g. from state accumulation)
                    existing_qs = {q.id for q in final_questions}
                    unique_new_qs = [q for q in new_qs if q.id not in existing_qs]
                    final_questions.extend(unique_new_qs)

                # DUPLICATES
                check_d = chunk.get("check_duplicates")
                if check_d and isinstance(check_d, dict):
                    duplicate_results.extend(check_d.get("duplicate_results", []))

                # CROSS DEDUP
                cross_d = chunk.get("cross_batch_deduplicate")
                if cross_d and isinstance(cross_d, dict):
                    final_questions = cross_d.get("questions", final_questions)

                # ANSWERS
                gen_a = chunk.get("generate_answer")
                if gen_a and isinstance(gen_a, dict):
                    ans_toadd = []
                    if gen_a.get("answers"):
                        ans_toadd = gen_a["answers"]
                    elif gen_a.get("answer") and getattr(gen_a["answer"], "answers", None):
                        ans_toadd = gen_a["answer"].answers
                    
                    if ans_toadd:
                        existing_ans = {a.question_id for a in final_answers}
                        unique_ans = [a for a in ans_toadd if a.question_id not in existing_ans]
                        final_answers.extend(unique_ans)

                # MEMORY SAVE (Catches exact Postgres DB duplicates)
                mem = chunk.get("create_memory")
                if mem and isinstance(mem, dict):
                    dropped = mem.get("discarded_db_qids", [])
                    discarded_db_qids.update(dropped)

                # Send node name for the frontend progress bar
                node_name = list(chunk.keys())[0] if isinstance(chunk, dict) and chunk else ""
                yield json.dumps({"step": "progress", "data": f"{{'{node_name}': '...'}}"}) + "\n"

                # Emit partial results incrementally
                if node_name in ["generate_answer", "check_duplicates"]:
                    dup_map_temp = {item["question_id"]: item["result"] for item in duplicate_results}
                    temp_filtered = [
                        q for q in final_questions
                        if not dup_map_temp.get(q.id) or not getattr(dup_map_temp.get(q.id), "exists", False)
                    ]
                    if selected_bloom_code:
                        temp_filtered = [q for q in temp_filtered if bloom_code(getattr(q, "bloom_level", "")) == selected_bloom_code]
                    
                    ans_map_temp = {a.question_id: a for a in final_answers}
                    temp_results = []
                    for idx, q in enumerate(temp_filtered, start=1):
                        ans = ans_map_temp.get(q.id)
                        temp_results.append({
                            "id": idx,
                            "question_text": str(getattr(q, "question_text", "")),
                            "difficulty": str(getattr(q, "difficulty", state.difficulty)),
                            "bloom_level": str(selected_bloom or getattr(q, "bloom_level", state.bloom_level)),
                            "topic_tags": getattr(q, "topic_tags", []),
                            "options": getattr(q, "options", {}),
                            "correct_option": str(getattr(q, "correct_option", "")),
                            "answer": str(getattr(ans, "answer_text", "Generating answer...")),
                        })
                    if temp_results:
                        yield json.dumps({"step": "partial_result", "result": temp_results}) + "\n"

            # ✅ Deduplication
            dup_map = {
                item["question_id"]: item["result"] for item in duplicate_results
            }

            filtered_questions = [
                q for q in final_questions
                if not dup_map.get(q.id) or not getattr(dup_map.get(q.id), "exists", False)
            ]

            # Enforce selected bloom level when client requested single-bloom mode.
            if selected_bloom_code:
                filtered_questions = [
                    q for q in filtered_questions
                    if bloom_code(getattr(q, "bloom_level", "")) == selected_bloom_code
                ]

            # Filter out DB-layer exact matching duplicates
            filtered_questions = [q for q in filtered_questions if q.id not in discarded_db_qids]

            # ✅ Final formatting
            result_list = []
            ans_map = {a.question_id: a for a in final_answers}

            for idx, q in enumerate(filtered_questions, start=1):
                ans = ans_map.get(q.id)
                result_list.append({
                    "id": idx,
                    "question_text": getattr(q, "question_text", ""),
                    "difficulty": getattr(q, "difficulty", state.difficulty),
                    "bloom_level": selected_bloom or getattr(q, "bloom_level", state.bloom_level),
                    "topic_tags": getattr(q, "topic_tags", []),
                    "options": getattr(q, "options", {}),
                    "correct_option": getattr(q, "correct_option", ""),
                    "answer": getattr(ans, "answer_text", "No answer generated."),
                })

            if result_list:
                with lock:
                    TOTAL_QUESTIONS_GENERATED += len(result_list)
                yield json.dumps({"step": "result", "result": result_list}) + "\n"

            # ✅ Update rate limit safely
            with lock:
                LAST_COMPLETED_TIME[user_id] = time.time()

            yield json.dumps({"step": "completed"}) + "\n"

        except Exception as e:
            yield json.dumps({"step": "error", "message": str(e)}) + "\n"
        finally:
            global ACTIVE_STREAMS
            with active_streams_lock:
                ACTIVE_STREAMS = max(0, ACTIVE_STREAMS - 1)
                ACTIVE_USER_IDS.discard(user_id)

    return StreamingResponse(event_stream(), media_type="text/plain")