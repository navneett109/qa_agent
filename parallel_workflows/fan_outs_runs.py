
from state.state import State
from langgraph.types import Send


DIFFICULTY_CYCLE = ("easy", "medium", "hard")


def _difficulty_for_run(run_id: int) -> str:
    return DIFFICULTY_CYCLE[run_id % len(DIFFICULTY_CYCLE)]


def fan_out_runs(state: State):
    workflows_count = int(state.get("parallel_workflows", 3) or 3)
    return [
        Send("generate_question", {
            **state,
            "run_id": i,
            "difficulty": _difficulty_for_run(i),
        })
        for i in range(workflows_count)
    ]