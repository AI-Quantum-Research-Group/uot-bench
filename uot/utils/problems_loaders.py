from pathlib import Path
from uot.problems.iterator import ProblemIterator
from uot.problems.store import ProblemStore


def load_problems_from_dir(path: str) -> list[ProblemIterator]:
    path_obj = Path(path)
    iterators = []
    for problemset_path in path_obj.iterdir():
        problems_store = ProblemStore(str(problemset_path))
        problems_iterator = ProblemIterator(problems_store)
        iterators.append(problems_iterator)
    return iterators
