import os
import yaml
import argparse

from uot.experiments.runner import run_pipeline
from uot.utils.yaml_helpers import load_solvers, load_problems, load_experiment
from uot.utils.problems_loaders import load_problems_from_dir
from uot.problems.hdf5_store import HDF5ProblemStore
from uot.problems.iterator import ProblemIterator, OnlineProblemIterator
from uot.utils.exceptions import InvalidConfigurationException
from uot.problems.problem_serializer import _resolve_references
from uot.utils.logging import setup_logger

logger = setup_logger(__name__)
logger.propagate = False


def get_problem_iterators(
        dataset: str | None, config
) -> list[ProblemIterator]:
    if dataset:
        if dataset.endswith(".h5"):
            store = HDF5ProblemStore(dataset)
            return [ProblemIterator(store)]
        return load_problems_from_dir(dataset)
    return load_problems(config=config)


def add_online_generators(generator_yaml: str) -> list[ProblemIterator]:
    with open(generator_yaml, encoding='utf8') as f:
        raw_cfg = yaml.safe_load(f)

    generators = raw_cfg.get('generators', {})
    generator_iters = []

    for name, gen_cfg in generators.items():
        # skip hidden or anchor defaults
        if name.startswith('_'):
            continue

        # resolve imports and parsing
        cfg = {'name': name, **gen_cfg}
        cfg = _resolve_references(cfg)

        generator_cls = cfg.pop('generator')
        cache_gt = cfg.pop('cache_gt', False)
        generator = generator_cls(**cfg)
        num_datasets = generator._num_datasets
        generator_iters.append(
            OnlineProblemIterator(generator, num_datasets, cache_gt)
        )
    return generator_iters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run time precision experiments.")

    parser.add_argument(
        "--save",
        help="Flag to save the results to a CSV file."
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=1,
        help="Number of folds for cross-validation or repeated experiments."
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to a configuration file with experiment parameters."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to the dataset directory containing sets of problems\
        as generated by the serialization utility."
    )

    parser.add_argument(
        "--generators-config",
        type=str,
        help="Path to the yaml configuration file of the generators\
        configuration that will be added to the dataset online (generated\
        as we run the experiment)."
    )

    parser.add_argument(
        "--export",
        type=str,
        default="gaussian_toy_results.csv",
        help="Path to export the results CSV file."
    )

    parser.add_argument(
        "--progress",
        type=bool,
        default=False,
        help="Show progress bar."
    )

    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    try:
        experiment = load_experiment(config=config)
        solver_configs = load_solvers(config=config)
        problems_iterators = get_problem_iterators(args.dataset, config)
        if args.generators_config:
            problems_iterators += add_online_generators(args.generators_config)
    except InvalidConfigurationException as ex:
        logger.exception(ex)
        exit(1)

    df = run_pipeline(
        experiment=experiment,
        solvers=solver_configs,
        iterators=problems_iterators,
        folds=args.folds,
        progress=args.progress,
    )

    logger.info(f"Exporting results to {args.export}")
    os.makedirs(os.path.dirname(args.export), exist_ok=True)
    df.to_csv(args.export, index=False)


if __name__ == "__main__":
    main()
