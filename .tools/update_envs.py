from copy import deepcopy
from pathlib import Path


def _keep_line(line: str, flag: str) -> bool:
    """Return True if line contains flag and does not include a comment.

    Examples:
    >>> assert _keep_line("  - jax  # tests", "tests") is True
    >>> assert _keep_line("name: env", "tests") is True
    >>> assert _keep_line("  - jax  # run", "tests") is False

    """
    return flag in line or "#" not in line


def main():
    lines = Path("environment.yml").read_text().splitlines()

    # create standard testing environments

    test_env = [line for line in lines if _keep_line(line, "tests")]
    test_env.append("      - -e ../../")  # add local installation

    # find index to insert additional dependencies
    _insert_idx = [i for i, line in enumerate(lines) if "dependencies:" in line][0] + 1

    ## linux
    test_env_linux = deepcopy(test_env)
    test_env_linux.insert(_insert_idx, "  - pygmo")
    test_env_linux.insert(_insert_idx, "  - jax")

    ## test environment others
    test_env_others = deepcopy(test_env)
    test_env_others.insert(_insert_idx, "  - cyipopt<=1.2.0")

    ## test environment for pandas version 1
    test_env_pandas = deepcopy(test_env)
    test_env_pandas = [line for line in test_env_pandas if "pandas" not in line]
    test_env_pandas.insert(_insert_idx, "  - pandas<2.0.0")

    # create docs testing environment

    docs_env = [line for line in lines if _keep_line(line, "docs")]
    docs_env.append("      - -e ../../")  # add local installation

    # write environments
    for name, env in zip(
        ["linux", "others", "pandas"],
        [test_env_linux, test_env_others, test_env_pandas],
        strict=False,
    ):
        # Specify newline to avoid wrong line endings on Windows.
        # See: https://stackoverflow.com/a/69869641
        Path(f".tools/envs/testenv-{name}.yml").write_text(
            "\n".join(env) + "\n", newline="\n"
        )


if __name__ == "__main__":
    main()
