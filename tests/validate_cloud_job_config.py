import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.cloud.run_job import parse_args


def main():
    args = parse_args(
        [
            "--config",
            "configs/fuchs_w1_quick_local.json",
            "--num-vecs",
            "7",
            "--local-output-dir",
            "outputs/config_validation",
        ]
    )
    assert args.execution_target == "local"
    assert args.recipe == "fuchs-w1"
    assert args.profile == "quick"
    assert args.static is True
    assert args.num_vecs == 7
    assert args.local_output_dir == "outputs/config_validation"
    print("SUCCESS: cloud/local job config loading and CLI overrides work.")


if __name__ == "__main__":
    main()
