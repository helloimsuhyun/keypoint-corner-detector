"""Fetch pinned author code and build only the headless dataset adapters."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "ahmed": ("https://github.com/Merium88/Point-cloud-Edge-and-Corner-Detection.git",
              "e8aa7a4254cf67c3bdd6a82a7efb67d51c47994c"),
    "orthogonal": ("https://github.com/c-sommer/orthogonal-planes.git",
                   "1a05583ce7913308c5a523f90bd34fa6608c8864"),
}


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"Upstream source changed; expected exactly one {old!r}")
    return source.replace(old, new, 1)


def adapt_ahmed(repo: Path, target: Path) -> None:
    """Keep every numerical statement, expose all ten L outputs, omit plots."""
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo / "edgedetection.m", target / "edgedetection.m")
    source = (repo / "corner_detection.m").read_text()
    source = replace_once(source, "function corner_detection(", "function predictions = corner_detection(")
    source = replace_once(source, "count = 1;num=1;", "count = 1;num=1;\npredictions = cell(10, 1);")
    start = source.index("\nfigure\nCloud = pointCloud(cloud);")
    if source[start:].strip().splitlines()[-2:] != ["end", "end"]:
        raise ValueError("Unexpected Ahmed plotting/function tail")
    source = source[:start] + "\npredictions{L} = E;\nend\nend\n"
    (target / "corner_detection.m").write_text(source)


def adapt_orthogonal(repo: Path, target: Path) -> None:
    """Author's main: replace viewer with CSV output; retain all stage calls."""
    source = (repo / "ply_corners/src/main.cpp").read_text()
    source = replace_once(source, '#include "visualize/pcshow.h"', '#include <iomanip>\n#include <cstdlib>')
    source = replace_once(source, "std::string ply_file;", "std::string ply_file, output_file;\n    unsigned seed = 1;")
    source = replace_once(source, '    app.add_option("--min-votes",',
                          '    app.add_option("--output", output_file)->required();\n'
                          '    app.add_option("--seed", seed);\n'
                          '    app.add_option("--min-votes",')
    source = replace_once(source, 'std::string filepath = "../data/";', 'std::string filepath = "";')
    source = replace_once(source, "    pcshow(points);",
                          '    if (points.empty() || points.size() != normals.size() || sampling < 2 ||\n'
                          '        !(d_min > 0 && d_max > d_min) || min_votes < 1) return 2;\n'
                          '    std::ofstream output(output_file);\n'
                          '    if (!output) return 2;\n'
                          '    output << "x_m,y_m,z_m\\n" << std::setprecision(17);\n'
                          '    std::srand(seed);')
    source = replace_once(source, "    pcshow_corners(points, corners);",
                          '    for (const auto& c : corners) {\n'
                          '        auto p = c.position();\n'
                          '        output << p[0] << "," << p[1] << "," << p[2] << "\\n";\n'
                          '    }\n    if (!output) return 2;')
    target.mkdir(parents=True, exist_ok=True)
    (target / "orthogonal_main.cpp").write_text(source)


def checkout(method: str) -> Path:
    url, revision = SOURCES[method]
    repo = ROOT / "vendor" / method
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        subprocess.run(["git", "clone", url, str(repo)], check=True)
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise RuntimeError(f"Refusing to overwrite local upstream edits: {repo}")
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", revision], check=True)
    if method == "orthogonal":
        subprocess.run(["git", "-C", str(repo), "submodule", "update", "--init", "--depth", "1",
                        "3rdParty/Sophus", "3rdParty/CLI11"], check=True)
    return repo


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["all", "harris3d", *SOURCES], default="all")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    methods = ["harris3d", *SOURCES] if args.method == "all" else [args.method]
    try:
        for method in methods:
            if method == "ahmed":
                adapt_ahmed(checkout(method), ROOT / "build" / "ahmed")
                print("Ahmed prepared. Runtime: MATLAB + Statistics and Machine Learning / Computer Vision toolboxes.")
            else:
                build = ROOT / "build" / method
                command = ["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"]
                if method == "orthogonal":
                    repo = checkout(method)
                    adapt_orthogonal(repo, build)
                    command += [f"-DORTHOGONAL_SOURCE={repo}"]
                subprocess.run(command, check=True)
                targets = ["pcl_baseline", "orthogonal_baseline"] if method == "orthogonal" else ["pcl_baseline"]
                subprocess.run(["cmake", "--build", str(build), "--target", *targets,
                                "--parallel", str(args.jobs)], check=True)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Setup failed: {exc}\nSee corner_dataset_tool/baselines/README.md for dependencies.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
