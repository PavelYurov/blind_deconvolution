# import sys
# from preflight.config import load_pyproject, resolve_dependencies
# from preflight.checks.python import check_python_version
# from preflight.checks.packages import check_package

# def main():
#     profile = sys.argv[1] if len(sys.argv) > 1 else "base"

#     cfg = load_pyproject()
#     results = []

#     py_spec = cfg["project"]["requires-python"]
#     results.append(check_python_version(py_spec))
#     deps = resolve_dependencies(cfg, profile)
#     for name, spec in deps.items():
#         results.append(check_package(name, spec))

#     for r in results:
#         print(f"{r.name}: {'OK' if r.ok else 'FAIL'}")
#         if not r.ok:
#             print(f"  {r.message}")
#             if r.recommendation:
#                 print(f"  Recommendation: {r.recommendation}")

#     if not all(r.ok for r in results):
#         sys.exit(1)

# if __name__ == "__main__":
#     main()


import sys
from preflight.config import load_pyproject, resolve_profile_dependencies
from preflight.checks.python import check_python_version
from preflight.checks.packages import check_requirement


def run(profile: str) -> int:
    cfg = load_pyproject()
    results = []

    results.append(
        check_python_version(cfg["project"]["requires-python"])
    )

    deps = resolve_profile_dependencies(cfg, profile)
    for req in deps:
        results.append(check_requirement(req))

    for r in results:
        print(f"{r.name}: {'OK' if r.ok else 'FAIL'}")
        if not r.ok:
            print(f"  {r.summary}")
            if r.details:
                print(f"  Details: {r.details}")
            if r.recommendation:
                print(f"  Recommendation: {r.recommendation}")

    return 0 if all(r.ok for r in results) else 1


def main() -> None:
    profile = sys.argv[1] if len(sys.argv) > 1 else "base"

    try:
        exit_code = run(profile)
    except RuntimeError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(2)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()