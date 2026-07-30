from __future__ import annotations

import json

from jitpi05.runtime import inspect_gpu_runtime


def main() -> int:
    try:
        report = inspect_gpu_runtime()
    except Exception as error:
        print(f"JitPi05 GPU runtime check failed: {error}")
        return 1

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
