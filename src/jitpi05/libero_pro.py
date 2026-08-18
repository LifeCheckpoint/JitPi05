"""LIBERO-Pro 扰动 suite 动态注册。

LIBERO-Pro（arXiv:2510.03827）在原始 LIBERO 之上引入 object / position /
semantic / task / environment 五个维度的扰动。其中 object（``_object``）、
position（``_swap``）、semantic（``_lan``）、task（``_task``）四维有独立
bddl/init 文件；environment 维度通过替换场景实现，无独立 bddl/init 目录，
本模块暂不注册 env 变体。

集成策略：在进程启动时把扰动 suite 动态注册进 libero benchmark 的
``BENCHMARK_MAPPING`` 与 ``task_maps``，避免覆盖 venv 内 libero 库的原始文件。
扰动 bddl/init 文件通过软链接放置在 libero 包目录的 ``bddl_files/`` 与
``init_files/`` 下（见 README 迁移说明）。
"""

from __future__ import annotations

from libero.libero import benchmark as _benchmark
from libero.libero.benchmark import (
    Benchmark,
    Task,
    grab_language_from_filename,
    register_benchmark,
)
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

# 扰动维度 → suite 后缀。
PERTURBATION_SUFFIXES: dict[str, str] = {
    "object": "_object",  # object 扰动：外观/颜色/尺度
    "swap": "_swap",  # position 扰动：物体位置互换/平移
    "lan": "_lan",  # semantic 扰动：语言指令改写
    "task": "_task",  # task 扰动：任务逻辑重定义
}

# 支持扰动的原始 suite。
BASE_SUITES: tuple[str, ...] = (
    "libero_10",
    "libero_goal",
    "libero_object",
    "libero_spatial",
)

_registered = False


def _register_suite_class(suite: str) -> None:
    """注册一个与 LIBERO-Pro 替换文件等价的 benchmark 类。"""

    def __init__(self, task_order_index: int = 0) -> None:
        Benchmark.__init__(self, task_order_index=task_order_index)
        self.name = suite
        self._make_benchmark()

    cls = type(suite.upper(), (Benchmark,), {"__init__": __init__})
    register_benchmark(cls)


def register_libero_pro_suites() -> None:
    """把 LIBERO-Pro 扰动 suite 注册进 libero benchmark（幂等）。"""
    global _registered
    if _registered:
        return

    for base in BASE_SUITES:
        for suffix in PERTURBATION_SUFFIXES.values():
            suite = f"{base}{suffix}"
            if suite in _benchmark.BENCHMARK_MAPPING:
                continue

            # 扰动 suite 的任务名与原始 suite 完全一致（problem_folder 不同）。
            if suite not in libero_task_map:
                libero_task_map[suite] = list(libero_task_map[base])

            _benchmark.task_maps[suite] = {}
            for task in libero_task_map[suite]:
                language = grab_language_from_filename(task + ".bddl")
                _benchmark.task_maps[suite][task] = Task(
                    name=task,
                    language=language,
                    problem="Libero",
                    problem_folder=suite,
                    bddl_file=f"{task}.bddl",
                    init_states_file=f"{task}.pruned_init",
                )
            _register_suite_class(suite)

    _registered = True


def libero_pro_task_specs(
    base_suite: str = "libero_10",
    max_steps: int = 520,
) -> tuple[dict, ...]:
    """生成 LIBERO-Pro 面板的任务描述（与 config 的 JITRL_TASKS 同构）。

    每个扰动维度作为一个独立 suite（例如 ``libero_10_object``），任务列表
    与原始 ``libero_10`` 一致。
    """
    from libero.libero.benchmark import libero_task_map as task_map  # noqa: F401

    base_tasks = list(libero_task_map[base_suite])
    specs: list[dict] = []
    for dim, suffix in PERTURBATION_SUFFIXES.items():
        suite = f"{base_suite}{suffix}"
        for task_id, task_name in enumerate(base_tasks):
            specs.append(
                {
                    "name": f"{suite}_task{task_id}",
                    "suite": suite,
                    "task_id": task_id,
                    "description": "",  # 运行时由 env.call("task_description") 提供
                    "max_steps": max_steps,
                    "zero_shot": True,
                    "perturbation": dim,
                }
            )
    return tuple(specs)
