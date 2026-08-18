# JitPi05

一个尽量简洁的分层 VLA 研究原型：

- **Qwen3.5-4B**：读取 LIBERO 双相机首帧，生成原有离线入口的高层 subtask。
- **显式 logits 调制**：逐 token 保存原始/调制后 logits，并在 argmax 前暴露可编辑回调。
- **LeRobot π₀.₅ LIBERO**：把总任务与 subtask 组成低层语言条件，显式执行 prefix 编码、KV cache、10 步 flow-matching 去噪和反归一化。
- **四组对照**：原始任务、人工 subtask、Qwen subtask、调制后的 Qwen subtask。
- **JitRL 第一阶段**：Qwen3.5-2B 基于固定九类语义动作工作集完成状态抽象、参数绑定和基础策略 logits；任务内在线记忆只估计当前 Qwen 动作的 advantage。Gemini 仅在 episode 结束后提供视觉 step reward。

代码采用可安装的 `src/jitpi05` 包结构：

- [`config.py`](src/jitpi05/config.py)：模型、数据集和实验常量。
- [`datasets.py`](src/jitpi05/datasets.py)：LIBERO episode 读取与图像转换。
- [`planning.py`](src/jitpi05/planning.py)：Qwen3.5 高层规划与 logits 调制。
- [`policy.py`](src/jitpi05/policy.py)：显式 π₀.₅ 流水线与动作指标。
- [`simulation.py`](src/jitpi05/simulation.py)：LIBERO 环境与闭环 rollout。
- [`jitrl/`](src/jitpi05/jitrl)：固定语义动作、Qwen 规划、记忆、Gemini evaluator、rollout 与统计。
- [`cli/`](src/jitpi05/cli)：离线、仿真和 JitRL 命令行入口。

支持平台为 Ubuntu 24.04/Linux NVIDIA CUDA。Windows 用户先运行
`powershell -ExecutionPolicy Bypass -File scripts/setup_wsl.ps1 -LinuxUser <name>`，
首次启动 Ubuntu
并把仓库 clone 到 `~/projects` 后，再运行 `bash scripts/bootstrap_ubuntu.sh`。
不要从 `/mnt/c` 或 `/mnt/d` 运行大模型实验。

```bash
# 默认安装完整 CUDA 运行环境
uv sync --frozen
uv run jitpi05-doctor

# 开发检查与报告工具按需安装
uv sync --group dev
uv run --group dev pytest -m "not gpu"
uv run --group report jitpi05-report --profile positive-reward
```

## JitRL 在线记忆实验

这是一个与原有 `jitpi05-offline` 和 `jitpi05-eval-sim` 分离的在线闭环入口。当前版本按 [`jitrl-reproduction-corrections.md`](docs/issues/jitrl-reproduction-corrections.md) 使用单一 Qwen3.5-2B 高层策略：固定语义动作工作集、Qwen 状态绑定与基础 logits、memory advantage、additive logit update。Gemini 不参与高层候选生成，仅在 episode 结束后评价视觉 step reward。

### 固定语义动作工作集与 Qwen 基础策略

每 30 个低层环境步视为一个高层决策步；π₀.₅ 每 10 步重新生成低层 action chunk，因此一个高层语义动作默认复用于连续三次低层重规划。

固定工作集定义在 [`semantic_actions.py`](src/jitpi05/jitrl/semantic_actions.py)，部署期间禁止新增动作类型：

```text
approach(target)
align(gripper, target)
grasp(target)
lift(target)
transport(target, destination)
place(target, destination)
release(target, destination)
retract(direction)
stop()
```

同一个 Qwen3.5-2B 完成两个步骤：

1. 读取外部相机、腕部相机、总任务、最近动作、固定 schema 和静态 ICL 示例，输出结构化 `state_summary`、共享对象/目标/方向参数及当前有效动作类型。代码按这些紧凑绑定构造完整且有序的九类工作集，避免要求 2B 重复抄写固定动作表。
2. 仅对当前部署有效实例构造固定标签选择 prompt，执行一次 forward，直接读取这些标签的原始 logits。状态绑定、有效性和基础 logits 均来自同一冻结 Qwen；不使用外部 proposal model，不对 logits 做均值中心化。

当前配置是 `environment_only_no_stop_v1` 诊断上限：固定九类工作集仍完整保留 `stop()`，但 prompt 要求 Qwen 不启用它，代码还会在部署候选中二次屏蔽。若 Qwen 只启用 `stop()`，该次绑定无可执行动作并进入既有 planner 重试；代码不会注入非 Qwen 恢复动作。JitRL 与 Static 都只能由 LIBERO 环境成功、环境结束或最大步数终止，因此本实验用于测量去除视觉假停止后的策略上限，不是最终部署停止机制。完整九类工作集、Qwen 原始有效实例、部署后实例、终止模式和标签 token 均写入 trace。

### Memory 与 JitRL 更新边界

JitRL memory 只包含同一 task、先前已结束 episode 的高层记录。当前 `state_summary` 用小写英文/数字 token 的 Jaccard 相似度检索 top-k=10：

- `V(s)` 是检索邻居 return 均值。
- `Q(s,a)` 只按当前 Qwen 候选的 `semantic_key` 聚合邻居 return。
- Memory 不能添加历史动作，不能改变当前候选数，也没有 memory-only `base_logit`。
- unseen action 以 `lambda=0.05` 使用 `Q=V+alpha/|N|` 的乐观分支，否则 `Q=0`；`alpha=5`。
- advantage 在当前 Qwen 候选集合内按最大绝对值归一化。

更新直接作用于 Qwen 原始 logits：`z'=z+beta*A`，其中 `beta=0.40`、采样 temperature 为 `0.8`。相同 uniform 同时得到当前状态下的 base choice 与 updated choice。Trace 额外记录基础/更新策略熵及 `KL(updated || base)`。

`static` 基线运行完全相同的 Qwen 工作集绑定、原始 logits、temperature、uniform 和 π₀.₅ 流程，但 advantage 恒为零且不读写 memory。因此两种方法的差别只在测试时 memory advantage modulation。

### 本地 Qwen evaluator、记忆生命周期与随机配对

JitRL episode 结束后，**本地 Qwen3.5-2B**（复用高层规划器的同一模型实例，`QwenChunkEvaluator`）按高层 chunk 读取动作前后的外部/腕部图像、Qwen 状态摘要、已执行语义动作、下一状态摘要和最终环境成功标记。`score∈[0,3]`；仅视觉上明确且与任务相关的进展可得正分，错误对象、错误目标、回退、重复、无效果或不确定动作均为 0。局部 reward 为 `score/3`，环境成功时最后一个 chunk 增加 `+1`，再按 `gamma=0.95` 计算 reward-to-go。Static 不调用 evaluator。

> **自评偏差说明**：评估器与高层策略共享同一 Qwen3.5-2B 模型，credit assignment 不再由独立模型（Gemini 3.6 Flash）提供，存在自评乐观偏差（冒烟观测 `mean_evaluator_score=0.9`，显著高于 Gemini 的 `0.1`）。这是为节省 API 成本、规避 4B 级显存溢出的取舍；报告应明确声明评估器与策略同源的局限。

Rollout 期间 memory 只读；episode 完成并评价后才批量写入，避免同 episode 信息泄漏。每个 task/method/seed run 从独立空 memory 开始。候选 uniform 与 π₀.₅ flow noise 使用不含 method 的稳定坐标 seed；轨迹分叉后视觉状态自然可能不同。

### 环境、默认实验与运行命令

运行要求 Linux、NVIDIA CUDA；无桌面服务器建议使用 EGL。4-bit NF4 Qwen3.5-2B 与 bf16 π₀.₅ 同时驻留显存；Gemini 仅通过网络 API 执行 episode 后 evaluator。凭据位于被忽略的 `.secrets/gemini.json`，格式为 `{"api_key":"..."}`，也可通过 `JITPI05_GEMINI_CREDENTIALS` 覆盖。

评测面板可选，通过环境变量 `JITPI05_JITRL_PANEL` 控制（默认 `libero90`）：

- `libero10`：完整 LIBERO-10 长任务 suite 全 10 任务（published pi0.5-LIBERO benchmark，max_steps 520），对应 HarnessVLA × JitRL 消融报告的四格配置。
- `libero90`：LIBERO-90 预注册候选池（难度筛选 + CycleVLA 正式比较，max_steps 400）。
- `libero_pro`：LIBERO-Pro 扰动面板（object / position(swap) / semantic(lan) / task 四维扰动，基于 libero_10 的 10 个长任务），用于泛化 robustness 消融。低层策略切换为 RLinf-Pi05-LIBERO-130-fullshot-SFT（见下方「RLinf π0.5 低层策略」）。

默认规模为 `tasks × methods × 10 episodes × 1 seed`；四方法完整面板（`jitrl-free`、`static-free`、`jitrl`、`static`）为 `10 tasks × 4 methods × 10 episodes = 400 rollouts`。结果仍然是当前 JitRL/Static 分层系统的评测，不等同于 direct `lerobot-eval` baseline。

建议先运行一个小型子集，确认环境、checkpoint 和高层接口正常（LIBERO-10 面板下固定工作集方法 `jitrl`/`static`）：

```bash
uv sync
MUJOCO_GL=egl JITPI05_JITRL_PANEL=libero10 uv run jitpi05-eval-jitrl \
  --task libero_10_task0 \
  --task libero_10_task1 \
  --method jitrl --method static --seed 17 --episodes 3
```

LIBERO-Pro 面板冒烟（单扰动任务、static 方法、1 episode，验证环境 + Qwen 规划 + RLinf π0.5 低层链路）：

```bash
MUJOCO_GL=egl JITPI05_JITRL_PANEL=libero_pro uv run jitpi05-eval-jitrl \
  --task libero_10_object_task0 --method static --seed 17 --episodes 1 \
  --output-dir artifacts/jitrl_libero_pro_smoke
```

完整 LIBERO-10 四方法面板（`jitrl-free`、`static-free`、`jitrl`、`static`，每方法 100 episodes）。建议用独立 `--output-dir`，避免与默认 `jitrl_cycle` 目录中的 CycleVLA 数据混淆：

```bash
OUT=artifacts/jitrl_eval_libero10_long_seed17_qwen2b_workspace_v2_rerun
MUJOCO_GL=egl JITPI05_JITRL_PANEL=libero10 uv run jitpi05-eval-jitrl \
  --method jitrl-free --method static-free --method jitrl --method static \
  --seed 17 --episodes 10 --output-dir "$OUT"

# 汇总必须与运行时使用相同的 method/seed/episodes/output-dir 参数
JITPI05_JITRL_PANEL=libero10 uv run jitpi05-eval-jitrl \
  --method jitrl-free --method static-free --method jitrl --method static \
  --seed 17 --episodes 10 --output-dir "$OUT" --summarize-only
```

每条 run 写入 `videos/`、`episodes.json`、`memory.json`、`memory_snapshots/`、`tensors/`、`run_summary.json`、`metrics.json` 和 `progress.json`。高层 trace 保存 Qwen binding prompt/原始 JSON、九类工作集、Qwen 原始有效实例、部署后实例、终止模式、原始/更新 logits 和概率、`V/Q/A`、UCB、选择变化、熵、KL，以及 Gemini evaluator 记录。指标额外报告 Qwen `stop` 被部署掩码的比例、最终 `stop` 选择率、`qwen_stop` 终止率和最大步数终止率；no-stop 诊断中后两项里的 `stop` 指标应严格为零。

统计包括 overall/final-10 success rate、成功步数、工作集有效率、邻居动作覆盖、非零 advantage、选择变化、UCB、基础/更新熵、更新 KL、evaluator 分布、logit shift、邻居数和 memory size。默认只有一个 seed，且同一 JitRL run 内 episode 顺序相关；结果只能描述当前任务面板。

### RLinf π0.5 低层策略与 LIBERO-Pro 集成

`libero_pro` 面板使用 [`RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT`](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT) 作为低层策略（OpenPI π0.5 架构，`action_horizon=10`、`action_dim=7` 环境维度 / 32 模型内部维度），替代 `lerobot/pi05-libero`。该 checkpoint 是 OpenPI（PyTorch 实现）格式，与当前项目（LeRobot PyTorch、numpy 2、transformers 5.5）依赖硬冲突，因此以**独立 openpi 子进程**运行：

- 服务端 [`scripts/rlinf_pi05_server.py`](scripts/rlinf_pi05_server.py) 在 `third_party/openpi/.venv`（Python 3.11 + torch 2.7.1 + transformers 4.53.2 + patch）中加载 checkpoint，通过 stdin/stdout JSON 行协议提供推理。
- 客户端 [`src/jitpi05/rlinf_client.py`](src/jitpi05/rlinf_client.py) 在当前项目（Python 3.12）中管理子进程生命周期，向 rollout 暴露与本地 `PI05Policy` 等价的 `predict_action_chunk`/`reset`/`config` 接口。
- 输入对齐在服务端完成：双相机 180° 旋转；客户端从 LeRobot 嵌套 `robot_state` 构造 8 维 state（eef_pos + axisangle + gripper_qpos）。
- flow-matching 噪声通过 `policy.infer(obs, noise=...)` 显式注入；服务端把 7 维环境噪声 pad 到 32 维模型内部维度，保证四方法共享同一噪声的实验边界。

LIBERO-Pro（arXiv:2510.03827）通过整体替换 libero 包集成：`third_party/libero-pro/libero` 软链接到 `.venv` 的 libero 包（原包备份为 `libero_orig_backup`），并把 [`zhouxueyang/LIBERO-Pro`](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro) 的扰动 bddl/init 软链接进 libero 包的 `bddl_files/`、`init_files/`。`~/.libero/config.yaml` 指向新包路径。扰动 suite 由 LIBERO-Pro 的替换版 benchmark 注册（`libero_10_object/swap/lan/task` 等）；[`src/jitpi05/libero_pro.py`](src/jitpi05/libero_pro.py) 保留幂等的动态注册作为兜底。object 扰动引入的 `bigger_alphabet_soup`、`red_coffee_mug` 等新物体随 LIBERO-Pro 的 `envs/objects/` 与 `assets/` 一并提供。

### 论文对齐边界

当前版本不再使用 Gemini 动态候选、memory-only 动作、人工 `base_logit=0` 或均值中心化。被 advantage 调制的候选和基础 logits 均由同一 Qwen3.5-2B 在固定工作集内产生，因此更新可解释为对冻结 Qwen 基础策略的概率重分配。

机器人领域仍保留两项适配：语义动作以文本条件驱动冻结 π₀.₅；credit assignment 使用 episode 后 Gemini 视觉 evaluator 和环境成功 bonus。这些设计不改变单一高层基础策略与固定动作空间的核心实验边界。

## 离线分析

本节所述原有离线入口以及后文四条件闭环入口要求 NVIDIA CUDA GPU，建议至少 16 GB 显存；两个模型按阶段顺序加载，不同时驻留显存。JitRL 独立入口则采用 Qwen 4-bit NF4 与 π₀.₅ bf16 同驻，资源需求见上节。

```bash
uv sync
uv run jitpi05-offline
```

第一次运行会下载：

- `Qwen/Qwen3.5-4B`
- `lerobot/pi05_libero_base`
- `HuggingFaceVLA/libero` 的 episode 0
- 一个与官方 PaliGemma 相同的非 gated tokenizer 镜像

官方 `google/paligemma-3b-pt-224` tokenizer 仓库需要先接受 Gemma 许可并登录 Hugging Face。为了让本实验的 tokenizer 阶段不依赖登录，默认使用 `nnh-pbbb/paligemma-3b-pt-224`；其 Gemma tokenizer 词表大小、BOS/PAD ID 与官方版本一致。若你已经获得官方访问权，可直接修改 `PI05_TOKENIZER_ID`。

## 最重要的研究入口

### 高层 prompt

修改 [`planning_prompt()`](src/jitpi05/planning.py)。当前要求 Qwen 只输出一个立即执行的英文动词短语，并关闭 thinking。

### logits 调制

修改 [`make_phrase_bias()`](src/jitpi05/planning.py)，或者直接向 `generate_subtask()` 传入自己的回调：

```python
def modifier(step, input_ids, logits):
    # logits: [batch, vocab_size]
    logits[:, target_token_id] += bias
    return logits
```

当前示例依次提升 `BIAS_PHRASE` 中各 token 的 logit。它只是验证“调制会传导到规划文本和 π₀.₅ 动作”的最小实验，不是完整控制算法。

逐 token 信息保存在：

- `artifacts/planning_logits.pt`：完整原始/调制后 logits，使用 bfloat16 保存。
- `artifacts/summary.json`：top-k token、概率、最终 token 与规划文本。

### π₀.₅ 显式流水线

[`explicit_pi05_flow()`](src/jitpi05/policy.py) 没有调用 `select_action()` 或黑盒 `predict_action_chunk()`，而是显式执行：

1. LeRobot preprocessor：批维、数据集统计归一化、状态离散化、PaliGemma tokenization、搬到 CUDA。
2. π₀.₅ 图像预处理与 VLM prefix embedding。
3. prefix attention mask、position IDs 与 KV cache。
4. 固定相同初始噪声的 10 步 flow-matching 去噪。
5. 裁剪到 LIBERO 7 维动作。
6. LeRobot postprocessor 反归一化并搬回 CPU。

函数同时保存每一步 action trajectory 和 vector field，方便后续研究低层去噪过程。

## 四组条件

同一首帧、同一 π₀.₅ 权重、同一初始噪声下运行：

1. `original_task`：原始完整任务。
2. `oracle_subtask`：人工提供 `pick up the white mug`。
3. `qwen_subtask`：未调制的 Qwen 规划结果。
4. `modulated_qwen_subtask`：经过 logits 调制的 Qwen 规划结果。

如果切换 episode，应同步修改 `EPISODE_INDEX` 和 `ORACLE_SUBTASK`。

## 输出

- `artifacts/summary.json`
  - 两次 Qwen 规划文本
  - 逐 token top-k
  - 四组实际语言条件
  - 动作数值统计、示范 MAE/RMSE、相对原始任务输出的 RMSE
- `artifacts/planning_logits.pt`
  - 完整 Qwen logits
- `artifacts/pi05_actions.pt`
  - 四组反归一化动作 chunk
  - episode 0 示范动作 chunk
  - 固定初始噪声
  - π₀.₅ 去噪 trajectory、vector fields 与 prefix tokens

## LIBERO 闭环评测

闭环入口只支持 Linux。无桌面服务器建议使用 EGL：

```bash
MUJOCO_GL=egl uv run jitpi05-eval-sim
```

第一次导入 LIBERO 时，上游包会询问数据目录；直接选择默认路径即可。评测模型为当前 LeRobot 0.6 / Transformers 5 兼容的 `lerobot/pi05-libero`，与离线分析使用的 base checkpoint 分开配置。

JitRL 的默认 benchmark 使用上文四个标准 suite；原有四条件入口仍默认运行两个小型诊断任务，每个任务 5 个固定初始状态，并在同一初始状态上配对运行四组语言条件：

1. `libero_object` task 0：`pick up the alphabet soup and place it in the basket`。这是 π₀.₅ LIBERO 微调分布内的闭环基准，用于确认环境、processor、动作空间和 checkpoint 均正常。
2. `libero_90` task 79：`pick up the book and place it in the left compartment of the caddy`。这一原有四条件入口仍只把 task 79 作为任务级零样本探针；独立的 JitRL 入口则使用上文四个标准 suite 面板。两种入口都不宣称跨 embodiment 的通用零样本控制。

高层 Qwen 只在每个 episode 的初始双相机观测上规划一次，然后释放显存。低层 π₀.₅ 显式预测 50 步 action chunk，只执行前 10 步，再用新观测重新预测。四组条件在同一 episode 的第 n 次重规划中复用相同 flow-matching 初始噪声。

仿真产物保存在 `artifacts/sim_eval/`：

- `summary.json`：逐条件 success rate、每个 episode 的 success/reward/steps、Qwen 文本和 token top-k。
- `planning_logits.pt`：所有 episode 的原始/调制后完整 Qwen logits。
- `rollouts.pt`：每次预测的完整 action chunk、实际执行动作和 reward。
- `videos/<task>/<condition>/episode_<id>.mp4`：逐条件 rollout 视频。

完整默认评测共 `2 tasks × 5 episodes × 4 conditions = 40` 个 rollout，并包含 10 次 Qwen 规划对，耗时会明显长于离线 sanity check。可直接修改 [`SIM_EPISODES`](src/jitpi05/config.py)、`SIM_ACTION_STEPS` 和 `SIM_TASKS` 缩小实验。JitRL 第二轮使用标准四 suite 面板，但不复用该原有四条件入口。

## 解释限制

`jitpi05-offline` 仍是**单帧离线 sanity check**，用于回答：

- Qwen 是否能输出可读的高层动作？
- logits 调制是否真的改变 token 分布和最终 subtask？
- 高层文本变化是否传导到 π₀.₅ 的低层动作？
- π₀.₅ 输出是否有限、尺度是否正常、是否与示范动作处在相近范围？

示范动作 MAE/RMSE 不是策略成功率，也不能单独证明动作语义正确。当前 JitRL 的主要比较指标是成功 episode 的步数减少，闭环成功率只作为环境健康度辅助指标；两者都应以 `jitpi05-eval-jitrl` 的 LIBERO rollout 为准。当前面板是完整 LIBERO-10 长任务 suite、只有一个 seed，属于可控诊断实验，不代表跨套件泛化；高层 Qwen/subtask、Static/JitRL 与 direct π₀.₅ 仍是不同条件，不能把 JitRL 结果直接当作 checkpoint 的官方成功率。
