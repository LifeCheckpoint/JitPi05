# JitPi05

一个尽量简洁的分层 VLA 研究原型：

- **Qwen3.5-4B**：读取 LIBERO 双相机首帧，生成当前高层 subtask。
- **显式 logits 调制**：逐 token 保存原始/调制后 logits，并在 argmax 前暴露可编辑回调。
- **LeRobot π₀.₅ LIBERO**：把总任务与 subtask 组成低层语言条件，显式执行 prefix 编码、KV cache、10 步 flow-matching 去噪和反归一化。
- **四组对照**：原始任务、人工 subtask、Qwen subtask、调制后的 Qwen subtask。
- **JitRL 第一阶段**：Gemini 3.6 Flash 负责无需 logits 的多模态状态理解、五候选生成和逐步评价；本地 Qwen3.5-4B 只负责五个候选编号的真实 logits，再以任务内在线记忆估计 advantage 并调制。

代码采用可安装的 `src/jitpi05` 包结构：

- [`config.py`](src/jitpi05/config.py)：模型、数据集和实验常量。
- [`datasets.py`](src/jitpi05/datasets.py)：LIBERO episode 读取与图像转换。
- [`planning.py`](src/jitpi05/planning.py)：Qwen3.5 高层规划与 logits 调制。
- [`policy.py`](src/jitpi05/policy.py)：显式 π₀.₅ 流水线与动作指标。
- [`simulation.py`](src/jitpi05/simulation.py)：LIBERO 环境与闭环 rollout。
- [`jitrl/`](src/jitpi05/jitrl)：记忆、Gemini VLM、规划、rollout 与统计。
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

这是一个与原有 `jitpi05-offline` 和 `jitpi05-eval-sim` 分离的在线闭环入口；两者保持原来的单帧离线与四条件闭环语义。当前版本在第一阶段候选级 logit 原型上恢复论文式 augmented candidate、随机 UCB unseen 探索和逐步 VLM evaluator，用于研究 JitRL 在 VLM-VLA 分层控制中的适用性。

### 高层决策映射与候选 logits

每 30 个低层环境步视为一个高层决策步；π₀.₅ 仍每 10 步重新生成一次低层 action chunk，因此一个高层 subtask 默认复用于连续三次 π₀.₅ 重规划：

- **state**：Gemini 3.6 Flash 根据高层决策点的外部相机与腕部相机生成结构化短文本，格式为 `robot=...; gripper=...; object=...; target=...; relations=...; stage=...`。
- **action**：候选中的完整自由文本 subtask，而不是某个文本 token；每个 action 另有规范化的 `semantic_key=skill|object|target`，用于跨措辞聚合同类动作经验。
- **低层执行**：选中的 subtask 与总任务共同作为 π₀.₅ 条件。π₀.₅ 在第 0、10、20、30… 个低层步读取最新视觉观测并生成新 action chunk；Gemini/Qwen 高层组合只在第 0、30、60… 个低层步更新 subtask。

每个高层决策包含两个分工明确的模型步骤：

1. Gemini 3.6 Flash 通过 PydanticAI 读取双相机图像，并以 Pydantic schema 输出一个 `state_summary` 和恰好 5 个候选；每个候选都含自由文本 `text` 与 `semantic_key`。候选 prompt 明确要求至少两个使用空间/方向描述、至少一个使用“中央物体/附近杯子/手中物体”等通用短语，并限制最多两个候选复述任务中的精细颜色限定词，避免低层 VLA 必须理解“红色杯子”才能执行全部候选。
2. 本地 4-bit Qwen3.5-4B 将 Gemini 的同一状态和五个候选编号后重新输入，只执行一次 forward，读取下一个 token 为单 token 编号 `1`、`2`、`3`、`4`、`5` 时的五个真实 logits。Qwen 不再承担复杂 JSON 候选生成，因此回归 4B 以降低显存和推理开销。后续 softmax、记忆 advantage 和采样都只在这五个候选槽上进行，**不是**把候选文本中的每个词表 token 当作 RL action。

### 记忆、augmented candidate 与 logit 更新

JitRL memory 只包含同一 task、先前已结束 episode 的高层决策记录。对当前 `state_summary`，用小写英文/数字 token 集合的 Jaccard 相似度检索 top-k=10；不设相似度阈值，因此记忆非空时即使相似度为 0 的记录也可能进入 top-k。局部估计为：

- `V(s)`：全部检索邻居 return 的均值；无邻居时为 0。
- `Q(s,a)`：邻居中具有相同 `semantic_key` 的 return 均值。
- 候选集合为 Gemini 当前 5 个候选与 top-k 邻居中去重后的历史 action 并集。memory-only action 使用最高相似度邻居保存的自由文本作为可执行 subtask。
- Qwen 五个原始编号 logits 先减去均值；memory-only action 的基础 logit 为 0。trace 同时保留 raw、centered 与 augmented logits。
- unseen action 使用论文随机规则：以 `lambda=0.05` 令 `Q=V+alpha/|N|`，其中 `alpha=5`；否则令 `Q=0`。空 memory 时不除零，所有 action 保持 `Q=V=A=0`。
- 在完整 augmented candidate set 内按 advantage 最大绝对值归一化；全为 0 时归一化 advantage 仍全为 0。

更新使用 `z' = z + beta * A`，其中 `beta=0.40`，基础与更新分布都使用 `temperature=0.8`。同一个 `[0, 1)` uniform 同时产生原始五候选、augmented base 和 JitRL updated 三种 counterfactual choice，便于拆分候选增广与 advantage 调制的影响。需要注意：evaluator reward 已改为非负，但相对 advantage 仍可为负；当某动作的 `Q(s,a)<V(s)` 时，其 logit 仍会被压低。

`static` 基线执行完全相同的 Gemini 双相机候选生成、本地 Qwen 候选打分、temperature、uniform 采样和 π₀.₅ 流程，但 advantage 恒为 0，并且不读取或写入 memory。

### 正奖励-only VLM evaluator、记忆生命周期与随机配对

JitRL episode 结束后，Gemini 3.6 Flash 按高层 chunk 顺序逐个读取动作前后的外部/腕部双相机关键帧、动作前状态摘要、subtask、下一状态摘要和最终环境成功标记。当前 Pydantic schema 严格限制 `score∈[0,3]`：只有视觉上明确、有益且与总体任务相关的进展可得 `1..3`；错误对象、错误目标、回退、重复、无效果、视觉不确定或无法确认的动作全部记为 `0`，不再允许负奖励。`score>0` 必须对应 `usefulness=useful`，`score=0` 必须对应 `usefulness=neutral`，不一致输出会触发最多 3 次 evaluator 重试。place/release/drop 只有在正确物体已脱离夹爪、由明确指定的容器、表面、方向区域或 compartment 真实支撑并稳定释放时才能给正分；二维重叠、悬空、错误物体或错误 compartment 一律为 0。局部 reward 为 `score/3`；失败 episode 不增加 terminal penalty，环境成功时最后一个 chunk 额外加 `+1`，随后按 `gamma=0.95` 计算非负 reward-to-go。Static 不执行 evaluator，避免无意义的额外 API 调用。

为避免同一 episode 内的信息泄漏，rollout 期间 memory 只读，episode 结束并完成逐步评价后才批量写入全部高层决策。

每个 task/method/seed run 都从独立空 memory 开始，默认不会加载旧 `memory.json`；`static` 的 memory 始终为空，不同任务之间也不会共享经验。当前多任务实验只使用 seed 17；每条 run 按 `init_state_id=0..14` 顺序运行，因此同一 JitRL run 的后续 episode 会依赖此前写入的经验。

候选 uniform 由 `(kind, seed, episode, high_level_step)`，π₀.₅ flow noise 由 `(kind, seed, episode, low_level_chunk)` 的稳定坐标 seed 即时重建，均不含 method。因而同一实验坐标在 `static` 与 `jitrl` 间使用相同随机数和初始 flow noise。轨迹一旦因高层选择而分叉，后续视觉状态和 Qwen 候选自然可能不同；所以 `choice_changed` 的含义是 **JitRL 当前状态**下 updated choice 与其 counterfactual base choice 的比较，不是与 Static 轨迹在同一 chunk 的直接动作比较。

### 环境、默认实验与运行命令

运行环境要求 Linux、NVIDIA CUDA；无桌面服务器建议使用 EGL。JitRL 入口让 4-bit NF4 的本地 Qwen3.5-4B（仅计算候选 logits）与 bf16 的 π₀.₅ 同时驻留显存；Gemini 通过网络 API 提供更强的多模态候选生成和 evaluator。`bitsandbytes`、`flash-linear-attention` 与 `pydantic-ai` 依赖由 `uv sync` 安装。当前 WSL2、CUDA 13、PyTorch 2.11 环境中，`causal-conv1d` CUDA kernel 会触发 segmentation fault，因此未保留该依赖；Qwen 仍使用已安装的 FLA kernels，但 causal conv 部分回退到 Transformers 的 PyTorch 实现。4B 相比上一轮 9B 显著降低了显存与候选打分开销，但本地 Qwen 和 π₀.₅ 仍会同时驻留显存，运行时不要并发占用 GPU。

Gemini 凭据存放于被 Git 忽略的 `.secrets/gemini.json`，格式为 `{"api_key":"..."}`；也可通过 `JITPI05_GEMINI_CREDENTIALS` 指定其他路径。API collection URL、模型名和超时配置位于 [`config.py`](src/jitpi05/config.py)。不要把密钥写入 README、命令行参数或提交记录。

当前实验固定 seed 17。根据上一轮的成功率上限/下限结果，保留中等难度的 task 79，并将过易的 task 19/60 与过难的 task 27/62 替换为四个结构上更可能位于中间成功区间的任务：

| task ID | 准确任务描述 | 主要难点 |
|---:|---|---|
| 18 | `put the frying pan on the stove` | 非对称长柄物体抓取，开放 stove 目标 |
| 53 | `pick up the orange juice and put it in the basket` | 多包装物干扰，纸盒抓取与宽容 basket |
| 59 | `pick up the tomato sauce and put it in the tray` | 拥挤场景对象消歧，宽容 tray |
| 69 | `put the chocolate pudding to the left of the plate` | 开放空间关系与左右方向理解 |
| 79 | `pick up the book and place it in the left compartment of the caddy` | 细粒度 compartment 对准与稳定释放 |

实验规模仍为 `5 tasks × 2 methods × 15 episodes × 1 seed = 150 rollouts`，共 10 条独立 run。默认循环顺序为 task 18→53→59→69→79，并在每个任务内部依次运行 JitRL→Static；每个 task/method run 使用独立目录，其中 JitRL 从新的空 memory 冷启动。新产物根目录为 `artifacts/jitrl_eval_libero90_mid5_seed17_qwen4b_positive_v3/`。

上一轮 task 19/27/60/62/79 的 signed-reward 结果仍保留在 `artifacts/jitrl_eval_libero90_5tasks_seed17_qwen4b_beta040/`，完整分析见 [`report.md`](reports/signed_reward/report.md)；新一轮不会覆盖旧 artifact 或报告。

```bash
# 安装锁定依赖（包括 bitsandbytes 与 flash-linear-attention）
uv sync

# 默认全量：5 个任务；每个任务内先 JitRL，再 Static
MUJOCO_GL=egl uv run jitpi05-eval-jitrl

# 分片运行一个 task/method/seed
MUJOCO_GL=egl uv run jitpi05-eval-jitrl --task libero_90_task18 --method jitrl --seed 17

# 单 episode API/rollout smoke test
MUJOCO_GL=egl uv run jitpi05-eval-jitrl --task libero_90_task18 --method jitrl --seed 17 --episodes 1

# 不加载模型，只从已有 episodes.json 与 memory.json 重算并汇总
uv run jitpi05-eval-jitrl --summarize-only
```

`--summarize-only` 可与 `--task`、`--method`、`--seed`、`--output-dir` 组合；`--task`、`--method` 和 `--seed` 均可重复指定，以运行或汇总选定分片。

运行时终端会同时显示三层进度：总 task/method/seed run、当前 run 的 episode、当前 episode 的低层环境步，并由 `tqdm` 根据实测速度给出 elapsed/ETA。默认顺序是 task→method→seed，每个 task 内 method 顺序为 `jitrl` 后 `static`。默认配置会在启动信息中打印 5 tasks、10 runs 和 150 rollouts。每个 episode 结束后还会立即输出一条 `[episode]` 结果，包含 task、SUCCESS/FAIL、完成步数、高层/低层重规划次数、累计成功率、最近 10 个 episode 成功率和当前 memory 大小。若 Gemini API、Pydantic schema、候选缺失 `text/subtask_label/label`、五候选数量/semantic key 或本地 Qwen logits 计算失败，当前高层决策会自动重试最多 7 次，并通过 `[planner-retry]` 实时播报；evaluator 仍通过 `[evaluator-retry]` 最多重试 3 次。只有高层连续 7 次或 evaluator 连续 3 次均失败才会终止该 run。模型加载阶段显示 `[load]`/`[ready]`；即使终端被断开，最新 task、method、episode、低层 chunk、高层决策和环境步仍持续写入 `progress.json`。

### 产物与统计口径

当前每条 run 写入 `artifacts/jitrl_eval_libero90_mid5_seed17_qwen4b_positive_v3/<task>/<method>/seed_<seed>/`：

- `videos/`：全部 episode 视频。
- `episodes.json`：逐 episode 结果、完整高层 trace 及其覆盖的低层 chunk 索引。
- `memory.json`：run 结束时的最终记忆；`memory_snapshots/` 保存每个 episode 后的快照。
- `tensors/`：逐 episode 的完整 action chunks、实际执行 actions 与环境 rewards。
- `run_summary.json`：模型、任务、配置、成功序列、步数和路径索引。
- `metrics.json`：该 method/seed 的统计指标。
- `progress.json`：当前 episode、低层 chunk、高层决策、环境步和完成状态；用于在另一个终端直接查看最新进度。

根目录 `artifacts/jitrl_eval_libero90_mid5_seed17_qwen4b_positive_v3/summary.json` 汇总全部 run。`tasks.<task_name>` 保留每个任务的 JitRL/Static 指标及相同 seed 的配对差值，`task_macro` 报告 5 个任务上的描述性 macro 均值。高层 trace 保存 Gemini proposal 的 backend/model/prompt/结构化原始输出、本地 Qwen raw/centered/augmented/updated logits、候选来源、UCB uniform 与分支、`V/Q/A`、三种 counterfactual choice、memory-only 选择信息，以及 Gemini evaluator 的 prompt、结构化原始输出、分数、certainty、局部 reward 和 discounted return；仍不保存完整词表 logits，也不会保存 API key。

每个 task/method/seed run 分别报告 overall success rate、final-10 success rate、长度为 10 的 moving success rate、成功 episode 的平均步数，以及候选覆盖率、非零 advantage 比例、候选增广/advantage 选择变化率、UCB 比例、memory-only 候选与选择率、evaluator 正分/零分比例、平均绝对 logit shift、平均邻居数和 memory size 等辅助指标。`negative_evaluator_score_rate` 仅为兼容旧 signed-reward artifact 而保留；在新实验中应恒为 0。任务内部仍保留 seed-level mean、sample std 和 bootstrap 字段；默认只有一个 seed 时 sample std 为 0，bootstrap 区间退化为该单个观测值。

统计解释必须保守：每个 task/method 默认只有 1 个 seed，不能进行跨 seed 稳健性判断；同一 JitRL run 内的 15 个 episode 通过在线 memory 顺序相关，也不能当作 15 个独立样本。5 个任务是按物体与结构有意筛选而非从 LIBERO-90 随机抽样，因此 task-macro 均值和 task-level bootstrap 区间只描述当前面板，不构成整个 suite 的总体推断。

### 论文对齐边界

当前已实现论文的动态非参数 memory、Jaccard top-k、augmented candidate、随机 UCB unseen exploration、逐步 VLM evaluator、discounted return 和 additive logit update。它仍不是逐字复刻：LIBERO 使用自由文本高层 subtask 驱动 π₀.₅，Qwen 编号 logits 需要先均值中心化才能与 base-logit=0 的 memory-only action 放在同一可采样尺度；成功环境标记还作为最后 chunk 的额外可靠 bonus。报告时应把这些 VLM-VLA 适配明确列为实验设计，而不是原论文默认设置。

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

默认运行两个任务，每个任务 5 个固定初始状态，并在同一初始状态上配对运行四组语言条件：

1. `libero_object` task 0：`pick up the alphabet soup and place it in the basket`。这是 π₀.₅ LIBERO 微调分布内的闭环基准，用于确认环境、processor、动作空间和 checkpoint 均正常。
2. `libero_90` task 79：`pick up the book and place it in the left compartment of the caddy`。这一原有四条件入口仍只把 task 79 作为任务级零样本探针；独立的 JitRL 入口则使用前文列出的 5-task 面板。两种入口都不宣称跨 embodiment 的通用零样本控制。

高层 Qwen 只在每个 episode 的初始双相机观测上规划一次，然后释放显存。低层 π₀.₅ 显式预测 50 步 action chunk，只执行前 10 步，再用新观测重新预测。四组条件在同一 episode 的第 n 次重规划中复用相同 flow-matching 初始噪声。

仿真产物保存在 `artifacts/sim_eval/`：

- `summary.json`：逐条件 success rate、每个 episode 的 success/reward/steps、Qwen 文本和 token top-k。
- `planning_logits.pt`：所有 episode 的原始/调制后完整 Qwen logits。
- `rollouts.pt`：每次预测的完整 action chunk、实际执行动作和 reward。
- `videos/<task>/<condition>/episode_<id>.mp4`：逐条件 rollout 视频。

完整默认评测共 `2 tasks × 5 episodes × 4 conditions = 40` 个 rollout，并包含 10 次 Qwen 规划对，耗时会明显长于离线 sanity check。可直接修改 [`SIM_EPISODES`](src/jitpi05/config.py)、`SIM_ACTION_STEPS` 和 `SIM_TASKS` 缩小实验。JitRL 第二轮不使用该原有四条件入口。

## 解释限制

`jitpi05-offline` 仍是**单帧离线 sanity check**，用于回答：

- Qwen 是否能输出可读的高层动作？
- logits 调制是否真的改变 token 分布和最终 subtask？
- 高层文本变化是否传导到 π₀.₅ 的低层动作？
- π₀.₅ 输出是否有限、尺度是否正常、是否与示范动作处在相近范围？

示范动作 MAE/RMSE 不是策略成功率，也不能单独证明动作语义正确；闭环成功率应以 `jitpi05-eval-sim` 或 `jitpi05-eval-jitrl` 的 LIBERO rollout 为准。原有 task 79 结果是单任务诊断探针；新的 5-task JitRL 面板扩大了任务覆盖，但任务经过有意筛选且只有一个 seed，仍不应外推成整个 LIBERO-90 或真实机器人上的普遍零样本能力。
