# Harness VLA × JitRL 消融实验完整结果

> 分支：`eliminate` ｜ Benchmark：LIBERO-10 全 10 任务 ｜ Seed：17 ｜ 每任务 10 episodes ｜ 每方法 100 episodes
> 数据来源：`artifacts/jitrl_eval_libero10_long_seed17_qwen2b_workspace_v2_rerun/`（400 rollouts 全部跑满）。
> 上一版结果（`..._no_stop_diagnostic/` 与 `..._free/`）固定工作集未跑满 100（static 83 / jitrl 76），结论已被本版修正。

## 1. 四格配置与定义

| # | 配置 | HarnessVLA（固定原语） | JitRL（在线记忆调制） | 对应 method |
|---|---|---|---|---|
| 1 | **Pi0.5** | —（无 Qwen 高层） | — | 纯 π₀.₅ 直接闭环 |
| 2 | **Pi0.5 + JitRL** | ✗（自由候选） | ✓ | `jitrl-free` |
| 3 | **Pi0.5 + HarnessVLA** | ✓（固定原语） | ✗（static） | `static`（固定工作集） |
| 4 | **Pi0.5 + HarnessVLA + JitRL** | ✓（固定原语） | ✓ | `jitrl`（固定工作集） |

术语：
- **HarnessVLA = 固定原语方案**（固定九类语义动作工作集，见 [`semantic_actions.py`](../src/jitpi05/jitrl/semantic_actions.py:9)）；非固定原语则为**自由候选方案**（Qwen 自由生成候选动作文本，见 [`free_proposal_prompt`](../src/jitpi05/jitrl/planner.py:506)）。
- **JitRL = 使用在线记忆 + advantage logit 调制**（`z'=z+β·Â`，见 [`apply_jitrl_update`](../src/jitpi05/jitrl/planner.py:447)）；否则为 **static**（advantage 恒 0、不读写记忆）。

所有配置共享：终止由环境判定；高层策略为 Qwen3.5-2B（NF4）；低层执行为冻结 π₀.₅（LeRobot `pi05-libero`）；`HIGH_LEVEL_STEPS=40`（本版四格统一，消除上一版固定 30 vs 自由 40 的不一致）。

## 2. 完整结果（每方法 100 episodes）

| 配置 | method | episodes | 成功率 | pooled 成功步数 | 终止分布 |
|---|---|---|---|---|---|
| Pi0.5 + JitRL（自由候选） | `jitrl-free` | 100 | **97.0%** (97/100) | **254.0** | env_success 97 / max_steps 3 |
| Pi0.5 + HarnessVLA | `static`（固定） | 100 | **93.0%** (93/100) | **247.0** | env_success 93 / max_steps 7 |
| Pi0.5 + HarnessVLA + JitRL | `jitrl`（固定） | 100 | **92.0%** (92/100) | **243.1** | env_success 92 / max_steps 8 |
| （参考）自由候选 + static | `static-free` | 100 | 96.0% (96/100) | 256.5 | env_success 96 / max_steps 4 |

### 2.1 配对差值（per-task 宏平均，static − jitrl 为正值即利 JitRL）

| 架构 | 成功率差（jitrl − static） | 成功步数缩减 | 方向 |
|---|---|---|---|
| 固定工作集（`jitrl` vs `static`） | −0.010 | +2.48 步 | 步数利 JitRL、成功率利 static，均不显著 |
| 自由候选（`jitrl-free` vs `static-free`） | +0.010 | +2.55 步 | 均不显著 |

补充（池化口径）：固定工作集 `jitrl` 243.1 vs `static` 247.0，池化差约 3.9 步；自由候选 `jitrl-free` 254.0 vs `static-free` 256.5，池化差约 2.5 步。两方向均跨 0 附近、单 seed，无可测效应。

### 2.2 机制指标

| method | free_action_match_rate | free_unseen_rate | choice_change_rate | 平均 logit shift | memory 规模 |
|---|---|---|---|---|---|
| `jitrl-free` | 0.49 | 0.51 | 11.9% | 0.18 | 72.3 条 |
| `jitrl`（固定） | — | — | 10.2% | 0.20 | 76.9 条 |
| `static-free` | 0.0 | 1.0 | 0.0% | 0.0 | 0 |
| `static`（固定） | — | — | 0.0% | 0.0 | 0 |

## 3. 逐格解读

1. **Pi0.5（纯 π₀.₅）**：LIBERO-10 面板**未评测**。参考 `sim_eval` 诊断任务的 `original_task` 条件（π₀.₅ 直接执行任务描述）：分布内 `libero_object_task0` 100%（145.6 步）、零样本 `libero_90_task79` 20%（387.4 步）。说明 π₀.₅ 在分布内强、零样本弱；该格需在 LIBERO-10 面板补充 direct baseline。
2. **Pi0.5 + JitRL（自由候选）**：97.0% / 254.0 步。JitRL 记忆调制改变 11.9% 的决策、累积约 72 条经验，但成功率与 static 对照（96.0%）仅 +1pt。
3. **Pi0.5 + HarnessVLA（固定原语 + static）**：93.0% / 247.0 步。固定词表成功率低于自由候选约 −3pt，但成功步数省约 −9.5 步。
4. **Pi0.5 + HarnessVLA + JitRL（固定原语 + jitrl）**：92.0% / 243.1 步。JitRL 调制改变 10.2% 决策，但成功率相对 static 反而 −1pt。

## 4. 消融分析

**HarnessVLA（固定原语）的贡献**：跑满 100 后，固定工作集成功率（92–93%）**低于**自由候选（96–97%）约 −3~−4pt，掉点集中在难任务（`libero_10_task8` 固定工作集 70–80%、`libero_10_task9` 四方法全 70%）。成功步数上固定工作集仍更省（243–247 vs 254–256，约 −9~−13 步）。上一版"固定原语 96.4% 无成功率差异"是未跑满（static 83、jitrl 76）造成的**系统性高估**，本版修正为：**固定词表约束省步数、但掉成功率**，不再支持"固定原语更可靠"的表述。

**JitRL 的贡献**：两种架构下 JitRL 相对 static 的成功率变化为 ±1pt（不显著）；成功步数上 JitRL 均略省（固定 +2.48 步、自由候选 +2.55 步 per-task paired 宏平均），幅度小、单 seed、CI 无跨 seed 推断力。**JitRL 的在线记忆 advantage 调制在本任务面板上未带来可测成功率增益**。

**综合结论**：四个配置的成功率落在 92–97%；固定原语词表在本面板上表现为"省步数但掉成功率"（约 −3pt），JitRL 记忆调制表现为近似零的独立效应（±1pt、步数略省）。即：在环境终止 + 冻结 π₀.₅ 的高层语义动作体系下，固定原语词表主要降低成功率、JitRL 记忆调制对成功率无可测影响。

## 5. 限定

- **单 seed（17）**：`sample_std=0`、seed-level bootstrap 塌缩为观测值，不能做跨 seed 推断；task 级 CI 仅描述性。
- **episode 顺序相关**：同一 JitRL run 内在线记忆使 episode 不独立，不能当独立重复样本。
- 本面板测的是"固定词表约束"这一简化骨架（无 HarnessVLA 的 Task Specific Memory / Global Memory / 解析原语层，无 JitRL 论文的增广候选集），结论应表述为：在本面板、单一冻结 π₀.₅ 低层、环境终止 + 固定动作词表/自由候选条件下成立。

## 6. LIBERO-Pro object 扰动对照（修复后结果）

> Benchmark：LIBERO-Pro `libero_10_object`（物体外观/尺度扰动），10 任务 × 4 方法 × 10 episodes × 1 seed，共 400 rollouts。
> 修复后数据由两个目录合并得到：Task 0–6 来自 [`artifacts/jitrl_libero_pro_object_rlinf_state_fix`](../../artifacts/jitrl_libero_pro_object_rlinf_state_fix)，Task 7–9 来自 [`artifacts/jitrl_libero_pro_object_rlinf_subtask_seed17_tasks7_9`](../../artifacts/jitrl_libero_pro_object_rlinf_subtask_seed17_tasks7_9)。两部分的配置完全一致。
> 旧版 raw-task 结果只作为故障定位历史，不与修复后结果合并。当前 RLinf 实验默认将 `Overall task + Current subtask` 传给低层；`JITPI05_RLINF_USE_RAW_TASK_PROMPT=true` 仅保留为官方 raw-task 兼容诊断模式。

### 6.1 与旧结果和 LeRobot 结果的关系

| 低层后端/协议 | jitrl-free | static-free | jitrl | static | 解释 |
|---|---:|---:|---:|---:|---|
| RLinf，旧 raw-task | 0% | 0% | 0% | 0% | 高层选择未进入低层，消融失效 |
| RLinf，修复后 subtask | **40%** | **42%** | **43%** | **45%** | 有效 400-rollout 面板 |
| LeRobot，上一代结果 | 69% | 73% | 67% | 65% | 不同 checkpoint/adapter/protocol，仅作参考 |

因此，旧 RLinf 的 0% 不能解释为 RLinf 模型在 object 扰动上完全失效：subtask 修复后成功率升至 40–45%。但它仍显著低于 LeRobot 的 65–73%，剩余差距需要由 RLinf checkpoint、输入/动作协议和 object OOD 泛化共同解释。

### 6.2 修复后完整面板总体结果

| 配置 | method | 成功率 | 全 episode 平均步数 | 成功 episode 平均步数 | 800 步上限比例 |
|---|---|---:|---:|---:|---:|
| 自由候选 + JitRL | `jitrl-free` | 40/100 = **40%** | 585.7 | 307.6 | 60% |
| 自由候选 + Static | `static-free` | 42/100 = **42%** | 572.6 | 284.4 | 58% |
| 固定工作集 + JitRL | `jitrl` | 43/100 = **43%** | 580.4 | 305.9 | 57% |
| 固定工作集 + Static | `static` | 45/100 = **45%** | 564.8 | 293.7 | 55% |

逐任务成功数为：

| 任务 | `jitrl-free` | `static-free` | `jitrl` | `static` |
|---|---:|---:|---:|---:|
| task0 | 0/10 | 0/10 | 0/10 | 0/10 |
| task1 | 9/10 | 9/10 | 10/10 | 10/10 |
| task2 | 7/10 | 6/10 | 9/10 | 9/10 |
| task3 | 3/10 | 3/10 | 5/10 | 4/10 |
| task4 | 4/10 | 4/10 | 1/10 | 2/10 |
| task5 | 6/10 | 9/10 | 5/10 | 7/10 |
| task6 | 10/10 | 9/10 | 10/10 | 10/10 |
| task7 | 0/10 | 0/10 | 1/10 | 1/10 |
| task8 | 1/10 | 2/10 | 2/10 | 2/10 |
| task9 | 0/10 | 0/10 | 0/10 | 0/10 |

### 6.3 修复有效性与机制指标

修复后的所有 400 rollouts 共包含 11,599 个高层 chunks；每个 method 的每个 chunk 都满足：

```text
Overall task: <overall task>.
Current subtask: <selected semantic action>.
```

`condition` 与 selected action 精确匹配率为 100%。每个任务的四方法 `actions.pt` 和 `action_chunks.pt` hash 均为 4/4 不同，说明高层更新已经真实传递到 RLinf 低层并导致轨迹分叉。

完整面板的平均机制指标：

| method | choice change | mean logit shift | mean update KL | nonzero advantage | mean neighbors |
|---|---:|---:|---:|---:|---:|
| `jitrl-free` | 9.34% | 0.1913 | 0.0157 | 90.00% | 9.03 |
| `jitrl` | 9.36% | 0.1922 | 0.0135 | 89.99% | 9.02 |
| Static 两种 | 0% | 0 | 0 | 0% | 0 |

这排除了“JitRL 没有更新”的解释：memory、advantage、logit update 都在执行，问题是更新方向没有转化为稳定的任务级收益。

### 6.4 JitRL 独立效应

| 比较 | 成功率差 | 全 episode 平均步数差 | 配对成功胜负 | 精确双侧 p |
|---|---:|---:|---:|---:|
| `jitrl - static` | −2pt | +15.6 步 | 5 胜 / 7 负 | 0.774 |
| `jitrl-free - static-free` | −2pt | +13.1 步 | 4 胜 / 6 负 | 0.754 |

步数差的 paired bootstrap 95% 区间分别为 `[-19.2, 51.95]` 和 `[-16.79, 43.72]`，均跨 0。正确结论是“未观察到可测增益”，而不是“JitRL 已被证明有害”。

按 episode index 聚合 10 个任务：

```text
jitrl:       [4, 5, 3, 4, 6, 5, 4, 5, 3, 4]  first5=22  last5=21
static:      [4, 3, 4, 4, 7, 5, 4, 5, 4, 5]  first5=22  last5=23
jitrl-free:  [4, 5, 4, 4, 5, 4, 3, 3, 5, 3]  first5=22  last5=18
static-free: [4, 6, 3, 4, 4, 3, 4, 5, 6, 3]  first5=21  last5=21
```

没有观察到 memory 增长带来的后半程学习曲线。

### 6.5 为什么 JitRL 作用不显著

#### （1）低层策略与 object OOD 是共同瓶颈（强证据）

Task 0 四方法均 0/10，Task 9 四方法均 0/10，Task 7 free 两种方法均 0/10，Task 8 仅 1–2/10。此时大部分失败来自低层视觉/接触控制，JitRL 只改变约 9%–10% 的高层选择，无法挽救共同失败。

报告中的 LIBERO-10 高成功率使用 LeRobot checkpoint；当前使用 RLinf fullshot-SFT checkpoint。两者不是同一个模型的简单实现替换，训练数据、prompt、state/action normalization、相机处理和 action horizon 都可能不同。

#### （2）固定工作集的候选/对象绑定质量不足（强证据）

trace 中出现过 `Current subtask: align the gripper with empty` 以及在 book/caddy 任务中绑定 `mug` 的情况。JitRL 只能重排当前候选，不能把错误对象替换成正确对象；因此错误 binding 会直接限制 JitRL 上限。

以高层动作是否提及任务实体作为粗粒度代理，fixed workspace 的目标实体提及率约 42%–44%，free candidates 约 98%；这不是严格准确率，但足以显示 fixed binding 是当前主要风险。

#### （3）Positive-only、同源 evaluator 可能污染 memory（强推断）

当前 reward 版本允许失败 episode 获得局部正分；Task 9 四方法最终成功率均为 0%，但 JitRL evaluator 均分仍约 1.15，Task 7 free JitRL 失败率 100% 时均分仍约 0.91。planner/evaluator 又共享 Qwen3.5-2B，存在同源自评乐观偏差。

可能的错误链为：

```text
局部视觉变化 → 正向 evaluator score → 正 return → 错误经验写入 memory → JitRL 强化错误动作
```

#### （4）更新幅度保守，且更新方向不可靠（中等证据）

`mean update KL` 仅约 0.013–0.016，choice change 约 9%–10%。这说明 JitRL 的更新确实存在但较保守；在低层困难和 binding 错误状态下，可能不足以切换关键错误动作。直接增大 `beta` 又可能放大错误 evaluator credit，因此不能把调大 beta 当作首要修复。

#### （5）状态表示和动作语义对机器人几何状态不充分（中等推断）

当前使用文本 state summary + Jaccard 检索；视觉遮挡、夹爪接触、相对位姿和物体是否已放稳等关键变量不一定能被短文本稳定表示。平均 neighbor 数约 9 只说明检索数量足够，不说明邻居的几何语义正确。

#### （6）面板处于 ceiling/floor 两种不利 regime（强实验设计判断）

普通 LIBERO-10 LeRobot 成功率 92%–97%，接近 ceiling，JitRL 可改善的空间很小；RLinf object 成功率 40%–45%，接近低层噪声/能力 floor，JitRL 的高层效应又被共同失败淹没。两种设置都不适合清晰测量小幅 test-time policy improvement。

#### （7）单 seed、每 task 10 episode 的统计功效不足（确定限制）

当前只有 seed 17，且同一 JitRL run 内 episode 顺序相关。−2pt 的点差配对检验 p≈0.75–0.77，不能支持显著负效应，也不能排除小幅正效应。
