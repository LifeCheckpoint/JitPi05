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

## 6. LIBERO-Pro object 扰动对照（新增）

> Benchmark：LIBERO-Pro `libero_10_object`（object 扰动：物体外观/尺度改变）10 任务 × 4 方法 × 10 episodes × 1 seed，共 400 rollouts/后端。
> 数据来源：`artifacts/jitrl_libero_pro_object/`（RLinf 后端）与 `artifacts/jitrl_libero_pro_object_lerobot/`（LeRobot 后端）。
> 说明：本阶段 credit assignment 已由 Gemini 3.6 Flash 切换为本地 Qwen3.5-2B 自评（复用高层规划器模型），存在自评乐观偏差。

### 6.1 结果对照（四方法宏平均成功率）

| 低层后端 | jitrl-free | static-free | jitrl | static |
|---|---|---|---|---|
| **RLinf-Pi05-LIBERO-130-fullshot-SFT** | 0.000 | 0.000 | 0.000 | 0.000 |
| **lerobot/pi05-libero（上一代）** | 0.690 | 0.730 | 0.670 | 0.650 |

### 6.2 每任务明细（LeRobot 后端成功率）

| 任务 | jitrl-free | static-free | jitrl | static |
|---|---|---|---|---|
| task0 | 0.00 | 0.00 | 0.00 | 0.00 |
| task1 | 1.00 | 1.00 | 1.00 | 0.90 |
| task2 | 1.00 | 1.00 | 1.00 | 1.00 |
| task3 | 1.00 | 0.90 | 0.80 | 0.80 |
| task4 | 1.00 | 1.00 | 1.00 | 1.00 |
| task5 | 0.80 | 1.00 | 1.00 | 1.00 |
| task6 | 0.90 | 0.90 | 0.80 | 0.70 |
| task7 | 0.00 | 0.10 | 0.00 | 0.00 |
| task8 | 1.00 | 1.00 | 1.00 | 1.00 |
| task9 | 0.20 | 0.40 | 0.10 | 0.10 |

### 6.3 结论

1. **RLinf 后端全灭、LeRobot 后端正常（65–73%）**：同一 LIBERO-Pro object 扰动环境下，上一代 `lerobot/pi05-libero` 大部分任务 80–100% 成功，而 RLinf fullshot-SFT 400 rollouts 无一成功，与观察到的"机械臂拿一下就胡乱扭动"一致。这**排除 LIBERO-Pro 集成问题**（环境、bddl、扰动物体注册均正确），问题定位于 RLinf 后端——待用 RLinf 在非扰动 `libero_10` 的单 episode 决定性测试区分"adapter bug"与"RLinf 模型对 object 扰动的 OOD 泛化崩溃"。
2. **JitRL 记忆调制在两个后端下均无可观测正向增益**：LeRobot 后端下 `jitrl vs static` = +2pt、`jitrl-free vs static-free` = −4pt；配对成功步数差 task1 为 −20.3（JitRL 反而多用步）、其余接近 0。与第 4 节 LIBERO-10 面板的"±1pt 不显著"结论一致。
3. **评估器切换影响**：本地 Qwen 自评的 `mean_evaluator_score≈1.2–1.3`、正分率≈0.95，显著高于此前 Gemini 的 0.1（自评乐观偏差），报告结论需在"同源自评"限定下解读。
