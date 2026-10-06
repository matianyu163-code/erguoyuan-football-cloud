# 二果园足球预测系统

Python 3.11，包名 `erguoyuan_football`，项目名称“二果园足球预测系统”。正式 PHASE 9 已建立递归 OOS 血缘核验、依赖感知的 `META_NO_MARKET_V1`、开发期概率校准、分离的 META/校准工件和 CORE V2 预览；阶段状态为 **PARTIAL / AWAITING_FINAL_HOLDOUT**。当前仍不是每日生产预测系统。

YY-CORE Desktop UI 已冻结。现有 `production.launcher` / Research Bridge 提供无界面试运行入口，由 Codex 调度真实代码和正式模型；Codex 不生成或替代 R3 概率。每日目标顺序为 Prediction → Lock → Result → Evaluation → Database。当前赛果录入、生产预测自动评分账本及 PRODUCTION/DEVELOPMENT 运行时硬隔离尚未完成，生产配置仍为 TRIAL，因此不得宣称 Headless 每日生产闭环已就绪。操作边界和解除阻断清单见 [Headless Prediction Engine 操作契约](docs/headless_prediction_engine.md)。
已登记 11 个可实例化模型：原有 9 个，以及 `CORE_XGBOOST_V1` 和 `CORE_CATBOOST_V1`。两种 ML 模型调用官方学习库，执行真实多分类训练和赛前预测；成功结果是未经校准的原始概率。特征快照、时间切分、OOS 校验、模型文件校验和回测代码已经建立。LIKE 模型始终明确标记为非官方实现。
Phase 8 使用 OpenFootball 固定提交的公开数据；数据库有 9,158 场赛程、9,137 场赛果、5 个联赛。Phase 8.2 扩展到 2022-08 至 2026-05 的 7,061 场 OOS 目标比赛；八个非市场底模/ML 模型在 2025–26 赛季有 1,453 场共同可比样本。没有精确 UTC 开球时间，因此只支持 DATE_SAFE_BATCH 历史评估。市场、xG、阵容、官方 Opta 和历史外部快照仍缺失。详见 [Phase 8.2 完成报告](reports/phase8_2_completion_report.md) 和 [自动自审](reports/phase8_2_self_review.md)。

## 安装与测试

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\scripts\test.ps1 -Full
.\.venv\Scripts\python.exe examples/phase2_demo.py
```

已有 `.venv` 时直接执行其中的 Python。运行依赖包括 DuckDB、Pydantic、penaltyblog、NumPy、pandas、SciPy、scikit-learn、joblib、PyYAML、HTTPX、XGBoost 和 CatBoost。
开发依赖包括 pytest、ruff 和 mypy；实际安装版本及许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
未安装 OCR 或视觉服务依赖。测试离线，无外部账户或网络调用。
示例使用 `SYNTHETIC_TEST` 测试夹具、临时数据库和测试赔率，退出即清理，不自动加载到业务数据库。
Phase 9.1 的本机缓存、pytest 临时目录、DuckDB 临时文件和库缓存统一定向到项目 `.workspace/`；
运行 `.\scripts\set_workspace_env.ps1` 只设置当前 PowerShell 进程，不更改 Windows 全局环境。
`.\scripts\test.ps1 -Full` 会先应用该目录约定，再运行完整 pytest。
Phase 9.1 C 盘临时目录迁移脚本默认为 dry-run；本次审计未发现可验证归属本项目的 C 盘临时项，
没有执行移动。性能基准与预测等价核验见 [Phase 9.1 性能报告](reports/phase9_1_performance_report.md)。

## 每日输入入口

```powershell
.\.venv\Scripts\python.exe -m erguoyuan_football --db data/football.duckdb --at 2026-09-29T12:00:00Z --text "周二001 阿森纳VS切尔西"
.\.venv\Scripts\python.exe -m erguoyuan_football --db data/football.duckdb --at 2026-09-29T12:00:00Z --text "阿森纳"
.\.venv\Scripts\python.exe -m erguoyuan_football --db data/football.duckdb --at 2026-09-29T12:00:00Z --image schedule.png
```

CLI 先打印包含请求状态、快照和依赖检查的审计对象，再打印 V5.1 对象。
时间必须带时区；`--match-date` 是可选的 **UTC 开赛日期**过滤，不猜测用户当地日期。
多场文本支持每行一场，也支持比赛编号独占一行、下一行为对阵。

解析依赖通过 `Store` 写入的真实球队、别名、联赛、联赛别名和赛程。
新数据库为空；缺少来源时返回 NOT_FOUND，不自动生成赛程。
仅使用已登记且置信度为 1 的确切别名，忽略大小写和多余空格，不做模糊择优。
多身份或多场候选返回 AMBIGUOUS；只剩一个赛程候选也不掩盖别名歧义。
历史关联使用 ID，不以显示名称 Join。

截图通过 `ScreenshotParser → VisionProvider` 接口接入。未配置 provider 返回 VISION_PROVIDER_UNAVAILABLE。
视觉置信门槛默认 1.0，可显式配置；低于门槛返回 AMBIGUOUS，阻断解析。
截图赔率必须有 as_of_time 才能成为市场记录，source=USER_SCREENSHOT，只记 CURRENT，不能当作 CLOSE。
晚于预测时间取得的历史截图不会进入当时快照。当前没有真实 OCR 或 OpenAI Vision 服务调用。

## 代码布局

```text
src/erguoyuan_football/
  input/          text_parser, batch_parser, screenshot_parser, match_resolver, schemas
  data/           schemas, store, snapshots, availability, canonical market storage
                  real_bootstrap（真实数据三层导入）, temporal（按数据类判断时点资格）,
                  kickoff_enrichment（独立UTC补时）, real_quality（质量与时间门槛）
  contracts/      common, predictions（ModelPrediction / CorePrediction）
  external/opta/  provider（默认 UNAVAILABLE）
  models/         base, training, Phase 3 模型, dynamic_bayes/, spi/, opta_like/, registry, runner
  network/        source registry, pooled client, retry, cache, audit, preflight
  external/market.py  CoreNetworkClient-backed authorized market provider adapter
  models/historical_market_bayes.py  market-anchored Bayesian Poisson base model
  ml/             PIT 特征构建/存储、XGBoost/CatBoost、时间切分、OOS 回测、消融、artifact
  markets/        event identity, odds normalization, de-vig, consensus, snapshots, settlements, movement, goal fit,
                  from_score_matrix（只做数学派生）
  backtesting/    base_model_backtest（rolling / walk-forward 与基础指标）,
                  date_safe_oos（真实历史赛果的按日批次OOS）
  meta/           Phase 9 递归血缘校验、无市场 META、开发期校准、双工件、CORE V2 预览与晋级门禁
  context/        Phase 10 赛事规则/积分榜、时点赛事状态、赛程与阵容证据门禁、特征 lineage、
                  上下文概率层、审计与 DuckDB 持久化
  output/v51/     schema, validator, renderer
  output_contract/ CORE_OUTPUT_V2 中间结果、候选/建议契约与预览 Adapter
  daily.py        DailyPredictionRequest / DailyService
  pipeline.py     每日生产 ModelPipeline / MetaStacking / Calibration，均 NOT_IMPLEMENTED
  __main__.py     CLI
  simulation/     模拟引擎占位
tests/            unit, pit, integration, fixtures, golden/v51
examples/phase2_demo.py
```

原有 features、training、backtesting、stacking、calibration、gates、audit、presentation 等目录保持不变。
当前展示契约位于 output/v51；原 presentation 占位保留。不另建重复的 core_football 包。

## Phase 8 真实数据导入

OpenFootball 本地仓库须保持官方固定提交并核对 Git blob；原始文件及哈希保留在 RAW 表。
导入例子：

```powershell
.\.venv\Scripts\python.exe -m erguoyuan_football.data.bootstrap --provider openfootball --repo data/raw/openfootball_repo --db data/football.duckdb --from-date 2021-07-01 --to-date 2026-09-30 --resume
```

CLI 支持 `--dry-run`、`--competition` 和 `--all-open`。本地赛果作为 `EVENT_IMMUTABLE` 事件事实，允许在其比赛日早于目标比赛日时参加 DATE_SAFE_BATCH 历史训练；这不回填或改写 `retrieved_at`。赔率、阵容、统计快照和外部赛前预测仍需有真实的历史 `as_of_time` / `retrieved_at` 证据。运行 `python -m erguoyuan_football.data.kickoff_provider_import` 可尝试经 CoreNetworkClient 使用授权 football-data.org 补充 UTC 开球时间；无凭据返回 `AUTH_NOT_CONFIGURED`。CORE_OUTPUT_V2 仍是 META 前中间契约，V5.1 不变，正式推荐和资金分配未启用。

## 数据与时点保证

见 [数据库与接口说明](docs/phase2_data_layer.md) 和 [审计契约](docs/audit_contract.md)。
输入只追加、不覆盖，赛程保留版本；PredictionSnapshot 保存完整冻结内容、来源时间和证据 ID。
DuckDB 为读写数据库，Parquet 为可复核导出格式，DuckDB 可直接读回。
内部时间统一使用 UTC。不可变历史事件按 `event_time < prediction_time` 判断；纯日期事件只允许此前日期，并要求目标日所有比赛先预测、再批量更新。快照、市场和外部预测仍要求当时已存在的 `as_of_time` 与 `retrieved_at`。迟到抓取只对不可变事件事实放行，不适用于可变快照。
12:00 和 18:00 可创建不同快照，新增数据不改写原快照。
AVAILABLE 只表示最低证据存在，不保证训练样本量、时效或模型质量。
静态联赛 tier 不作为历史证据，league_hierarchy 暂为 UNAVAILABLE，等待版本化来源。

## 模型引擎边界与实现状态

Phase 3 已实现并登记为 `REAL_IMPLEMENTATION`：`DIXON_COLES_V1`、`BIVARIATE_POISSON_V1`、
`BAYESIAN_HIERARCHICAL_V1`、`ELO_V1`、`PI_RATING_V1`。它们通过独立 wrapper 访问 penaltyblog 或自有评级逻辑，
不互相读取预测结果；每次训练记录数据哈希、配置哈希和训练截止时间，预测记录快照版本和执行状态。

Phase 4 新增 `DYNAMIC_BAYESIAN_POISSON_V1`，以逐场 Poisson 似然和 Laplace 后验状态更新跟踪球队攻防强度，
并把赛后状态延迟到结果真实可用时点。Python 3.11 环境没有 PyMC；当前实现不宣称 MCMC，
方法、先验、状态转移、诊断和限制见 [Dynamic Bayesian 设计](docs/dynamic_bayesian_design.md)。

Phase 5 新增 `CORE_SPI_LIKE_V1`（GOALS_ONLY / 有完整来源时 GOALS_XG）及
`CORE_OPTA_XG_ELO_LIKE_V1`（团队 Elo、有效期层级、可选 provenance-checked xG delta）。它们是 CORE 自有 LIKE 实现，
不冒充 FiveThirtyEight 或 Opta。没有 xG 时 SPI 保持一致的 goals-only 特征；Opta-like 走 result-only 路径。
暂无登记的真实来源，且 80/20 xG 权重只是参考配置，尚无真实来源 OOS 验证。

Phase 6 新增规范化的 `odds_quotes`、`market_snapshots`、`market_consensus`、`market_movements`、
`market_quality_reports`、`market_implied_goals` 和事件身份审计表 `market_event_bindings`，以及 `HISTORICAL_MARKET_BAYESIAN_POISSON_V1`。
报价需通过 CoreNetworkClient-backed Provider，按单一预测时点冻结；先逐 bookmaker 去水、过滤，再建立共识。
市场隐含进球来自对市场 1X2 与大小球概率的数值拟合，融合模型用训练期拟合的市场锚参数，而非与历史模型固定平均。
模型支持 FUSION 与 HISTORICAL_ONLY 消融及按预测 horizon 分开的合成 walk-forward 测试；没有真实来源/历史赔率时市场特征与正式生产数据仍不可用。
Phase 7 新增独立 ML wrapper、固定 `ML_FEATURE_SCHEMA_V1`、`NO_MARKET` / `WITH_MARKET` 两套模式和 append-only 特征库。训练仅使用已冻结的赛前向量；底模概率特征必须有 OOS 证据。训练、验证和最终测试按时间隔离，early stopping 只看验证集。模型文件含原生权重、依赖版本、配置/特征/训练数据哈希和校验和。`WITH_MARKET` 必须有合法 PIT 报价；没有市场时不补造。Phase 8.2 已让 XGBoost 与 CatBoost 各生成 1,472 条真实 `NO_MARKET` OOS；正式生产调用仍受数据源和生产 Gate 限制。
Opta-like Supercomputer 属于未来 Simulation Engine 规划：只消费最终概率/比分分布，不独立创造概率，也不作为额外信号输入 META；当前只保留 `simulation/` 占位目录。

Phase 8.2 真实 OOS 扩展为 5 个联赛、多个赛季的结果型信号，并新增非市场 ML 的真实 OOS；八模型共同可比样本为 2025–26 赛季的 1,453 场。Phase 9 核验全部 6,895 条候选及 6,844 条嵌套 ML 训练特征的 OOS 血缘，使用统一 epsilon 的 log-ratio 与缺失掩码训练无市场 META，并将校准结果传至独立 CORE 契约及 167 条 CORE V2 开发预览。历史市场/xG/阵容/官方 Opta 快照、精确 UTC 评估、完整市场 META、最终留出集验证、带真实证据的 Context/Lineup 调整和投注策略仍未完成。Phase 10 已建立上下文契约与基于现有开发期真实数据的重建/审计路径，但不会在外部证据缺失时调整概率。每日生产 V5.1 输出继续关闭。

Phase 9 的离线入口：

```powershell
python -m erguoyuan_football.meta.run_phase9 --db data/football.duckdb --config config/phase9_development.yaml --artifact-root artifacts/phase9/meta_no_market --report-dir reports
```

程序校验 Phase 8.2 不可变候选、三个分文件哈希和递归 OOS 祖先，只读取真实历史预测；生成开发期 META、独立校准工件、指标及 167 条 `CORE_OUTPUT_V2` 预览。每条 `FINAL_CORE_CALIBRATED` 均为 `DEVELOPMENT_ONLY / NOT_PROMOTED`。原候选集不重建，`META_FULL_V1` 因历史市场数据为零而封锁。2026-08-01 至 2027-06-30 的最终留出集尚无结果，所有 Phase 9 指标标为 `DEVELOPMENT_ONLY`。在 2026 年 5 月的 167 场共同样本中，校准后 META 的 Log Loss 为 1.06495，高于 Dixon–Coles 的 1.05240；未证明优于该单模型。详见 [正式 Phase 9 报告](reports/phase9_completion_report.md)。

Simulation Engine：Opta-like Supercomputer，只消费最终概率/比分分布，不创造独立概率，不能重复输入 META；只预留职责，不作为底层概率模型实现。
ModelRequirements 为 14 项未来模型/引擎需求，包含 11 个已注册模型以及未实现模型和 Simulation Engine 的声明。
缺 REQUIRED 返回 UNAVAILABLE；必需数据存在仍返回 SKIPPED / NOT_IMPLEMENTED。
依赖仅是最小接口要求，未来模型需明确样本窗口、数量、特征和训练资格。

外部产品接口预留 Opta Power Ranking、xG、xGA、Team Stats、Player Stats、Prediction。
仅允许公开页面或合法授权 API；目前没有合法 endpoint/credentials，Provider 返回 UNAVAILABLE。LIKE 对应 LIKE_IMPLEMENTATION，不冒充官方输出。

```text
用户输入 → MatchResolver → PredictionSnapshot → DataAvailabilityReport
→ Base Models V1 → 时序 OOS Predictions → META_NO_MARKET_V1（开发期）→ Calibration（开发期）
→ Context/Lineup Gates → CALIBRATED CORE P → V5.1
```

META 只使用严格时间顺序生成的 OOS 预测，不使用底模训练样本上的拟合预测。
校准、调参和最终测试时间隔离，不反复使用最终测试集调参。

## Phase 3 底模运行示例

底层模型 CLI 只接受已有数据库中的真实赛程、赛果和可用快照；`--trained-until` 必须早于预测时间，训练数据严格限制为该时间点以前可用的记录：

```powershell
.\.venv\Scripts\python.exe -m erguoyuan_football.models.run_base_models `
  --db data/football.duckdb `
  --match-id MATCH_ID `
  --competition-id COMPETITION_ID `
  --as-of 2026-09-29T12:00:00Z `
  --trained-until 2026-09-29T11:00:00Z
```

每个模型独立返回 `SUCCESS`、`UNAVAILABLE` 或 `FAILED`。空数据库或不完整历史数据会输出 `INSUFFICIENT_DATA`，不会输出示例数字。

## Phase 4 Dynamic Bayesian 运行示例

```powershell
.\.venv\Scripts\python.exe -m erguoyuan_football.models.run_dynamic_bayes `
  --db data/football.duckdb `
  --match-id MATCH_ID `
  --competition-id COMPETITION_ID `
  --as-of 2026-09-29T12:00:00Z `
  --trained-until 2026-09-29T11:00:00Z
```

当前命令使用 DuckDB 中已登记的赛程、赛果和冻结快照；缺少真实历史时返回 `INSUFFICIENT_DATA`，不输出概率。
该模型仅为 `FULL_REFIT`，同配置、数据哈希和训练截止时间的合法 artifact 可由 Runner 复用。
阶段实现细节、OOS 证据状态与限制见 [Phase 4 报告](reports/phase4_dynamic_bayes_report.md)。

## V5.1 与阶段边界

[V5.1 契约](docs/v51_output_contract.md) 已实现字段顺序和 Golden Test。
按上阶段约定保留“总进球数”“投注方向”，对应本阶段描述的“总进球”“预测方向”。
渲染器只序列化已提供字段，不选注、不计算概率、Edge、MDI、URS、评级或资金。
未运行预测时全部组合 NO-BET，未知值为 null，不以 0 或均匀概率占位。
Golden 样例是合成结构样例，不是真实预测。

## Phase 10 赛事状态与上下文

Phase 10 已建立开发期上下文接口和真实数据的 DATE_SAFE 重建路径。运行时只接受现有、真实、PIT 通过的
Phase 9 `FINAL_CORE_CALIBRATED` CORE V2 开发预测；赛果只取目标日期之前的不可变记录，且拒绝访问
Phase 9 锁定的 2026-08-01 至 2027-06-30 最终留出集。运行结果只追加到 `context_*` 表，不覆盖 Phase 9 工件。

```powershell
.\.venv\Scripts\python.exe -m erguoyuan_football.context.run_phase10 `
  --db data/football.duckdb `
  --config config/context.yaml `
  --from-date 2026-05-01 `
  --to-date 2026-05-31 `
  --dry-run
```

去掉 `--dry-run` 可持久化上下文快照、特征、概率调整账本和 CORE V2 附加上下文；`--resume` 可复用相同
数据、规则、配置和预测身份的已提交结果。历史竞赛规则必须同时满足目标日期有效以及 `retrieved_at`
不晚于预测时间。当前登记的 EPL 规则记录是在 2026-09-30 抓取，故不会被倒用于 2026-05 历史预测。

现有数据不含完整跨赛事赛程、合法历史首发/伤停快照或球员强度来源。疲劳、阵容和伤停等字段保持空值/不可用；
也没有经时序 OOS 验证的上下文残差模型，因此该阶段严格保持基线概率不变，状态为
`COMPLETE_WITH_EXTERNAL_CONTEXT_BLOCKERS`、`DEVELOPMENT_ONLY / NOT_PROMOTED`。该接口不代表每日生产预测、
投注建议或资金配置已经启用。架构、证据边界与限制见 [Phase 10 设计说明](docs/phase10_context_engine.md)，
结果见 [完成报告](reports/phase10_completion_report.md)、[自审](reports/phase10_self_review.md) 和
[性能报告](reports/phase10_performance_report.md)。

Phase 9.1 对 Phase 9 冻结 META/校准数学、特征、样本切分和模型选择未作调整，
只加固运行时缓存、批量 CORE V2 转换、工件索引、原子工件保存、恢复检查点、遥测和 D 盘临时空间。
该阶段的 `1.064948` 校准 META Log Loss 与 `1.052403` Dixon–Coles 对照保持不变，
Phase 9 仍是 `PARTIAL / AWAITING_FINAL_HOLDOUT`，生产状态仍为 `NOT_PROMOTED`。
参见 [Phase 9.1 完成报告](reports/phase9_1_completion_report.md) 与 [自动自审](reports/phase9_1_self_review.md)。

## Phase 11 最终玩法概率与 CORE REPORT V2

Phase 11 将概率、候选、购买建议和参考账户分开。真实开发期 CORE 概率与经时序核验的 OOS 比分矩阵经过结果质量重分配后，可输出胜平负、比分 TOP2、总进球和无相关性模型时明确标注独立假设的组合。没有真实官方让球盘口、独立半全场模型或当前可购买赔率时，相应玩法、建议和账户保持 `UNAVAILABLE`，不会生成假赔率或 EV。CORE REPORT V2 是独立的新报告；V5.1 结构未修改。

```powershell
.\.venv\Scripts\python.exe -m erguoyuan_football.report.cli --count 20
```

命令生成真实历史开发期报告与性能记录，状态始终为 `DEVELOPMENT_ONLY / NOT_PROMOTED / DATE_SAFE_BATCH`；不代表线上投注可用。详见 [Phase 11 设计说明](docs/phase11_final_heads.md) 与 [完成报告](reports/phase11_completion_report.md)。

仍未实现：新的真实数据采集、OCR 服务、完整市场 META、最终留出集验证、具有历史证据的 Context/Lineup 调整模型、生产模拟、经真实市场验证的投注策略和生产部署。Phase 9/11 的开发期结果没有证明未来赛季预测质量或投注收益；详见 [Phase 9 完成报告](reports/phase9_completion_report.md)。

## Phase 12 桌面审阅程序

桌面程序可输入单场或多场比赛，严格核对本地真实开发期比赛目录，然后调用现有 CORE REPORT V2 流水线。截图入口保留人工确认门槛；未配置 OCR 时只显示 `INPUT_REVIEW_REQUIRED`。不在目录中的今日比赛显示 `UNAVAILABLE`，不生成实时概率。报告及失败输入写入追加式 DuckDB 历史，可导出 JSON、HTML、PDF。生产门禁始终关闭。

```powershell
.\.venv\Scripts\python.exe -m erguoyuan_football.app --check
.\.venv\Scripts\python.exe -m erguoyuan_football.app
.\.venv\Scripts\python.exe -m erguoyuan_football.app --input-text 700fedf625427e26697df271
```

命令行示例中的 ID 是仓库已有的真实开发期样本，仅用于复核历史报告。Windows 打包命令为
`.\.venv\Scripts\python.exe -m erguoyuan_football.app.installer.build`，打包目录位于
`.workspace/dist/COREFootballEngine/`；可执行文件读取项目中的只读历史数据与工件。桌面环境依赖放在 `pyproject.toml` 的 `desktop` 可选组。详见 [Phase 12 完成报告](reports/phase12_completion_report.md)。

## SMART_BRIDGE 桌面输入

桌面普通输入默认使用 `SMART_BRIDGE`。只输入 `法国VS比利时` 会生成
`CORE_RESEARCH_REQUEST_V1` 并写入 `data/bridge/inbox/`，同时记录
`BRIDGE_RESEARCH_REQUESTED`。当前桌面没有实时连接 ChatGPT Work/Codex Research Agent，
因此状态会明确显示 `BRIDGE_AGENT_NOT_CONNECTED`，不会声称已联网研究。Research Agent
准备好 `CORE_DATA_PACKET_V1` 后放到 `data/bridge/ready/<request_id>.packet.json`，桌面
“检查研究结果”会由 CORE 执行验证、PIT、快照、模型规划及可运行底模；结果按请求编号
写入 `data/bridge/results/<request_id>.result.json`。多个可能场次必须先由用户明确选择。

包含竞彩编号、赛事、对阵和开球时间的完整输入会直接进入既有 `JC_PRODUCTION`；
高级模式 `JC_PRODUCTION` 和 `AUTO_RESEARCH` 仍保留。该桥接输入流程不代表校准概率或
长期实战表现已通过验证。请求与数据包字段见
[Research Agent 协议](docs/CORE_RESEARCH_AGENT_PROTOCOL.md)。
