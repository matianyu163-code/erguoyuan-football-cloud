# 二果园足球预测系统开发约束

## 阶段范围

当前工作处于 PHASE 10 — TOURNAMENT STATE + LINEUP + CONTEXT ENGINE；阶段状态和外部证据阻断见
`reports/phase10_completion_report.md` 与 `reports/phase10_self_review.md`。PHASE 9 仍为
`PARTIAL / AWAITING_FINAL_HOLDOUT`，实测限制见 `reports/phase9_completion_report.md`。
Phase 9 只训练 `META_NO_MARKET_V1` 并开发校准和 CORE V2 集成；所有候选与 ML 嵌套训练特征
必须通过递归 OOS 祖先校验。开发期校准 CORE 输出必须标记 `NOT_PROMOTED`，晋级门禁保持关闭；
`META_FULL_V1` 因历史市场行数为 0 而 `BLOCKED_MARKET_DATA`。
最终留出集固定为 2026-08-01 至 2027-06-30，仍为 0 行；不得移动边界、读取其结果或把开发指标称为最终表现。
正式投注建议、资金分配与正式 V5.1 生产链路仍禁止。Phase 10 完成后必须停止等待审核；未经用户明确授权不得开始 PHASE 11。
OpenFootball 原始文件必须校验固定 Git 提交与内容；未声明时区的比赛不可虚构 UTC 开球时间。
公开 OpenFootball Git 快照可通过 Git 工具取得固定版本，随后仅从本地校验过的文件导入；
运行时 HTTP/API Provider 仍统一通过 `CoreNetworkClient`，不得用 Git 快照冒充历史时点 API 可用性。
新下载的历史文件不得把 retrieved_at 倒填为历史预测时点；不足以证明 PIT 的数据不得生成真实 OOS。
CORE_OUTPUT_V2 的底模输出必须标记 BASE_MODEL/ML_MODEL/PRE_META；Phase 9 的 `FINAL_CORE_CALIBRATED`
只能标记 `DEVELOPMENT_ONLY`，不得冒充最终留出集验证或实时生产概率。
正式预测采用 ENHANCED_ONLY；未通过国际网络预检与必需外部源 Gate 时不得输出正式预测。
Opta-like 始终标记 LIKE；缺少真实 xG/层级来源时必须走明确的无 xG 路径或 UNAVAILABLE。
市场赔率必须在 prediction_time 时真实可用；Closing 仅可作为事后 benchmark，不可泄漏至早期特征。
不同 bookmaker 必须先分别标准化、去水和质量过滤，然后才能聚合；去水策略须固定版本，禁止逐场择法。
所有线上市场 Provider 只能经过 `CoreNetworkClient`，本地/竞彩截图不可绕过 ENHANCED_ONLY 市场来源 Gate。
市场派生值需保留信息依赖标签和来源 lineage；市场锚定 Bayesian Poisson 必须内部拟合市场影响，不可与历史概率作固定平均。
正式 V5.1 生产输出仍 NOT_READY_FOR_FINAL_PRODUCTION；不改变格式和 Golden Test。
第三方模型必须经 Adapter；所有新增实现配套类型提示、docstring、ruff、mypy、pytest。
沿用 erguoyuan_football 包名；不得建立重复的 core_football 包。测试样例必须标记 SYNTHETIC_TEST。
每阶段结束必须运行测试并完成 Automated Self Review，报告新增文件、测试结果、未完成内容。
Phase 9 对应报告为 `reports/phase9_self_review.md` 与 `reports/phase9_completion_report.md`；
Phase 9.1 对应 `reports/phase9_1_self_review.md`、`reports/phase9_1_completion_report.md`
和 `reports/phase9_1_performance_report.md`。阶段报告完成后停止。
未经用户明确授权，不自动开始下一阶段。所有后续模块必须配套 pytest 测试。

## 真实执行与缺失数据

使用 Python 3.11。概率必须来自真实执行的程序，禁止用 LLM 臆测概率。
缺少必要数据时返回 UNAVAILABLE，不得伪造 Elo、xG、Opta、赔率、阵容或模型结果。
未实现的模块不得输出占位概率或被登记为已实现。
模型分类限定为 REAL_IMPLEMENTATION、LIKE_IMPLEMENTATION、EXTERNAL、UNAVAILABLE。

## 审计契约（后续实现）

所有模型输出必须留下以下字段；如不适用，需明确说明，不得伪造值：

- model_id
- model_version
- training_end_time
- prediction_time
- input_data_version
- p_home
- p_draw
- p_away
- lambda_home（如适用）
- lambda_away（如适用）
- score_matrix（如适用）
- data_source
- data_status
- execution_status

外部数据必须保存 source、retrieved_at、as_of_time。
历史训练和回测严格遵守 point-in-time，并先按 `TEMPORAL_DATA_CLASS` 分类。
EVENT_IMMUTABLE 赛果只要求事件时间早于预测时间；纯日期事件仅可在 DATE_SAFE_BATCH 中使用，
且训练只纳入目标比赛日期之前的整日批次。目标日期所有比赛必须先预测完毕，再统一更新模型。
SNAPSHOT_TIME_SERIES、MARKET_TIME_SERIES、EXTERNAL_FORECAST 仍须有当时的 `as_of_time`
和 `retrieved_at` 证据；POST_MATCH_ONLY 不得用于目标比赛。不得只依赖事件时间为可变数据背书；
派生特征、训练标签、赔率快照、阵容、OOS、校准与模型选择均禁止未来信息泄漏。
DATE_SAFE_BATCH 不是 T-24h 或小时级预测，不得用于市场、阵容和 intraday context。
后续实现须明确时间字段语义、UTC 时区、版本与可用性证据，并通过测试验证。
具体字段语义、UTC 时间约定、空值与状态规则见 [审计契约](docs/audit_contract.md)。

## META 与时间隔离

META STACKING 只能使用严格按时间顺序生成的 Out-of-Sample Predictions 训练。
每条 OOS 预测必须来自仅使用该预测时点以前可用训练数据的底模，保留对应训练窗口和预测时点证据。
禁止使用底模训练样本上的拟合预测训练 META；随机划分不能替代时间顺序隔离。
校准、调参及最终测试必须有明确的时间隔离，记录各阶段时间窗口与使用目的。
最终测试集不得参与校准或调参，不得反复使用最终测试集调参。

## 数据与模型边界

Opta 只允许使用公开页面或合法授权 API；无真实 Opta 数据则返回 UNAVAILABLE。
Opta-like 必须标记 LIKE，不得冒充 Opta 官方模型或输出。
LIKE 对应 LIKE_IMPLEMENTATION；实现类型、数据可用状态和执行状态必须分别记录，不能互相替代。
目标模型与市场范围见 README.md；名单不代表当前已有实现。
训练后的最终链路必须为：
Base Models → OOS Predictions → META STACKING → Probability Calibration
→ Context/Lineup Gates → CALIBRATED CORE P。
Opta-like Supercomputer 属于 Simulation Engine 规划，只消费最终概率/比分分布，
不独立创造概率，不得作为额外独立信号重复输入 META。本阶段仅预留 simulation/ 目录。

## Phase 11 边界

Phase 11 的预测、候选、购买建议和参考资金账户必须保持独立。`NO_BET` 不得隐藏合法候选；缺赔率、官方让球或独立半全场模型时使用 `UNAVAILABLE`，不得伪造 EV 或概率。比分和总进球只能从可追溯的 REAL OOS、PIT 合法且与有效 CORE 1X2 重分配一致的同一比分矩阵派生。正式报告使用独立 [CORE REPORT V2 说明](docs/phase11_final_heads.md)，不得修改 V5.1。Phase 9 最终留出集仍锁定，当前开发输出仍为 `DEVELOPMENT_ONLY / NOT_PROMOTED / DATE_SAFE_BATCH`。

## Phase 12 桌面应用边界

桌面入口仅调用现有 Phase 11 开发期 CORE REPORT V2 路径，不修改 Phase 9/10/11 数学与选择逻辑。
当前输入只可严格解析到已有真实开发期预测；新比赛、歧义比赛与未人工确认截图必须显示
`UNAVAILABLE`、`AMBIGUOUS` 或 `INPUT_REVIEW_REQUIRED`，不得伪造概率。
应用历史保存在 `.workspace/app/history.duckdb`，使用追加式记录，禁止覆盖报告或删除历史。
Windows 打包和导出不得把缺失市场、让球、半全场或投注建议填成假数。
`LIVE_PRODUCTION_READY` 恒为 `FALSE`，直到最终留出集、合法市场数据与生产验证完成。
Phase 12 完成后停止，未经用户明确授权不得进入下一阶段。

## V5.1 输出冻结

最终前端输出结构必须与用户项目定义的 V5.1 完全一致。
已确定的展示顺序与字段见 [V5.1 输出契约](docs/v51_output_contract.md)，必须遵守。
MDI、URS、评级阈值、资金分配等计算细节仍待定义，不得自行补造，也不得编造真实预测样例。
没有合格结果时明确显示“无合格组合 / NO-BET”，禁止为了凑齐五类而编造结果。
本阶段实现固定格式契约与 Golden Test；未定义的计算规则保持空值，不在渲染器中计算或选注。
底层模型升级不得擅自改变该结构。

## Phase 5 网络规则

外部请求统一经 `network/CoreNetworkClient`；TLS 验证始终开启，来源 URL 必须来自登记表，
密钥不得进入配置、日志或审计记录。AUTO 只使用操作系统现有路由与 HTTPX 环境代理；
CORE 不检测、启动或修改 VPN。缺少真实 Provider 或合法 endpoint 时明确不可用，不猜 URL。
Network preflight 与 production gate 默认 fail-closed；离线开发测试必须显式标注，不能成为生产输出。

## Desktop 冻结与 Headless 日常操作（用户指令，2026-10-05）

YY-CORE Desktop UI 已冻结。未经用户明确要求解冻，不得继续修复、扩展、打包或新增桌面界面功能；后续日常操作以本地 Headless 流程为准，细则见 [Headless Prediction Engine 操作契约](docs/headless_prediction_engine.md)。

Codex 只负责调度现有正式代码：读取输入和数据、调用正式模型入口、保存并锁定原始结果、按有来源的实际赛果评分、更新追加式记录。禁止由 Codex/LLM 生成、改写或代替正式模型产生 R3 概率；模型不可用时原样报告 UNAVAILABLE/BLOCKED。

日常生命周期顺序固定为 Prediction → Lock → Result → Evaluation → Database。不得覆盖既有预测；赛果须在比赛结束后从可追溯来源取得，评估记录须引用原预测 ID 和赛果来源。

PRODUCTION 仅允许读取冻结代码、配置、模型和特征工件并运行正式预测，禁止临时改代码、调参或训练。DEVELOPMENT 用于研究、模型升级、回测和实验，写入独立数据与工件目录；未经验证的模型不得晋级。当前仓库仍共享源代码树与部分存储位置，Headless 物理隔离尚未完成，必须如实保持 BLOCKED/NOT_READY，不得把文档约定说成运行时强制隔离。

目前 `config/production.yaml` 仍标记 `TRIAL`；`TrialRecordStore` 对预测记录实施追加且禁止更新/删除，可作为保存时锁定的基础。生产预测记录关联的赛果录入、自动评分台账和 PRODUCTION/DEVELOPMENT 硬隔离尚未完整实现。在这些能力经过独立验证前，不得宣称每日 Headless 正式闭环已就绪，不得借用开发期 OOS 指标冒充生产表现。

每次开工先读取 `docs/repository_map.md`、`config/phase_manifest.yaml`、
`config/model_registry.yaml`，然后只检查当前阶段相关模块。禁止扫描原始数据、缓存、模型工件和历史报告目录。

## 测试与安全

基础测试离线执行，不依赖真实账户、外部数据或网络。
测试用合成数据须明确标记，不得流入真实预测结果。
禁止提交 API 密钥、私有原始数据、训练产物或环境目录。
不得把脚手架测试通过描述为预测系统已可用或已达到生产质量。
