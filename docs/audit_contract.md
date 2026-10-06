# 审计契约

项目：二果园足球预测系统。PHASE 3 在既有时点数据层和契约校验上增加了五个独立底层模型，
代码位于 contracts/predictions.py、data/schemas.py、data/snapshots.py 和 models/。

## 必须保留的模型审计字段

| 字段 | 含义 |
| --- | --- |
| model_id | 模型的稳定标识，不冒充外部官方模型。 |
| model_version | 本次执行对应的模型版本。 |
| training_end_time | 该模型训练数据窗口的截止时点，不是训练进程完成的墙钟时间。 |
| prediction_time | 本次预测的信息截止与决策时点；历史回放中为被还原的历史预测时点。 |
| input_data_version | 本次输入数据的可追溯版本。 |
| p_home | 主胜概率。 |
| p_draw | 平局概率。 |
| p_away | 客胜概率。 |
| lambda_home | 主队进球强度，仅适用模型提供。 |
| lambda_away | 客队进球强度，仅适用模型提供。 |
| score_matrix | 比分概率矩阵，仅适用模型提供。 |
| data_source | 本次模型输入所用的实际数据来源及其来源记录关联。 |
| data_status | 输入数据的可用性状态，独立于实现类型和执行状态。 |
| execution_status | 本次执行的实际结果状态，不能用模型是否存在来代替。 |

上述字段均须保留；未知或不适用的值为空并记录原因，不得省略字段来掩盖缺失。
外部结果如未披露训练截止时点，不得编造 training_end_time，须记录不可核验的原因。

## 外部数据溯源

每份外部数据必须保存：

| 字段 | 含义 |
| --- | --- |
| source | 实际来源身份与可追溯定位，例如公开页面地址或合法授权 API 的提供方及端点；不得包含密钥。 |
| retrieved_at | 系统实际取得该数据版本的时间，不得倒填成历史预测时间。 |
| as_of_time | 来源明确说明的该版本信息截至或生效时点；不是抓取时间，也不自动等于比赛时间。 |

无法确认 as_of_time 时使用空值并说明原因，不以 retrieved_at 冒充。
source 描述外部记录的来源；data_source 将模型输入关联到这些实际来源记录和版本。
公开页面或合法授权 API 是 Opta 的唯一允许来源；无真实 Opta 数据返回 UNAVAILABLE。

## 时间与 point-in-time

内部时间统一为带时区的 UTC；原始时区信息应留存以便溯源。
无时区且无法可靠解释的时间不得默认为本地时间用于历史预测。
`prediction_timestamp` 是约束表达式中对审计字段 `prediction_time` 的称呼，两者是同一时点。
历史回放的实际运行时间不得覆盖 prediction_time；实际执行时间应另行留痕，具体扩展字段待后续定义。

任何用于预测比赛的数据必须满足：

`data_timestamp <= prediction_timestamp < kickoff_time`

data_timestamp 表示该具体数据版本在预测中可用的时点，需要真实可用性证据。
as_of_time 较早不代表该版本当时已公开或已可获取；事后修订、回填不得混入历史输入。
retrieved_at 晚于预测时点的历史数据，必须有可核验的当时版本及可用性证据，否则不能用于该历史预测。
PHASE 2 采用更保守实现：排除全部 retrieved_at > prediction_time 记录，尚无历史档案豁免。
训练截止和全部训练标签、派生特征、赔率快照、阵容同样不得越过预测时点。

META STACKING 只能使用严格按时间顺序生成的 Out-of-Sample Predictions 训练，
禁止使用底模训练样本上的拟合预测。校准、调参、最终测试必须明确时间隔离，
不得反复使用最终测试集调参；具体规则同时受 [AGENTS.md](../AGENTS.md) 约束。

## 概率与空值

成功输出的 p_home、p_draw、p_away 必须均为有限数值、位于 [0,1]，且三者之和为 1。
不得使用 NaN 或正负无穷。PHASE 2 求和绝对容差为 1e-8，相对容差为 0。
比分矩阵必须非空、矩形、总和为 1，胜平负边际必须与对应概率一致。
Phase 3 的 ScoreMatrix 对截断矩形执行显式条件归一化，并记录 retained_mass、tail_mass 和 tail_policy；
校验器不把未记录的尾部质量静默丢弃。

UNAVAILABLE 时，概率及不适用数值使用空值（例如 JSON null / Python None），并记录原因。
lambda_home、lambda_away、score_matrix 不适用时为空；不能用 0、空矩阵或均匀概率伪装预测。
真实成功计算得到的 0 概率与缺失值不同，不能把二者混用。
已知的来源、版本和时间等审计信息仍须保留；未知信息不得编造。
原因保存在 reason 字段，当前明确 MISSING_REQUIRED_DATA、NOT_IMPLEMENTED 等入口状态。
完整供应商和模型错误码留待对应模块实现。

## 分类与状态的分离

保留分类 REAL_IMPLEMENTATION、LIKE_IMPLEMENTATION、EXTERNAL、UNAVAILABLE：

| 分类 | 语义 |
| --- | --- |
| REAL_IMPLEMENTATION | 真实实现并执行的相应模型；不是数据完整或本次执行成功的保证。 |
| LIKE_IMPLEMENTATION | 明确标识的近似或自定义实现；显示标记 LIKE 对应此分类。 |
| EXTERNAL | 可追溯的外部模型结果，不宣称为本地训练执行。 |
| UNAVAILABLE | 没有可用实现或结果；不包含占位预测。 |

实现类型描述实现身份，data_status 描述输入可用性，execution_status 描述本次执行结果，
三者必须分别记录，不能相互推断或替代。已有 REAL_IMPLEMENTATION 也可能因缺数据而无可用结果；
数据可用也不代表执行成功。未执行或执行失败不能记为成功。
implementation_type 仅允许 REAL_IMPLEMENTATION / LIKE_IMPLEMENTATION / EXTERNAL。
UNAVAILABLE 保留为不可用分类，通过 data_status / execution_status 表达，不伪装成技术实现类型。
data_status 为 AVAILABLE / UNAVAILABLE，execution_status 为 SUCCESS / UNAVAILABLE / FAILED / SKIPPED。
缺必需数据时 UNAVAILABLE，依赖齐全但未实现时 SKIPPED，reason=NOT_IMPLEMENTED。
trained_until 与 training_end_time 均保留且必须相同，训练截止不能晚于预测时点。
ModelPrediction 保留 match_id、prediction_snapshot_id、input_data_version；存储时核验冻结关联。
CorePrediction 固定未来阶段字段，未执行时派生概率、调整和评级为 null。

所有概率来自真实执行，禁止 LLM 臆测或伪造 Elo、xG、Opta、赔率、阵容和模型结果。
Opta-like 必须标记 LIKE / LIKE_IMPLEMENTATION，不得冒充 Opta 官方模型。
