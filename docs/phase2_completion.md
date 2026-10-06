# PHASE 2 交付报告

项目：二果园足球预测系统。包名保持 erguoyuan_football。
本阶段完成输入、标准化、数据存储、时点快照、完整性报告、未来模型契约和 V5.1 格式契约；不进入 PHASE 3。

## 1. 新增文件

完整逐文件清单见 [phase2_file_manifest.json](phase2_file_manifest.json)。主要分组：

- src/erguoyuan_football/input/：schemas、text_parser、batch_parser、screenshot_parser、match_resolver、包入口。
- src/erguoyuan_football/data/：schemas、store、snapshots、availability、market、包入口。
- src/erguoyuan_football/contracts/：common、predictions、包入口。
- src/erguoyuan_football/external/opta/：provider 和包入口；external 包入口。
- src/erguoyuan_football/models/：requirements 和包入口。
- src/erguoyuan_football/output/v51/：schema、validator、renderer 和包入口；output 包入口。
- src/erguoyuan_football/：daily.py、pipeline.py、__main__.py。
- tests/unit/：test_input、test_contracts、test_v51、test_additional_boundaries。
- tests/pit/test_snapshots.py、tests/integration/test_daily.py。
- tests/fixtures/catalog.json；tests/golden/v51/ 的固定字段、NO-BET 样例和说明。
- examples/phase2_demo.py；docs/phase2_data_layer.md、本报告和文件清单。

生成产物另存：data/football.duckdb、artifacts/phase2-schema.sql、artifacts/phase2-pytest.xml、artifacts/phase2-demo-output.txt。
数据库和运行产物保持 gitignore 排除；未新增模型依赖或真实账户配置。

## 2. 修改文件

1. AGENTS.md：授权范围更新到第二阶段，继续禁止正式模型和自动跨阶段。
2. README.md：安装、每日入口、数据库、截图接口、时点策略与未实现边界。
3. pyproject.toml：DuckDB、Pydantic、pytz 轻量运行依赖。
4. docs/audit_contract.md：具体枚举、空值、时间别名、浮点容差与快照关联。
5. docs/v51_output_contract.md：固定序列化和 Golden Test 约定。
6. src/erguoyuan_football/__init__.py：当前阶段说明。
7. tests/conftest.py：离线合成赛程、数据库和赔率测试夹具。

## 3. 数据库 Schema

实际初始化 18 张表，业务数据库 0 行测试数据。
包含用户指定的 16 张表，另加 competition_aliases、prediction_snapshots。
完整表、键、列、JSON payload 与接口见 [数据库说明](phase2_data_layer.md)。
实际数据库导出的 DDL： [phase2-schema.sql](../artifacts/phase2-schema.sql)。
输入快照具有 snapshot_id、match_id、source、retrieved_at、as_of_time、data_version；
PredictionSnapshot 冻结完整内容，模型与 CORE 存储强制关联该快照。

## 4. Match Input 实际运行示例

运行命令：

```powershell
.\.venv\Scripts\python.exe examples/phase2_demo.py
```

使用明确标记 SYNTHETIC_TEST 的临时数据，不代表真实赛程：

```text
周二001 阿森纳VS切尔西       → test_match_1 / RESOLVED
周二002 皇马VS巴萨          → test_match_2 / RESOLVED
周二003 AC米兰VS国际米兰    → test_match_3 / RESOLVED
```

空库或未知球队返回 NOT_FOUND，多身份/多场候选返回 AMBIGUOUS。
截图没有 provider 时返回 VISION_PROVIDER_UNAVAILABLE，不会生成 OCR 文本。

## 5. Snapshot 实际运行示例

本次合成演示生成：

| prediction_time（UTC） | prediction_snapshot_id |
| --- | --- |
| 2026-09-29 12:00 | 96906b63-5c2b-4e99-920d-fa10c4287858 |
| 2026-09-29 18:00 | 1253a64a-e189-42ae-8aac-6919d3669fe0 |

完整哈希及证据 ID 见 [实际运行输出](../artifacts/phase2-demo-output.txt)。
示例的临时数据库已清理；项目 data/football.duckdb 保持空业务库。
回归测试另外验证 18:00 新报价不改写 12:00 快照，同时间同输入的重复冻结具有相同输入哈希。

## 6. Data Availability 实际运行示例

| 数据项 | 本次合成示例结果 |
| --- | --- |
| Current Odds / Market Odds | AVAILABLE：3 条同版本合成赔率证据 |
| Historical Results / Historical Goals | UNAVAILABLE |
| Opening Odds / Closing Odds | UNAVAILABLE |
| Asian Handicap / Over-Under / Sports Lottery | UNAVAILABLE |
| xG / Lineup / Team Stats / Injuries | UNAVAILABLE |
| Opta 全部产品 | UNAVAILABLE |
| League Hierarchy / Final Probability | UNAVAILABLE |

缺少 REQUIRED 的模型为 UNAVAILABLE；具备最低证据也只返回 SKIPPED / NOT_IMPLEMENTED。
12 个名称全部登记，其中 Supercomputer 是 Simulation Engine，不是独立 META 信号。

## 7. V5.1 Schema 示例

固定 [NO-BET Golden 样例](../tests/golden/v51/no_bet.json) 包含：冻结时间、总表、固定五类、资金汇总。
总表模型概率/MDI/URS/Edge/评级为 null；五类均显示“无合格组合 / NO-BET”。
单场字段保留“投注方向”，总进球栏目保留“总进球数”，不擅自改变上一阶段固定格式。
渲染器不计算赔率、联合概率、EV、评级或资金，也不做投注准入决策。

## 8–9. 测试数量与结果

实际命令（项目根目录）：

```powershell
.\.venv\Scripts\python.exe -m pytest --junitxml=artifacts/phase2-pytest.xml
```

Python 3.11.16；pytest 8.4.2；DuckDB 1.5.6；Pydantic 2.13.5；pytz 2026.4。
**收集 118 项，118 通过，0 失败，0 跳过；29.74 秒。**
机器可读记录：[phase2-pytest.xml](../artifacts/phase2-pytest.xml)。

覆盖输入解析、别名、联赛、歧义、截图置信与来源、跨时区、未来与迟到数据排除、开赛边界、赛程修订、
快照冻结、矛盾版本、市场完整性、历史结果、Opta 不可用、依赖阻断、概率与矩阵校验、快照关联、
V5.1 栏目/字段/顺序变更拒绝、Parquet 读回、数据库重开、CLI 与每日入口。
测试使用合成数据；没有验证真实模型准确率、生产数据覆盖率或业务投注效果。

## 10. 当前尚未实现

- 正式数据源采集、OCR/视觉 provider、真实 Opta 接入；目前只有合法接入接口和显式不可用结果。
- 所有正式预测模型、训练、回测、模拟、META STACKING、Calibration、投注策略。
- 去水计算、MDI/URS、评级阈值、Edge/EV 和资金分配规则。
- 历史档案证明豁免、版本化联赛层级、数据新鲜度和训练样本充分性规则。
- 生产部署、服务鉴权、多进程写入调度及数据库升级迁移。

本次没有开始第三阶段；未来阶段须另行授权。
