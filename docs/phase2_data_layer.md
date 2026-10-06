# PHASE 2 数据库与接口

此阶段提供本地单进程类型化接口，不是生产部署或真实数据连接器。
`Store(path)` 创建 DuckDB，支持上下文管理。公用写入接口只追加；管理员直接执行 SQL 不在接口安全边界内。
记录保留完整 JSON payload；索引时间列使用 TIMESTAMPTZ，连接统一 UTC。

## 18 张正式表

| 表 | 键及主要列 | 接口 |
| --- | --- | --- |
| teams | team_id PK，team_name | add_team / team |
| team_aliases | (alias, team_id, source, created_at) PK，normalized_alias、language、confidence；team_id FK | add_team_alias / alias_ids |
| competitions | competition_id PK，competition_name、country、competition_type、season_format、tier | add_competition / competition |
| competition_aliases | (alias, competition_id, source, created_at) PK，language、confidence；competition_id FK | add_competition_alias / alias_ids(competition=True) |
| matches | (match_id, data_version) PK，lottery_match_no、competition_id、home_team_id、away_team_id、kickoff_time、source、retrieved_at、as_of_time | add_fixture / fixtures_at / fixture_at |
| match_requests | request_id PK，created_at、input_type、match_id、resolution_status，payload 保存所有原始与解析字段 | save_request |
| odds_snapshots | snapshot_id PK，统一快照列，OddsSnapshot payload | add_snapshot / snapshots_at |
| market_snapshots | snapshot_id PK，统一快照列，OddsSnapshot payload | add_snapshot / snapshots_at |
| team_stats_snapshots | snapshot_id PK，统一快照列，team_id 与数值 metrics payload | add_snapshot / snapshots_at |
| xg_snapshots | snapshot_id PK，统一快照列，team_id 与 xg/xga metrics payload | add_snapshot / snapshots_at |
| lineup_snapshots | snapshot_id PK，统一快照列，team_id、player_ids、confirmed payload | add_snapshot / snapshots_at |
| injury_snapshots | snapshot_id PK，统一快照列，team_id、player_ids payload | add_snapshot / snapshots_at |
| opta_snapshots | snapshot_id PK，统一快照列，product、access_basis、payload | add_snapshot / snapshots_at |
| prediction_snapshots | prediction_snapshot_id PK，match_id、created_at、prediction_time，完整冻结 payload | SnapshotService.create / load_prediction_snapshot |
| model_predictions | prediction_id PK，match_id、prediction_snapshot_id FK、prediction_time，ModelPrediction payload | save_prediction |
| oos_predictions | 同上，另验证 is_oos 与较早 training_end_time | save_prediction(table="oos_predictions") |
| final_predictions | 同上，CorePrediction payload | save_prediction(table="final_predictions") |
| match_results | result_id PK，(match_id, data_version) 唯一，home_goals、away_goals、completed_at、source、retrieved_at、as_of_time | add_result / results_at |

统一输入快照列：`snapshot_id, match_id, source, retrieved_at, as_of_time, data_version, availability, payload`。
prediction_snapshots 为聚合冻结表，使用 prediction_snapshot_id，来源时间保存在嵌入记录中。
球队和联赛使用数据库外键；输入快照与结果的 match_id 通过 Store 验证，因为 matches 保留多版本，match_id 单列不唯一。
`schema_description()` 可查看实际类型与约束。

OddsSnapshot payload 完整保存 bookmaker、market_type、phase、selection、line、odds、raw_implied_probability、devig_probability。
market_type 为 1X2 / ASIAN_HANDICAP / OVER_UNDER / SPORTS_LOTTERY；phase 为 OPEN / CURRENT / CLOSE。
亚洲盘和大小球必须有 line，去水字段默认为 null，转换接口明确 NOT_IMPLEMENTED。
记录可写入 odds_snapshots 或 market_snapshots，聚合按同一语义流去重。同时间矛盾版本报错，不择优。

## 真实数据写入和 Parquet

依次构造 Team、TeamAlias、Competition、CompetitionAlias、Fixture 并调用 add 方法，
再按确定的 match_id 写入市场、统计、xG、阵容、伤病和外部记录。
source 必须是真实来源定位，不能以测试夹具替代。数据提供者须确认来源真实性及访问权限。
没有合法 Opta 配置使用 UnavailableOptaProvider；此版本不自动下载真实赛程。

`export_parquet(table, destination)` 导出白名单表并拒绝覆盖文件，路径通过关系 API 传入。
`store.connection.read_parquet(path)` 可直接读回；导入正式表须经过类型化写入接口重验。
Parquet 保存 JSON payload 与时间索引，不需要 PyArrow 或 pandas。

## 快照示例与时点规则

```python
from datetime import datetime, timezone
from erguoyuan_football.data.store import Store
from erguoyuan_football.data.snapshots import SnapshotService

with Store("data/football.duckdb") as store:
    # 必须已有真实来源赛程；空库不会生成该比赛。
    snapshot = SnapshotService(store).create(
        "已确认的比赛ID", datetime(2026, 9, 29, 12, tzinfo=timezone.utc))
    print(snapshot.prediction_snapshot_id)
    print(snapshot.input_data_version)
    print(snapshot.data_completeness.model_dump_json())
```

快照在同一事务内读写，保存实际内容而非可变引用。先按 as_of_time 和 retrieved_at 筛选，再选当时最新版本。
所有输入满足 `max(as_of_time, retrieved_at) <= prediction_time < kickoff_time`。
历史结果额外要求 completed_at < prediction_time，且排除当前待预测比赛结果。
开赛时及之后的预测请求拒绝；未来原始快照可存储，但不能被更早预测使用。
input_data_version 为冻结输入 SHA-256，不依赖新建 UUID 和创建时间；同数据重复冻结可比对。
保存预测时校验 match_id、snapshot_id、prediction_time 与输入版本，不允许跨快照混用。

本阶段没有训练。OOS 的 is_oos 与训练截止核验只是接口防线，不能证明未来训练算法确实排除了样本；
模型阶段仍须逐折审计训练样本 ID 和时间窗口，不能仅靠 is_oos 标志声明无泄漏。

## DataAvailabilityReport

每项保存 availability、reason、evidence_ids，不填造默认数值。

- 历史进球/结果：有已结束、当时可取得的结果，覆盖本场双方。
- 市场：同来源、公司、玩法、阶段、盘口、as_of_time、data_version 的完整选项集合。
- xG：双方均有 xg 与 xga；阵容：双方均有 confirmed 名单；统计、伤病同样要求双方来源记录。
- Opta：只有带合法访问依据的真实载荷可为 AVAILABLE，否则 UNAVAILABLE。
- league_hierarchy：静态 tier 不是历史特征证明，暂为 UNAVAILABLE，等待版本化来源。
- final_probability：没有执行模型，保持 UNAVAILABLE。

AVAILABLE 只表示最低证据完整，不等于可训练。样本量、时效、首发人数等领域规则尚未定义，不擅自设阈值。
缺 REQUIRED 返回 UNAVAILABLE；全部必需证据存在仍是 SKIPPED / NOT_IMPLEMENTED。

## 可运行合成示例

`python examples/phase2_demo.py` 演示三行输入、标准 match_id、12:00 / 18:00 快照、可用性报告、空概率 CORE 和 V5.1 NO-BET。
示例始终标记 SYNTHETIC_TEST，使用临时目录，不是真实比赛或赔率。实际输出保存在本阶段交付记录中。
