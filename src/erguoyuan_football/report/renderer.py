"""Chinese human renderer with fixed section and candidate-before-advice order."""

from __future__ import annotations

from erguoyuan_football.report.schemas import CoreReportV2Schema


class CoreReportV2Renderer:
    """Render a report without querying data or changing selections."""

    def render(self, report: CoreReportV2Schema) -> str:
        lines = ["CORE REPORT V2（开发期；非生产预测）", "一、当日全部比赛概率表"]
        for match in report.all_matches:
            label = (f"{match.match_id}｜{match.competition_id}｜{match.home_team_name} VS "
                f"{match.away_team_name}｜开赛 {match.kickoff_time.isoformat() if match.kickoff_time else match.kickoff_status}")
            lines.append(label)
            if match.probability is None:
                lines.append("胜平负：UNAVAILABLE")
            else:
                p = match.probability
                highest = max((("主胜", p.p_home), ("平", p.p_draw), ("客胜", p.p_away)),
                              key=lambda pair: pair[1])[0]
                lines.append(f"胜平负：主胜 {p.p_home:.2%}｜平 {p.p_draw:.2%}｜客胜 {p.p_away:.2%}｜最高方向 {highest}")
            if match.handicap_probabilities is None:
                lines.append("让球胜平负：UNAVAILABLE")
            else:
                h = match.handicap_probabilities
                lines.append(f"让球胜平负：官方主队让球 {match.official_home_handicap:+d}｜"
                    f"让胜 {h['HOME']:.2%}｜让平 {h['DRAW']:.2%}｜让负 {h['AWAY']:.2%}")
            for name, top in (("半全场", match.htft_top2), ("比分", match.score_top2),
                              ("总进球", match.totals_top2)):
                lines.append(f"{name}：UNAVAILABLE" if top is None else
                    f"{name}：TOP1 {top.selections[0]} {top.probabilities[0]:.2%}｜"
                    f"TOP2 {top.selections[1]} {top.probabilities[1]:.2%}")
        lines.append("二、当日最高命中率组合")
        advice_by_id = {item.candidate_id: item for item in report.bet_advice_ledger.advices}
        for slot in report.highest_hit:
            if slot.candidate is None:
                lines.append(f"{slot.play_type}：UNAVAILABLE（{slot.reason}）")
                continue
            candidate = slot.candidate
            lines.append(f"{slot.play_type}｜Candidate {candidate.candidate_id}｜"
                         f"方向 {','.join(candidate.selections)}｜覆盖概率 {candidate.coverage_probability:.2%}｜"
                         f"方法 {candidate.probability_method}")
            advice = advice_by_id.get(candidate.candidate_id)
            lines.append(f"Advice：{advice.advice_status}｜"
                         f"{advice.recommendation or 'UNAVAILABLE'}｜建议金额 {advice.recommended_stake:.2f}" if advice
                         else "Advice：UNAVAILABLE")
        lines.append("三、400元高命中率账户")
        for entry in report.high_hit_400.entries:
            lines.append(f"{entry.slot}｜Candidate {entry.candidate.candidate_id if entry.candidate else 'UNAVAILABLE'}｜"
                f"参考分配 {entry.reference_allocation:.2f}｜建议金额 {entry.recommended_stake:.2f}｜"
                f"Advice {entry.advice.recommendation or entry.advice.advice_status if entry.advice else 'UNAVAILABLE'}")
        lines.append("四、100元自由Value实验账户")
        lines.extend(self._value_lines(report.value_100))
        lines.append("五、20元高赔率小博大账户")
        lines.extend(self._value_lines(report.longshot_20))
        lines.append("六、模型与数据状态")
        for key, value in report.model_data_status.items():
            lines.append(f"{key}：{value}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _value_lines(account) -> list[str]:
        if not account.entries:
            return [account.status]
        lines = []
        for entry in account.entries:
            candidate = entry.candidate
            advice = entry.advice
            if candidate is None:
                lines.append(f"{entry.slot}：UNAVAILABLE")
                continue
            lines.append(f"{entry.slot}｜Candidate {candidate.candidate_id}｜玩法 {candidate.play_type}｜"
                f"方向 {','.join(candidate.selections)}｜模型概率 {candidate.joint_probability:.2%}｜"
                f"市场概率 {candidate.market_probability if candidate.market_probability is not None else 'UNAVAILABLE'}｜"
                f"赔率 {candidate.odds if candidate.odds is not None else 'UNAVAILABLE'}｜"
                f"公平赔率 {candidate.fair_odds if candidate.fair_odds is not None else 'UNAVAILABLE'}｜"
                f"EV {candidate.expected_value if candidate.expected_value is not None else 'UNAVAILABLE'}｜"
                f"不确定性 {candidate.uncertainty if candidate.uncertainty is not None else 'UNAVAILABLE'}｜"
                f"数据质量 {candidate.data_quality}｜参考金额 {entry.reference_allocation:.2f}")
            lines.append(f"Advice：{advice.recommendation or advice.advice_status if advice else 'UNAVAILABLE'}｜"
                         f"建议金额 {entry.recommended_stake:.2f}")
        return lines
