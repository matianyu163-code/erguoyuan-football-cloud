"""Compatibility entry point for point-in-time fatigue and schedule features."""

from erguoyuan_football.context.schedule_context import ScheduleContextBuilder


class FatigueContextBuilder(ScheduleContextBuilder):
    """Expose verified rest and congestion features; no heuristic fatigue score."""
