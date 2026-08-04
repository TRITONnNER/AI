"""Поведение: лепет, цели, навыки, контуры, размыкатель эффекторов.

В вехе 0 этого не было по плану. Здесь оно появляется потому, что без ответа мира
на действие невозможно проверить ни обратимость, ни ошибку предсказания — а без
них проверять нечего и в вехе 1.
"""

from .babbling import Babbler, Probe, ProbeResult, run_babbling
from .contours import Contour, ContourError, Ran, Scheduler, budgeted, fixed
from .goals import (Candidate, Goal, GoalError, GoalStack, State as GoalState,
                    candidates_from_beliefs, candidates_from_body,
                    candidates_from_error, candidates_from_places, choose)
from .imagination import Breaker, GuardedEffectors, Loop, Mode, Step as LoopStep
from .skills import Library, Skill, SkillError, Step as SkillStep, mine, try_undo, verify

__all__ = ["Babbler", "Probe", "ProbeResult", "run_babbling",
           "Contour", "ContourError", "Ran", "Scheduler", "budgeted", "fixed",
           "Candidate", "Goal", "GoalError", "GoalStack", "GoalState",
           "candidates_from_beliefs", "candidates_from_body",
           "candidates_from_error", "candidates_from_places", "choose",
           "Breaker", "GuardedEffectors", "Loop", "Mode", "LoopStep",
           "Library", "Skill", "SkillError", "SkillStep", "mine", "try_undo", "verify"]
