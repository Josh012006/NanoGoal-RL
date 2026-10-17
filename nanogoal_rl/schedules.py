"""Learning-rate schedule that survives a stop/resume.

SB3 hands a schedule the value `progress_remaining = 1 - num_timesteps / total`,
where `total` is the target of the CURRENT `learn()` call. With
`learn(total_timesteps=remaining, reset_num_timesteps=False)` after a resume
that target is `session_end` (the absolute step the session aims for), so a
plain `LinearSchedule` would be re-stretched over the new session: resume at
20M of a 35M budget and it would restart at the schedule's start value.

`StageLinearSchedule` is a function of the stage's ABSOLUTE step counter
instead. It goes linearly from `start` to `end` over `budget` timesteps and then
stays at `end`, whatever the session length. A resumed run therefore continues
exactly where it stopped, and a run extended beyond its original budget simply
keeps `end` until the end.

For a fresh run (`session_end == budget`, the default) it computes the same
values as `stable_baselines3.common.utils.LinearSchedule(start, end, 1.0)`.
"""


class StageLinearSchedule:
    def __init__(self, start: float, end: float, budget: int, session_end: int = None) -> None:
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        self.start = float(start)
        self.end = float(end)
        self.budget = int(budget)
        # Absolute step the current learn() call is aiming for (the `total`
        # SB3 divides by). Defaults to the budget, i.e. a fresh run.
        self.session_end = int(session_end) if session_end is not None else int(budget)

    def _value_at_fraction(self, fraction: float) -> float:
        # Same expression as SB3's LinearSchedule (end_fraction=1.0).
        if fraction > 1.0:
            return self.end
        return self.start + fraction * (self.end - self.start)

    def value_at(self, step: float) -> float:
        """Learning rate at an absolute timestep of the stage."""
        return self._value_at_fraction(step / self.budget)

    def __call__(self, progress_remaining: float) -> float:
        elapsed = 1.0 - progress_remaining  # fraction of the CURRENT session's target
        if self.session_end == self.budget:
            # Fresh run: identical, bit for bit, to SB3's LinearSchedule.
            return self._value_at_fraction(elapsed)
        return self.value_at(elapsed * self.session_end)

    def spec(self) -> dict:
        """What must be stored in a checkpoint to rebuild the same schedule."""
        return {"type": "linear", "start": self.start, "end": self.end, "budget": self.budget}

    @classmethod
    def from_spec(cls, spec: dict, session_end: int = None) -> "StageLinearSchedule":
        if spec.get("type") != "linear":
            raise ValueError(f"Unsupported learning-rate schedule type: {spec.get('type')!r}")
        return cls(spec["start"], spec["end"], spec["budget"], session_end=session_end)

    def __repr__(self) -> str:
        return (f"StageLinearSchedule(start={self.start}, end={self.end}, "
                f"budget={self.budget}, session_end={self.session_end})")
