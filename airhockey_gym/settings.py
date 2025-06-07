from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EnvSettings:
    width: int = 690
    height: int = 370
    fps: int = 30

    puck_radius: int = 20
    paddle_radius: int = 27
    goal_height: int = 115

    friction: float = 0.99
    max_velocity: int = 750
    puck_stopped_threshold: float = 10
    max_steps: int = 1000
    action_scale: float = 333

    background_color: List[int] = field(default_factory=lambda: [255, 255, 255])
    border_color: List[int] = field(default_factory=lambda: [0, 0, 0])
    puck_color: List[int] = field(default_factory=lambda: [0, 0, 255])
    paddle_color: List[int] = field(default_factory=lambda: [255, 0, 0])

    reward_win: float = +100.0
    reward_lose: float = -100.0
    reward_default: float = -0.1
    reward_alignment: float = 0.2
    reward_y_alignment: float = 0.1

    render_mode: Optional[str] = 'human'
