"""
OpenGL-based 3D world viewer integrated with the simulation.

Renders terrain, arena, and UAVs using GLRenderer and exposes a RGB frame
that can be embedded into the UI.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.rendering.renderer import GLRenderer
from src.simulation.arena import TeknofestArena


class GLWorldViewer:
    """Render a 3D overview of the simulation world to an RGB frame."""

    def __init__(
        self,
        width: int,
        height: int,
        world,
        arena_config: Optional[Dict[str, Any]] = None,
        camera_config: Optional[Dict[str, Any]] = None,
    ):
        self.width = width
        self.height = height
        self._world = world
        self._camera_config = camera_config or {}

        self.follow_distance = float(self._camera_config.get("follow_distance", 180.0))
        self.follow_height = float(self._camera_config.get("follow_height", 70.0))
        self.look_height = float(self._camera_config.get("look_height", 20.0))

        self.renderer = GLRenderer(width, height)
        self.renderer.camera.set_projection(
            fov=60.0,
            aspect_ratio=width / height,
            near=0.1,
            far=2000.0,
        )

        if getattr(world, "environment", None) is not None:
            self.renderer.init_environment(world.environment)

        arena_config = arena_config or {}
        self.renderer.init_arena(TeknofestArena(arena_config))

    @staticmethod
    def _sim_to_gl(position: np.ndarray) -> np.ndarray:
        """Map sim coords (x, y, z-alt) -> GL coords (x, y-up, z-depth)."""
        return np.array([position[0], position[2], position[1]], dtype=np.float32)

    def _select_target(self, world_state: Dict[str, Any], target_id: Optional[str]) -> Optional[Dict[str, Any]]:
        uavs = world_state.get("uavs", {})
        if target_id and target_id in uavs:
            return uavs[target_id]

        player_id = world_state.get("player_id")
        if player_id and player_id in uavs:
            return uavs[player_id]

        if uavs:
            return next(iter(uavs.values()))

        return None

    def _update_camera(self, target: Optional[Dict[str, Any]]):
        if target is None:
            eye = np.array([0.0, 80.0, -220.0], dtype=np.float32)
            target_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            pos = np.array(target.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
            heading_deg = float(target.get("heading", 0.0))
            heading_rad = np.radians(heading_deg)

            forward = np.array([np.cos(heading_rad), np.sin(heading_rad), 0.0], dtype=np.float32)
            offset = -forward * self.follow_distance
            offset[2] += self.follow_height

            eye = self._sim_to_gl(pos + offset)
            target_pos = self._sim_to_gl(pos + np.array([0.0, 0.0, self.look_height], dtype=np.float32))

        self.renderer.camera.look_at(eye, target_pos, up=np.array([0.0, 1.0, 0.0], dtype=np.float32))

    def _render_uavs(self, world_state: Dict[str, Any], program=None):
        uavs = world_state.get("uavs", {})
        player_id = world_state.get("player_id")

        for uav_id, uav in uavs.items():
            pos = np.array(uav.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
            orientation = uav.get("orientation") or [0.0, 0.0, np.radians(uav.get("heading", 0.0))]
            roll, pitch, yaw = orientation

            team = uav.get("team")
            if uav_id == player_id:
                color = (0.2, 0.9, 0.2)
            elif team == "blue":
                color = (0.2, 0.4, 0.9)
            elif team == "red":
                color = (0.9, 0.2, 0.2)
            else:
                color = (0.8, 0.8, 0.8)

            self.renderer.render_aircraft(
                position=pos,
                heading=yaw,
                roll=roll,
                pitch=pitch,
                color=color,
                program=program,
            )

    def render(self, world_state: Dict[str, Any], target_id: Optional[str] = None) -> np.ndarray:
        """Render the world and return an RGB frame (H, W, 3)."""
        target = self._select_target(world_state, target_id)
        self._update_camera(target)

        # Shadow pass (optional for better depth cues)
        if hasattr(self.renderer, "begin_shadow_pass"):
            self.renderer.begin_shadow_pass()
            self._render_uavs(world_state, program=self.renderer.prog_shadow)

        self.renderer.begin_frame()
        self._render_uavs(world_state)
        self.renderer.end_frame()

        buffer = self.renderer.fbo.read(components=3, alignment=1)
        frame = np.frombuffer(buffer, dtype=np.uint8).reshape((self.height, self.width, 3))
        frame = np.flipud(frame)
        return frame
