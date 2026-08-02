"""Qt list model projecting `aim_fsm.worldmap` objects for QML."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Optional

from PyQt6.QtCore import QAbstractListModel, QModelIndex, Qt, pyqtProperty, pyqtSignal, pyqtSlot

from aim_fsm.worldmap import (
    AprilTagObj,
    ArucoMarkerObj,
    BarrelObj,
    BlueBarrelObj,
    DominoObj,
    OrangeBarrelObj,
    SportsBallObj,
    WallObj,
)

RoleMap = Dict[int, bytes]
Item = Dict[str, Any]


def _role(index: int) -> int:
    return int(Qt.ItemDataRole.UserRole) + index


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _pose_attr(obj: Any, attr: str, default: float = 0.0) -> float:
    pose = getattr(obj, "pose", None)
    if pose is None and isinstance(obj, Mapping):
        pose = obj.get("pose")
    if pose is None:
        return float(default)

    value = getattr(pose, attr, None)
    if value is None and isinstance(pose, Mapping):
        value = pose.get(attr)
    if value is None and hasattr(pose, "__getitem__"):
        try:
            value = pose[attr]
        except Exception:
            value = None
    return _to_float(value, default)


def _theta_attr(obj: Any) -> float:
    theta = _pose_attr(obj, "theta", 0.0)
    if theta is None:
        return 0.0
    return _to_float(theta, 0.0)


class WorldMapModel(QAbstractListModel):
    """`QAbstractListModel` exposing canonical world objects to QML."""

    countChanged = pyqtSignal()

    ROLE_NAMES: tuple[str, ...] = (
        "id",
        "type",
        "x",
        "y",
        "z",
        "theta",
        "visible",
        "missing",
        "diameter_mm",
        "height_mm",
        "length_mm",
        "width_mm",
        "thickness_mm",
        "size_mm",
        "marker_id",
        "doorways",
        "holding",
        "face_label",
        "domino_halves",
        "is_fallen",
    )

    _ROLE_MAP: RoleMap = {
        _role(i + 1): name.encode("utf-8") for i, name in enumerate(ROLE_NAMES)
    }

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._items: list[Item] = []

    @pyqtProperty(int, notify=countChanged)
    def count(self) -> int:
        return len(self._items)

    def roleNames(self) -> RoleMap:
        return self._ROLE_MAP

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self._items)

    def data(self, index: QModelIndex, role: int = 0) -> Any:
        if not index.isValid():
            return None
        row = index.row()
        if row < 0 or row >= len(self._items):
            return None
        name = self._ROLE_MAP.get(role)
        if name is None:
            return None
        key = name.decode("utf-8")
        return self._items[row].get(key)

    def sync_from(self, robot: Any, objects: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> None:
        entries: list[Item] = []

        robot_entry = self._build_robot(robot)
        if robot_entry is not None:
            entries.append(robot_entry)

        if hasattr(objects, "items"):
            iterator = getattr(objects, "items")()
        else:
            iterator = objects

        for key, obj in sorted(iterator, key=lambda pair: str(pair[0])):
            item = self._build_object(str(key), obj)
            if item is not None:
                entries.append(item)

        self.beginResetModel()
        self._items = entries
        self.endResetModel()
        self.countChanged.emit()

    @pyqtSlot(int, result="QVariant")
    def get(self, row: int) -> Optional[Item]:
        if row < 0 or row >= len(self._items):
            return None
        return dict(self._items[row])

    def _build_robot(self, robot: Any) -> Optional[Item]:
        if robot is None:
            return None
        entry: Item = {
            "id": "robot#1",
            "type": "robot",
            "x": _pose_attr(robot, "x", 0.0),
            "y": _pose_attr(robot, "y", 0.0),
            "z": _pose_attr(robot, "z", 0.0),
            "theta": _theta_attr(robot),
            "visible": True,
            "missing": False,
            "diameter_mm": 64.0,
            "height_mm": 72.0,
            "length_mm": None,
            "width_mm": None,
            "thickness_mm": None,
            "size_mm": None,
            "marker_id": None,
            "doorways": [],
            "holding": bool(getattr(robot, "holding", False)),
            "face_label": None,
            "domino_halves": [],
            "is_fallen": False,
        }
        return entry

    def _build_object(self, object_id: str, obj: Any) -> Optional[Item]:
        if obj is None:
            return None

        type_name = self._resolve_type(obj)
        if type_name is None or type_name == "robot":
            return None

        entry: Item = {
            "id": object_id,
            "type": type_name,
            "x": _pose_attr(obj, "x", 0.0),
            "y": _pose_attr(obj, "y", 0.0),
            "z": _pose_attr(obj, "z", 0.0),
            "theta": _theta_attr(obj),
            "visible": bool(getattr(obj, "is_visible", False)),
            "missing": bool(getattr(obj, "is_missing", False)),
            "diameter_mm": None,
            "height_mm": None,
            "length_mm": None,
            "width_mm": None,
            "thickness_mm": None,
            "size_mm": None,
            "marker_id": None,
            "doorways": [],
            "holding": None,
            "face_label": None,
            "domino_halves": [],
            "is_fallen": bool(getattr(obj, "is_fallen", False)),
        }

        if type_name == "domino":
            is_fallen = bool(getattr(obj, "is_fallen", False))
            length = _to_float(getattr(obj, "length", getattr(obj, "length_mm", 48.0)), 48.0)
            width = _to_float(getattr(obj, "width", getattr(obj, "width_mm", 24.0)), 24.0)
            thickness = _to_float(getattr(obj, "thickness", getattr(obj, "thickness_mm", 8.0)), 8.0)

            if is_fallen:
                # Fallen flat geometry:
                # Flat on table top face up. Z-center = 4.0mm
                entry["length_mm"] = thickness  # X dimension
                entry["width_mm"] = length      # Y dimension 
                entry["height_mm"] = width      # Z dimension
                entry["thickness_mm"] = thickness
                entry["z"] = 4.0
            else:
                # Upright Landscape standing geometry:
                entry["length_mm"] = thickness  # X dimension (Depth)
                entry["width_mm"] = length      # Y dimension (Horizontal Width)
                entry["height_mm"] = width      # Z dimension (Standing Height)
                entry["thickness_mm"] = thickness
                entry["z"] = width / 2.0        # Elevate 12mm
                
            face_label = getattr(obj, "face_label", None)
            entry["face_label"] = face_label
            entry["domino_halves"] = self._build_domino_halves(obj, length, face_label)

        return entry

    def _build_domino_halves(self, obj: Any, length: float, face_label: Optional[str]) -> list[Item]:
        half_span = float(length) / 4.0 if length else 12.0
        counts = getattr(obj, "half_counts", None)
        if face_label == "?" or not counts or len(counts) != 2 or any(count is None for count in counts):
            return [
                {"count": None, "local_y": -half_span},
                {"count": None, "local_y": half_span},
            ]

        return [
            {"count": int(counts[0]), "local_y": -half_span},
            {"count": int(counts[1]), "local_y": half_span},
        ]

    @staticmethod
    def _resolve_type(obj: Any) -> Optional[str]:
        if isinstance(obj, DominoObj):
            return "domino"
        name = getattr(obj, "name", "")
        if isinstance(name, str) and "domino" in name.lower():
            return "domino"
        return None


__all__ = ["WorldMapModel"]