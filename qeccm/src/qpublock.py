# Visualization
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


class QPUBlock:
    """Class representing a QPU chiplet with no-placement zones"""

    def __init__(
        self,
        width: int,
        height: int,
        block_coord: tuple,
        no_placement_zones: list[tuple] = None,
        patch_initialization: str = "center",
    ):

        self.width = width
        self.height = height
        self.coord = block_coord

        # List of (x, y) points that cannot be used
        self.no_placement_zones = no_placement_zones if no_placement_zones is not None else []

        # free rectangles inside block
        self.free_rects = [(0, 0, width, height)]

        # list of (partition_id, x, y, w, h)
        self.placed_partitions = []

        # Location where to assign the first partition:
        # - center: Place at the center
        # - origin: Place at the origin
        if patch_initialization == "" or None:
            self.patch_initialization = "center"
        else:
            self.patch_initialization = patch_initialization

    def _overlaps_partitions(self, x: int, y: int, w: int, h: int) -> bool:
        """Check if partitions overlap

        :param self: QPUBlock
        :param x: x location
        :type x: int
        :param y: y location
        :type y: int
        :param w: width of partition
        :type w: int
        :param h: height of partition
        :type h: int
        :return: Whether partition overlaps or not
        :rtype: bool
        """

        for pid, px, py, pw, ph in self.placed_partitions:
            if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y):
                return True
        return False

    def _overlaps_forbidden(self, x: int, y: int, w: int, h: int) -> bool:
        """Check if any forbidden (fx, fy) lies inside the placement rectangle.

        :param x: x location
        :type x: int
        :param y: y location
        :type y: int
        :param w: width
        :type w: int
        :param h: height
        :type h: int
        :return: Check overlap with no placement zone
        :rtype: bool
        """
        for fx, fy in self.no_placement_zones:
            if x <= fx < x + w and y <= fy < y + h:
                return True

        print("not allowed")
        return False

    def _overlaps(self, x: int, y: int, w: int, h: int) -> bool:
        """Check if partition overlaps other partitions or no placement zones

        :param x: x location
        :type x: int
        :param y: y location
        :type y: int
        :param w: width
        :type w: int
        :param h: height
        :type h: int
        :return: Check overlap
        :rtype: bool
        """

        return self._overlaps_partitions(x, y, w, h) or self._overlaps_forbidden(x, y, w, h)

    def _find_covering_free_rect(
        self, x: int, y: int, w: int, h: int
    ) -> Optional[Tuple[int, Tuple[int, int, int, int]]]:
        """Find existing free rectangle that completely contains the target area.

        :param self: Description
        :param x: x location
        :type x: int
        :param y: y location
        :type y: int
        :param w: width of partition
        :type w: int
        :param h: height of partition
        :type h: int
        :return: A tuple containing the index and the geometry (x, y, w, h) of the covering rectangle, or (None, None)
          if no valid container is found.
        :rtype: Tuple[int, Tuple[int, int, int, int]] | None
        """

        for i, (fx, fy, fw, fh) in enumerate(self.free_rects):
            if x >= fx and y >= fy and x + w <= fx + fw and y + h <= fy + fh:
                return i, (fx, fy, fw, fh)
        return None, None

    def _split_free_rect(self, index: int, fx: int, fy: int, fw: int, fh: int, x: int, y: int, w: int, h: int) -> None:
        """Splits a free rectangle into smaller sub-rectangles after a partition is placed.

        :param self: Description
        :param index: The index of the free rectangle to be split.
        :type index: int
        :param fx: The x-coordinate of the existing free rectangle.
        :type fx: int
        :param fy: The y-coordinate of the existing free rectangle.
        :type fy: int
        :param fw: The width of the existing free rectangle.
        :type fw: int
        :param fh: The height of the existing free rectangle.
        :type fh: int
        :param x: x location
        :type x: int
        :param y: y location
        :type y: int
        :param w: width of partition
        :type w: int
        :param h: height of partition
        :type h: int
        """

        del self.free_rects[index]

        # left side
        if x > fx:
            self.free_rects.append((fx, fy, x - fx, fh))

        # right side
        if x + w < fx + fw:
            self.free_rects.append((x + w, fy, (fx + fw) - (x + w), fh))

        # top side (above the partition) - constrained to partition's width
        if y > fy:
            self.free_rects.append((x, fy, w, y - fy))

        # bottom side (below the partition) - constrained to partition's width
        if y + h < fy + fh:
            self.free_rects.append((x, y + h, w, (fy + fh) - (y + h)))

    def place_partition(self, partition_id: int, pw: int, ph: int) -> Optional[Tuple[int, int]]:
        """Attempt to place a partition within the available free space

        :param self: Description
        :param partition_id: id of partition
        :type partition_id: int
        :param pw: width of partition
        :type pw: int
        :param ph: height of partition
        :type ph: int
        :return: The absolute (x, y) coordinates of the placed partition if successful, otherwise None.
        :rtype: Tuple[int, int] | None
        """

        if self.patch_initialization == "center":
            # Preferred center-based placement
            preferred_x = (self.width - pw) // 2
            preferred_y = (self.height - ph) // 2
        elif self.patch_initialization == "size_aware":
            # Preferred origin placement
            preferred_x = 0
            preferred_y = self.height - ph

        idx, rect = self._find_covering_free_rect(preferred_x, preferred_y, pw, ph)
        if idx is not None and not self._overlaps(preferred_x, preferred_y, pw, ph):
            fx, fy, fw, fh = rect
            self.placed_partitions.append((partition_id, preferred_x, preferred_y, pw, ph))
            self._split_free_rect(idx, fx, fy, fw, fh, preferred_x, preferred_y, pw, ph)

            print(f"Placed {partition_id} at centered ({preferred_x}, {preferred_y})")
            return (self.coord[0] + preferred_x, self.coord[1] + preferred_y)

        # Iterate over free rectangles to search for a free place
        for i, (fx, fy, fw, fh) in enumerate(self.free_rects):
            if pw <= fw and ph <= fh:
                # Use preferred_y if it fits vertically into this free-rect
                if fy <= preferred_y and preferred_y + ph <= fy + fh:
                    y_to_check = preferred_y
                else:
                    y_to_check = fy

                # Use preferred_x if it fits horizontally into this free-rect
                if fx <= preferred_x and preferred_x + pw <= fx + fw:
                    x_to_check = preferred_x
                else:
                    x_to_check = fx

                if not self._overlaps(x_to_check, y_to_check, pw, ph):
                    x, y = x_to_check, y_to_check

                    self.placed_partitions.append((partition_id, x, y, pw, ph))
                    self._split_free_rect(i, fx, fy, fw, fh, x, y, pw, ph)

                    print(f"Placed {partition_id} at ({x}, {y}) using heuristic")
                    return (self.coord[0] + x, self.coord[1] + y)

        # Brute-force search over the whole block
        print(f"Standard placement failed for {partition_id}, performing grid search...")

        for y in range(self.height - ph + 1):
            for x in range(self.width - pw + 1):
                # Check for overlaps
                if self._overlaps(x, y, pw, ph):
                    continue

                idx, rect = self._find_covering_free_rect(x, y, pw, ph)
                if idx is None:
                    continue

                fx, fy, fw, fh = rect
                self.placed_partitions.append((partition_id, x, y, pw, ph))
                self._split_free_rect(idx, fx, fy, fw, fh, x, y, pw, ph)

                print(f"Placed {partition_id} at ({x}, {y}) via grid search")
                return (self.coord[0] + x, self.coord[1] + y)

        print(f"Placement for {partition_id} failed: no free slot available.")
        return None

    def place_relative(
        self, partition_id: int, pw: int, ph: int, anchor_id: str, direction: str, max_shift: int = 5
    ) -> Optional[Tuple[int, int]]:
        """Place a partition relative to an existing anchor

        :param self: Description
        :param partition_id: partition oid
        :type partition_id: int
        :param pw: partition width
        :type pw: int
        :param ph: partition height
        :type ph: int
        :param anchor_id: id of partition to place relative to
        :type anchor_id: str
        :param direction: direction of placement
        :type direction: str
        :param max_shift: maximum offset to test for placement on the same QPUBlock
        :type max_shift: int
        :return: The absolute (x, y) coordinates of the newly placed partition if successful, or None if no valid space
          is found within the shift constraints.
        :rtype: Tuple[int, int] | None
        """

        anchor = next((p for p in self.placed_partitions if p[0] == anchor_id), None)
        if anchor is None:
            raise ValueError(f"Anchor partition {anchor_id} not found.")

        _, ax, ay, aw, ah = anchor

        # Initial target position (shift=0)
        if direction == "right":
            base_x, base_y = ax + aw, ay
        elif direction == "left":
            base_x, base_y = ax - pw, ay
        elif direction == "below":
            base_x, base_y = ax, ay + ah
        elif direction == "above":
            base_x, base_y = ax, ay - ph
        else:
            raise ValueError("Direction must be one of: right/left/above/below")

        # Search starting from 0 shift up to max_shift
        for shift in range(max_shift + 1):
            # Calculate current placement attempt (x, y) based on shift
            x, y = base_x, base_y

            if shift > 0:
                if direction in ("below", "above"):  # ("right", "left"):
                    # Shift vertically (up first)
                    y += shift

                elif direction in ("right", "left"):  # ("below", "above"):
                    # Shift horizontally (right first)
                    x += shift

            # Bounds check
            if x < 0 or y < 0 or x + pw > self.width or y + ph > self.height:
                continue

            # Overlap check (with partitions AND forbidden zones)
            if self._overlaps(x, y, pw, ph):
                continue  # Try next shift

            # Find free rect that fully contains this placement
            idx, rect = self._find_covering_free_rect(x, y, pw, ph)
            if idx is None:
                continue  # Try next shift

            # Valid placement found
            fx, fy, fw, fh = rect

            # Place partition
            self.placed_partitions.append((partition_id, x, y, pw, ph))

            # Split the free rectangle
            self._split_free_rect(idx, fx, fy, fw, fh, x, y, pw, ph)

            print(f"Placed {partition_id} at ({x}, {y}) with shift {shift} (Direction: {direction})")

            return (self.coord[0] + x, self.coord[1] + y)

        # If the loop finishes without finding a valid position
        print(f"Placement for {partition_id} failed: No valid position found within {max_shift} units of shift.")
        return None


def plot_block_counts(width: int, height: int, block_assignments: dict, filename: str) -> None:
    """Plot 2D grid of QPUs with the number of assigned partitions shown

    :param width: _description_
    :type width: int
    :param height: _description_
    :type height: int
    :param block_assignments: _description_
    :type block_assignments: dict
    :param filename: _description_
    :type filename: str
    """

    fig, ax = plt.subplots(figsize=(width, height))

    # Loop through each grid block and get number of partitions per QPU
    for y in range(height):
        for x in range(width):
            count = len(block_assignments.get((x, y), []).placed_partitions)
            ax.text(x + 0.5, height - y - 0.5, str(count), ha="center", va="center", fontsize=12)

    # Draw grid lines
    ax.set_xticks(np.arange(0, width + 1, 1))
    ax.set_yticks(np.arange(0, height + 1, 1))
    ax.grid(True)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def dimension_to_linear_index(x, w, h):
    x1, x2 = x

    idx = x2 * h + x1
    return idx


def linear_index_to_dimension(idx, w):
    x = idx % w
    y = idx // w
    return x, y
