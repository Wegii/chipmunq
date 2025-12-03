# Visualization
import matplotlib.pyplot as plt

import numpy as np


class QPUBlock:
    """Class representing a QPU chiplet with no-placement zones"""

    def __init__(self,
                 width: int,
                 height: int,
                 block_coord: tuple,
                 no_placement_zones: list[tuple] = None,
                 patch_initialization: str = "center"):
        
        self.width = width
        self.height = height
        self.coord = block_coord

        print(f"block has width {width} and height {height}")

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

    
    def _overlaps_partitions(self, x, y, w, h):
        for pid, px, py, pw, ph in self.placed_partitions:
            if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y):
                return True
        return False

    def _overlaps_forbidden(self, x, y, w, h):
        """ Check if any forbidden (fx, fy) lies inside the placement rectangle.

        :param x: _description_
        :type x: _type_
        :param y: _description_
        :type y: _type_
        :param w: _description_
        :type w: _type_
        :param h: _description_
        :type h: _type_
        :return: _description_
        :rtype: _type_
        """

        for fx, fy in self.no_placement_zones:
            if x <= fx < x + w and y <= fy < y + h:
                return True
            
        print("not allowed")
        return False

    def _overlaps(self, x, y, w, h):
        return self._overlaps_partitions(x, y, w, h) or self._overlaps_forbidden(x, y, w, h)

    def _find_covering_free_rect(self, x, y, w, h):
        for i, (fx, fy, fw, fh) in enumerate(self.free_rects):
            if (x >= fx and y >= fy and 
                x + w <= fx + fw and 
                y + h <= fy + fh):
                return i, (fx, fy, fw, fh)
        return None, None

    def _split_free_rect(self, index, fx, fy, fw, fh, x, y, w, h):
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
    
    def place_partition(self, partition_id, pw, ph):
        
        if self.patch_initialization == "center":
            # Preferred center-based placement
            preferred_x = (self.width - pw) // 2
            preferred_y = (self.height - ph) // 2
        elif self.patch_initialization == "size_aware":
            # Preferred origin placement
            preferred_x = 0
            preferred_y = (self.height - ph)

        # ------------------------------------------------
        # 1. Try fully centered placement (preferred_x, preferred_y)
        # ------------------------------------------------
        idx, rect = self._find_covering_free_rect(preferred_x, preferred_y, pw, ph)
        if idx is not None and not self._overlaps(preferred_x, preferred_y, pw, ph):
            fx, fy, fw, fh = rect
            self.placed_partitions.append((partition_id, preferred_x, preferred_y, pw, ph))
            self._split_free_rect(idx, fx, fy, fw, fh, preferred_x, preferred_y, pw, ph)

            print(f"Placed {partition_id} at centered ({preferred_x}, {preferred_y})")
            return (self.coord[0] + preferred_x, self.coord[1] + preferred_y)

        # ------------------------------------------------
        # 2. Try the original free-rectangle-based heuristic
        # ------------------------------------------------
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

        # ------------------------------------------------
        # 3. Fallback: full-grid brute-force search
        # ------------------------------------------------
        print(f"Standard placement failed for {partition_id}, performing grid search...")

        for y in range(0, self.height - ph + 1):
            for x in range(0, self.width - pw + 1):

                # Overlaps forbidden/partitions?
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

    # ----------------------------------------------------------
    # MODIFIED: Relative placement with iterative search (shift)
    # ----------------------------------------------------------
    def place_relative(self, partition_id, pw, ph, anchor_id, direction, max_shift=5):
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

        # --- Search Loop ---
        # Search starting from 0 shift up to max_shift
        for shift in range(max_shift + 1):
            
            # Calculate current placement attempt (x, y) based on shift
            x, y = base_x, base_y
            
            if shift > 0:
                if direction in ("below", "above"):#("right", "left"):
                    # Shift vertically
                    y += shift # Try shifting 'up' first (lower y value is higher on screen/chip)
                    
                    # NOTE: You could also implement a strategy to try shifting in the
                    # opposite direction (e.g., y -= shift) or both, but a single
                    # incremental shift is a common simple heuristic.
                
                elif direction in ("right", "left"):#("below", "above"):
                    # Shift horizontally
                    x += shift # Try shifting 'right' first
                    
                    # NOTE: As above, a strategy could include x -= shift

            # 1. Bounds check
            if x < 0 or y < 0 or x + pw > self.width or y + ph > self.height:
                # If even the base position (shift=0) is out of bounds, we fail immediately.
                # For shift > 0, we simply stop this iteration.
                continue

            # 2. Overlap check (with partitions AND forbidden zones)
            if self._overlaps(x, y, pw, ph):
                continue # Try next shift

            # 3. Find free rect that fully contains this placement
            idx, rect = self._find_covering_free_rect(x, y, pw, ph)
            if idx is None:
                continue # Try next shift

            # If all checks pass, we have found a valid placement!
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
    

def plot_block_counts(width: int,
                      height: int,
                      block_assignments: dict,
                      filename: str
                    ) -> None:
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
            ax.text(x + 0.5, height - y - 0.5, str(count),
                    ha='center', va='center', fontsize=12)

    # Draw grid lines
    ax.set_xticks(np.arange(0, width + 1, 1))
    ax.set_yticks(np.arange(0, height + 1, 1))
    ax.grid(True)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def dimension_to_linear_index(x, w, h):
    x1, x2 = x 

    idx = x2 * h + x1
    return idx

def linear_index_to_dimension(idx, w):
    x = idx % w
    y = idx // w
    return x, y