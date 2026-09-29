"""Scalable GHZ-state preparation via lattice surgery (tqec), with patch partitions for Chipmunq.

Construction (ZX picture)
-------------------------
An N-qubit GHZ state is a single Z spider with N output legs. We realise it as

* a *routing bus* of spatial ``XXZ`` cubes (a spatial Z spider = the ancilla region of a
  multi-patch ZZ...Z measurement), and
* N *data patches* of kind ``XZZ`` attached to the bus by a Y-pipe. Each data cube has
  exactly two legs (bus + time), so as an X spider it is an identity wire and the whole
  structure fuses into one N-legged Z spider, i.e. a GHZ state.

Why not a plain Z-spider tree with 2D junctions? A time-extended cube only admits pipes
along two directions, so 2D junctions must be spatial cubes; connecting a spatial Z
junction to a time-extended Z spider forces a Hadamard pipe, which breaks the fusion.
The bus construction avoids this and is also the standard "data block + routing
region" layout of Litinski's Game of Surface Codes.

Layouts
-------
``"bus"`` (default, 2D, scalable)
    ``num_bands`` bands stacked in y. Each band = data row / bus row / data row.
    For more than one band, the bus rows are joined by a vertical spine at x = 0.

        y=3b+2   D D D D           D = data patch (logical qubit)
        y=3b+1 S B B B B           B = bus block, S = spine block
        y=3b     D D D D

    With more than one band, the top of the spine is capped by one extra data patch
    (keeps all tqec blocks regular, see build_ghz_block_graph).

``"line"``
    1D chain of ``XZX`` patches merged directly with their neighbours (no bus). Useful as
    a minimal baseline.

Timeline (in tqec blocks, each block = d syndrome rounds)
    t = 0                      : data initialisation + multi-patch merge through the bus
    t = 1 .. memory_blocks     : idle memory on the data patches
    t = memory_blocks + 1      : final block, transversal readout in ``basis``

Observables: ``basis="Z"`` gives the N-1 independent Z_i Z_j stabilisers of the GHZ
state; ``basis="X"`` gives X^{(x)N}.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import stim
from tqec import compile_block_graph
from tqec.computation.block_graph import BlockGraph
from tqec.utils.noise_model import NoiseModel
from tqec.utils.position import Position3D

ROLE_DATA = "data"
ROLE_BUS = "bus"


@dataclass(frozen=True)
class GHZLayout:
    """Spatial description of the generated block graph (independent of the distance)."""

    block_graph: BlockGraph
    # (x, y) block position -> role ("data" or "bus")
    roles: dict[tuple[int, int], str]
    # (x, y) block position of a data patch -> logical qubit index
    logical_index: dict[tuple[int, int], int]


# --------------------------------------------------------------------------------------
# Block graph construction
# --------------------------------------------------------------------------------------


def _split_evenly(n: int, parts: int) -> list[int]:
    base, rem = divmod(n, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _default_num_bands(n: int) -> int:
    # Aim for a roughly square footprint: width ~ n/(2B) blocks, height ~ 3B blocks.
    return max(1, round(math.sqrt(n / 6)))


def build_ghz_block_graph(
    n: int,
    layout: str = "bus",
    num_bands: int | None = None,
    memory_blocks: int = 1,
    basis: str = "Z",
) -> GHZLayout:
    """Build the tqec block graph of an N-qubit GHZ preparation.

    :param n: number of logical qubits (>= 2)
    :param layout: ``"bus"`` (2D, scalable) or ``"line"`` (1D direct merges)
    :param num_bands: number of data/bus/data bands for the bus layout (default: ~sqrt(n/6))
    :param memory_blocks: idle blocks between the merge and the final readout (>= 0)
    :param basis: readout basis, ``"Z"`` or ``"X"``
    """
    if n < 2:
        raise ValueError("A GHZ state needs n >= 2 logical qubits.")
    if memory_blocks < 0:
        raise ValueError("memory_blocks must be >= 0.")
    basis = basis.upper()
    if basis not in ("Z", "X"):
        raise ValueError("basis must be 'Z' or 'X'.")

    g = BlockGraph(f"GHZ_{n}_{layout}")
    roles: dict[tuple[int, int], str] = {}
    logical_index: dict[tuple[int, int], int] = {}
    t_final = memory_blocks + 1

    def add_data_column(x: int, y: int, merge_kind: str, idle_kind: str, final_kind: str) -> None:
        """Data patch at (x, y): merge at t=0, idle blocks, readout block on top."""
        g.add_cube(Position3D(x, y, 0), merge_kind)
        for t in range(1, t_final + 1):
            kind = final_kind if t == t_final else idle_kind
            g.add_cube(Position3D(x, y, t), kind)
            g.add_pipe(Position3D(x, y, t - 1), Position3D(x, y, t))
        roles[(x, y)] = ROLE_DATA
        logical_index[(x, y)] = len(logical_index)

    if layout == "line":
        # XZX = Z spider with pipes allowed along X and time -> directly fusable chain.
        final = "XZZ" if basis == "Z" else "XZX"
        for x in range(n):
            add_data_column(x, 0, "XZX", "XZX", final)
            if x > 0:
                g.add_pipe(Position3D(x - 1, 0, 0), Position3D(x, 0, 0))

    elif layout == "bus":
        bands = num_bands if num_bands is not None else _default_num_bands(n)
        bands = max(1, min(bands, (n - 1) // 2))  # every band needs >= 2 data patches
        # With several bands, one logical qubit sits on top of the spine (see below).
        per_band = _split_evenly(n - 1 if bands > 1 else n, bands)
        x0 = 1 if bands > 1 else 0  # column 0 is reserved for the spine
        final = "XZZ" if basis == "Z" else "XZX"

        def add_bus(x: int, y: int) -> None:
            g.add_cube(Position3D(x, y, 0), "XXZ")  # spatial Z spider
            roles[(x, y)] = ROLE_BUS

        for b, nb in enumerate(per_band):
            y_bot, y_bus, y_top = 3 * b, 3 * b + 1, 3 * b + 2
            n_cols = math.ceil(nb / 2)
            placed = 0
            for c in range(n_cols):
                x = x0 + c
                add_bus(x, y_bus)
                if c > 0:
                    g.add_pipe(Position3D(x - 1, y_bus, 0), Position3D(x, y_bus, 0))
                for y in (y_bot, y_top):
                    if placed == nb:
                        break
                    # XZZ = X spider, pipes along Y and time -> identity wire onto the bus.
                    add_data_column(x, y, "XZZ", "XZZ", final)
                    g.add_pipe(Position3D(x, y, 0), Position3D(x, y_bus, 0))
                    placed += 1

        if bands > 1:
            # Vertical spine joining the bus rows of all bands.
            y_last_bus = 3 * (bands - 1) + 1
            for y in range(1, y_last_bus + 1):
                add_bus(0, y)
                if y > 1:
                    g.add_pipe(Position3D(0, y - 1, 0), Position3D(0, y, 0))
            for b in range(bands):
                g.add_pipe(Position3D(0, 3 * b + 1, 0), Position3D(1, 3 * b + 1, 0))
            # A spatial bus block whose only vertical pipe points to -y is emitted by tqec
            # with one unused boundary qubit missing (24 instead of 25 at d=3), which breaks
            # Chipmunq's column-wise patch placement. Capping the spine with a data patch
            # keeps every block regular; that patch is simply one of the N GHZ qubits.
            add_data_column(0, y_last_bus + 1, "XZZ", "XZZ", final)
            g.add_pipe(Position3D(0, y_last_bus, 0), Position3D(0, y_last_bus + 1, 0))
    else:
        raise ValueError(f"Unknown layout '{layout}' (use 'bus' or 'line').")

    g.validate()
    return GHZLayout(block_graph=g, roles=roles, logical_index=logical_index)


# --------------------------------------------------------------------------------------
# Partition extraction (replaces the hand-written index lists)
# --------------------------------------------------------------------------------------


def extract_partitions(stim_circuit: stim.Circuit, layout: GHZLayout, distance_scale: int) -> list[dict]:
    """Derive Chipmunq patch partitions from the qubit coordinates of a tqec circuit.

    tqec places the block at spatial position (X, Y) on qubit coordinates
    [p*X, p*X + 2d] x [p*Y, p*Y + 2d] with pitch p = 2d + 2. The single coordinate
    line p*X + 2d + 1 between two neighbouring blocks holds the d measurement qubits that
    only exist when the blocks are joined by a pipe. Qubits are therefore assigned to

    * the block (patch) they lie in -> type ``rotated_surface_code`` (role data / bus), or
    * the pipe strip between two blocks -> type ``rotated_surface_code_ancilla``.

    The union over time is taken implicitly: a block position is one partition no matter
    how many time slices it occupies.

    ``width``/``height`` follow Chipmunq's mapper convention (identical to the hand-written
    CNOT partitions at d=3 and d=5): patches are (d+1) x (2d+1); pipe strips are (d+1) x 1 when
    horizontal and 1 x (2d+1) when vertical. The vertical value deliberately differs from
    the hand-written d=3/d=5 CNOT partitions (1 x (d+1)), which under-reserve rows for the
    mapper's vertical placement walk; it matches the d=7 hand-written partition.
    """
    d = 2 * distance_scale + 1
    pitch = 2 * d + 2
    coords = stim_circuit.get_final_qubit_coordinates()

    patch_qubits: dict[tuple[int, int], list[int]] = defaultdict(list)
    strip_qubits: dict[tuple[tuple[int, int], tuple[int, int]], list[int]] = defaultdict(list)
    xy_of: dict[int, tuple[float, float]] = {}

    for q, c in coords.items():
        x, y = int(round(c[0])), int(round(c[1]))
        xy_of[q] = (x, y)
        bx, rx = divmod(x, pitch)
        by, ry = divmod(y, pitch)
        in_x, in_y = rx <= 2 * d, ry <= 2 * d
        if in_x and in_y:
            patch_qubits[(bx, by)].append(q)
        elif not in_x and in_y:
            strip_qubits[((bx, by), (bx + 1, by))].append(q)
        elif in_x and not in_y:
            strip_qubits[((bx, by), (bx, by + 1))].append(q)
        else:
            raise RuntimeError(f"Qubit {q} at {c} lies on a strip corner; unexpected for this layout.")

    # Consistency checks against the block graph.
    unknown = set(patch_qubits) - set(layout.roles)
    if unknown:
        raise RuntimeError(f"Qubits found at unoccupied block positions {sorted(unknown)}; "
                           "the coordinate convention of tqec may have changed.")
    g = layout.block_graph
    spatial_pipes = set()
    for pipe in g.pipes:
        u, v = pipe.u.position, pipe.v.position
        if u.z == v.z:  # spatial pipe, at any time step
            spatial_pipes.add(tuple(sorted(((u.x, u.y), (v.x, v.y)))))
    for (a, b) in strip_qubits:
        if (a, b) not in spatial_pipes:
            raise RuntimeError(f"Pipe-strip qubits between {a} and {b}, but no pipe exists there.")

    def _by_coord(qs: list[int]) -> list[int]:
        # Chipmunq walks the indices column by column (x, then y); make that explicit.
        return sorted(qs, key=lambda q: (xy_of[q][0], xy_of[q][1]))

    partitions: list[dict] = []
    for pos in layout.roles:
        part = {
            "indices": _by_coord(patch_qubits[pos]),
            # Chipmunq convention (mapper.map_partition_on_qpu): a patch occupies d+1
            # physical columns of 2d+1 rows (tqec columns 2c and 2c+1 share column c).
            "width": d + 1,
            "height": 2 * d + 1,
            "distance": d,
            "type": "rotated_surface_code",
            "role": layout.roles[pos],
            "position": pos,
        }
        if pos in layout.logical_index:
            part["logical_qubit"] = layout.logical_index[pos]
        partitions.append(part)
    for (a, b) in sorted(strip_qubits):
        vertical = a[1] == b[1]  # pipe along x -> strip is a column of qubits (fixed x)
        partitions.append({
            "indices": _by_coord(strip_qubits[(a, b)]),
            # width == 1 -> placed vertically, height == 1 -> placed horizontally.
            # A vertical strip must reserve 2d+1 rows: the mapper writes it at rows
            # local_y + 2d, local_y + 2d - 2, ... (as the d=7 hand-written partition does);
            # reserving only d+1 rows lets it overlap neighbours or leave the chiplet.
            "width": 1 if vertical else d + 1,
            "height": 2 * d + 1 if vertical else 1,
            "distance": d,
            "type": "rotated_surface_code_ancilla",
            "role": "pipe",
            "position": (a, b),
        })

    # Order partitions column-major by block position (x, then y), with every pipe strip
    # sorted between the two blocks it joins. Chipmunq's placement heuristic anchors each
    # partition on its BFS predecessor, so the order matters: on the 2-band N=7 GHZ this
    # ordering needed 34% fewer SWAPs than "data first, then bus, then strips".
    def _order_key(p: dict) -> tuple[float, float]:
        pos = p["position"]
        if isinstance(pos[0], tuple):
            (ax, ay), (bx, by) = pos
            return ((ax + bx) / 2, (ay + by) / 2)
        return (float(pos[0]), float(pos[1]))

    partitions.sort(key=_order_key)

    validate_partitions(stim_circuit, partitions)
    check_chipmunq_compatibility(stim_circuit, partitions)
    return partitions


# Code distances for which Chipmunq's mapper defines a placement offset
# (mapper.map_partition_on_qpu; the branches meant for d=11 and d=13 test d == 9).
CHIPMUNQ_SUPPORTED_DISTANCES = (3, 5, 7, 9, 15)


def check_chipmunq_compatibility(stim_circuit: stim.Circuit, partitions: list[dict]) -> None:
    """Assert that every partition matches the shapes Chipmunq's mapper places.

    * ``rotated_surface_code``: rectangular (2d+1) x (2d+1) checkerboard, i.e. tqec
      columns alternating d+1 / d qubits, ordered column by column. This is exactly the
      order in which the mapper fills its d+1 x (2d+1) physical rectangle.
    * ``rotated_surface_code_ancilla``: d qubits on one straight line (a single
      coordinate column for width == 1, a single coordinate row for height == 1).
    """
    coords = stim_circuit.get_final_qubit_coordinates()
    for p in partitions:
        d = p["distance"]
        if d not in CHIPMUNQ_SUPPORTED_DISTANCES:
            raise ValueError(f"Chipmunq's mapper has no placement offsets for d={d} "
                             f"(supported: {CHIPMUNQ_SUPPORTED_DISTANCES}).")
        xy = [(int(round(coords[q][0])), int(round(coords[q][1]))) for q in p["indices"]]
        if xy != sorted(xy):
            raise AssertionError(f"Partition {p['position']}: indices not in column-major order.")
        if p["type"] == "rotated_surface_code":
            cols: dict[int, list[int]] = defaultdict(list)
            for x, y in xy:
                cols[x].append(y)
            xs = sorted(cols)
            x0, y0 = xs[0], min(y for _, y in xy)
            expected = {x0 + i: [y0 + j for j in range(i % 2, 2 * d + 1, 2)] for i in range(2 * d + 1)}
            if {x: sorted(v) for x, v in cols.items()} != expected:
                raise AssertionError(f"Patch {p['role']} at {p['position']} is not a full (2d+1)x(2d+1) "
                                     f"rotated patch ({len(xy)} qubits).")
        else:
            line = {x for x, _ in xy} if p["width"] == 1 else {y for _, y in xy}
            if len(xy) != d or len(line) != 1:
                raise AssertionError(f"Pipe strip at {p['position']} is not a straight line of d qubits.")




def validate_partitions(stim_circuit: stim.Circuit, partitions: list[dict]) -> None:
    """Check that the partitions are an exact, disjoint cover of all circuit qubits."""
    seen: set[int] = set()
    for p in partitions:
        overlap = seen.intersection(p["indices"])
        if overlap:
            raise AssertionError(f"Qubits {sorted(overlap)[:10]} are in more than one partition.")
        seen.update(p["indices"])
    all_q = set(stim_circuit.get_final_qubit_coordinates())
    if seen != all_q:
        raise AssertionError(f"Partitions miss {len(all_q - seen)} qubits / contain {len(seen - all_q)} extra.")


# --------------------------------------------------------------------------------------
# Public entry point (same return convention as QECCircuit.single_cnot_full_memory)
# --------------------------------------------------------------------------------------


def generate_ghz_lattice_surgery(
    n: int,
    distance_scale: int = 1,
    layout: str = "bus",
    num_bands: int | None = None,
    memory_blocks: int = 1,
    basis: str = "Z",
    noise_model: NoiseModel | None = None,
    manhattan_radius: int = 2,
    output_file: str | None = None,
) -> tuple[stim.Circuit, list[dict]]:
    """Generate an N-qubit GHZ lattice-surgery circuit and its Chipmunq partitions.

    :param n: number of logical qubits
    :param distance_scale: tqec scale k, code distance d = 2k + 1
    :param noise_model: optional tqec noise model (e.g. ``NoiseModel.uniform_depolarizing(1e-3)``);
        leave ``None`` if Chipmunq inserts noise after routing
    :param output_file: if given, the stim circuit is written to this path
    :return: (stim circuit, partitions)
    """
    ghz = build_ghz_block_graph(n, layout=layout, num_bands=num_bands, memory_blocks=memory_blocks, basis=basis)
    compiled = compile_block_graph(ghz.block_graph)
    circuit = compiled.generate_stim_circuit(k=distance_scale, noise_model=noise_model, manhattan_radius=manhattan_radius)
    partitions = extract_partitions(circuit, ghz, distance_scale)

    if output_file is not None:
        with open(output_file, "w") as f:
            print(circuit, file=f)
    return circuit, partitions


def get_tqec_ghz(n: int, distance_scale: int = 1, **kwargs) -> tuple[stim.Circuit, list[dict]]:
    """Drop-in analogue of ``get_tqec_cnot_rotated`` for the experiment scripts."""
    return generate_ghz_lattice_surgery(n, distance_scale=distance_scale,
                                        output_file=f"stim_ghz_n_{n}_d_{2 * distance_scale + 1}.stim", **kwargs)