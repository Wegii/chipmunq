"""Scalable Bernstein-Vazirani (BV) circuits via lattice surgery (tqec), with patch partitions
for Chipmunq.

Construction (ZX picture)
-------------------------
The textbook BV circuit for a secret s in {0,1}^n is

    |0>^n |1>  --H^(n+1)--  oracle: CNOT(i -> anc) for every s_i = 1  --H^n--  measure Z

Conjugating the oracle by the Hadamards flips every CNOT, H CNOT(i -> a) H = CNOT(a -> i), and
the H's cancel on the data qubits. The circuit is therefore equivalent to

    ancilla |1>,  data |0>^n,  fan-out CNOT(anc -> i) for every s_i = 1,  measure data in Z,

so data qubit i ends in |s_i>. In ZX the fan-out is a single Z spider on the ancilla wire with one
leg to an X spider on every data wire with s_i = 1. In lattice surgery this is a single multi-patch
Z...Z merge, realised exactly like the GHZ bus:

* a *routing bus* of spatial ``XXZ`` cubes (a spatial Z spider = the ancilla region of a
  multi-patch Z...Z measurement),
* one *data patch* of kind ``XZZ`` per secret bit. Patches with s_i = 1 are attached to the bus by
  a Y-pipe (X spider with two legs = identity wire, the fan-out target). Patches with s_i = 0 are
  plain memory and are *not* merged; they still occupy their slot so the floorplan of the data
  register does not depend on the secret,
* one *ancilla patch* (the fan-out control) of kind ``XZZ`` attached to the bus. It is a leaf
  cube living for a single block: initialised in the Z basis, merged, and read out in Z. This
  one-legged X spider on the Z bus is what distinguishes BV from GHZ: the control enters in a
  known Z eigenstate instead of |+>, so every data qubit gets an individually deterministic
  Z value instead of only pairwise Z_i Z_j correlations.

Layout
------
``num_bands`` bands stacked in y, each data row / bus row / data row, joined by a vertical spine at
x = 0 whose top is capped by the ancilla patch:

        y=3b+2   A D D D D         A = ancilla (fan-out control, one block in time), on top
        y=3b+1   S B B B B             of the spine (shown here for the top band)
        y=3b       D D D D         D = data patch (logical qubit i <-> secret bit s_i)
                                   B = bus block, S = spine block (lower bands: S at y=3b+2 too)

With ``trim_bus=True`` (default) bus blocks that serve no merged data patch are dropped, so the
footprint shrinks for sparse secrets. With ``trim_bus=False`` the bus spans every data column and
the floorplan is independent of the secret.

Timeline (in tqec blocks, each block = d syndrome rounds)
    t = 0                      : data init (Z), ancilla init (Z) + multi-patch Z...Z merge
    t = 1 .. memory_blocks     : idle memory on the data patches
    t = memory_blocks + 1      : final block, transversal Z readout of the data patches

Observables
    Exactly n, with observable i belonging to secret bit i:
        s_i = 1 : Z_i (x) Z_anc  (plus the merge / bus stabiliser outcomes that form the frame)
        s_i = 0 : Z_i
    These generate the logical stabiliser group of the BV output (data qubit i ends in |s_i>,
    ancilla in |1>), so observable i flipping is exactly "bit i of the secret decoded wrongly".
    tqec only sees the Clifford skeleton: whether the ancilla starts in |0> or |1> is a logical
    Pauli frame that changes neither detectors nor observable definitions.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import stim
from tqec import compile_block_graph
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface
from tqec.utils.noise_model import NoiseModel
from tqec.utils.position import Position3D

ROLE_DATA = "data"
ROLE_BUS = "bus"
ROLE_ANCILLA = "ancilla"


@dataclass(frozen=True)
class BVLayout:
    """Spatial description of the generated block graph (independent of the distance)."""

    block_graph: BlockGraph
    secret: tuple[int, ...]
    # (x, y) block position -> role ("data", "bus" or "ancilla")
    roles: dict[tuple[int, int], str]
    # (x, y) block position of a data patch -> logical qubit index (= index of the secret bit)
    logical_index: dict[tuple[int, int], int]
    # (x, y) block position of the ancilla patch
    ancilla: tuple[int, int]


# --------------------------------------------------------------------------------------
# Block graph construction
# --------------------------------------------------------------------------------------


def parse_secret(secret: str | Sequence[int]) -> tuple[int, ...]:
    """Accept ``"1011"`` or ``[1, 0, 1, 1]``; bit i of the result is secret bit s_i."""
    bits = tuple(int(c) for c in secret) if isinstance(secret, str) else tuple(int(b) for b in secret)
    if not bits:
        raise ValueError("The secret needs at least one bit.")
    if any(b not in (0, 1) for b in bits):
        raise ValueError(f"Secret must be a bit string, got {secret!r}.")
    return bits


def _split_evenly(n: int, parts: int) -> list[int]:
    base, rem = divmod(n, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _default_num_bands(n: int) -> int:
    # Aim for a roughly square footprint: width ~ n/(2B) blocks, height ~ 3B blocks.
    return max(1, round(math.sqrt(n / 6)))


def build_bv_block_graph(
    secret: str | Sequence[int],
    num_bands: int | None = None,
    memory_blocks: int = 1,
    trim_bus: bool = True,
) -> BVLayout:
    """Build the tqec block graph of a Bernstein-Vazirani circuit for ``secret``.

    :param secret: secret bit string s (``"1011"`` or a sequence of 0/1); n = len(s) data qubits
    :param num_bands: number of data/bus/data bands (default: ~sqrt(n/6))
    :param memory_blocks: idle blocks between the oracle merge and the final readout (>= 0)
    :param trim_bus: drop bus blocks that serve no merged data patch
    """
    s = parse_secret(secret)
    n = len(s)
    if memory_blocks < 0:
        raise ValueError("memory_blocks must be >= 0.")

    bands = num_bands if num_bands is not None else _default_num_bands(n)
    bands = max(1, min(bands, max(1, n // 2)))  # every band gets >= 2 data slots when n >= 2
    per_band = _split_evenly(n, bands)

    g = BlockGraph(f"BV_{''.join(map(str, s))}")
    roles: dict[tuple[int, int], str] = {}
    logical_index: dict[tuple[int, int], int] = {}
    t_final = memory_blocks + 1

    # --- data register: fixed slots, logical qubit i <-> secret bit s_i ------------------
    # slots[b] = list of (x, y, i) for band b, filled column by column (bottom, then top).
    slots: list[list[tuple[int, int, int]]] = []
    i = 0
    for b, nb in enumerate(per_band):
        y_bot, y_top = 3 * b, 3 * b + 2
        band_slots = []
        for c in range(math.ceil(nb / 2)):
            for y in (y_bot, y_top):
                if len(band_slots) == nb:
                    break
                band_slots.append((1 + c, y, i))
                i += 1
        slots.append(band_slots)

    for band_slots in slots:
        for x, y, q in band_slots:
            g.add_cube(Position3D(x, y, 0), "XZZ")
            for t in range(1, t_final + 1):
                g.add_cube(Position3D(x, y, t), "XZZ")
                g.add_pipe(Position3D(x, y, t - 1), Position3D(x, y, t))
            roles[(x, y)] = ROLE_DATA
            logical_index[(x, y)] = q

    # --- oracle: bus rows, spine, ancilla ------------------------------------------------
    def add_bus(x: int, y: int) -> None:
        g.add_cube(Position3D(x, y, 0), "XXZ")  # spatial Z spider
        roles[(x, y)] = ROLE_BUS

    band_has_bus = []
    for b, band_slots in enumerate(slots):
        y_bus = 3 * b + 1
        linked_cols = [x for x, _, q in band_slots if s[q]]
        all_cols = sorted({x for x, _, _ in band_slots})
        if trim_bus:
            last = max(linked_cols, default=0)
        else:
            last = max(all_cols, default=0)
        band_has_bus.append(last > 0)
        for x in range(1, last + 1):
            add_bus(x, y_bus)
            if x > 1:
                g.add_pipe(Position3D(x - 1, y_bus, 0), Position3D(x, y_bus, 0))
        for x, y, q in band_slots:
            if s[q]:  # fan-out target: XZZ = X spider, identity wire onto the bus
                g.add_pipe(Position3D(x, y, 0), Position3D(x, y_bus, 0))

    # Spine at x = 0, up to the highest band that has a bus row, capped by the ancilla.
    # Bus blocks that tqec emits with a missing corner qubit are padded afterwards
    # (pad_irregular_patches).
    top_band = max((b for b, has in enumerate(band_has_bus) if has), default=0)
    y_top_spine = 3 * top_band + 1
    for y in range(1, y_top_spine + 1):
        add_bus(0, y)
        if y > 1:
            g.add_pipe(Position3D(0, y - 1, 0), Position3D(0, y, 0))
    for b, has in enumerate(band_has_bus):
        if has:
            g.add_pipe(Position3D(0, 3 * b + 1, 0), Position3D(1, 3 * b + 1, 0))

    # Ancilla (fan-out control): XZZ leaf = one-legged X spider on the Z bus.
    anc = (0, y_top_spine + 1)
    g.add_cube(Position3D(anc[0], anc[1], 0), "XZZ")
    g.add_pipe(Position3D(0, y_top_spine, 0), Position3D(anc[0], anc[1], 0))
    roles[anc] = ROLE_ANCILLA

    g.validate()
    return BVLayout(block_graph=g, secret=s, roles=roles, logical_index=logical_index, ancilla=anc)


# --------------------------------------------------------------------------------------
# Observables
# --------------------------------------------------------------------------------------


def find_observables(g: BlockGraph) -> list[CorrelationSurface]:
    """Correlation surfaces of *every* connected component of ``g``.

    ``BlockGraph.find_correlation_surfaces`` (tqec 0.2) combines disconnected components by a
    Cartesian product, which yields only prod_c |gens_c| surfaces instead of sum_c |gens_c|. With
    two lone memory columns it returns one observable (Z_1 Z_2) instead of two. The unmerged s_i = 0
    patches are exactly such components, so we collect the generators component by component and
    pad each with the identity surface on the rest of the graph.
    """
    # Private tqec API (pinned to the behaviour of tqec 0.2.x); checked by the observable count
    # assertion in generate_bv_lattice_surgery.
    from tqec.computation._correlation import _CorrelationSurfaceView, _find_correlation_surfaces
    from tqec.utils.enums import Pauli

    zx = g.to_zx_graph().g
    identity = {u: {v: Pauli.I for v in zx.neighbors(u)} for u in zx.vertices()}
    surfaces = []
    for component in _find_correlation_surfaces(zx):
        for cs in component:
            surfaces.append(_CorrelationSurfaceView(cs, identity)._to_immutable_public_representation(zx))
    return surfaces


def _measurement_qubits(circuit: stim.Circuit) -> list[int]:
    """Qubit measured by every measurement record, in record order."""
    out: list[int] = []
    for inst in circuit.flattened():
        if stim.gate_data(inst.name).produces_measurements:
            if inst.name.startswith("MPP"):
                raise NotImplementedError("MPP is not expected in tqec circuits.")
            out.extend(t.value for t in inst.targets_copy())
    return out


def _observable_records(circuit: stim.Circuit) -> dict[int, set[int]]:
    """Absolute measurement indices of each observable (XOR semantics)."""
    obs: dict[int, set[int]] = defaultdict(set)
    count = 0
    for inst in circuit.flattened():
        if stim.gate_data(inst.name).produces_measurements:
            count += len(inst.targets_copy())
        elif inst.name == "OBSERVABLE_INCLUDE":
            o = int(inst.gate_args_copy()[0])
            for t in inst.targets_copy():
                obs[o] ^= {count + t.value}
    return obs


def _block_of(coord: Sequence[float], pitch: int) -> tuple[int, int]:
    return int(round(coord[0])) // pitch, int(round(coord[1])) // pitch


def canonicalize_observables(circuit: stim.Circuit, layout: BVLayout, distance_scale: int) -> stim.Circuit:
    """Rewrite the observables so that observable i is the one of secret bit i.

    tqec returns an arbitrary generating set (for merged qubits it looks like GHZ generators,
    Z_i Z_j). Over GF(2) we recombine them so that observable i contains the readout of data
    patch i and of no other data patch, which gives Z_i (s_i = 0) or Z_i Z_anc (s_i = 1).
    """
    d = 2 * distance_scale + 1
    pitch = 2 * d + 2
    coords = circuit.get_final_qubit_coordinates()
    mq = _measurement_qubits(circuit)
    obs = _observable_records(circuit)
    n = len(layout.secret)
    if sorted(obs) != list(range(n)):
        raise RuntimeError(f"Expected {n} observables from tqec, got {len(obs)}.")

    pos_of_index = {q: pos for pos, q in layout.logical_index.items()}

    def data_signature(recs: set[int]) -> int:
        sig = 0
        for r in recs:
            pos = _block_of(coords[mq[r]], pitch)
            if pos in layout.logical_index:
                # Membership, not parity: one logical readout contributes d records.
                sig |= 1 << layout.logical_index[pos]
        return sig

    # Gaussian elimination over GF(2). Record sets are XORed and the signature is recomputed
    # from the result, so partially cancelled readouts would show up (and fail the check below).
    basis: dict[int, set[int]] = {}
    for o in range(n):
        recs = set(obs[o])
        sig = data_signature(recs)
        while sig:
            pivot = sig.bit_length() - 1
            if pivot not in basis:
                break
            recs ^= basis[pivot]
            sig = data_signature(recs)
        if not sig:
            raise RuntimeError("tqec observables are not independent on the data readouts.")
        basis[pivot] = recs
    # Back-substitute to reduce every row to a single data patch.
    for pivot in sorted(basis):
        for other in basis:
            if other != pivot and data_signature(basis[other]) >> pivot & 1:
                basis[other] ^= basis[pivot]
    for q in range(n):
        if data_signature(basis[q]) != 1 << q:
            raise RuntimeError(f"Could not isolate data qubit {q} ({pos_of_index[q]}).")

    # Rebuild the circuit: drop tqec's OBSERVABLE_INCLUDEs, append the canonical ones at the end.
    new = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            if any(i.name == "OBSERVABLE_INCLUDE" for i in inst.body_copy().flattened()):
                raise NotImplementedError("OBSERVABLE_INCLUDE inside a REPEAT block.")
            new.append(inst)
        elif inst.name != "OBSERVABLE_INCLUDE":
            new.append(inst)
    total = len(mq)
    for q in range(n):
        targets = [stim.target_rec(r - total) for r in sorted(basis[q])]
        new.append("OBSERVABLE_INCLUDE", targets, q)
    return new


def _expected_patch_coords(pos: tuple[int, int], d: int) -> set[tuple[int, int]]:
    """Qubit coordinates of a full (2d+1) x (2d+1) rotated patch at block position ``pos``."""
    pitch = 2 * d + 2
    x0, y0 = pitch * pos[0], pitch * pos[1]
    return {(x0 + i, y0 + j) for i in range(2 * d + 1) for j in range(i % 2, 2 * d + 1, 2)}


def pad_irregular_patches(circuit: stim.Circuit, layout: BVLayout, distance_scale: int) -> tuple[stim.Circuit, int]:
    """Declare the unused boundary qubit that tqec omits on some spatial bus blocks.

    tqec drops one unused corner position of a spatial (``XXZ``) block whose pipes are
    {one pipe only}, {+x, -y} or {-x, +y}. In the GHZ layout this was avoided by construction;
    for BV the bus shape depends on the secret (unmerged s_i = 0 patches, trimmed bus), so these
    configurations do occur. The qubit carries no operation, so declaring it (QUBIT_COORDS only)
    changes no detector or observable, but it makes every patch the regular (2d+1)^2
    checkerboard that Chipmunq's mapper expects.
    """
    d = 2 * distance_scale + 1
    present = {(int(round(c[0])), int(round(c[1]))) for c in circuit.get_final_qubit_coordinates().values()}
    next_q = circuit.num_qubits
    header = stim.Circuit()
    added = 0
    for pos in layout.roles:
        missing = _expected_patch_coords(pos, d) - present
        if not missing:
            continue
        if layout.roles[pos] != ROLE_BUS or len(missing) > 1:
            raise RuntimeError(f"Unexpected missing qubits {sorted(missing)} in {layout.roles[pos]} "
                               f"patch at {pos}; tqec's layout convention may have changed.")
        for x, y in sorted(missing):
            header.append("QUBIT_COORDS", [next_q], [x, y])
            next_q += 1
            added += 1
    return header + circuit, added


# --------------------------------------------------------------------------------------
# Partition extraction (unchanged from the GHZ generator; roles now include "ancilla")
# --------------------------------------------------------------------------------------


def extract_partitions(stim_circuit: stim.Circuit, layout: BVLayout, distance_scale: int) -> list[dict]:
    """Derive Chipmunq patch partitions from the qubit coordinates of a tqec circuit.

    tqec places the block at spatial position (X, Y) on qubit coordinates
    [p*X, p*X + 2d] x [p*Y, p*Y + 2d] with pitch p = 2d + 2. The single coordinate
    line p*X + 2d + 1 between two neighbouring blocks holds the d measurement qubits that
    only exist when the blocks are joined by a pipe. Qubits are therefore assigned to

    * the block (patch) they lie in -> type ``rotated_surface_code`` (role data / bus / ancilla), or
    * the pipe strip between two blocks -> type ``rotated_surface_code_ancilla``.

    ``width``/``height`` follow Chipmunq's mapper convention: patches are (d+1) x (2d+1);
    pipe strips are (d+1) x 1 when horizontal and 1 x (2d+1) when vertical.
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

    unknown = set(patch_qubits) - set(layout.roles)
    if unknown:
        raise RuntimeError(f"Qubits found at unoccupied block positions {sorted(unknown)}; "
                           "the coordinate convention of tqec may have changed.")
    g = layout.block_graph
    spatial_pipes = set()
    for pipe in g.pipes:
        u, v = pipe.u.position, pipe.v.position
        if u.z == v.z:
            spatial_pipes.add(tuple(sorted(((u.x, u.y), (v.x, v.y)))))
    for (a, b) in strip_qubits:
        if (a, b) not in spatial_pipes:
            raise RuntimeError(f"Pipe-strip qubits between {a} and {b}, but no pipe exists there.")

    def _by_coord(qs: list[int]) -> list[int]:
        return sorted(qs, key=lambda q: (xy_of[q][0], xy_of[q][1]))

    partitions: list[dict] = []
    for pos in layout.roles:
        part = {
            "indices": _by_coord(patch_qubits[pos]),
            "width": d + 1,
            "height": 2 * d + 1,
            "distance": d,
            "type": "rotated_surface_code",
            "role": layout.roles[pos],
            "position": pos,
        }
        if pos in layout.logical_index:
            q = layout.logical_index[pos]
            part["logical_qubit"] = q
            part["secret_bit"] = layout.secret[q]
        partitions.append(part)
    for (a, b) in sorted(strip_qubits):
        vertical = a[1] == b[1]  # pipe along x -> strip is a column of qubits (fixed x)
        partitions.append({
            "indices": _by_coord(strip_qubits[(a, b)]),
            "width": 1 if vertical else d + 1,
            "height": 2 * d + 1 if vertical else 1,
            "distance": d,
            "type": "rotated_surface_code_ancilla",
            "role": "pipe",
            "position": (a, b),
        })

    # Column-major by block position, pipe strips between the blocks they join (see GHZ notes:
    # Chipmunq anchors each partition on its BFS predecessor, so the order matters).
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
      columns alternating d+1 / d qubits, ordered column by column.
    * ``rotated_surface_code_ancilla``: d qubits on one straight line.
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
# Public entry point (same return convention as the GHZ generator)
# --------------------------------------------------------------------------------------


def generate_bv_lattice_surgery(
    secret: str | Sequence[int],
    distance_scale: int = 1,
    num_bands: int | None = None,
    memory_blocks: int = 1,
    trim_bus: bool = True,
    noise_model: NoiseModel | None = None,
    manhattan_radius: int = 2,
    output_file: str | None = None,
) -> tuple[stim.Circuit, list[dict]]:
    """Generate a Bernstein-Vazirani lattice-surgery circuit and its Chipmunq partitions.

    :param secret: secret bit string s; one logical data qubit per bit (+ one ancilla)
    :param distance_scale: tqec scale k, code distance d = 2k + 1
    :param noise_model: optional tqec noise model (e.g. ``NoiseModel.uniform_depolarizing(1e-3)``);
        leave ``None`` if Chipmunq inserts noise after routing
    :param output_file: if given, the stim circuit is written to this path
    :return: (stim circuit, partitions); observable i of the circuit is secret bit i
    """
    bv = build_bv_block_graph(secret, num_bands=num_bands, memory_blocks=memory_blocks, trim_bus=trim_bus)
    compiled = compile_block_graph(bv.block_graph, observables=find_observables(bv.block_graph))
    circuit = compiled.generate_stim_circuit(k=distance_scale, noise_model=noise_model,
                                             manhattan_radius=manhattan_radius)
    circuit = canonicalize_observables(circuit, bv, distance_scale)
    circuit, _ = pad_irregular_patches(circuit, bv, distance_scale)
    partitions = extract_partitions(circuit, bv, distance_scale)

    if output_file is not None:
        with open(output_file, "w") as f:
            print(circuit, file=f)
    return circuit, partitions


def get_tqec_bv(secret: str | Sequence[int], distance_scale: int = 1, **kwargs) -> tuple[stim.Circuit, list[dict]]:
    """Drop-in analogue of ``get_tqec_ghz`` for the experiment scripts."""
    s = "".join(map(str, parse_secret(secret)))
    return generate_bv_lattice_surgery(secret, distance_scale=distance_scale,
                                       output_file=f"stim_bv_s_{s}_d_{2 * distance_scale + 1}.stim", **kwargs)