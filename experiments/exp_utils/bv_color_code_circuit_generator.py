"""Bernstein-Vazirani (BV) on triangular 6.6.6 color codes with transversal gates, with Chipmunq partitions.

Why transversal
---------------
The color code is self-dual and has transversal H and CNOT (and S). The textbook BV circuit uses
only H and CNOT, so every logical gate is a transversal layer of physical gates:

    data_i in |0>_L, ancilla in |->_L
    H^(x)n on every data patch                       (transversal H)
    CNOT(data_i -> ancilla) for every s_i = 1        (transversal CNOT, qubit j -> qubit j)
    H^(x)n on every data patch
    measure every data patch in Z                     ->  data_i = s_i

No lattice surgery is needed. For the mapper this is a very different workload from the surface-
code BV: a transversal CNOT couples *every* physical qubit of a data patch to the matching qubit of
the ancilla patch, so the ancilla patch has to "meet" each data patch with s_i = 1 in turn.

Code and syndrome extraction
----------------------------
* Geometry and face supports are taken from stim's ``color_code:memory_xyz`` generator (triangular
  6.6.6 code, distance d odd, n = (3 d^2 + 1) / 4 data qubits, (n - 1) / 2 faces, one ancilla per
  face at the centre of its hexagon).
* Every round measures, for every face, first the X stabiliser (ancilla in |+>, CX ancilla -> data,
  MX) and then the Z stabiliser (ancilla in |0>, CX data -> ancilla, M). X and Z checks are never
  interleaved, so every round measures the stabilisers exactly.
* Flag qubits (default): a bare ancilla measuring a weight-6 face spreads a single fault to up to 3
  data qubits (hook error). With bare ancillas the circuit distance of this construction is only
  2 / 3 / 4 for d = 3 / 5 / 7 (checked with stim; no CNOT order avoids it). Every face therefore
  gets a flag qubit: CX(ancilla, flag) after the face's first and before its last data CNOT
  (mirrored for Z checks), flag measured every sub-round (deterministic -> detector). With flags
  the circuit distance is d (checked with stim for d = 3 and 5).
* Logical operators: Zbar = Z^(x)n, Xbar = X^(x)n (n is odd, all stabilisers have even weight).

Detectors are derived by tracking, for every (patch, face, basis), which measurements its current
value equals, through the transversal gates:
    H on patch p:           X_f(p) <-> Z_f(p)
    CNOT(c -> t):           X_f(c) -> X_f(c) X_f(t),   Z_f(t) -> Z_f(c) Z_f(t)
    Z^(x)n (|+> -> |->):    no change (even-weight X stabilisers)

Observables: observable i = Zbar of data patch i (its value is s_i), observable n = Xbar of the
ancilla patch (value -1, the |-> state).

Qubit layout (for Chipmunq)
---------------------------
Patch p occupies qubits [p * q_patch, (p + 1) * q_patch), ordered [data qubits..., face ancillas...,
flag qubits...]. stim places a patch on a triangular lattice with coordinates (x, y), x in Z/2;
``row = y, col = x - y / 2`` maps it onto a square grid in which the triangular-lattice neighbours
are (0, +-1), (+-1, 0) and the diagonal (+1, -1).

* Without flags a patch is packed on that grid; a chiplet with ``topology="grid",
  connectivity="long_range", long_range_offsets=[(1, -1)]`` hosts it natively.
* With flags the lattice is spread by 2 (data / ancillas at (2 row, 2 col)) and the flag of a face
  sits at (2 row, 2 col + 1), next to its ancilla. The chiplet then needs
  ``long_range_offsets=[(0, 2), (2, 0), (2, -2)]`` (data - ancilla) in addition to the nearest-
  neighbour links (ancilla - flag). ``chiplet_long_range_offsets(flags)`` returns the right list.

Every patch is one ``grid`` partition that carries these local coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import stim

ROLE_DATA = "data"
ROLE_ANCILLA = "ancilla"



def chiplet_long_range_offsets(flags: bool = True) -> list[tuple[int, int]]:
    """Extra chiplet links (``BackendChipletV2(..., connectivity="long_range", long_range_offsets=...)``)
    under which a patch with the coordinates of ``bv_color_code_partitions`` needs no SWAPs."""
    return [(0, 2), (2, 0), (2, -2)] if flags else [(1, -1)]


# --------------------------------------------------------------------------------------
# Color code geometry (from stim's generator)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ColorCode:
    distance: int
    data: list[int]                        # local indices 0..n-1
    faces: list[list[int]]                 # data support of every face (local data indices)
    layers: list[list[tuple[int, int]]]    # 6 CNOT layers of (local data index, face index)
    coords: list[tuple[int, int]]          # (row, col) of local qubit k (data first, then faces)

    @property
    def n(self) -> int:
        return len(self.data)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def qubits_per_patch(self, flags: bool) -> int:
        return self.n + self.num_faces * (2 if flags else 1)

    def patch_coords(self, flags: bool) -> list[tuple[int, int]]:
        """(row, col) of every local qubit: data, face ancillas, (flags)."""
        if not flags:
            return list(self.coords)
        base = [(2 * r, 2 * c) for r, c in self.coords]
        return base + [(2 * r, 2 * c + 1) for r, c in self.coords[self.n:]]

    def width(self, flags: bool) -> int:
        return max(c for _, c in self.patch_coords(flags)) + 1

    def height(self, flags: bool) -> int:
        return max(r for r, _ in self.patch_coords(flags)) + 1

    def face_order(self) -> list[list[int]]:
        """Data qubits of every face in CNOT order (stim's layer order)."""
        order = [[] for _ in self.faces]
        for layer in self.layers:
            for k, f in layer:
                order[f].append(k)
        return order


def color_code(distance: int) -> ColorCode:
    """Triangular 6.6.6 color code of odd ``distance`` (geometry and CNOT order from stim)."""
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be odd and >= 3.")
    ref = stim.Circuit.generated("color_code:memory_xyz", distance=distance, rounds=2)
    coords = ref.get_final_qubit_coordinates()
    body = next(inst for inst in ref if isinstance(inst, stim.CircuitRepeatBlock)).body_copy()
    anc = [t.value for inst in body if inst.name == "MR" for t in inst.targets_copy()]
    layers_raw = []
    for inst in body:
        if inst.name == "CX":
            t = [x.value for x in inst.targets_copy()]
            layers_raw.append(list(zip(t[::2], t[1::2])))  # (data control, ancilla target)
    data = sorted(q for q in coords if q not in set(anc))
    dpos = {q: k for k, q in enumerate(data)}
    fpos = {q: k for k, q in enumerate(anc)}
    faces = [[] for _ in anc]
    layers = []
    for layer in layers_raw:
        out = []
        for c, t in layer:
            if c in fpos or t not in fpos:
                raise RuntimeError("Unexpected CX orientation in stim's color code circuit.")
            faces[fpos[t]].append(dpos[c])
            out.append((dpos[c], fpos[t]))
        layers.append(out)
    faces = [sorted(f) for f in faces]

    def rc(q):
        x, y = coords[q]
        r, c = int(round(y)), x - y / 2
        if abs(c - round(c)) > 1e-9:
            raise RuntimeError("Unexpected color code coordinates.")
        return (r, int(round(c)))

    local_coords = [rc(q) for q in data] + [rc(q) for q in anc]
    code = ColorCode(distance, list(range(len(data))), faces, layers, local_coords)
    _check_code(code)
    return code


def _check_code(code: ColorCode) -> None:
    n, d = code.n, code.distance
    if n != (3 * d * d + 1) // 4 or code.num_faces != (n - 1) // 2:
        raise AssertionError("Unexpected color code parameters.")
    for i, f in enumerate(code.faces):
        if len(f) % 2:
            raise AssertionError("Face with odd weight.")
        for g in code.faces[i + 1:]:
            if len(set(f) & set(g)) % 2:
                raise AssertionError("Faces overlap in an odd number of qubits.")
    if len(set(code.coords)) != len(code.coords):
        raise AssertionError("Two qubits share a grid position.")
    # every face ancilla must be a triangular-lattice neighbour of its data qubits
    nbrs = {(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1)}
    for f, supp in enumerate(code.faces):
        ra, ca = code.coords[code.n + f]
        for q in supp:
            rq, cq = code.coords[q]
            if (rq - ra, cq - ca) not in nbrs:
                raise AssertionError("Face ancilla is not adjacent to its data qubits on the lattice.")


# --------------------------------------------------------------------------------------
# Circuit
# --------------------------------------------------------------------------------------


def parse_secret(secret: str | Sequence[int]) -> tuple[int, ...]:
    bits = tuple(int(c) for c in secret) if isinstance(secret, str) else tuple(int(b) for b in secret)
    if not bits or any(b not in (0, 1) for b in bits):
        raise ValueError(f"Secret must be a non-empty bit string, got {secret!r}.")
    return bits


class _Circuit:
    def __init__(self, noise: float):
        self.c = stim.Circuit()
        self.num_meas = 0
        self.p = noise

    def gate1(self, name: str, qubits: list[int]) -> None:
        if qubits:
            self.c.append(name, qubits)
            if self.p and name not in ("R", "RX"):
                self.c.append("DEPOLARIZE1", qubits, self.p)
            elif self.p:
                self.c.append("X_ERROR" if name == "R" else "Z_ERROR", qubits, self.p)

    def cx(self, pairs: list[tuple[int, int]]) -> None:
        if pairs:
            flat = [q for pr in pairs for q in pr]
            self.c.append("CX", flat)
            if self.p:
                self.c.append("DEPOLARIZE2", flat, self.p)

    def measure(self, basis: str, qubits: list[int]) -> list[int]:
        if self.p:
            self.c.append("X_ERROR" if basis == "Z" else "Z_ERROR", qubits, self.p)
        self.c.append("M" if basis == "Z" else "MX", qubits)
        idx = list(range(self.num_meas, self.num_meas + len(qubits)))
        self.num_meas += len(qubits)
        return idx

    def tick(self) -> None:
        self.c.append("TICK")

    def detector(self, meas: set[int] | list[int], coords: list[float]) -> None:
        self.c.append("DETECTOR", [stim.target_rec(m - self.num_meas) for m in sorted(meas)], coords)

    def observable(self, index: int, meas: list[int]) -> None:
        self.c.append("OBSERVABLE_INCLUDE", [stim.target_rec(m - self.num_meas) for m in sorted(meas)], index)


def _greedy_layers(ops: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Pack 2-qubit gates into layers (each qubit at most once per layer), keeping the order of the
    gates on every qubit."""
    layers: list[list[tuple[int, int]]] = []
    busy: list[set[int]] = []
    last: dict[int, int] = {}
    for a, b in ops:
        i = max(last.get(a, -1), last.get(b, -1)) + 1
        while i < len(layers) and (a in busy[i] or b in busy[i]):
            i += 1
        if i == len(layers):
            layers.append([])
            busy.append(set())
        layers[i].append((a, b))
        busy[i].update((a, b))
        last[a] = last[b] = i
    return layers


def build_bv_color_code_circuit(secret: str | Sequence[int], distance: int = 3, rounds_per_step: int = 1,
                                flags: bool = True, noise: float = 0.0,
                                patch_spacing: int = 2) -> tuple[stim.Circuit, ColorCode]:
    """Stim circuit of transversal BV on color code patches (see module docstring).

    :param flags: one flag qubit per face (full circuit distance); ``False`` = bare ancillas

    :param rounds_per_step: syndrome rounds after initialisation and after every transversal CNOT
    :param noise: circuit-level noise for testing (leave 0 if Chipmunq inserts noise after routing)
    :param patch_spacing: gap (in grid columns) between patches in the QUBIT_COORDS (visualisation only)
    """
    s = parse_secret(secret)
    if rounds_per_step < 1:
        raise ValueError("rounds_per_step must be >= 1.")
    code = color_code(distance)
    num_data_patches = len(s)
    patches = list(range(num_data_patches + 1))
    anc_patch = num_data_patches
    qp = code.qubits_per_patch(flags)
    pcoords = code.patch_coords(flags)
    pwidth = code.width(flags)

    def dq(p: int, k: int) -> int:
        return p * qp + k

    def fq(p: int, f: int) -> int:
        return p * qp + code.n + f

    def flq(p: int, f: int) -> int:
        return p * qp + code.n + code.num_faces + f

    b = _Circuit(noise)
    for p in patches:
        for k, (r, c) in enumerate(pcoords):
            b.c.append("QUBIT_COORDS", [p * qp + k], [c + p * (pwidth + patch_spacing), r])

    # CNOT schedule of one sub-round, as (local data or flag, face) steps per face:
    #   face f: d_0, [flag], d_1 .. d_{w-2}, [flag], d_{w-1}
    face_seq = []
    for f, order in enumerate(code.face_order()):
        seq = [("d", order[0])] + ([("f", f)] if flags else []) + [("d", k) for k in order[1:-1]]
        seq += ([("f", f)] if flags else []) + [("d", order[-1])]
        face_seq.append(seq)

    def sub_round_layers(basis: str) -> list[list[tuple[int, int]]]:
        ops = []
        for step in range(max(len(q) for q in face_seq)):
            for p in patches:
                for f, seq in enumerate(face_seq):
                    if step < len(seq):
                        kind, k = seq[step]
                        other = dq(p, k) if kind == "d" else flq(p, k)
                        ops.append((fq(p, f), other) if basis == "X" else (other, fq(p, f)))
        return _greedy_layers(ops)

    round_layers = {basis: sub_round_layers(basis) for basis in ("X", "Z")}

    # ref[(basis, patch, face)] = set of measurements whose XOR is the current stabiliser value,
    # or None if the value is random.
    ref: dict[tuple[str, int, int], set[int] | None] = {}
    rnd = [0]

    def data_qubits(p: int) -> list[int]:
        return [dq(p, k) for k in range(code.n)]

    def syndrome_round() -> None:
        faces_all = [fq(p, f) for p in patches for f in range(code.num_faces)]
        flags_all = [flq(p, f) for p in patches for f in range(code.num_faces)] if flags else []
        for basis in ("X", "Z"):
            flag_basis = "Z" if basis == "X" else "X"
            b.gate1("RX" if basis == "X" else "R", faces_all)
            b.gate1("R" if flag_basis == "Z" else "RX", flags_all)
            b.tick()
            for layer in round_layers[basis]:
                b.cx(layer)
                b.tick()
            meas = b.measure(basis, faces_all)
            if flags:
                for i, m in enumerate(b.measure(flag_basis, flags_all)):
                    p, f = divmod(i, code.num_faces)
                    b.detector({m}, [p, f, rnd[0], 2])  # flag: deterministic 0
            for i, (p, f) in enumerate((p, f) for p in patches for f in range(code.num_faces)):
                key = (basis, p, f)
                if ref.get(key) is not None:
                    b.detector(ref[key] ^ {meas[i]}, [p, f, rnd[0], 0 if basis == "X" else 1])
                ref[key] = {meas[i]}
            b.tick()
        rnd[0] += 1

    def transversal_h(ps: list[int]) -> None:
        b.gate1("H", [q for p in ps for q in data_qubits(p)])
        b.tick()
        for p in ps:
            for f in range(code.num_faces):
                ref[("X", p, f)], ref[("Z", p, f)] = ref[("Z", p, f)], ref[("X", p, f)]

    def transversal_cnot(ctrl: int, tgt: int) -> None:
        b.cx([(dq(ctrl, k), dq(tgt, k)) for k in range(code.n)])
        b.tick()
        for f in range(code.num_faces):
            xc, xt = ref[("X", ctrl, f)], ref[("X", tgt, f)]
            zc, zt = ref[("Z", ctrl, f)], ref[("Z", tgt, f)]
            ref[("X", ctrl, f)] = None if xc is None or xt is None else xc ^ xt
            ref[("Z", tgt, f)] = None if zc is None or zt is None else zc ^ zt

    # 1) initialisation: data |0>_L, ancilla |->_L = Zbar |+>_L
    b.gate1("R", [q for p in patches[:-1] for q in data_qubits(p)])
    b.gate1("RX", data_qubits(anc_patch))
    b.gate1("Z", data_qubits(anc_patch))
    b.tick()
    for p in patches:
        for f in range(code.num_faces):
            det_basis = "X" if p == anc_patch else "Z"
            ref[(det_basis, p, f)] = set()
            ref[("Z" if det_basis == "X" else "X", p, f)] = None
    for _ in range(rounds_per_step):
        syndrome_round()

    # 2) BV
    data_patches = patches[:-1]
    transversal_h(data_patches)
    for i, bit in enumerate(s):
        if bit:
            transversal_cnot(i, anc_patch)
            for _ in range(rounds_per_step):
                syndrome_round()
    transversal_h(data_patches)

    # 3) readout: data patches in Z, ancilla patch in X
    final = {}
    for p in patches:
        basis = "X" if p == anc_patch else "Z"
        m = b.measure(basis, data_qubits(p))
        final[p] = m
        for f, supp in enumerate(code.faces):
            r = ref[(basis, p, f)]
            if r is not None:
                b.detector(r ^ {m[k] for k in supp}, [p, f, rnd[0], 0 if basis == "X" else 1])
    for i in data_patches:
        b.observable(i, final[i])              # Zbar of data patch i  (= s_i)
    b.observable(num_data_patches, final[anc_patch])  # Xbar of the ancilla (= -1)
    return b.c, code


def bv_color_code_partitions(secret: str | Sequence[int], code: ColorCode, flags: bool = True) -> list[dict]:
    """One ``grid`` partition per patch, with local (row, col) coordinates for the placement."""
    s = parse_secret(secret)
    qp = code.qubits_per_patch(flags)
    parts = []
    for p in range(len(s) + 1):
        part = {
            "indices": list(range(p * qp, (p + 1) * qp)),
            "coords": code.patch_coords(flags),
            "width": code.width(flags),
            "height": code.height(flags),
            "distance": code.distance,
            "type": "grid",
            "role": ROLE_DATA if p < len(s) else ROLE_ANCILLA,
        }
        if p < len(s):
            part["logical_qubit"] = p
            part["secret_bit"] = s[p]
        parts.append(part)
    return parts


def generate_bv_color_code(secret: str | Sequence[int], distance: int = 3, rounds_per_step: int = 1,
                           flags: bool = True, noise: float = 0.0,
                           output_file: str | None = None) -> tuple[stim.Circuit, list[dict]]:
    """Generate the transversal color-code BV circuit and its Chipmunq partitions.

    :return: (stim circuit, partitions); observable i is Zbar of data patch i (value s_i),
        the last observable is Xbar of the ancilla patch
    """
    circuit, code = build_bv_color_code_circuit(secret, distance=distance, rounds_per_step=rounds_per_step,
                                                flags=flags, noise=noise)
    partitions = bv_color_code_partitions(secret, code, flags=flags)
    if output_file is not None:
        with open(output_file, "w") as f:
            print(circuit, file=f)
    return circuit, partitions


def get_color_code_bv(secret: str | Sequence[int], distance: int = 3, **kwargs) -> tuple[stim.Circuit, list[dict]]:
    """Drop-in analogue of ``get_tqec_bv`` for the experiment scripts."""
    s = "".join(map(str, parse_secret(secret)))
    return generate_bv_color_code(secret, distance=distance,
                                  output_file=f"stim_bv_color_s_{s}_d_{distance}.stim", **kwargs)
