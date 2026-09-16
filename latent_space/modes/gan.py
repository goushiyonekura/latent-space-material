"""GAN-type controller (spec §10): co-adaptive generator / discriminator over *whole* acoustic
compositions, with a small reference distribution, a joint low-rank proposal basis and lineage.

  r_b     ∝ exp[ -J_ref(Xi_b^ref, F, H, o) / tau_ref ],  J_ref = E_form + 0.2 E_hist   (40)
  Xi_hat  = Xi_b^parent + B_ref w,   p_Theta(b, w) = alpha_b N(w; mu_b, diag sigma_b^2) (41)
  D_phi(Xi) = sigmoid(phi^T psi(Xi)),  0 < D < 1                                       (42)
  L_D     = -E_ref log D(Xi^ref) - E_gen log[1 - D(Xi^actual)] + lambda_D ||phi||^2     (43)
  l(b, w) = -log D(Xi^actual) + lambda_fit <d_xi^2(xi^actual, xi_hat)> + lambda_f E_form(44)
  grad_Theta E l = E[(l - b_past) grad_Theta log p_Theta(b, w)]                         (45)

The discriminator only ever sees *realized* compositions (cand.xi, computed from the actual
summed PCM) and the reference compositions; ideal proposals are never scored by D.  Positive and
negative examples come from the same unit, the same phase grid and the same analysis scales, and
only free (non GOAL_HOLD) windows enter psi.  No candidate id / origin label is a feature.

This is the tool's lineage / mutation / contraction cycle, not standard GAN inference: nothing
here claims convergence, an optimal equilibrium or a match to a real acoustic distribution
(spec §10.5 / structural contract G)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..bank import Bank
from ..curves import MotionLimits
from ..types import Candidate, Realization, Target, UnitContext
from .base import ModeController, fix_hold_rows


try:      # the engine is fully imported before any mode module is; no cycle at runtime
    from ..engine import HardConstraintFailure as _EngineFailure
except Exception:                                                         # noqa: BLE001
    _EngineFailure = RuntimeError

_BASES = (_EngineFailure,) if issubclass(_EngineFailure, RuntimeError) else (_EngineFailure, RuntimeError)


class NoFeasibleReference(*_BASES):    # type: ignore[misc]
    """No legal reference trajectory exists for this unit: a hard feasibility problem, not a soft
    criterion (spec §10.2).  A RuntimeError, and also the engine's HardConstraintFailure so the
    job reports HARD_CONSTRAINT_FAILURE / NO_FEASIBLE_PLAN_FOUND instead of crashing."""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic; strictly inside (0, 1) after the clip."""
    z = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class GANMode(ModeController):
    cli_name = "gan"
    internal_id = "gan_coadaptive"

    # ------------------------------------------------------------------ construction
    def __init__(self, cfg, analyzer, fs, rng, objective):
        super().__init__(cfg, analyzer, fs, rng, objective)
        md = cfg["mode_defaults"]
        self.p = dict(md["gan"])
        g = self.p.get
        self.n_ref_target = int(md.get("gan_reference_target", 8))
        self.n_hist_parents = max(1, int(md.get("gan_components", 2)))
        self.rounds_max = int(md.get("gan_adversarial_rounds_max", 4))
        self.tau_ref = max(1e-6, float(g("reference_temperature", 0.5)))
        self.l2 = float(g("l2", 1e-4))
        self.lr_D = float(g("lr_D", 0.001))
        self.lr_G = float(g("lr_G", 0.001))
        self.lambda_fit = float(g("lambda_fit", 0.25))
        self.lambda_f = float(g("lambda_f", 0.5))
        self.C_fail = float(g("C_fail", 100.0))
        self.sigma_init = float(g("sigma_init", 0.2))
        self.sigma_min = float(g("sigma_min", 0.01))
        self.sigma_max = float(g("sigma_max", 1.0))
        self.basis_rank = max(1, int(g("basis_rank", 3)))
        self.samples_per_round = max(1, int(g("samples_per_round", 4)))
        # not in config.py DEFAULTS (shared file, not edited): read with .get + documented default
        self.inner_steps_D = max(1, int(g("inner_steps_D", 40)))
        self.inner_steps_G = max(1, int(g("inner_steps_G", 10)))
        self.n_fresh_parents = max(1, int(g("fresh_parents", 2)))
        self.step_clip = float(g("generator_step_clip", 0.05))
        self.grad_clip = float(g("generator_gradient_clip", 10.0))
        self.hint_min_corr = float(g("hint_min_correlation", 0.3))
        self.max_hints = int(g("max_hints", 4))
        self.ess_min_fraction = float(g("importance_ess_min_fraction", 0.5))
        self.psi_scale_source = "uninitialised"

        self.lim = MotionLimits.from_config(cfg["motion"], cfg["numerics"])
        # psi layout (eq. 42): fixed for the whole job, so phi carries over between units
        oi, oj = [], []
        for i in range(self.M):
            for j in range(self.M):
                if i != j:
                    oi.append(i)
                    oj.append(j)
        self.pair_i = np.array(oi, dtype=int)
        self.pair_j = np.array(oj, dtype=int)
        iu = analyzer.iu
        names = ([f"mean_phi[{k}]" for k in range(self.d_phi)]
                 + [f"halfdiff_phi[{k}]" for k in range(self.d_phi)]
                 + [f"mean_c[{i}]" for i in range(self.M)]
                 + [f"mean_R[{int(a)},{int(b)}]" for a, b in zip(iu[0], iu[1])]
                 + [f"lag_c[{i}]xc[{j}]" for i, j in zip(oi, oj)]
                 + ["bias"])
        self.psi_names = names
        self.d_psi = len(names)
        self.phi = np.zeros(self.d_psi)              # discriminator parameters (eq. 42)
        self.n_parents = self.n_hist_parents + self.n_fresh_parents
        self.r = self.basis_rank
        # generator parameters Theta (eq. 41) - re-shaped per unit, initialised here for safety
        self.logits = np.zeros(self.n_parents)
        self.mu = np.zeros((self.n_parents, self.r))
        self.log_sigma = np.full((self.n_parents, self.r), np.log(max(self.sigma_init, 1e-6)))
        self.parent_ids: List[str] = []
        self.parent_xi = np.zeros((self.n_parents, 1, self.d_xi))
        self.psi_mean = np.zeros(self.d_psi)
        self.psi_std = np.ones(self.d_psi)
        self.B_ref = np.zeros((self.r, 1, self.d_xi))
        self.extra_candidate_evaluations = 0
        # traces
        self.generator_traces: List[Dict[str, Any]] = []
        self.discriminator_traces: List[Dict[str, Any]] = []
        self.update_counts: List[Dict[str, Any]] = []
        self.reference_summaries: List[Dict[str, Any]] = []
        self.lineage: List[Dict[str, Any]] = []
        self._round_targets: Dict[int, List[Target]] = {}
        self._round_loss_mean: Optional[float] = None
        self._unit_counts = {"D": 0, "G": 0, "D_rejected": 0, "rounds": 0}
        self._total_counts = {"D": 0, "G": 0, "D_rejected": 0, "rounds": 0}

    # ================================================================== psi / D (eq. 42)
    def _psi_raw(self, xi: np.ndarray) -> np.ndarray:
        """Interaction features of one realized/reference composition on the free windows only."""
        x = np.asarray(xi, dtype=np.float64)[self.fm_idx]
        phi_b, c, Rup = self.analyzer.split(x)
        n = len(x)
        mean_phi = phi_b.mean(axis=0)
        if n >= 2:
            h = max(1, n // 2)
            half = phi_b[:h].mean(axis=0) - phi_b[h:].mean(axis=0)
            lag = np.einsum("ti,tj->ij", c[1:], c[:-1]) / float(n - 1)
        else:
            half = np.zeros(self.d_phi)
            lag = np.outer(c[0], c[0])
        inter = lag[self.pair_i, self.pair_j]
        return np.concatenate([mean_phi, half, c.mean(axis=0), Rup.mean(axis=0), inter, [1.0]])

    def _psi(self, xi: np.ndarray) -> np.ndarray:
        return (self._psi_raw(xi) - self.psi_mean) / self.psi_std

    def _log_p(self, b: int, w: np.ndarray) -> float:
        """log p_Theta(b, w) = log alpha_b + log N(w; mu_b, diag sigma_b^2)."""
        a = self.alphas()
        sig = np.maximum(np.exp(self.log_sigma[b]), 1e-12)
        z = (np.asarray(w, dtype=np.float64) - self.mu[b]) / sig
        return float(np.log(max(float(a[b]), 1e-300)) - 0.5 * float((z * z).sum())
                     - float(np.log(sig).sum()) - 0.5 * self.r * np.log(2.0 * np.pi))

    def discriminator_score(self, xi: np.ndarray) -> float:
        """D_phi(Xi) in (0, 1) for a *realized* composition trajectory (eq. 42)."""
        return float(_sigmoid(float(self.phi @ self._psi(xi))))

    def _d_loss(self, Psi_ref: np.ndarray, w_ref: np.ndarray, Psi_gen: np.ndarray,
                phi: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Eq. (43) with the reference weights r_b on the positive term."""
        D_ref = _sigmoid(Psi_ref @ phi)
        D_gen = _sigmoid(Psi_gen @ phi) if len(Psi_gen) else np.zeros(0)
        pos = -float((w_ref * np.log(np.clip(D_ref, 1e-12, 1.0))).sum())
        neg = -float(np.log(np.clip(1.0 - D_gen, 1e-12, 1.0)).mean()) if len(D_gen) else 0.0
        return pos + neg + self.l2 * float(phi @ phi), D_ref, D_gen

    # ================================================================== 10.2 reference set
    def _draw_references(self, unit: UnitContext, history) -> List[Candidate]:
        bank = Bank(unit, self.lim, self.cfg, self.rng, self.objective, history)
        refs: List[Candidate] = []
        attempts = 0
        max_attempts = 4 * self.n_ref_target + 8
        while len(refs) < self.n_ref_target and attempts < max_attempts:
            attempts += 1
            c = bank.random_candidate(None, origin="gan_reference")
            if c is not None:
                refs.append(c)
        if not refs:
            # legal reference trajectories are a hard feasibility problem, not a soft criterion
            raise NoFeasibleReference(
                f"NO_FEASIBLE_PLAN_FOUND: unit {unit.index}: no legal reference composition "
                f"trajectory for the GAN reference distribution in {attempts} attempts "
                f"(bank rejected {bank.n_failed} illegal plans)")
        if len(refs) < min(4, self.n_ref_target):
            self.warnings.append(
                f"unit {unit.index}: only {len(refs)} legal reference trajectories "
                f"(target {self.n_ref_target}); proceeding with a small reference distribution")
        return refs

    # ================================================================== conditioning
    def begin_unit(self, unit: UnitContext, history) -> None:
        self.unit = unit
        fm = unit.free_mask
        if fm.any():
            self.fm_idx = np.where(fm)[0]
        else:
            self.fm_idx = np.arange(unit.J)
            self.warnings.append(f"unit {unit.index}: no free windows; psi/basis use all windows")
        self.o = unit.o

        # ---------------------------------------------------------- (40) reference distribution
        refs = self._draw_references(unit, history)
        self.extra_candidate_evaluations = len(refs)
        self.ref_xi = np.stack([c.xi for c in refs])                      # (B, J, d)
        j_ref = np.array([float(c.e_form) + 0.2 * float(c.e_hist) for c in refs])
        z = -(j_ref - j_ref.min()) / self.tau_ref
        w = np.exp(z - z.max())
        self.ref_w = w / w.sum()
        self.ref_J = j_ref
        order = np.argsort(-self.ref_w)
        self.ref_order = order
        div = 0.0
        if len(refs) > 1:
            xs = self.ref_xi[:, self.fm_idx, :]
            ds = [float(self.analyzer.dist2(xs[a], xs[b]).mean())
                  for a in range(len(xs)) for b in range(a + 1, len(xs))]
            div = float(np.mean(ds))
        if div < 1e-6 and len(refs) > 1:
            self.warnings.append(f"unit {unit.index}: reference distribution has very low acoustic "
                                 f"diversity (mean pair dist2 {div:.2e}); proceeding")
        ent = float(-(self.ref_w * np.log(np.clip(self.ref_w, 1e-12, 1.0))).sum())
        self.reference_summaries.append({
            "unit": unit.index, "n_references": len(refs), "weights": self.ref_w.tolist(),
            "J_ref": j_ref.tolist(), "J_ref_definition": "E_form + 0.2 * E_hist (no discriminator)",
            "temperature": self.tau_ref, "weight_entropy_nats": ent,
            "diversity_mean_pair_dist2": div,
            "e_form": [float(c.e_form) for c in refs], "e_hist": [float(c.e_hist) for c in refs]})

        # ---------------------------------------------------------- (41) joint basis B_ref
        self._build_basis(unit)
        # ---------------------------------------------------------- parents + lineage
        self._build_parents(unit, history)
        # ---------------------------------------------------------- psi scales + warm start
        Psi = np.stack([self._psi_raw(x) for x in self.ref_xi])
        m = Psi.mean(axis=0)
        s = np.maximum(Psi.std(axis=0), 1e-3)
        m[-1] = 0.0
        s[-1] = 1.0                                  # bias stays 1
        st_prev = history.mode_state.get("gan") if hasattr(history, "mode_state") else None
        if st_prev and isinstance(st_prev.get("psi_mean"), list) and len(st_prev["psi_mean"]) == self.d_psi:
            # the discriminator input standardisation is fixed for the whole generation so that the
            # inherited coefficients phi keep their meaning (audit §8.1)
            self.psi_mean = np.array(st_prev["psi_mean"], dtype=np.float64)
            self.psi_std = np.array(st_prev["psi_std"], dtype=np.float64)
            self.psi_scale_source = "frozen_from_first_unit"
        else:
            self.psi_mean, self.psi_std = m, s
            self.psi_scale_source = "reference_set_of_this_unit"
        self._init_theta(history)
        self._round_targets = {}
        self._round_loss_mean = None
        self._unit_counts = {"D": 0, "G": 0, "D_rejected": 0, "rounds": 0}
        self._round_records: List[Dict[str, Any]] = []

    def _build_basis(self, unit: UnitContext) -> None:
        """Few joint directions from the reference differences (small SVD, eq. 41).
        The basis acts on the whole composition trajectory at once, never on a single track."""
        X = self.ref_xi[:, self.fm_idx, :]                                # (B, Jf, d)
        mean = np.einsum("b,bjd->jd", self.ref_w, X)
        Dif = (X - mean[None]).reshape(len(X), -1)
        # the rank is a fixed configuration constant (unused directions stay zero) so that the
        # generator parameter shapes - and therefore signature() - do not depend on how many
        # references happened to be legal in this unit
        self.r = max(1, int(self.basis_rank))
        B = np.zeros((self.r, unit.J, self.d_xi))
        sv_all: List[float] = []
        if len(X) >= 2 and np.any(np.abs(Dif) > 1e-12):
            _U, sv, Vt = np.linalg.svd(Dif, full_matrices=False)
            sv_all = sv.tolist()
            scale = 1.0 / np.sqrt(float(len(X)))
            n_dir = 0
            for k in range(self.r):
                if k < len(sv) and sv[k] > 1e-12:
                    B[k][self.fm_idx] = (sv[k] * scale) * Vt[k].reshape(len(self.fm_idx), self.d_xi)
                    n_dir += 1
            self.basis_source = "svd_of_reference_differences"
            if n_dir < self.r:
                self.warnings.append(f"unit {unit.index}: only {n_dir} of {self.r} basis directions "
                                     f"are supported by the reference set; the rest stay zero")
        else:
            # explicit reference differences (spec allows this instead of the SVD)
            for k in range(self.r):
                b = self.ref_order[min(k + 1, len(self.ref_order) - 1)]
                B[k][self.fm_idx] = X[b] - mean
            self.basis_source = "explicit_reference_differences"
            self.warnings.append(f"unit {unit.index}: reference differences degenerate; "
                                 f"basis built from explicit reference differences")
        self.B_ref = B
        self.basis_sv = sv_all
        self.basis_norms = [float(np.linalg.norm(B[k])) for k in range(self.r)]

    def _build_parents(self, unit: UnitContext, history) -> None:
        """Parents: past non-hold composition lines from the lineage (mapped onto this unit's
        grid through the phase-relative blocks) plus a few fresh reference lines, so re-enabling
        keeps exploring instead of collapsing onto the single goal point (spec §10.6)."""
        parents: List[np.ndarray] = []
        ids: List[str] = []
        origins: List[str] = []
        hp = [p for p in getattr(history, "parents", []) if isinstance(p, dict) and "blocks" in p]
        hp = sorted(hp, key=lambda q: (-int(q.get("unit", -1)), float(q.get("score", 1e9))))
        for q in hp[: self.n_hist_parents]:
            try:
                xi_p = self.objective.blocks_to_unit(unit, q["blocks"])
            except Exception as e:                                        # noqa: BLE001
                self.warnings.append(f"unit {unit.index}: parent {q.get('id')} could not be mapped "
                                     f"onto this unit ({e}); replaced by a fresh reference")
                continue
            parents.append(fix_hold_rows(unit, xi_p))
            ids.append(str(q.get("id")))
            origins.append("history_parent")
        k = 0
        while len(parents) < self.n_parents and k < len(self.ref_order):
            b = int(self.ref_order[k])
            parents.append(fix_hold_rows(unit, self.ref_xi[b]))
            ids.append(f"gan:u{unit.index}:ref{b}")
            origins.append("fresh_reference")
            k += 1
        while len(parents) < self.n_parents:                              # only if refs < parents
            parents.append(fix_hold_rows(unit, self.ref_xi[int(self.ref_order[0])]))
            ids.append(f"gan:u{unit.index}:ref{int(self.ref_order[0])}:dup{len(parents)}")
            origins.append("fresh_reference_duplicate")
        self.parent_xi = np.stack(parents[: self.n_parents])
        self.parent_ids = ids[: self.n_parents]
        self.parent_origins = origins[: self.n_parents]
        self.n_history_parents_used = sum(1 for o in self.parent_origins if o == "history_parent")

    def _init_theta(self, history) -> None:
        """Warm start mu / log sigma / logits by parent id and phi directly (fixed dimension)."""
        st = history.mode_state.get("gan") if hasattr(history, "mode_state") else None
        self.logits = np.zeros(self.n_parents)
        self.mu = np.zeros((self.n_parents, self.r))
        self.log_sigma = np.full((self.n_parents, self.r), np.log(max(self.sigma_init, 1e-9)))
        self.warm_started: List[str] = []
        if not st:
            self.phi = np.zeros(self.d_psi)
            self.warm_start_source = "cold"
            return
        phi = np.array(st.get("phi", []), dtype=np.float64)
        if phi.shape == (self.d_psi,) and np.all(np.isfinite(phi)):
            self.phi = phi
            self.warm_start_source = "history_mode_state"
        else:
            self.phi = np.zeros(self.d_psi)
            self.warm_start_source = "history_mode_state_without_phi"
        by_id: Dict[str, Dict[str, Any]] = {}
        for e in st.get("generator", []):
            if not isinstance(e, dict):
                continue
            by_id[str(e.get("parent_id"))] = e
            # lineage: a parent record created from component b inherits that component's Theta
            for d in e.get("descendant_ids", []) or []:
                by_id.setdefault(str(d), e)
        for b, pid in enumerate(self.parent_ids):
            e = by_id.get(str(pid))
            if e is None:
                continue
            mu = np.array(e.get("mu", []), dtype=np.float64)
            ls = np.array(e.get("log_sigma", []), dtype=np.float64)
            if mu.shape == (self.r,) and np.all(np.isfinite(mu)):
                self.mu[b] = mu
            if ls.shape == (self.r,) and np.all(np.isfinite(ls)):
                self.log_sigma[b] = np.clip(ls, np.log(self.sigma_min), np.log(self.sigma_max))
            lg = e.get("logit")
            if lg is not None and np.isfinite(float(lg)):
                self.logits[b] = float(lg)
            self.warm_started.append(f"{pid}<-{e.get('parent_id')}")
        self._clip_sigma()

    def _clip_sigma(self) -> None:
        self.log_sigma = np.clip(self.log_sigma, np.log(self.sigma_min), np.log(self.sigma_max))

    def alphas(self) -> np.ndarray:
        z = self.logits - self.logits.max()
        e = np.exp(z)
        return e / e.sum()

    # ================================================================== hints
    def hints(self, unit: UnitContext, history) -> List[tuple]:
        """Joint structure read off the top parent's realized composition: materials whose
        contribution trajectories move together get 'sync', opposed ones 'counter'."""
        if not len(self.parent_ids):
            return []
        b = int(np.argmax(self.alphas()))
        c = self.analyzer.split(self.parent_xi[b][self.fm_idx])[1]        # (Jf, M)
        out: List[Tuple[int, int, str, float]] = []
        sd = c.std(axis=0)
        for i in range(1, self.M):
            for j in range(i + 1, self.M):
                if sd[i] < 1e-9 or sd[j] < 1e-9:
                    continue
                rho = float(np.corrcoef(c[:, i], c[:, j])[0, 1])
                if not np.isfinite(rho) or abs(rho) < self.hint_min_corr:
                    continue
                out.append((i, j, "sync" if rho > 0 else "counter", abs(rho)))
        out.sort(key=lambda t: -t[3])
        self._last_hints = [(a, b2, rel) for (a, b2, rel, _) in out[: self.max_hints]]
        return list(self._last_hints)

    # ================================================================== proposals (eq. 41)
    def _sample_theta(self) -> Tuple[int, np.ndarray]:
        a = self.alphas()
        b = int(self.rng.choice(self.n_parents, p=a))
        w = self.mu[b] + np.exp(self.log_sigma[b]) * self.rng.standard_normal(self.r)
        return b, w

    def ideal(self, unit: UnitContext, b: int, w: np.ndarray) -> np.ndarray:
        """Xi_hat = Xi_b^parent + B_ref w, openness-scaled, contracted toward the goal, holds pinned."""
        pert = np.einsum("k,kjd->jd", np.asarray(w, dtype=np.float64), self.B_ref)
        xi_hat = self.parent_xi[b] + self.o[:, None] * pert
        contract = (unit.phase_names == "CONTRACT")
        if contract.any():
            lam = np.zeros(unit.J)
            lam[contract] = (1.0 - self.o[contract]) ** 2
            xi_hat = (1.0 - lam[:, None]) * xi_hat + lam[:, None] * unit.xi_goal
        return fix_hold_rows(unit, xi_hat)

    def propose(self, unit: UnitContext, history, round_index: int, n_targets: int) -> List[Target]:
        n = max(1, min(int(n_targets), self.samples_per_round))
        targets: List[Target] = []
        for a in range(n):
            b, w = self._sample_theta()
            targets.append(Target(
                f"gan:u{unit.index}:r{round_index}:{a}", self.ideal(unit, b, w),
                meta={"b": int(b), "w": [float(x) for x in w], "parent_id": self.parent_ids[b],
                      "parent_origin": self.parent_origins[b], "alpha": float(self.alphas()[b])}))
        self._round_targets[int(round_index)] = list(targets)
        return targets

    # ================================================================== mode error
    def mode_error(self, unit: UnitContext, cand: Candidate, target: Target) -> float:
        """Fit of the *realized* composition to the ideal one (free windows).  The discriminator
        deliberately does not enter the search score; it enters the generator update (eq. 44)."""
        return unit.mean_dist2(cand.xi, target.xi_hat)

    # ================================================================== adversarial update
    def update(self, unit: UnitContext, history, realizations: Sequence[Realization],
               round_index: int) -> Dict[str, Any]:
        if not realizations:
            return {}
        if round_index >= self.rounds_max:
            return {"skipped": "gan_adversarial_rounds_max reached", "round": int(round_index)}
        act = [np.asarray(r.candidate.xi, dtype=np.float64) for r in realizations]
        Psi_ref = np.stack([self._psi(x) for x in self.ref_xi])
        Psi_gen = np.stack([self._psi(x) for x in act])

        # ---------------------------------------------------- (43) discriminator, Theta fixed
        L0, D_ref0, D_gen0 = self._d_loss(Psi_ref, self.ref_w, Psi_gen, self.phi)
        nD = 0
        rejected = 0
        for _ in range(self.inner_steps_D):
            D_ref = _sigmoid(Psi_ref @ self.phi)
            D_gen = _sigmoid(Psi_gen @ self.phi)
            grad = (-(self.ref_w[:, None] * (1.0 - D_ref)[:, None] * Psi_ref).sum(axis=0)
                    + (D_gen[:, None] * Psi_gen).mean(axis=0) + 2.0 * self.l2 * self.phi)
            if not np.all(np.isfinite(grad)):
                rejected += 1
                break
            prev = self.phi
            cand_phi = self.phi - self.lr_D * grad
            L_try, _, _ = self._d_loss(Psi_ref, self.ref_w, Psi_gen, cand_phi)
            if not (np.isfinite(L_try) and np.all(np.isfinite(cand_phi))):
                self.phi = prev
                rejected += 1
                break
            self.phi = cand_phi
            nD += 1
        L1, D_ref1, D_gen1 = self._d_loss(Psi_ref, self.ref_w, Psi_gen, self.phi)

        # ---------------------------------------------------- (44) losses on realized compositions
        losses: List[float] = []
        samples: List[Tuple[int, np.ndarray]] = []
        detail: List[Dict[str, Any]] = []
        for k, r in enumerate(realizations):
            meta = r.target.meta or {}
            b = int(meta.get("b", 0))
            w = np.asarray(meta.get("w", np.zeros(self.r)), dtype=np.float64)
            if w.shape != (self.r,):
                w = np.zeros(self.r)
            d = float(_sigmoid(float(self.phi @ Psi_gen[k])))
            fit = float(r.mode_error)
            ef = float(r.candidate.e_form)
            ell = -np.log(max(d, 1e-12)) + self.lambda_fit * fit + self.lambda_f * ef
            losses.append(float(ell))
            samples.append((b, w))
            r.extra["gan_D"] = d
            r.extra["gan_neg_log_D"] = float(-np.log(max(d, 1e-12)))
            r.extra["gan_loss"] = float(ell)
            detail.append({"target": r.target.id, "b": b, "D_actual": d, "fit": fit,
                           "e_form": ef, "loss": float(ell), "realized": True})
        realized_ids = {r.target.id for r in realizations}
        for tg in self._round_targets.get(int(round_index), []):
            if tg.id in realized_ids:
                continue
            meta = tg.meta or {}
            b = int(meta.get("b", 0))
            w = np.asarray(meta.get("w", np.zeros(self.r)), dtype=np.float64)
            if w.shape != (self.r,):
                w = np.zeros(self.r)
            losses.append(self.C_fail)                      # finite cost for unrealizable proposals
            samples.append((b, w))
            detail.append({"target": tg.id, "b": b, "loss": self.C_fail, "realized": False})
        ell = np.array(losses, dtype=np.float64)

        # ---------------------------------------------------- (45) generator, D / basis fixed
        baseline = self._round_loss_mean
        baseline_source = "previous_round_mean"
        if baseline is None:
            baseline = float(ell.mean())
            baseline_source = "first_round_own_mean"
        adv = ell - float(baseline)
        nG = 0
        clipped = 0
        ess_stops = 0
        ess_trace: List[float] = []
        n_s = float(max(1, len(samples)))
        ess_min = self.ess_min_fraction * n_s
        logp_old = np.array([self._log_p(b, w) for (b, w) in samples], dtype=np.float64)
        for _ in range(self.inner_steps_G):
            # eq. (45) needs samples from the *current* p_Theta.  The batch was drawn from p_old, so
            # every re-use is corrected with self-normalised importance ratios p_Theta / p_old
            # (all ratios are 1 on the first step) and the inner loop stops when the effective
            # sample size falls below the trust region (audit §8).
            logp_new = np.array([self._log_p(b, w) for (b, w) in samples], dtype=np.float64)
            ratio = np.exp(np.clip(logp_new - logp_old, -30.0, 30.0))
            ess = float(ratio.sum() ** 2 / max(1e-300, float((ratio ** 2).sum())))
            ess_trace.append(ess)
            if ess < ess_min:
                ess_stops += 1
                break
            omega = ratio / max(1e-300, float(ratio.sum())) * n_s
            g_mu = np.zeros_like(self.mu)
            g_ls = np.zeros_like(self.log_sigma)
            g_lg = np.zeros_like(self.logits)
            a = self.alphas()
            sig = np.exp(self.log_sigma)
            for (b, w), adv_i, om in zip(samples, adv, omega):
                dw = (w - self.mu[b]) / np.maximum(sig[b] ** 2, 1e-12)
                g_mu[b] += om * adv_i * dw
                g_ls[b] += om * adv_i * (((w - self.mu[b]) ** 2) / np.maximum(sig[b] ** 2, 1e-12) - 1.0)
                oh = np.zeros(self.n_parents)
                oh[b] = 1.0
                g_lg += om * adv_i * (oh - a)
            n = n_s
            g_mu, g_ls, g_lg = g_mu / n, g_ls / n, g_lg / n
            gn = float(np.sqrt((g_mu ** 2).sum() + (g_ls ** 2).sum() + (g_lg ** 2).sum()))
            if not np.isfinite(gn):
                break
            if gn > self.grad_clip:
                f = self.grad_clip / gn
                g_mu, g_ls, g_lg = g_mu * f, g_ls * f, g_lg * f
                clipped += 1
            s_mu = np.clip(-self.lr_G * g_mu, -self.step_clip, self.step_clip)
            s_ls = np.clip(-self.lr_G * g_ls, -self.step_clip, self.step_clip)
            s_lg = np.clip(-self.lr_G * g_lg, -self.step_clip, self.step_clip)
            if not (np.all(np.isfinite(s_mu)) and np.all(np.isfinite(s_ls)) and np.all(np.isfinite(s_lg))):
                break
            self.mu = self.mu + s_mu
            self.log_sigma = self.log_sigma + s_ls
            self.logits = self.logits + s_lg
            self._clip_sigma()
            nG += 1
        self._round_loss_mean = float(ell.mean())
        self._unit_counts["D"] += nD
        self._unit_counts["G"] += nG
        self._unit_counts["D_rejected"] += rejected
        self._unit_counts["rounds"] += 1
        stats = {
            "D_loss_before": float(L0), "D_loss_after": float(L1),
            "D_mean_ref": float(D_ref1.mean()), "D_mean_gen": float(D_gen1.mean()),
            "D_mean_ref_before": float(D_ref0.mean()), "D_mean_gen_before": float(D_gen0.mean()),
            "G_loss_mean": float(ell.mean()), "updates_D": int(nD), "updates_G": int(nG),
            "baseline": float(baseline), "baseline_source": baseline_source,
            "rejected_D_steps": int(rejected), "clipped_G_steps": int(clipped),
            "importance_ess": ess_trace, "ess_stopped_G_steps": int(ess_stops),
            "importance_correction": "self-normalised p_theta/p_old on the re-used batch; stop when ESS < fraction * n",
            "psi_scale_source": self.psi_scale_source,
            "n_generated": int(len(act)), "n_failed_targets": int(len(ell) - len(act)),
            "phi_norm": float(np.linalg.norm(self.phi)),
            "alpha": self.alphas().tolist(), "sigma_mean": float(np.exp(self.log_sigma).mean()),
            "note": "no convergence / equilibrium is claimed (spec §10.5)",
        }
        self._round_records.append(dict(stats, unit=int(unit.index), round=int(round_index),
                                        per_sample=detail))
        return stats

    # ================================================================== 10.6 lineage
    def end_unit(self, unit: UnitContext, history, chosen: Realization, alternatives) -> None:
        meta = chosen.target.meta or {}
        chosen_parent = str(meta.get("parent_id", ""))
        kept: List[Dict[str, Any]] = []
        def _add(real: Realization, origin: str) -> str:
            pid = f"gan:u{unit.index}:c{real.candidate.id}"
            rec = {"id": pid, "unit": int(unit.index),
                   "blocks": self.objective.phase_blocks(unit, real.candidate.xi),
                   "score": float(real.total), "origin": origin,
                   "mode_error": float(real.mode_error), "e_form": float(real.e_form),
                   "D": float(self.discriminator_score(real.candidate.xi)),
                   "from_parent": str((real.target.meta or {}).get("parent_id", ""))}
            history.add_parent(rec)
            kept.append({k: v for k, v in rec.items() if k != "blocks"})
            return pid
        chosen_id = _add(chosen, "gan_chosen")
        alts = sorted([a for a in alternatives if a is not chosen], key=lambda z: z.total)[:2]
        for a in alts:
            _add(a, "gan_alternative")
        history.selected_parent_ids.append(chosen_parent or chosen_id)

        self.lineage.append({"unit": int(unit.index), "chosen_parent_id": chosen_parent,
                             "chosen_candidate_id": int(chosen.candidate.id),
                             "new_parent_id": chosen_id, "target_id": chosen.target.id,
                             "parent_origin": str(meta.get("parent_origin", "")),
                             "w": [float(x) for x in meta.get("w", [])],
                             "kept_parents": [k["id"] for k in kept]})
        for k in ("D", "G", "D_rejected", "rounds"):
            self._total_counts[k] += self._unit_counts[k]
        gen = [{"parent_id": str(pid), "origin": str(org), "mu": self.mu[b].tolist(),
                "log_sigma": self.log_sigma[b].tolist(),
                "sigma": np.exp(self.log_sigma[b]).tolist(),
                "logit": float(self.logits[b]), "alpha": float(self.alphas()[b]),
                # lineage link: parents stored this unit that descend from this component
                "descendant_ids": [str(k["id"]) for k in kept if str(k["from_parent"]) == str(pid)]}
               for b, (pid, org) in enumerate(zip(self.parent_ids, self.parent_origins))]
        rs = self.reference_summaries[-1]
        history.mode_state["gan"] = {
            "phi": [float(x) for x in self.phi],
            "psi_mean": [float(x) for x in self.psi_mean], "psi_std": [float(x) for x in self.psi_std],
            "generator": gen,
            "update_counts": {"D": int(self._total_counts["D"]), "G": int(self._total_counts["G"]),
                              "rejected_D": int(self._total_counts["D_rejected"]),
                              "adversarial_rounds": int(self._total_counts["rounds"])},
            "reference_weights_summary": {
                "unit": int(unit.index), "n": int(rs["n_references"]),
                "max_weight": float(max(rs["weights"])), "entropy_nats": float(rs["weight_entropy_nats"]),
                "mean_J_ref": float(np.mean(rs["J_ref"])),
                "diversity_mean_pair_dist2": float(rs["diversity_mean_pair_dist2"])},
            "lineage": [{"unit": int(l["unit"]), "chosen_parent_id": str(l["chosen_parent_id"]),
                         "chosen_candidate_id": int(l["chosen_candidate_id"]),
                         "new_parent_id": str(l["new_parent_id"])} for l in self.lineage][-8:],
            "last_unit": int(unit.index),
        }
        self.generator_traces.append({
            "unit": int(unit.index), "logits": self.logits.tolist(), "alpha": self.alphas().tolist(),
            "mu": self.mu.tolist(), "sigma": np.exp(self.log_sigma).tolist(),
            "parent_ids": list(self.parent_ids), "parent_origins": list(self.parent_origins),
            "history_parents_used": int(self.n_history_parents_used),
            "warm_started_parents": list(self.warm_started), "warm_start_source": self.warm_start_source,
            "basis_rank": int(self.r), "basis_source": self.basis_source,
            "basis_singular_values": [float(x) for x in self.basis_sv],
            "basis_frobenius_norms": self.basis_norms})
        self.discriminator_traces.append({
            "unit": int(unit.index), "phi": self.phi.tolist(), "phi_norm": float(np.linalg.norm(self.phi)),
            "feature_names": list(self.psi_names), "feature_scale_mean": self.psi_mean.tolist(),
            "feature_scale_std": self.psi_std.tolist(), "feature_scale_source": self.psi_scale_source,
            "D_chosen_realized": float(self.discriminator_score(chosen.candidate.xi)),
            "D_mean_reference": float(np.mean([self.discriminator_score(x) for x in self.ref_xi]))})
        self.update_counts.append({"unit": int(unit.index), "per_round": list(self._round_records),
                                   "unit_totals": dict(self._unit_counts),
                                   "cumulative": dict(self._total_counts)})
        # re-enabling a past composition line at new absolute times: whatever feasibility has
        # changed shows up as the realization gap of the proposals built from history parents
        gaps = [float(d["fit"]) for rec in self._round_records for d in rec["per_sample"]
                if d.get("realized") and self.parent_origins[int(d["b"])] == "history_parent"]
        self.unit_traces.append({
            "unit": int(unit.index), "chosen_target": chosen.target.id,
            "chosen_parent_id": chosen_parent, "normalized_mode_error": float(chosen.normalized_mode_error),
            "D_chosen": float(self.discriminator_score(chosen.candidate.xi)),
            "adversarial_rounds": int(self._unit_counts["rounds"]),
            "updates_D": int(self._unit_counts["D"]), "updates_G": int(self._unit_counts["G"]),
            "n_references": int(rs["n_references"]),
            "history_parents_used": int(self.n_history_parents_used),
            "hints": [list(h) for h in getattr(self, "_last_hints", [])],
            "history_parent_realization_gap_mean_dist2": (float(np.mean(gaps)) if gaps else None),
            "history_parent_proposals": len(gaps)})

    # ================================================================== checks / trace
    def signature(self) -> np.ndarray:
        parents_mean = self.parent_xi[:, self.fm_idx, :].mean(axis=(0, 1))
        return np.concatenate([self.phi, self.mu.ravel(), self.log_sigma.ravel(),
                               self.logits, parents_mean])

    def trace(self) -> Dict[str, Any]:
        return {
            "generator_parameters": self.generator_traces,
            "discriminator_parameters": self.discriminator_traces,
            "update_counts": self.update_counts,
            "lineage": self.lineage,
            "reference_summary": self.reference_summaries,
            "equations": {
                "40": "r_b ∝ exp(-J_ref/tau_ref), J_ref = E_form + 0.2 E_hist (discriminator not used)",
                "41": "Xi_hat = Xi_b^parent + B_ref w, p(b,w) = alpha_b N(w; mu_b, diag sigma_b^2)",
                "42": "D_phi(Xi) = sigmoid(phi^T psi(Xi)) on free windows of realized compositions",
                "43": "L_D = -E_ref log D - E_gen log(1-D) + l2 ||phi||^2",
                "44": "l = -log D(Xi^actual) + lambda_fit <d_xi^2> + lambda_f E_form (C_fail if unrealizable)",
                "45": "grad E l = E[(l - b_past) grad log p_Theta(b, w)]"},
            "claims": "bounded co-adaptive updates on realized compositions; no convergence, no "
                      "optimal GAN, no discriminator threshold used as a goal test",
            "warnings": list(self.warnings), "units": self.unit_traces}
