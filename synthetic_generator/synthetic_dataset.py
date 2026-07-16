"""Synthetic 2D Gaussian-blob datasets with several temporal-drift recipes.

Each ``DirichletSyntheticDataset`` method produces a flat ``(X, y, t, blob_ids,
blob_names)`` tuple covering ``n_bags`` bags at ``t = linspace(0, 1, n_bags)``.
Splitting into train/val/test is left to ``data.py`` (driven by VARIABLES.py).
Invariant: the first bag is at ``t = 0``, so ``TRAIN_SIZE = 1`` in
``VARIABLES.py`` yields ``t = 0`` training rows.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pdb


import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class BlobGeometry:
    """Static blob layout: means, covariances, names, and blob->class labels.

    ``means`` is (K, 2), ``covs`` is (K, 2, 2), ``names`` and ``labels`` are
    length-K. ``names`` is used only for the ``blob`` column in the CSV;
    ``labels`` maps each blob to its binary class (0 / 1).
    """
    means: tuple
    covs: tuple
    names: tuple
    labels: tuple

    @property
    def n_blobs(self):
        return len(self.names)


# Isotropic covariance shared by the binary (2-blob) approaches.
ISO_COV = ((0.20, 0.0), (0.0, 0.20))


class DirichletSyntheticDataset:
    """Generates 2D blob datasets with several time-drift recipes.

    Parameters
    ----------
    n_bags : int
        Number of bags emitted at ``t = linspace(0, 1, n_bags)``.
    samples_per_bag : int
        Samples per bag. ``data.py:725`` hardcodes 320 for the synthetic
        loader, so keep this at 320 unless that is also changed.
    random_state : int or None
        Seed for the internal numpy ``Generator``.
    label_shift : bool
        If True, approaches 1-4 also drift positive prevalence from
        ``LABEL_SHIFT_P0`` to ``LABEL_SHIFT_P1`` on top of their spatial
        shift. Stored as ``self.LABEL_SHIFT_ENABLED``.
    """

    SAMPLES_PER_BAG_DEFAULT = 320

    # --- Approach 0: original 3-blob joint shift --------------------------
    JOINT_GEOMETRY = BlobGeometry(
        means=((-0.2, -0.1), (0.8, 0.9), (1.0, -1.0)),
        covs=(
            ((0.50, -0.01), (-0.01, 0.40)),
            ((0.32,  0.08), ( 0.08, 0.20)),
            ((0.38, -0.06), (-0.06, 0.18)),
        ),
        names=("neg", "D", "B"),
        labels=(0, 1, 1),
    )
    JOINT_ALPHA = (20.0, 20.0, 20.0)

    # --- Approaches 1-5: shared 2-blob binary geometry --------------------
    BINARY_GEOMETRY = BlobGeometry(
        means=((-0.5, 0.0), (1.5, 0.0)),
        covs=(ISO_COV, ISO_COV),
        names=("neg", "pos"),
        labels=(0, 1),
    )

    # --- Approach 6: 4-blob covariate-shift geometry ----------------------
    # Two columns (left = negative, right = positive), each with an upper and
    # a lower blob stacked vertically:
    #     neg_up   pos_up
    #     neg_low  pos_low
    # Order below is (neg_up, neg_low, pos_up, pos_low); labels follow.
    COVARIATE_GEOMETRY = BlobGeometry(
        means=((-0.5, 0.5), (-0.5, -0.5), (0.5, 0.5), (0.5, -0.5)),
        covs=(ISO_COV, ISO_COV, ISO_COV, ISO_COV),
        names=("neg_up", "neg_low", "pos_up", "pos_low"),
        labels=(0, 0, 1, 1),
    )

    # Dirichlet concentration. Higher -> each bag's class proportions hug the
    # target prevalence trajectory more tightly. Governs PROPORTION-variation
    # smoothness only; it does NOT affect blob spread (that is the covariance).
    # With the balanced (1, 1) target used by approaches 1-4, the resulting
    # alpha is (c, c); concentration = 1 reproduces uniform Dir(1, 1) sampling.
    PREVALENCE_CONCENTRATION = 1.0

    # Per-approach motion knobs.
    SHIFT_RATE = 3.0        # translation units per unit t (approaches 1-3)
    SPIN_TURNS = 1.0        # full revolutions over t in [0, 1] (approach 4)
    LABEL_SHIFT_P0 = 0.3    # positive prevalence at t=0 (approach 5)
    LABEL_SHIFT_P1 = 0.7    # positive prevalence at t=1 (approach 5)
    HORIZONTAL_MIX_END = 0.5  # fraction of the horizontal gap closed at t=1 (horizontal_mix)
    # Approach 7: the four blob proportions (the Dirichlet target means).
    # All start at COVARIATE_SHIFT_START at t=0; the two rising blobs
    # (neg_up, pos_low) climb to COVARIATE_SHIFT_RISE while the two decaying
    # blobs (neg_low, pos_up) fall to COVARIATE_SHIFT_DECAY at t=1. The four
    # endpoints must sum to 1 and each class stays at 50% throughout.
    COVARIATE_SHIFT_START = 0.25
    COVARIATE_SHIFT_RISE = 0.45
    COVARIATE_SHIFT_DECAY = 0.05

    def __init__(
        self, n_bags=50, samples_per_bag=320, random_state=None,
        label_shift=False,
    ):
        self.n_bags = int(n_bags)
        self.samples_per_bag = int(samples_per_bag)
        self.random_state = random_state
        self.rng_ = np.random.default_rng(random_state)
        # When True, approaches 1-4 also drift positive prevalence from
        # LABEL_SHIFT_P0 to LABEL_SHIFT_P1 (same drift as temporal_label_shift)
        # on top of their spatial shift. Default off keeps existing CSVs
        # reproducible.
        self.LABEL_SHIFT_ENABLED = bool(label_shift)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _alpha_for_prevalence(prevalence, concentration):
        """Dirichlet alpha whose mean equals ``prevalence``.

        ``Dir(concentration * prevalence)`` has mean ``prevalence`` and
        per-component variance ``p(1-p) / (concentration + 1)``, so a higher
        concentration yields smoother per-bag proportions.
        """
        return concentration * np.asarray(prevalence, dtype=float)

    def _alpha_at(self, t):
        """Dirichlet alpha at time ``t`` for binary approaches.

        When ``LABEL_SHIFT_ENABLED`` is False (default), returns the balanced
        alpha used by approaches 1-4 today. When True, returns the linearly
        drifting alpha used by ``temporal_label_shift``.
        """
        if self.LABEL_SHIFT_ENABLED:
            p_pos = self.LABEL_SHIFT_P0 + (self.LABEL_SHIFT_P1 - self.LABEL_SHIFT_P0) * t
            return self._alpha_for_prevalence(
                (1.0 - p_pos, p_pos), self.PREVALENCE_CONCENTRATION
            )
        return self._alpha_for_prevalence((1, 1), self.PREVALENCE_CONCENTRATION)

    def _generate_bags(self, geometry, params_at):
        """Loop over ``n_bags`` evenly spaced ``t``s and concatenate bags.

        ``geometry`` supplies the static ``names`` / ``labels``. ``params_at(t)``
        returns a dict with ``means`` (K,2), ``covs`` (K,2,2), ``alpha`` (K,)
        and optionally ``weights`` (K,) — a multiplicative tilt on the sampled
        Dirichlet proportions (defaults to ones).
        Returns ``(X, y, t, blob_ids, blob_names)``.
        """
        labels = np.asarray(geometry.labels, dtype=np.int64)
        names = geometry.names
        ts = np.linspace(0.0, 1.0, self.n_bags)

        X_parts, y_parts, t_parts, blob_parts = [], [], [], []
        for t in ts:
            params = params_at(float(t))
            means = np.asarray(params["means"], dtype=float)
            covs = np.asarray(params["covs"], dtype=float)
            alpha = np.asarray(params["alpha"], dtype=float)
            weights = np.asarray(
                params.get("weights", np.ones(alpha.shape[0])), dtype=float
            )

            pi_raw = self.rng_.dirichlet(alpha)
            pi_eff = weights * pi_raw
            pi_eff = pi_eff / pi_eff.sum()
            counts = self.rng_.multinomial(self.samples_per_bag, pi_eff)

            for k, n_k in enumerate(counts):
                if n_k == 0:
                    continue
                X_parts.append(self.rng_.multivariate_normal(means[k], covs[k], size=n_k))
                y_parts.append(np.full(n_k, labels[k], dtype=np.int64))
                blob_parts.append(np.full(n_k, k, dtype=np.int64))
                t_parts.append(np.full(n_k, float(t), dtype=float))

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        t = np.concatenate(t_parts, axis=0)
        blob_ids = np.concatenate(blob_parts, axis=0)
        return X, y, t, blob_ids, tuple(names)

    def _translation_shift(self, shift_at):
        """Shared body for the translation approaches (horizontal/diagonal/vertical).

        ``shift_at(t)`` returns the ``(dx, dy)`` offset added to both blobs of
        ``BINARY_GEOMETRY``. Alpha is balanced by default; with
        ``LABEL_SHIFT_ENABLED`` it drifts via ``_alpha_at``.
        """
        mu0 = np.asarray(self.BINARY_GEOMETRY.means, dtype=float)

        def params_at(t):
            shift = np.asarray(shift_at(t), dtype=float)
            return {
                "means": tuple(map(tuple, mu0 + shift)),
                "covs": self.BINARY_GEOMETRY.covs,
                "alpha": self._alpha_at(t),
            }
        return self._generate_bags(self.BINARY_GEOMETRY, params_at)

    # ------------------------------------------------------------------
    # Approach 0 - temporal joint shift
    # ------------------------------------------------------------------
    def temporal_joint_shift(self):
        """Original 3-blob recipe: prior shift between D and B; fixed geometry.

        This is the only approach that uses the multiplicative ``weights`` tilt.
        """
        def params_at(t):
            return {
                "means": self.JOINT_GEOMETRY.means,
                "covs": self.JOINT_GEOMETRY.covs,
                "alpha": self.JOINT_ALPHA,
                "weights": (1.0, 1.0 + t, 1.0 - t),
            }
        return self._generate_bags(self.JOINT_GEOMETRY, params_at)

    # ------------------------------------------------------------------
    # Approach 1 - horizontal shift
    # ------------------------------------------------------------------
    def horizontal_shift(self):
        """Two blobs translated together along +x1 as t grows."""
        return self._translation_shift(lambda t: (self.SHIFT_RATE * t, 0.0))

    # ------------------------------------------------------------------
    # Approach 2 - diagonal shift
    # ------------------------------------------------------------------
    def diagonal_shift(self):
        """Two blobs translated together along (+x1, +x2) as t grows."""
        return self._translation_shift(
            lambda t: (self.SHIFT_RATE * t, self.SHIFT_RATE * t)
        )

    # ------------------------------------------------------------------
    # Approach 3 - vertical shift
    # ------------------------------------------------------------------
    def vertical_shift(self):
        """Two blobs translated together along +x2 as t grows."""
        return self._translation_shift(lambda t: (0.0, self.SHIFT_RATE * t))

    # ------------------------------------------------------------------
    # Approach 4 - spinning shift
    # ------------------------------------------------------------------
    def spinning_shift(self):
        """Two blobs rotate around their midpoint as t grows."""
        mu0 = np.asarray(self.BINARY_GEOMETRY.means, dtype=float)
        center = mu0.mean(axis=0)
        offsets = mu0 - center

        def params_at(t):
            theta = 2.0 * np.pi * self.SPIN_TURNS * t
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            means = center + offsets @ R.T
            return {
                "means": tuple(map(tuple, means)),
                "covs": self.BINARY_GEOMETRY.covs,
                "alpha": self._alpha_at(t),
            }
        return self._generate_bags(self.BINARY_GEOMETRY, params_at)

    # ------------------------------------------------------------------
    # Approach 8 - horizontal mix
    # ------------------------------------------------------------------
    def horizontal_mix(self):
        """Two blobs start separated and close the horizontal gap as t grows.

        Unlike ``horizontal_shift`` (both blobs translate together to the same
        side), here the negative and positive blobs move toward their shared
        midpoint along x, so at ``t = 1`` they are almost fully overlapping
        (``HORIZONTAL_MIX_END`` of the gap closed). There is no prevalence
        trend: alpha is the balanced ``Dir(1, 1) * PREVALENCE_CONCENTRATION``.
        """
        mu0 = np.asarray(self.BINARY_GEOMETRY.means, dtype=float)
        center_x = mu0[:, 0].mean()

        def params_at(t):
            frac = self.HORIZONTAL_MIX_END * t
            means = mu0.copy()
            means[:, 0] = center_x + (mu0[:, 0] - center_x) * (1.0 - frac)
            return {
                "means": tuple(map(tuple, means)),
                "covs": self.BINARY_GEOMETRY.covs,
                "alpha": self._alpha_for_prevalence(
                    (1, 1), self.PREVALENCE_CONCENTRATION
                ),
            }
        return self._generate_bags(self.BINARY_GEOMETRY, params_at)

    # ------------------------------------------------------------------
    # Approach 5 - temporal label shift
    # ------------------------------------------------------------------
    def temporal_label_shift(self):
        """Stationary blobs; the positive prevalence grows smoothly with t.

        Per-bag prevalence is drawn from ``Dir(concentration * (1-p, p))`` with
        ``p`` rising linearly from ``LABEL_SHIFT_P0`` to ``LABEL_SHIFT_P1``.
        """
        def params_at(t):
            p_pos = self.LABEL_SHIFT_P0 + (self.LABEL_SHIFT_P1 - self.LABEL_SHIFT_P0) * t
            return {
                "means": self.BINARY_GEOMETRY.means,
                "covs": self.BINARY_GEOMETRY.covs,
                "alpha": self._alpha_for_prevalence(
                    (1.0 - p_pos, p_pos), self.PREVALENCE_CONCENTRATION
                ),
            }
        return self._generate_bags(self.BINARY_GEOMETRY, params_at)

    # ------------------------------------------------------------------
    # Approach 6 - label shift
    # ------------------------------------------------------------------
    def label_shift(self):
        """Stationary blobs; the positive prevalence grows smoothly with t.

        Per-bag prevalence is drawn from ``Dir(concentration * (1-p, p))`` with
        ``p`` rising linearly from ``LABEL_SHIFT_P0`` to ``LABEL_SHIFT_P1``.
        """
        def params_at(t):
            return {
                "means": self.BINARY_GEOMETRY.means,
                "covs": self.BINARY_GEOMETRY.covs,
                "alpha": self._alpha_for_prevalence(
                    (1, 1), self.PREVALENCE_CONCENTRATION
                ),
            }
        return self._generate_bags(self.BINARY_GEOMETRY, params_at)

    # ------------------------------------------------------------------
    # Approach 7 - global covariate shift
    # ------------------------------------------------------------------
    def global_covariate_shift(self):
        """Four stationary blobs (2 per class, stacked); mass shifts vertically.

        Layout (columns = class, rows stacked vertically)::

            neg_up   pos_up
            neg_low  pos_low

        At ``t = 0`` ``neg_up`` and ``pos_low`` start high
        (``COVARIATE_SHIFT_RISE``) while ``neg_low`` and ``pos_up`` start low
        (``COVARIATE_SHIFT_DECAY``). As ``t`` grows all four converge to
        ``COVARIATE_SHIFT_START``: ``neg_up`` / ``pos_low`` decay while
        ``neg_low`` / ``pos_up`` rise. These four proportions are the Dirichlet
        target means (exactly as the class prevalence is in the binary
        approaches). Each class keeps a fixed 50/50 total, so ``P(y)`` is
        stationary while ``P(x)`` drifts — a pure covariate shift. Blobs never
        move; only the Dirichlet proportions change with ``t``.
        """
        start = self.COVARIATE_SHIFT_START

        def lerp(begin, t):
            return begin + (start - begin) * t

        def params_at(t):
            high = lerp(self.COVARIATE_SHIFT_RISE, t)
            low = lerp(self.COVARIATE_SHIFT_DECAY, t)
            # order: neg_up, neg_low, pos_up, pos_low
            prevalence = (
                high,  # neg_up decays  0.45 -> 0.25
                low,   # neg_low rises  0.05 -> 0.25
                low,   # pos_up rises   0.05 -> 0.25
                high,  # pos_low decays 0.45 -> 0.25
            )
            return {
                "means": self.COVARIATE_GEOMETRY.means,
                "covs": self.COVARIATE_GEOMETRY.covs,
                "alpha": self._alpha_for_prevalence(
                    prevalence, self.PREVALENCE_CONCENTRATION
                ),
            }
        return self._generate_bags(self.COVARIATE_GEOMETRY, params_at)

    # ------------------------------------------------------------------
    # Experiment folder builder
    # ------------------------------------------------------------------
    def build_experiment_folder(
        self, method_name, concentrations, samples_per_bag, out_dir,
    ):
        """Write the (concentration x samples_per_bag) grid of CSVs for one approach.

        For each pair ``(c, s)``, instantiates a fresh dataset with
        ``samples_per_bag=s`` and ``label_shift=self.LABEL_SHIFT_ENABLED``,
        sets ``PREVALENCE_CONCENTRATION=c``, runs ``method_name``, and saves
        to ``out_dir / f"{method_name}_experiment" / f"c{c}_s{s}.csv"``. When
        ``self.LABEL_SHIFT_ENABLED`` is True, the output folder is suffixed
        ``_labelshift_experiment`` instead, so the balanced CSVs already on
        disk are not clobbered.
        Returns the experiment directory.
        """
        suffix = "_labelshift_experiment" if self.LABEL_SHIFT_ENABLED else "_experiment"
        approach_dir = Path(out_dir) / f"{method_name}{suffix}"
        for c in tqdm(concentrations, desc=f"{method_name} concentrations"):
            for s in tqdm(
                samples_per_bag,
                desc=f"c={c} samples_per_bag",
                leave=False,
            ):
                ds = type(self)(
                    samples_per_bag=s, random_state=0,
                    label_shift=self.LABEL_SHIFT_ENABLED,
                )
                ds.PREVALENCE_CONCENTRATION = float(c)
                X, y, t, blob_ids, blob_names = getattr(ds, method_name)()
                save_dataset(
                    X, y, t, blob_ids,
                    approach_dir / f"c{c}_s{s}.csv",
                    blob_names=blob_names,
                )
        return approach_dir


def save_dataset(X, y, t, blob_ids, out_path, blob_names):
    """Write a flat synthetic dataset to CSV.

    Columns: ``x1, x2, label, t, blob``. The ``split`` column is intentionally
    omitted (``data.py:712`` ignores it). Returns ``(out_path, n_rows)``.
    """
    import pandas as pd

    name_arr = np.asarray(blob_names)
    df = pd.DataFrame({
        "x1": X[:, 0],
        "x2": X[:, 1],
        "label": y.astype(np.int64),
        "t": t,
        "blob": name_arr[blob_ids.astype(np.int64)],
    })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path, len(df)


def prevalence_per_timestep(csv_path):
    """Per-timestep positive-class prevalence for a horizontal-sweep CSV.

    The Dirichlet concentration is parsed from the filename pattern
    ``_c<int>.csv``. Prevalence is computed per unique ``t`` as the fraction
    of ``label == 1`` rows in that bag.

    Returns
    -------
    concentration : int or None
        Concentration parsed from the filename, or ``None`` if absent.
    ts : np.ndarray
        Unique ``t`` values in ascending order, shape ``(n_bags,)``.
    positive_prevalence : np.ndarray
        Positive-class prevalence at each ``t``, shape ``(n_bags,)``.
    """
    import re
    import pandas as pd

    csv_path = Path(csv_path)
    match = re.search(r"_c(\d+)\.csv$", csv_path.name)
    concentration = int(match.group(1)) if match else None

    df = pd.read_csv(csv_path)
    series = df.groupby("t")["label"].mean().sort_index()
    # pdb.set_trace()
    return concentration, series.index.to_numpy(), series.to_numpy()


APPROACHES = (
    # ("temporal_joint_shift",   "temporal_joint_shift7"),
    # ("horizontal_shift",       "temporal_joint_shift_horizontal2"),
    # ("diagonal_shift",         "temporal_joint_shift_diagonal7"),
#     ("vertical_shift",         "temporal_joint_shift_vertical7"),
    # ("spinning_shift",         "temporal_joint_shift_spinning_single"),
#     ("temporal_label_shift",   "temporal_label_shift7"),
    # ("label_shift",             "label_shift2"),
    # ("global_covariate_shift",  "global_covariate_shift2"),
    ("horizontal_mix",          "horizontal_mix"),
)


# HORIZONTAL_SWEEP_CONCENTRATIONS = list(range(1, 101))


# Concentration x samples-per-bag experiment grids.
EXPERIMENT_APPROACHES = (
    # "horizontal_shift",
    # "diagonal_shift",
    # "vertical_shift",
    "spinning_shift",
    # "temporal_label_shift",
)
EXPERIMENT_CONCENTRATIONS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]   # 11 values
EXPERIMENT_SAMPLES_PER_BAG = list(range(320, 3201, 320))                  # 10 values: 320..3200


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "data"
    # Flip label_shift=True here to drift positive prevalence on top of the
    # spatial shift for approaches 1-4. Output goes to a _labelshift_experiment
    # folder so the balanced CSVs already on disk are not overwritten.
    # builder = DirichletSyntheticDataset(label_shift=True)
    # for method_name in EXPERIMENT_APPROACHES:
    #     approach_dir = builder.build_experiment_folder(
    #         method_name,
    #         EXPERIMENT_CONCENTRATIONS,
    #         EXPERIMENT_SAMPLES_PER_BAG,
    #         out_dir,
    #     )
    #     print(
    #         f"{method_name}: wrote "
    #         f"{len(EXPERIMENT_CONCENTRATIONS) * len(EXPERIMENT_SAMPLES_PER_BAG)} CSVs "
    #         f"to {approach_dir}"
    #     )

    # --- Previous APPROACHES loop (commented out, kept for reference) -------
    out_dir = Path(__file__).resolve().parent / "data"
    for method_name, csv_stem in APPROACHES:
        ds = DirichletSyntheticDataset(random_state=0, samples_per_bag=1000, label_shift=True)
        # global_covariate_shift has 4 blobs, so it needs a higher concentration
        # than the 2-blob approaches for each bag to track the target proportions
        # (with c=1 the Dirichlet mass collapses onto a single blob per bag).
        ds.PREVALENCE_CONCENTRATION = 50.0
        ds.SPIN_TURNS = 1.0
        X, y, t, blob_ids, blob_names = getattr(ds, method_name)()
        out_path, n_rows = save_dataset(
            X, y, t, blob_ids, out_dir / f"{csv_stem}.csv", blob_names=blob_names
        )
        print(f"wrote {out_path} ({n_rows} rows, blobs={blob_names})")