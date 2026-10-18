"""
Phase Space Visualization for Tail Risk
========================================

Phase space representations from physics provide powerful ways
to visualize the state of complex systems.

In physics:
- Position × Momentum space (classical mechanics)
- State variables that fully describe system

In finance (our mapping):
- Returns × Volatility × Correlation space
- State variables that describe market regime

Key Insight:
The market's "trajectory" through phase space reveals:
- Current regime
- Direction of evolution
- Proximity to critical points (crashes)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from typing import Tuple, Optional, Dict, List
import warnings


def _vol_regime_labels(k: int) -> List[str]:
    """Regime names for k clusters ordered from low to high volatility."""
    if k <= 1:
        return ['NORMAL'][:max(k, 0)]
    if k == 2:
        return ['LOW_VOL', 'HIGH_VOL']
    if k == 3:
        return ['LOW_VOL', 'NORMAL', 'HIGH_VOL']
    return ['LOW_VOL'] + [f'NORMAL_{i}' for i in range(1, k - 1)] + ['HIGH_VOL']


def _points_on_lines(lines: List[np.ndarray], min_len: int) -> int:
    """Number of True entries that lie on runs of length >= min_len.

    The 1-D boolean arrays are concatenated with a False separator so that runs
    never join across arrays; run lengths come from one vectorised diff.
    """
    if not lines:
        return 0
    parts = []
    for line in lines:
        parts.append(np.asarray(line, dtype=np.int8))
        parts.append(np.zeros(1, dtype=np.int8))
    flat = np.concatenate([np.zeros(1, dtype=np.int8)] + parts)
    edges = np.diff(flat)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    lengths = ends - starts
    return int(lengths[lengths >= min_len].sum())


class Arrow3D(FancyArrowPatch):
    """3D arrow for phase space visualization."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class PhaseSpaceAnalyzer:
    """
    Analyze market dynamics in phase space.
    """

    def __init__(self, returns: np.ndarray, window: int = 20):
        """
        Initialize phase space analyzer.

        Args:
            returns: Time series of returns
            window: Rolling window for state calculation
        """
        self.returns = np.asarray(returns)
        self.window = window
        self.n = len(returns)

        # Compute phase space coordinates
        self._compute_coordinates()

    def _compute_coordinates(self):
        """Compute phase space coordinates."""
        n = self.n
        window = self.window

        # X: Mean return (momentum)
        self.X = np.array([np.mean(self.returns[max(0, i-window):i+1])
                          for i in range(n)])

        # Y: Volatility (energy)
        self.Y = np.array([np.std(self.returns[max(0, i-window):i+1])
                          for i in range(n)])

        # Z: Skewness (asymmetry) of the `window` returns before i
        # (same values as stats.skew(returns[i-window:i]) per index, vectorised)
        from scipy import stats
        self.Z = np.zeros(n)
        if 0 < window < n:
            windows = np.lib.stride_tricks.sliding_window_view(
                np.asarray(self.returns, dtype=float), window)[:n - window]
            self.Z[window:] = stats.skew(windows, axis=1)

    def compute_velocity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute velocity in phase space (rate of change of coordinates).

        High velocity = rapid regime change.
        """
        dX = np.gradient(self.X)
        dY = np.gradient(self.Y)
        dZ = np.gradient(self.Z)

        return dX, dY, dZ

    def compute_acceleration(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute acceleration in phase space.

        Acceleration toward instability = warning sign.
        """
        dX, dY, dZ = self.compute_velocity()

        ddX = np.gradient(dX)
        ddY = np.gradient(dY)
        ddZ = np.gradient(dZ)

        return ddX, ddY, ddZ

    def phase_space_distance(self, reference_point: Optional[Tuple[float, float, float]] = None,
                             standardize: bool = False) -> np.ndarray:
        """
        Compute distance from a reference point in phase space.

        Default reference: stable equilibrium (0, historical_vol, 0)

        Args:
            reference_point: (X, Y, Z) in the original units.
            standardize: If True, every axis is divided by its standard deviation
                before the Euclidean distance is taken. In raw units the skewness
                axis (order 1) dwarfs mean return and volatility (order 1e-2), so
                the unscaled distance is essentially |skewness - Z_ref|.
        """
        if reference_point is None:
            reference_point = (0, np.mean(self.Y), 0)

        scale = self._axis_scale() if standardize else np.ones(3)

        distance = np.sqrt(
            ((self.X - reference_point[0]) / scale[0])**2 +
            ((self.Y - reference_point[1]) / scale[1])**2 +
            ((self.Z - reference_point[2]) / scale[2])**2
        )

        return distance

    def _axis_scale(self) -> np.ndarray:
        """Standard deviation of X, Y, Z (1 for a constant axis)."""
        coords = np.column_stack([self.X, self.Y, self.Z])
        scale = np.nanstd(coords, axis=0)
        return np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)

    def detect_attractor_regions(self, n_clusters: int = 3, standardize: bool = True) -> Dict:
        """
        Identify attractor regions in phase space.

        Markets tend to cluster around certain "attractor" states:
        - Normal regime attractor
        - High volatility attractor
        - Crisis attractor

        Args:
            n_clusters: Number of clusters (Ward linkage, ``maxclust``).
            standardize: Cluster on z-scored coordinates (default). The raw axes
                differ by two orders of magnitude (mean return and volatility
                ~1e-2, skewness ~1), so unscaled Ward clustering splits almost
                only on skewness and ignores volatility. ``False`` restores the
                old unscaled behaviour.

        Returns:
            dict with ``cluster_labels`` (1..k per valid row), ``centers`` (k x 3,
            row i = cluster i+1, original units), ``regime_order`` (0-based
            cluster indices sorted by centre volatility, low to high),
            ``regime_labels`` (one label per entry of ``regime_order``) and
            ``cluster_regimes`` ({cluster id: label}).
        """
        from scipy.cluster.hierarchy import fcluster, linkage

        # Stack coordinates
        coords = np.column_stack([self.X, self.Y, self.Z])

        # Remove NaN
        valid_mask = ~np.any(np.isnan(coords), axis=1)
        valid_coords = coords[valid_mask]

        if len(valid_coords) < n_clusters:
            return {'error': 'Insufficient valid data'}

        features = valid_coords
        if standardize:
            scale = np.std(valid_coords, axis=0)
            scale = np.where(scale > 0, scale, 1.0)
            features = (valid_coords - np.mean(valid_coords, axis=0)) / scale

        # Hierarchical clustering
        Z = linkage(features, method='ward')
        clusters = fcluster(Z, n_clusters, criterion='maxclust')

        # Compute cluster centers (original units)
        centers = []
        for i in range(1, n_clusters + 1):
            cluster_points = valid_coords[clusters == i]
            centers.append(np.mean(cluster_points, axis=0) if len(cluster_points)
                           else np.full(3, np.nan))

        # Identify regime by volatility
        centers = np.array(centers)
        vol_order = np.argsort(centers[:, 1])  # Sort by Y (volatility)

        regime_labels = _vol_regime_labels(n_clusters)

        return {
            'cluster_labels': clusters,
            'centers': centers,
            'regime_order': vol_order,
            'regime_labels': regime_labels,
            'cluster_regimes': {int(c) + 1: label for c, label in zip(vol_order, regime_labels)},
        }

    def plot_phase_portrait(self, figsize: Tuple[int, int] = (14, 10),
                           show_velocity: bool = True,
                           show_attractors: bool = True) -> plt.Figure:
        """
        Create 3D phase portrait visualization.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Main trajectory
        time_color = np.arange(self.n)
        scatter = ax.scatter(self.X, self.Y, self.Z,
                            c=time_color, cmap='viridis', s=20, alpha=0.6)

        # Path
        ax.plot(self.X, self.Y, self.Z, 'b-', alpha=0.2, linewidth=0.5)

        # Velocity vectors (subsampled)
        if show_velocity:
            dX, dY, dZ = self.compute_velocity()
            step = max(1, self.n // 30)

            for i in range(0, self.n - step, step):
                if not np.any(np.isnan([dX[i], dY[i], dZ[i]])):
                    arrow = Arrow3D(
                        [self.X[i], self.X[i] + dX[i] * 10],
                        [self.Y[i], self.Y[i] + dY[i] * 10],
                        [self.Z[i], self.Z[i] + dZ[i] * 10],
                        mutation_scale=5, lw=1, arrowstyle='->', color='red', alpha=0.5
                    )
                    ax.add_artist(arrow)

        # Attractors
        if show_attractors:
            try:
                attractors = self.detect_attractor_regions()
                if 'centers' in attractors:
                    centers = attractors['centers']
                    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                              c='red', s=200, marker='*', label='Attractors')
            except Exception as exc:  # clustering is optional decoration
                warnings.warn(f"Attractor detection failed: {exc}", RuntimeWarning)

        # Mark current state
        ax.scatter([self.X[-1]], [self.Y[-1]], [self.Z[-1]],
                  c='green', s=150, marker='D', label='Current State')

        ax.set_xlabel('Return (Momentum)', fontsize=12)
        ax.set_ylabel('Volatility (Energy)', fontsize=12)
        ax.set_zlabel('Skewness (Asymmetry)', fontsize=12)
        ax.set_title('Market Phase Portrait', fontsize=14, fontweight='bold')

        plt.colorbar(scatter, ax=ax, label='Time', shrink=0.6)
        ax.legend()

        return fig


class LyapunovExponentEstimator:
    """
    Estimate Lyapunov exponents from time series.

    Positive Lyapunov exponent = chaotic dynamics = unpredictable
    This relates to tail risk: chaotic regimes have higher extreme event probability.
    """

    def __init__(self, data: np.ndarray, embedding_dim: int = 3, tau: int = 1):
        """
        Initialize Lyapunov estimator.

        Args:
            data: Time series
            embedding_dim: Embedding dimension for phase space reconstruction
            tau: Time delay for embedding
        """
        self.data = np.asarray(data)
        self.m = embedding_dim
        self.tau = tau

        # Create embedded phase space
        self._embed()

    def _embed(self):
        """Create delay embedding of the time series."""
        n = len(self.data)
        m = self.m
        tau = self.tau

        # Embedding (empty if the series is shorter than the embedding window)
        n_vectors = max(0, n - (m - 1) * tau)
        self.embedded = np.zeros((n_vectors, m))

        for i in range(n_vectors):
            for j in range(m):
                self.embedded[i, j] = self.data[i + j * tau]

    def estimate_largest_lyapunov(self, n_neighbors: int = 5,
                                   max_iter: int = 100,
                                   seed: Optional[int] = None) -> float:
        """
        Estimate largest Lyapunov exponent using Rosenstein method.

        λ > 0: Chaotic (sensitive to initial conditions)
        λ ≈ 0: Quasi-periodic
        λ < 0: Stable fixed point

        Args:
            n_neighbors: Neighbour candidates (plus 10) queried per start point.
            max_iter: Number of random start points.
            seed: Seed for the start points. ``None`` uses the global
                ``np.random`` state (previous behaviour).

        Returns:
            Mean of log(d_t / d_0) / t for t = 1..9, or NaN if the embedding has
            too few vectors (fewer than 2 * n_neighbors, or at most 10).
        """
        from scipy.spatial import cKDTree

        embedded = self.embedded
        n = len(embedded)

        # Start points are drawn from [0, n - 10): needs more than 10 vectors
        if n < n_neighbors * 2 or n <= 10:
            return np.nan

        if seed is None:
            draw_start = lambda: np.random.randint(0, n - 10)
        else:
            rng = np.random.default_rng(seed)
            draw_start = lambda: int(rng.integers(0, n - 10))

        # Build KD-tree for nearest neighbor search
        tree = cKDTree(embedded)
        k_query = min(n_neighbors + 10, n)

        # Track divergence
        divergence = []

        for _ in range(max_iter):
            # Random starting point
            i = draw_start()

            # Find nearest neighbor (excluding temporal neighbors)
            distances, indices = tree.query(embedded[i], k=k_query)

            # Find first neighbor not temporally close
            for j, idx in enumerate(indices[1:]):
                if abs(idx - i) > 5:  # Temporal separation
                    neighbor_idx = idx
                    initial_dist = distances[j + 1]
                    break
            else:
                continue

            # Track separation
            for t in range(1, min(10, n - max(i, neighbor_idx))):
                try:
                    final_dist = np.linalg.norm(embedded[i + t] - embedded[neighbor_idx + t])
                    if initial_dist > 0 and final_dist > 0:
                        divergence.append(np.log(final_dist / initial_dist) / t)
                except (IndexError, FloatingPointError):
                    pass

        if len(divergence) > 0:
            return np.mean(divergence)
        else:
            return np.nan

    def classify_dynamics(self, n_neighbors: int = 5, max_iter: int = 100,
                          seed: Optional[int] = None) -> Dict:
        """Classify system dynamics based on Lyapunov exponent."""
        lyap = self.estimate_largest_lyapunov(n_neighbors=n_neighbors,
                                              max_iter=max_iter, seed=seed)

        if np.isnan(lyap):
            return {'classification': 'UNKNOWN', 'lyapunov': lyap,
                    'risk_implication': 'Not enough data for an estimate'}

        if lyap > 0.1:
            classification = 'CHAOTIC'
            risk_implication = 'High unpredictability, elevated tail risk'
        elif lyap > 0:
            classification = 'WEAKLY_CHAOTIC'
            risk_implication = 'Moderate unpredictability'
        elif lyap > -0.1:
            classification = 'QUASI_PERIODIC'
            risk_implication = 'Some predictability, moderate tail risk'
        else:
            classification = 'STABLE'
            risk_implication = 'Low unpredictability, lower tail risk'

        return {
            'classification': classification,
            'lyapunov': lyap,
            'risk_implication': risk_implication
        }


class RecurrencePlotAnalyzer:
    """
    Recurrence plot analysis for phase space dynamics.

    Recurrence plots reveal hidden patterns and regime changes
    in complex time series.
    """

    def __init__(self, data: np.ndarray, embedding_dim: int = 3, tau: int = 1):
        self.data = np.asarray(data)
        self.m = embedding_dim
        self.tau = tau

        self._embed()

    def _embed(self):
        """Create delay embedding."""
        n = len(self.data)
        m = self.m
        tau = self.tau

        n_vectors = max(0, n - (m - 1) * tau)
        self.embedded = np.zeros((n_vectors, m))

        for i in range(n_vectors):
            for j in range(m):
                self.embedded[i, j] = self.data[i + j * tau]

    def compute_recurrence_matrix(self, epsilon: Optional[float] = None) -> np.ndarray:
        """
        Compute recurrence matrix.

        R[i,j] = 1 if ||x_i - x_j|| < epsilon, else 0
        """
        from scipy.spatial.distance import pdist, squareform

        # Distance matrix
        distances = squareform(pdist(self.embedded))

        if epsilon is None:
            epsilon = np.percentile(distances, 10)

        return (distances < epsilon).astype(int)

    def recurrence_quantification(self, epsilon: Optional[float] = None,
                                  min_line: int = 3) -> Dict:
        """
        Compute Recurrence Quantification Analysis (RQA) metrics.

        These metrics reveal system dynamics:
        - RR: Recurrence rate (density of recurrence points)
        - DET: Determinism (predictability)
        - LAM: Laminarity (intermittency)

        Args:
            epsilon: Recurrence threshold (default: 10th percentile of all
                pairwise distances, see ``compute_recurrence_matrix``).
            min_line: Minimum diagonal / vertical line length counted by DET and
                LAM (default 3, i.e. "length > 2").

        DET = points on diagonal lines of length >= min_line / recurrence points,
        LAM = points on vertical lines of length >= min_line / recurrence points,
        both without the line of identity (main diagonal), so 0 <= DET, LAM <= 1.
        """
        R = self.compute_recurrence_matrix(epsilon)
        return self._rqa(R, min_line)

    @staticmethod
    def _rqa(R: np.ndarray, min_line: int = 3) -> Dict:
        """RQA metrics of a recurrence matrix (see recurrence_quantification)."""
        R = np.asarray(R).astype(bool)
        n = len(R)

        # Recurrence rate
        RR = np.sum(R) / (n * n) if n else 0.0

        # Line of identity (main diagonal) is excluded from DET and LAM
        R_off = R.copy()
        np.fill_diagonal(R_off, False)
        total_recurrence = int(R_off.sum())

        # Determinism: fraction of recurrence points on diagonal lines
        diagonals = [np.diagonal(R_off, k) for k in range(-(n - 1), n) if k != 0]
        diag_points = _points_on_lines(diagonals, min_line)

        # Laminarity: fraction of recurrence points on vertical lines
        vert_points = _points_on_lines(list(R_off.T), min_line)

        DET = diag_points / total_recurrence if total_recurrence > 0 else 0
        LAM = vert_points / total_recurrence if total_recurrence > 0 else 0

        return {
            'recurrence_rate': RR,
            'determinism': DET,
            'laminarity': LAM,
            'n_points': n,
            'stability_index': DET / (RR + 1e-10)  # Higher = more predictable
        }

    def plot_recurrence(self, figsize: Tuple[int, int] = (10, 10),
                        epsilon: Optional[float] = None) -> plt.Figure:
        """Create recurrence plot."""
        R = self.compute_recurrence_matrix(epsilon)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(R, cmap='binary', origin='lower')
        ax.set_xlabel('Time i', fontsize=12)
        ax.set_ylabel('Time j', fontsize=12)
        ax.set_title('Recurrence Plot', fontsize=14, fontweight='bold')

        # Add RQA metrics as text
        metrics = self._rqa(R)
        text = f"RR: {metrics['recurrence_rate']:.3f}\nDET: {metrics['determinism']:.3f}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        return fig
