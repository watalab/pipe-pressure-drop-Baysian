# -*- coding: utf-8 -*-
"""
2D非圧縮性 SMAC 法ソルバーによるパイプ形状最適化
======================================================

条件（デフォルト）
------------------
- 直管、全長 L = 1.0 m（水平配置 → 重力による圧力勾配は考慮不要）
- 直径 D0 = 0.05 m
- 入口流速 UIN = 1.0 m/s（一様流入、出口は大気圧）
- くびれ：x = 0.3 m で開始、x = 0.5 m で直径が半分（D0/2）、x = 0.7 m で
  元の直径に戻る。X1, X2, X3 の座標とそこでの直径（D0, Dmin, D0）は固定。
- 流体：軽水（常温の水, H2O）  ρ = 998 kg/m3, μ = 1.002e-3 Pa.s
- 目的関数：入口平均圧力 − 出口平均圧力（= 圧力損失 ΔP）を最小化する
- X1〜X2, X2〜X3 の「区間内部」の形状を設計変数として最適化する

形状の表現方法
--------------
- X1→X2, X2→X3 の各区間内に、内部制御点を n_left / n_right 個ずつ配置
  - 各点のx座標は、区間を n 個の等幅ビンに分けてビンごとに1点ずつランダムに
    選ぶ「層化ランダムサンプリング」で決める。ビンの端に寄りすぎて隣の点と
    ほぼ重なることがないよう、各ビンの中央80%の範囲から選ぶ
  - 1段階（1つの点数）の最適化中はx座標を固定し、y座標（直径）だけを
    設計変数として動かす（x座標まで同時に動かすとBOの探索空間が
    非定常になり収束しにくくなるため）
- 各制御点の直径（y座標）に加えて、区間ごとに「滑らかさ」パラメータ
  smooth_left / smooth_right（0〜1）も設計変数にする
    - 区間ごとに PCHIP（形状保存の3次エルミート補間）で結ぶ。
      曲線は各区間内で隣り合う2点の値の間に収まるため、制御点が
      [Dmin, D0] の範囲内なら、曲線も必ずその範囲内に収まる
      （D0 を超える／Dmin を下回る部分は原理的に出ない）
    - smooth は、内部制御点の直径を「隣り合う点を結ぶ弦」の方向へ均す強さ
      （区間端 X1,X2,X3 は動かさない）。0 なら指定した点をそのまま通り、
      1 に近いほど凹凸が均されて滑らかな（直線に近い）曲線になる
    - 曲線は「均した後の点」を通る（図の赤丸は曲線が実際に通る点）

計算方式（ソルバー）
--------------------
- 既定は境界適合格子（座標変換 ξ=x, η=y/h(x), h=D/2）による2D非圧縮SMAC法
  （cfg.solver="bodyfit"）。壁が格子線(η=±1)の上に乗るので、形状を変えても
  ΔPが連続に変化する（階段近似のような、セルの反転による飛びが出ない）。
  圧力補正の行列は 離散発散D × 離散勾配G として組むため、格子が歪んでいても
  補正後の速度は厳密に発散ゼロ（検証で 1e-13 程度）。
- ΔP は「流れが管を通り抜ける回数（L/U を1回）」で測った時間 cfg.n_flow_through
  （既定6回）だけ時間発展させ、最後の cfg.avg_flow_through（既定2回）の時間平均場から
  求める。直線形状では約3〜4回で定常に落ち着くことを確認している。
  ※ 以前の版は「1ステップの圧力変化が小さい」ことを収束とみなして、立ち上げ直後
    （0.2〜0.4回分）で打ち切っていたため、ΔPは定常値ではなく途中経過だった。
- cfg.solver="cartesian" は従来の階段近似版（比較用）。上記の早期打ち切りの
  問題を残しているので、最適化には使わないこと。

段階的リファインメント（点数を増やしていく手順）
--------------------------------------------------
1. n_left=n_right=0（直線）から開始
2. 1段階ごとに n_left, n_right を1ずつ増やし、前段階の最良形状を
   新しい点数・新しいx位置の表現に変換して初期値（warm start）として引き継ぐ
3. 各段階でベイズ最適化（GP + Expected Improvement）を実行
4. 「これまでの最良（running best）」より悪化した結果は絶対に
   採用しない（前段階の結果をそのまま維持する）
5. 改善率が閾値を patience 回連続で下回ったら打ち切る
   （固定の点数上限は設けず、安全のための上限のみ設定）

注意（重要な限界）
------------------
実際の Re = ρ U D / μ = 998×1.0×0.05/1.002e-3 ≈ 5×10^4 であり、配管の一般的な
臨界Re（約2300）をはるかに超えています。本コードは乱流モデルなしの層流
Navier-Stokes をそのまま解く簡易ソルバーで、移流は1次風上差分（数値粘性が大きい）、
粘性項は主要項のみです。実際の乱流の圧力損失とは一致しない可能性があり、
工学的に検証された計算ではない点にご注意ください（形状どうしの相対比較の道具
としての利用を想定しています）。
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json, time, warnings, os

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")


# ============================================================
# 配管条件・物性値 ： ここを変更すれば全条件を再設定できる
# ============================================================
@dataclass
class PipeConfig:
    L: float = 1.0          # 全長 [m]
    D0: float = 0.05        # くびれ前後の直径 [m]
    UIN: float = 0.1        # 入口流速 [m/s]（一様流入）
    X1: float = 0.3         # くびれ開始位置 [m]
    X2: float = 0.5         # くびれ最小径位置 [m]
    X3: float = 0.7         # くびれ終了位置 [m]
    RHO: float = 998.0      # 密度 [kg/m3]（軽水）
    MU: float = 1.002e-3    # 粘性係数 [Pa.s]（軽水）
    P_ATM: float = 0.0      # 出口ゲージ圧 [Pa]（大気圧基準）
    solver: str = "bodyfit"   # "bodyfit"=境界適合格子(座標変換)  "cartesian"=従来の階段近似
    eta_stretch: float = 0.0  # 壁近傍への格子の寄せ具合(bodyfit用, 0=等間隔, 1〜2程度で壁側が細かくなる)
    n_flow_through: float = 10.0   # bodyfit: 流れが管を通り抜ける回数(L/U を1回とする)だけ時間発展させる
    avg_flow_through: float = 2.0  # bodyfit: 最後のこの回数分で時間平均した場(圧力・速度)を結果とする

    @property
    def DMIN(self):
        return self.D0 / 2.0

    @property
    def NU(self):
        return self.MU / self.RHO

    def __post_init__(self):
        if not (0.0 < self.X1 < self.X2 < self.X3 < self.L):
            raise ValueError(
                f"くびれ位置が全長と矛盾しています: 0 < X1({self.X1}) < X2({self.X2}) "
                f"< X3({self.X3}) < L({self.L}) を満たす必要があります。"
            )
        if not (self.D0 > 0 and self.UIN > 0):
            raise ValueError("D0 と UIN は正の値にしてください。")
        if self.solver not in ("bodyfit", "cartesian"):
            raise ValueError("solver は 'bodyfit' か 'cartesian' を指定してください。")
        if self.eta_stretch < 0:
            raise ValueError("eta_stretch は 0 以上にしてください。")
        if not (self.n_flow_through > self.avg_flow_through > 0):
            raise ValueError("n_flow_through > avg_flow_through > 0 としてください。")


DEFAULT_CFG = PipeConfig()


# ============================================================
# 直径分布 D(x)：区間ごとに PCHIP（形状保存補間）でつなぐ
# ============================================================
def _smooth_nodes(xs, ys, strength, n_pass=3):
    """
    区間端（最初と最後のノード）を固定したまま、内部ノードを「両隣のノードを
    結ぶ弦（直線）上の値」の方向へ引き寄せて均す（ラプラシアン平滑化）。
    strength=0 でそのまま、1 で1回ごとに弦の値へ置き換える。n_pass 回繰り返す。
    各ノードの新しい値は、元の値と隣のノード値の凸結合なので、
    ノード値の最大・最小の範囲（＝[Dmin, D0]）を超えることはない。
    """
    ys = np.array(ys, dtype=float)
    if len(ys) < 3 or strength <= 0.0:
        return ys
    h = np.diff(xs)
    hl, hr = h[:-1], h[1:]
    for _ in range(n_pass):
        chord = (hr * ys[:-2] + hl * ys[2:]) / (hl + hr)
        new = ys.copy()
        new[1:-1] = (1.0 - strength) * ys[1:-1] + strength * chord
        ys = new
    return ys


def _segment_profile(x_end0, y_end0, x_end1, y_end1, x_interior, y_interior, smooth_frac):
    """
    1区間分の D(x) を返す。x_interior/y_interior が空なら直線。
    返り値の関数には、曲線が実際に通るノード(nodes_x, nodes_y)を属性として付ける。
    """
    xs = np.array([x_end0] + list(x_interior) + [x_end1], dtype=float)
    ys = np.array([y_end0] + list(y_interior) + [y_end1], dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    n = len(xs)

    strength = float(np.clip(smooth_frac, 0.0, 1.0))
    ys_s = _smooth_nodes(xs, ys, strength)

    if n == 2 or np.any(np.diff(xs) <= 1e-12):
        # 内部点なし（直線）、またはx位置が重複している異常ケースは線形補間
        def D(x):
            return np.interp(x, xs, ys_s)
    else:
        pchip = PchipInterpolator(xs, ys_s)

        def D(x):
            return pchip(x)

    D.nodes_x = xs
    D.nodes_y = ys_s
    return D


def make_profile(cfg: PipeConfig, x_left, y_left, x_right, y_right,
                  smooth_left=0.0, smooth_right=0.0):
    """
    x_left, y_left   : X1->X2 区間の内部制御点のx座標・直径
    x_right, y_right : X2->X3 区間の内部制御点のx座標・直径
    smooth_left, smooth_right : 各区間の滑らかさ（0〜1）
    """
    x_left = np.asarray(x_left, dtype=float).ravel()
    y_left = np.asarray(y_left, dtype=float).ravel()
    x_right = np.asarray(x_right, dtype=float).ravel()
    y_right = np.asarray(y_right, dtype=float).ravel()

    seg1 = _segment_profile(cfg.X1, cfg.D0, cfg.X2, cfg.DMIN, x_left, y_left, smooth_left)
    seg2 = _segment_profile(cfg.X2, cfg.DMIN, cfg.X3, cfg.D0, x_right, y_right, smooth_right)

    # 曲線が実際に通るノード（平滑化後）。図の赤丸はこれ。
    control_x = np.concatenate([[0.0], seg1.nodes_x, seg2.nodes_x[1:], [cfg.L]])
    control_y = np.concatenate([[cfg.D0], seg1.nodes_y, seg2.nodes_y[1:], [cfg.D0]])

    def D(x):
        x = np.asarray(x, dtype=float)
        d = np.full_like(x, cfg.D0)
        m1 = (x >= cfg.X1) & (x <= cfg.X2)
        m2 = (x > cfg.X2) & (x <= cfg.X3)
        d[m1] = seg1(x[m1])
        d[m2] = seg2(x[m2])
        # PCHIP＋凸結合の平滑化なので原理的にはみ出さないが、浮動小数点の
        # 丸め誤差に対する安全装置として [Dmin, D0] にクリップしておく
        d = np.clip(d, cfg.DMIN, cfg.D0)
        return d

    D.control_x = control_x
    D.control_y = control_y
    return D


def generate_stage_x_positions(cfg: PipeConfig, n_left, n_right, seed=0):
    """
    各区間を n 個の等幅ビンに分け、ビンごとに1点ずつランダムに選ぶ
    （層化ランダムサンプリング）。1段階につき1回だけ生成し、その段階の
    最適化中は固定する（seed固定で再現性あり）。
    ビンの端に寄って隣の点とほぼ重なる（急な傾きの原因になる）のを避けるため、
    各ビンの中央80%の範囲から選ぶ（隣り合う点の間隔は最低でもビン幅の0.2倍）。
    """
    rng = np.random.default_rng(seed)

    def _one_segment(x0, x1, n):
        if n == 0:
            return np.array([])
        edges = np.linspace(x0, x1, n + 1)
        xs = np.array([rng.uniform(edges[i] + 0.1 * (edges[i + 1] - edges[i]),
                                    edges[i] + 0.9 * (edges[i + 1] - edges[i]))
                       for i in range(n)])
        return np.sort(xs)

    return _one_segment(cfg.X1, cfg.X2, n_left), _one_segment(cfg.X2, cfg.X3, n_right)


def straight_line_y(cfg: PipeConfig, x_left, x_right):
    """与えられたx位置の上に、直線テーパ上のy値（初期値 = 直線）を返す。"""
    y_left = np.interp(x_left, [cfg.X1, cfg.X2], [cfg.D0, cfg.DMIN])
    y_right = np.interp(x_right, [cfg.X2, cfg.X3], [cfg.DMIN, cfg.D0])
    return y_left, y_right


def param_bounds(cfg: PipeConfig, n_left, n_right):
    """設計変数の並び： [y_left(n_left), y_right(n_right), smooth_left?, smooth_right?]"""
    lo, hi = cfg.DMIN, cfg.D0
    bounds = [(lo, hi)] * (n_left + n_right)
    if n_left > 0:
        bounds.append((0.0, 1.0))
    if n_right > 0:
        bounds.append((0.0, 1.0))
    return bounds


def unpack_params(x, n_left, n_right):
    x = np.asarray(x, dtype=float).ravel()
    y_left = x[:n_left]
    y_right = x[n_left:n_left + n_right]
    i = n_left + n_right
    smooth_left = float(x[i]) if n_left > 0 else 0.0
    i += 1 if n_left > 0 else 0
    smooth_right = float(x[i]) if n_right > 0 else 0.0
    return y_left, y_right, smooth_left, smooth_right


def pack_params(y_left, y_right, smooth_left, smooth_right):
    vals = list(y_left) + list(y_right)
    if len(y_left) > 0:
        vals.append(smooth_left)
    if len(y_right) > 0:
        vals.append(smooth_right)
    return np.array(vals, dtype=float)


# ============================================================
# 格子生成とマスク（免震境界：くびれ壁の表現）
# ============================================================
def build_grid_cartesian(cfg: PipeConfig, Nx, Ny, Dfunc):
    dx = cfg.L / Nx
    dy = cfg.D0 / Ny
    xc = (np.arange(Nx) + 0.5) * dx
    yc = (np.arange(Ny) + 0.5) * dy - cfg.D0 / 2.0
    xu = np.arange(Nx + 1) * dx
    yv = np.arange(Ny + 1) * dy - cfg.D0 / 2.0

    Dx_c = Dfunc(xc)
    p_mask = np.abs(yc)[None, :] <= (Dx_c / 2.0)[:, None]

    u_mask = np.zeros((Nx + 1, Ny), dtype=bool)
    u_mask[0, :] = p_mask[0, :]
    u_mask[Nx, :] = p_mask[Nx - 1, :]
    u_mask[1:Nx, :] = p_mask[:-1, :] & p_mask[1:, :]

    v_mask = np.zeros((Nx, Ny + 1), dtype=bool)
    v_mask[:, 0] = False
    v_mask[:, Ny] = False
    v_mask[:, 1:Ny] = p_mask[:, :-1] & p_mask[:, 1:]

    return dict(type="cartesian", dx=dx, dy=dy, xc=xc, yc=yc, xu=xu, yv=yv,
                p_mask=p_mask, u_mask=u_mask, v_mask=v_mask, Nx=Nx, Ny=Ny)


# ============================================================
# 圧力 Poisson 行列（壁：ノイマン(流束0), 出口：ディリクレ phi=0）
# ============================================================
def build_poisson(p_mask, dx, dy):
    Nx, Ny = p_mask.shape
    idx_map = -np.ones((Nx, Ny), dtype=int)
    fluid_idx = []
    k = 0
    for i in range(Nx):
        for j in range(Ny):
            if p_mask[i, j]:
                idx_map[i, j] = k
                fluid_idx.append((i, j))
                k += 1
    N = k
    A = sp.lil_matrix((N, N))
    ax = 1.0 / dx ** 2
    ay = 1.0 / dy ** 2
    for (i, j) in fluid_idx:
        k = idx_map[i, j]
        diag = 0.0
        if i + 1 < Nx and p_mask[i + 1, j]:
            A[k, idx_map[i + 1, j]] += ax
            diag -= ax
        elif i + 1 == Nx:
            diag -= ax
        if i - 1 >= 0 and p_mask[i - 1, j]:
            A[k, idx_map[i - 1, j]] += ax
            diag -= ax
        if j + 1 < Ny and p_mask[i, j + 1]:
            A[k, idx_map[i, j + 1]] += ay
            diag -= ay
        if j - 1 >= 0 and p_mask[i, j - 1]:
            A[k, idx_map[i, j - 1]] += ay
            diag -= ay
        A[k, k] += diag
    return A.tocsr(), fluid_idx, idx_map


# ============================================================
# 移流拡散予測子ステップ（陽解法・風上差分＋中心差分拡散、圧力勾配込み）
# ============================================================
def predictor(u, v, p, dx, dy, dt, nu, rho, both_fluid_x, both_fluid_y):
    Nxp1, Ny = u.shape
    Nx, Nyp1 = v.shape

    dpdx = np.zeros_like(u)
    dpdx[1:Nx, :] = np.where(both_fluid_x, (p[1:, :] - p[:-1, :]) / dx, 0.0)
    dpdy = np.zeros_like(v)
    dpdy[:, 1:Ny] = np.where(both_fluid_y, (p[:, 1:] - p[:, :-1]) / dy, 0.0)

    u_pad_y = np.pad(u, ((0, 0), (1, 1)), mode='constant', constant_values=0.0)
    uc = u[1:-1, :]
    ue = u[2:, :]
    uw = u[:-2, :]
    un = u_pad_y[1:-1, 2:]
    us = u_pad_y[1:-1, :-2]

    dudx_b = (uc - uw) / dx
    dudx_f = (ue - uc) / dx
    dudx = np.where(uc >= 0, dudx_b, dudx_f)

    v_l = v[0:-1, :]
    v_r = v[1:, :]
    v_avg = 0.25 * (v_l[:, :-1] + v_l[:, 1:] + v_r[:, :-1] + v_r[:, 1:])

    dudy_b = (uc - us) / dy
    dudy_f = (un - uc) / dy
    dudy = np.where(v_avg >= 0, dudy_b, dudy_f)

    lap_u = (ue - 2 * uc + uw) / dx ** 2 + (un - 2 * uc + us) / dy ** 2

    u_star = u.copy()
    u_star[1:-1, :] = uc + dt * (-uc * dudx - v_avg * dudy + nu * lap_u - dpdx[1:-1, :] / rho)

    v_pad_x = np.pad(v, ((1, 1), (0, 0)), mode='edge')
    vc = v[:, 1:-1]
    vn = v[:, 2:]
    vs = v[:, :-2]
    ve = v_pad_x[2:, 1:-1]
    vw = v_pad_x[:-2, 1:-1]

    dvdy_b = (vc - vs) / dy
    dvdy_f = (vn - vc) / dy
    dvdy = np.where(vc >= 0, dvdy_b, dvdy_f)

    u_l = u[0:Nx, :]
    u_r = u[1:Nx + 1, :]
    u_avg = 0.25 * (u_l[:, 0:Ny - 1] + u_l[:, 1:Ny] + u_r[:, 0:Ny - 1] + u_r[:, 1:Ny])

    dvdx_b = (vc - vw) / dx
    dvdx_f = (ve - vc) / dx
    dvdx = np.where(u_avg >= 0, dvdx_b, dvdx_f)

    lap_v = (ve - 2 * vc + vw) / dx ** 2 + (vn - 2 * vc + vs) / dy ** 2

    v_star = v.copy()
    v_star[:, 1:-1] = vc + dt * (-u_avg * dvdx - vc * dvdy + nu * lap_v - dpdy[:, 1:-1] / rho)

    return u_star, v_star


# ============================================================
# SMAC 法メインソルバー（疑似時間発展で定常解に収束させる）
# ============================================================
def solve_flow_cartesian(cfg: PipeConfig, grid, n_steps=3000, cfl=0.2):
    dx, dy = grid['dx'], grid['dy']
    Nx, Ny = grid['Nx'], grid['Ny']
    u_mask, v_mask, p_mask = grid['u_mask'], grid['v_mask'], grid['p_mask']

    u = np.zeros((Nx + 1, Ny))
    v = np.zeros((Nx, Ny + 1))
    p = np.zeros((Nx, Ny))

    dt = cfl * min(dx, dy) / cfg.UIN

    A, fluid_idx, idx_map = build_poisson(p_mask, dx, dy)
    lu = spla.splu(A.tocsc())

    fluid_idx_arr = np.array(fluid_idx)
    ii = fluid_idx_arr[:, 0]
    jj = fluid_idx_arr[:, 1]

    both_fluid_x = p_mask[:-1, :] & p_mask[1:, :]
    both_fluid_y = p_mask[:, :-1] & p_mask[:, 1:]

    n_ramp = max(20, n_steps // 10)

    for step in range(n_steps):
        uin_now = cfg.UIN * min(1.0, (step + 1) / n_ramp)

        u[0, :] = uin_now
        u[0, :][~u_mask[0, :]] = 0.0
        u[~u_mask] = 0.0
        v[~v_mask] = 0.0
        u[Nx, :] = u[Nx - 1, :]

        u_star, v_star = predictor(u, v, p, dx, dy, dt, cfg.NU, cfg.RHO, both_fluid_x, both_fluid_y)

        u_star[0, :] = uin_now
        u_star[0, :][~u_mask[0, :]] = 0.0
        u_star[~u_mask] = 0.0
        v_star[~v_mask] = 0.0
        u_star[Nx, :] = u_star[Nx - 1, :]

        div = (u_star[1:, :] - u_star[:-1, :]) / dx + (v_star[:, 1:] - v_star[:, :-1]) / dy

        rhs = (cfg.RHO / dt) * div[ii, jj]
        phi_vec = lu.solve(rhs)
        phi = np.zeros((Nx, Ny))
        phi[ii, jj] = phi_vec

        dphidx = np.zeros_like(u)
        dphidx[1:Nx, :] = np.where(both_fluid_x, (phi[1:, :] - phi[:-1, :]) / dx, 0.0)
        dphidy = np.zeros_like(v)
        dphidy[:, 1:Ny] = np.where(both_fluid_y, (phi[:, 1:] - phi[:, :-1]) / dy, 0.0)

        u = u_star - (dt / cfg.RHO) * dphidx
        v = v_star - (dt / cfg.RHO) * dphidy
        p = p + phi

        u[~u_mask] = 0.0
        v[~v_mask] = 0.0

        if not np.isfinite(u).all() or not np.isfinite(p).all():
            raise FloatingPointError(f"SMAC solver diverged at step {step}")

        if step > n_ramp and np.abs(phi[p_mask]).max() < 1e-3 * max(1.0, np.abs(p[p_mask]).max()):
            break

    return u, v, p


# ============================================================
# 境界適合格子（座標変換）版 SMAC ソルバー
# ------------------------------------------------------------
#   座標変換  ξ = x ,  η = y / h(x) ,  h(x) = D(x)/2
#   → 壁は η = ±1 の「格子線の上」に厳密に乗る（階段状にならない）。
#   → 形状を変えると変換の係数（h, h'）が連続的に変わるだけなので、
#      圧力損失は形状に対して連続に変化する（カットセル/階段近似のような
#      セルの反転による飛びが出ない）。
#
#   基礎式（物理座標の速度成分 u,v をそのまま使い、微分だけ変換する）
#     ∂/∂x|y = ∂/∂ξ − (η h'/h) ∂/∂η ,   ∂/∂y = (1/h) ∂/∂η
#     対流:  u ∂u/∂ξ + (V/h) ∂u/∂η ,   V = v − η h' u  （η方向の流束速度）
#     質量保存（保存形）:  (1/h)[ ∂(h u)/∂ξ + ∂V/∂η ] = 0
#   スタガード配置: u は ξ面中心、v は η面中心、p はセル中心。
#   壁(η=±1)では u=v=0（滑りなし）、壁を通る流束は 0。
#   圧力補正の Poisson 行列は A = D・G（D:離散発散, G:離散勾配）として
#   組み立てるので、補正後の速度は格子が歪んでいても厳密に発散ゼロになる。
#   粘性項は主要項 u_ξξ + (η_x² + 1/h²) u_ηη のみを使う（交差微分項と
#   h'' を含む低次項は省略。壁の傾きが小さいくびれ管では寄与が小さい）。
# ============================================================
def build_grid_bodyfit(cfg: PipeConfig, Nx, Ny, Dfunc, stretch=None):
    if stretch is None:
        stretch = cfg.eta_stretch
    dx = cfg.L / Nx
    xf = np.arange(Nx + 1) * dx                 # ξ面（= x）
    xc = (np.arange(Nx) + 0.5) * dx             # セル中心
    hf = np.asarray(Dfunc(xf), dtype=float) / 2.0   # 面での半径 h
    hc = np.asarray(Dfunc(xc), dtype=float) / 2.0   # セル中心での半径 h

    # η方向の格子（-1〜1）。stretch>0 で壁(η=±1)側が細かくなる
    sf = np.linspace(-1.0, 1.0, Ny + 1)
    sc = 0.5 * (sf[:-1] + sf[1:])
    if stretch > 1e-12:
        etaf = np.tanh(stretch * sf) / np.tanh(stretch)
        etac = np.tanh(stretch * sc) / np.tanh(stretch)
    else:
        etaf, etac = sf, sc
    deta = np.diff(etaf)          # η面の間隔（セル幅） (Ny,)
    dec = np.diff(etac)           # セル中心間の距離 (Ny-1,)

    # 壁形状の傾き h'（差分）。η面の流束は幾何学的保存則を満たすよう面の差分 Δh を使う
    dh = hf[1:] - hf[:-1]                       # (Nx,)
    hp_c = dh / dx                              # セル中心での h'
    hp_f = np.empty(Nx + 1)                     # ξ面での h'
    hp_f[1:-1] = (hc[1:] - hc[:-1]) / dx
    hp_f[0] = (hc[0] - hf[0]) / (0.5 * dx)
    hp_f[-1] = (hf[-1] - hc[-1]) / (0.5 * dx)

    # 補間重み：v をξ面中心へ(wv)、u をη面へ(wu)
    wv = (etac - etaf[:-1]) / deta              # (Ny,)
    wu = (etaf[1:-1] - etac[:-1]) / dec         # (Ny-1,)

    Ix = sp.identity(Nx, format="csr")
    Iy = sp.identity(Ny, format="csr")

    # ---- セル中心でのη微分 (Ny x Ny) ----
    Deta = sp.lil_matrix((Ny, Ny))
    for j in range(Ny):
        if 0 < j < Ny - 1:
            d = etac[j + 1] - etac[j - 1]
            Deta[j, j + 1], Deta[j, j - 1] = 1.0 / d, -1.0 / d
        elif j == 0:
            d = etac[1] - etac[0]
            Deta[0, 1], Deta[0, 0] = 1.0 / d, -1.0 / d
        else:
            d = etac[-1] - etac[-2]
            Deta[j, j], Deta[j, j - 1] = 1.0 / d, -1.0 / d
    Deta = Deta.tocsr()

    # ---- 勾配演算子 G ：セル圧力 → 面での物理圧力勾配 ----
    #  u面(i=1..Nx-1):  p_x = p_ξ − (η h'/h) p_η
    Mdiff = sp.diags([-np.ones(Nx - 1), np.ones(Nx - 1)], [0, 1], shape=(Nx - 1, Nx))
    Mavg = 0.5 * sp.diags([np.ones(Nx - 1), np.ones(Nx - 1)], [0, 1], shape=(Nx - 1, Nx))
    kappa = -(etac[None, :] * hp_f[1:Nx, None] / hf[1:Nx, None])       # (Nx-1, Ny)
    G_u_int = (sp.kron(Mdiff / dx, Iy)
               + sp.diags(kappa.ravel()) @ sp.kron(Mavg, Iy) @ sp.kron(Ix, Deta))
    #  出口面(i=Nx): 出口の圧力 p=0（ディリクレ）。面とセル中心の距離は dx/2
    G_out = sp.csr_matrix((-2.0 / dx * np.ones(Ny),
                           (np.arange(Ny), (Nx - 1) * Ny + np.arange(Ny))),
                          shape=(Ny, Nx * Ny))
    #  v面(j=1..Ny-1):  p_y = p_η / h
    Mv = sp.lil_matrix((Ny - 1, Ny))
    for jp in range(Ny - 1):
        Mv[jp, jp], Mv[jp, jp + 1] = -1.0 / dec[jp], 1.0 / dec[jp]
    G_v = sp.diags(np.repeat(1.0 / hc, Ny - 1)) @ sp.kron(Ix, Mv.tocsr())
    G = sp.vstack([G_u_int, G_out, G_v]).tocsr()

    # ---- 発散演算子 D ：面の速度 → セルの正味流出流束（保存形） ----
    #  ξ面の流束 = h·Δη·u ,  η面の流束 = v·Δξ − η·Δh·ū
    Mxi = sp.diags([-np.ones(Nx), np.ones(Nx)], [0, 1], shape=(Nx, Nx + 1))
    Wu = (hf[:, None] * deta[None, :]).ravel()                       # ((Nx+1)*Ny,)
    Dxi = sp.kron(Mxi, Iy) @ sp.diags(Wu)
    Mxavg = 0.5 * sp.diags([np.ones(Nx), np.ones(Nx)], [0, 1], shape=(Nx, Nx + 1))
    Meta = sp.lil_matrix((Ny - 1, Ny))
    for jp in range(Ny - 1):
        Meta[jp, jp], Meta[jp, jp + 1] = 1.0 - wu[jp], wu[jp]
    Au = sp.kron(Mxavg, Meta.tocsr())                                # u → η面
    coef = (-etaf[None, 1:Ny] * dh[:, None]).ravel()                 # (Nx*(Ny-1),)
    Mdiv = sp.lil_matrix((Ny, Ny - 1))
    for j in range(Ny):
        if j <= Ny - 2:
            Mdiv[j, j] = 1.0
        if j >= 1:
            Mdiv[j, j - 1] = -1.0
    Ddiv = sp.kron(Ix, Mdiv.tocsr())                                 # η面流束 → セル
    D_uall = (Dxi + Ddiv @ sp.diags(coef) @ Au).tocsr()
    D_v = (dx * Ddiv).tocsr()
    D_in = D_uall[:, :Ny]                                            # 入口面(既知)の寄与
    D_unk = sp.hstack([D_uall[:, Ny:], D_v]).tocsr()                 # 未知の面（u: i=1..Nx, v）

    area = 0.5 * (hf[:-1] + hf[1:])[:, None] * dx * deta[None, :]    # セル面積 (Nx, Ny)

    Xn = np.repeat(xf[:, None], Ny + 1, axis=1)                      # 描画用の節点座標
    Yn = hf[:, None] * etaf[None, :]

    return dict(type="bodyfit", Nx=Nx, Ny=Ny, dx=dx, xf=xf, xc=xc, hf=hf, hc=hc,
                etaf=etaf, etac=etac, deta=deta, dec=dec, hp_f=hp_f, hp_c=hp_c, dh=dh,
                wv=wv, wu=wu, G=G, D_unk=D_unk, D_in=D_in, area=area,
                Xn=Xn, Yn=Yn, p_mask=np.ones((Nx, Ny), dtype=bool))


def solve_flow_bodyfit(cfg: PipeConfig, grid, n_steps=None, cfl=0.4, history=None, record_every=None):
    """
    n_steps は互換性のため残してあるが使わない。時間発展の長さは
    cfg.n_flow_through（流れが管を通り抜ける回数）で決め、最後の
    cfg.avg_flow_through 回分で時間平均した (u, v, p) を返す。
    （1ステップごとの圧力変化が小さいことを「収束」とみなす判定は、dtが小さいと
      途中経過でも満たされてしまうため使わない。ΔPは流れが十分発達してからの
      時間平均で評価する。）
    """
    Nx, Ny, dx = grid["Nx"], grid["Ny"], grid["dx"]
    hf, hc = grid["hf"], grid["hc"]
    etaf, etac, deta = grid["etaf"], grid["etac"], grid["deta"]
    hp_f, hp_c, wv, wu = grid["hp_f"], grid["hp_c"], grid["wv"], grid["wu"]
    G, D_unk, D_in = grid["G"], grid["D_unk"], grid["D_in"]
    rho, nu = cfg.RHO, cfg.NU
    n_u = Nx * Ny

    # 圧力補正の Poisson 行列（A = D・G）を LU 分解（形状ごとに1回）
    lu = spla.splu((D_unk @ G).tocsc())

    u = np.zeros((Nx + 1, Ny))
    v = np.zeros((Nx, Ny + 1))
    p = np.zeros((Nx, Ny))

    # 時間刻み：移流(ξ,η)・粘性の安定条件
    dy_min = float(hc.min() * deta.min())
    slope = max(1.0, float(np.abs(hp_f).max()))
    dt = cfl * min(dx, dy_min) / (cfg.UIN * slope)
    dt = min(dt, 0.25 / (nu * (1.0 / dx ** 2 + 1.0 / dy_min ** 2)))

    eta_ext = np.concatenate([[etaf[0]], etac, [etaf[-1]]])
    hf_i = hf[1:Nx]
    hp_f_i = hp_f[1:Nx]
    cu = (etac[None, :] * hp_f_i[:, None] / hf_i[:, None]) ** 2 + 1.0 / hf_i[:, None] ** 2
    cv = (etaf[None, 1:Ny] * hp_c[:, None] / hc[:, None]) ** 2 + 1.0 / hc[:, None] ** 2

    tau = cfg.L / cfg.UIN                              # 流れが管を1回通り抜ける時間
    total_steps = int(np.ceil(cfg.n_flow_through * tau / dt))
    avg_start = total_steps - int(np.ceil(cfg.avg_flow_through * tau / dt))
    n_ramp = max(20, int(0.2 * tau / dt))              # 入口流速の立ち上げ（0.2τ）
    if record_every is None:
        record_every = max(1, int(tau / dt / 20))
    sum_u = np.zeros_like(u)
    sum_v = np.zeros_like(v)
    sum_p = np.zeros_like(p)
    n_avg = 0

    for step in range(total_steps):
        uin = cfg.UIN * min(1.0, (step + 1) / n_ramp)
        u[0, :] = uin

        gp = G @ p.ravel()
        gu = gp[:n_u].reshape(Nx, Ny)           # u面 i=1..Nx
        gv = gp[n_u:].reshape(Nx, Ny - 1)       # v面 j=1..Ny-1

        # ---------- u の予測子（内部のξ面 i=1..Nx-1） ----------
        uc, ue, uw = u[1:Nx, :], u[2:, :], u[:Nx - 1, :]
        dudxi = np.where(uc >= 0.0, (uc - uw) / dx, (ue - uc) / dx)
        vbar = 0.5 * ((1.0 - wv) * (v[:Nx - 1, :Ny] + v[1:Nx, :Ny])
                      + wv * (v[:Nx - 1, 1:] + v[1:Nx, 1:]))
        Vc = vbar - etac[None, :] * hp_f_i[:, None] * uc
        u_ext = np.zeros((Nx - 1, Ny + 2))
        u_ext[:, 1:-1] = uc                      # 壁(η=±1)では u=0
        d_b = (u_ext[:, 1:-1] - u_ext[:, :-2]) / (eta_ext[1:-1] - eta_ext[:-2])
        d_f = (u_ext[:, 2:] - u_ext[:, 1:-1]) / (eta_ext[2:] - eta_ext[1:-1])
        dudeta = np.where(Vc >= 0.0, d_b, d_f)
        conv_u = uc * dudxi + (Vc / hf_i[:, None]) * dudeta
        u_ee = 2.0 * (d_f - d_b) / (eta_ext[2:] - eta_ext[:-2])
        visc_u = nu * ((ue - 2.0 * uc + uw) / dx ** 2 + cu * u_ee)

        # ---------- v の予測子（内部のη面 j=1..Ny-1） ----------
        vc = v[:, 1:Ny]
        vpx = np.pad(vc, ((1, 1), (0, 0)), mode="edge")
        ve, vw = vpx[2:], vpx[:-2]
        ubar = 0.5 * ((1.0 - wu) * (u[:Nx, :Ny - 1] + u[1:, :Ny - 1])
                      + wu * (u[:Nx, 1:] + u[1:, 1:]))
        dvdxi = np.where(ubar >= 0.0, (vc - vw) / dx, (ve - vc) / dx)
        Vg = vc - etaf[None, 1:Ny] * hp_c[:, None] * ubar
        vm, vpl = v[:, 0:Ny - 1], v[:, 2:Ny + 1]
        e_b = (vc - vm) / deta[None, 0:Ny - 1]
        e_f = (vpl - vc) / deta[None, 1:Ny]
        dvdeta = np.where(Vg >= 0.0, e_b, e_f)
        conv_v = ubar * dvdxi + (Vg / hc[:, None]) * dvdeta
        v_ee = 2.0 * (e_f - e_b) / (deta[None, 0:Ny - 1] + deta[None, 1:Ny])
        visc_v = nu * ((ve - 2.0 * vc + vw) / dx ** 2 + cv * v_ee)

        u_star = u.copy()
        u_star[1:Nx, :] = uc + dt * (-conv_u + visc_u) - (dt / rho) * gu[:Nx - 1, :]
        u_star[Nx, :] = u_star[Nx - 1, :]        # 出口：ゼロ勾配で外挿（後で圧力補正を受ける）
        u_star[0, :] = uin
        v_star = v.copy()
        v_star[:, 1:Ny] = vc + dt * (-conv_v + visc_v) - (dt / rho) * gv
        v_star[:, 0] = 0.0
        v_star[:, Ny] = 0.0

        # ---------- 圧力補正：A φ = (ρ/dt)·div(u*) ----------
        unk = np.concatenate([u_star[1:, :].ravel(), v_star[:, 1:Ny].ravel()])
        rhs = (rho / dt) * (D_unk @ unk + D_in @ u_star[0, :])
        phi = lu.solve(rhs)
        gphi = G @ phi
        u[1:, :] = u_star[1:, :] - (dt / rho) * gphi[:n_u].reshape(Nx, Ny)
        v[:, 1:Ny] = v_star[:, 1:Ny] - (dt / rho) * gphi[n_u:].reshape(Nx, Ny - 1)
        u[0, :] = uin
        v[:, 0] = 0.0
        v[:, Ny] = 0.0
        p = p + phi.reshape(Nx, Ny)

        if history is not None and step % record_every == 0:
            wgt = grid["deta"]
            history.append((step, (step + 1) * dt,
                            float(np.sum(p[0, :] * wgt) / np.sum(wgt) - np.sum(p[-1, :] * wgt) / np.sum(wgt))))

        if not np.isfinite(u).all() or not np.isfinite(p).all():
            raise FloatingPointError(f"body-fitted SMAC solver diverged at step {step}")

        if step >= avg_start:
            sum_u += u
            sum_v += v
            sum_p += p
            n_avg += 1

    return sum_u / n_avg, sum_v / n_avg, sum_p / n_avg


def build_grid(cfg: PipeConfig, Nx, Ny, Dfunc):
    """cfg.solver に応じて格子を作る（'bodyfit' または 'cartesian'）。"""
    if cfg.solver == "bodyfit":
        return build_grid_bodyfit(cfg, Nx, Ny, Dfunc)
    return build_grid_cartesian(cfg, Nx, Ny, Dfunc)


def solve_flow(cfg: PipeConfig, grid, n_steps=3000, cfl=None, history=None, record_every=None):
    """cfl=None のときは方式ごとの既定値（bodyfit: 0.4, cartesian: 0.2）を使う。"""
    if grid.get("type") == "bodyfit":
        return solve_flow_bodyfit(cfg, grid, n_steps=n_steps, cfl=0.4 if cfl is None else cfl,
                                  history=history, record_every=record_every)
    return solve_flow_cartesian(cfg, grid, n_steps=n_steps, cfl=0.2 if cfl is None else cfl)


def pressure_loss(cfg: PipeConfig, p, grid):
    if grid.get("type") == "bodyfit":
        # 同じ断面(同じi)内ではhが共通なので、η方向のセル幅deta（=断面積に比例）で重み付け平均
        w = grid["deta"]
        p_in = float(np.sum(p[0, :] * w) / np.sum(w))
        p_out = float(np.sum(p[-1, :] * w) / np.sum(w))
        return p_in - p_out + cfg.P_ATM
    pm = grid['p_mask']
    p_in = p[0, pm[0, :]].mean()
    p_out = p[-1, pm[-1, :]].mean()
    return p_in - p_out + cfg.P_ATM


# ============================================================
# 目的関数
# ============================================================
def objective(x, cfg: PipeConfig, x_left, x_right, n_left, n_right,
              Nx=90, Ny=18, n_steps=3000):
    y_left, y_right, s_left, s_right = unpack_params(x, n_left, n_right)

    lo, hi = cfg.DMIN, cfg.D0
    penalty = 0.0
    for yv in np.concatenate([y_left, y_right]):
        if yv < lo:
            penalty += 1e4 * (lo - yv)
        if yv > hi:
            penalty += 1e4 * (yv - hi)
    y_left_c = np.clip(y_left, lo, hi)
    y_right_c = np.clip(y_right, lo, hi)

    Dfunc = make_profile(cfg, x_left, y_left_c, x_right, y_right_c, s_left, s_right)

    xs_check = np.linspace(cfg.X1, cfg.X3, 200)
    Ds_check = Dfunc(xs_check)
    if Ds_check.min() < cfg.DMIN * 0.995:
        penalty += 1e4 * (cfg.DMIN - Ds_check.min())
    if Ds_check.max() > cfg.D0 * 1.005:
        penalty += 1e4 * (Ds_check.max() - cfg.D0)

    grid = build_grid(cfg, Nx, Ny, Dfunc)
    # 万一ソルバーが発散したら、時間刻みを半分にして再試行する（最適化全体が止まらないように）。
    # それでもだめなら、評価不能な候補として大きな損失を返す。
    dp = None
    for cfl in (None, 0.2, 0.1):
        try:
            u, v, p = solve_flow(cfg, grid, n_steps=n_steps, cfl=cfl)
            dp = pressure_loss(cfg, p, grid)
            break
        except FloatingPointError:
            continue
    if dp is None:
        print("  [警告] ソルバーが発散したため、この候補は評価不能として扱います")
        return 1e4, 1e4
    return dp + penalty, dp  # (最適化用ロス, 罰則なしの純粋なdP)


# ============================================================
# ベイズ最適化（ガウス過程回帰 + Expected Improvement、次元数自由）
# ============================================================
def _normalize(X, lo, hi):
    return (X - lo) / (hi - lo)


def _denormalize(Xn, lo, hi):
    return Xn * (hi - lo) + lo


def _expected_improvement(Xn_cand, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(Xn_cand, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    imp = y_best - mu - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


def bayesian_optimize(obj_fun, bounds, n_init=8, n_iter=22, xi=0.01,
                       n_candidates=3000, seed=0, x_seed=None, y_seed=None):
    """
    次元数自由のベイズ最適化（最小化）。obj_fun(x)->(loss, dp) を返す想定。
    x_seed, y_seed(=[(loss,dp),...]) : 前段階の最良点を最初から観測データに
                                        含める（warm start）。
    """
    rng = np.random.default_rng()
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    dim = len(bounds)

    Xn_init = np.zeros((n_init, dim))
    for d in range(dim):
        Xn_init[:, d] = (rng.permutation(n_init) + rng.random(n_init)) / n_init
    X_init = _denormalize(Xn_init, lo, hi)

    X_obs, y_obs, extra_obs = [], [], []
    if x_seed is not None:
        for xs_, (ys_, ex_) in zip(x_seed, y_seed):
            X_obs.append(np.asarray(xs_, dtype=float))
            y_obs.append(float(ys_))
            extra_obs.append(ex_)
    for x in X_init:
        loss, dp = obj_fun(x)
        X_obs.append(x)
        y_obs.append(loss)
        extra_obs.append(dp)

    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(
        length_scale=[0.3] * dim, length_scale_bounds=(1e-2, 10.0), nu=2.5
    ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e2))

    for it in range(n_iter):
        X_arr = np.array(X_obs)
        y_arr = np.array(y_obs)
        Xn_arr = _normalize(X_arr, lo, hi)
        y_mean, y_std = y_arr.mean(), y_arr.std() + 1e-9
        y_norm = (y_arr - y_mean) / y_std

        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                       n_restarts_optimizer=2, random_state=seed + it)
        gp.fit(Xn_arr, y_norm)

        Xn_cand = rng.random((n_candidates, dim))
        ei = _expected_improvement(Xn_cand, gp, y_norm.min(), xi=xi)
        x_next = _denormalize(Xn_cand[np.argmax(ei)], lo, hi)

        loss, dp = obj_fun(x_next)
        X_obs.append(x_next)
        y_obs.append(loss)
        extra_obs.append(dp)

    X_obs = np.array(X_obs)
    y_obs = np.array(y_obs)
    extra_obs = np.array(extra_obs)
    best_i = int(np.argmin(y_obs))
    return dict(x_best=X_obs[best_i], y_best=y_obs[best_i], dp_best=extra_obs[best_i],
                X_obs=X_obs, y_obs=y_obs, dp_obs=extra_obs)


# ============================================================
# 1段階分の最適化（点数 n_left, n_right とx座標を固定して実行）
# ============================================================
def run_stage(cfg: PipeConfig, x_left, x_right, n_left, n_right, Nx, Ny, n_steps,
              n_init, n_iter, x_seed_raw=None, loss_dp_seed=None, seed=0):
    bounds = param_bounds(cfg, n_left, n_right)

    def obj_wrapper(x):
        return objective(x, cfg, x_left, x_right, n_left, n_right, Nx=Nx, Ny=Ny, n_steps=n_steps)

    x_seed_packed, y_seed_packed = None, None
    if x_seed_raw is not None:
        x_seed_packed = [x_seed_raw]
        y_seed_packed = [loss_dp_seed]

    res = bayesian_optimize(obj_wrapper, bounds, n_init=n_init, n_iter=n_iter,
                             seed=seed, x_seed=x_seed_packed, y_seed=y_seed_packed)
    return res


def upsample_shape(cfg, x_prev, x_left_prev, x_right_prev, n_left_prev, n_right_prev,
                    x_left_new, x_right_new):
    """前段階の最良形状（古いx位置での制御点）を、新しいx位置で評価し直す
    ことで、新しい表現に引き継ぐための初期値(y)を作る。
    サンプルする値はすでに平滑化済みの曲線上の値なので、引き継ぎ初期値の
    smooth は 0 にする（再度平滑化がかかって形が変わるのを防ぐ）。"""
    y_left_p, y_right_p, s_left_p, s_right_p = unpack_params(x_prev, n_left_prev, n_right_prev)
    Dfunc_prev = make_profile(cfg, x_left_prev, y_left_p, x_right_prev, y_right_p, s_left_p, s_right_p)

    y_left_new = Dfunc_prev(x_left_new) if len(x_left_new) > 0 else np.array([])
    y_right_new = Dfunc_prev(x_right_new) if len(x_right_new) > 0 else np.array([])

    return pack_params(y_left_new, y_right_new, 0.0, 0.0)


# ============================================================
# 段階的リファインメント：改善されなくなるまで点数を増やす
# ============================================================
def run_progressive_refinement(cfg: PipeConfig = DEFAULT_CFG,
                                Nx=90, Ny=18, n_steps=3000,
                                n_init_per_stage=6, n_iter_per_stage=14,
                                improvement_tol_percent=0.3, patience=2,
                                max_points_per_side=15, seed=0,
                                n_x_restarts=3,
                                plot_each_stage=True, save_rejected=True, out_dir="."):
    """
    n_left=n_right=0（直線）から始め、両側の点数を1つずつ増やしながら
    ベイズ最適化を繰り返す。running best は改善したときしか更新しない
    （悪化した段階の結果は絶対に採用しない）。改善率(%)が
    improvement_tol_percent を patience 回連続で下回ったら打ち切る。

    n_x_restarts : 各段階で、制御点のx位置（層化ランダム）の抽選を
                    この回数だけ繰り返し、その中で一番良かった結果を
                    その段階の結果として採用する。1回のx位置抽選が
                    たまたま悪い配置だったせいで「改善なし」と誤判定
                    されるのを防ぐための対策（計算コストはほぼ
                    n_x_restarts倍になる点に注意）。
    save_rejected : True なら、採用されなかった候補（①restartのうち選ばれ
                    なかったもの、②その段階の最良候補が running best より
                    悪かったもの）の図を out_dir/rejected/ に保存する。
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    stage_log = []

    # ---- ステージ0：直線 ----
    x_left_best, x_right_best = np.array([]), np.array([])
    x_best = pack_params([], [], 0.0, 0.0)
    loss0, dp_best = objective(x_best, cfg, x_left_best, x_right_best, 0, 0,
                                Nx=Nx, Ny=Ny, n_steps=n_steps)
    nl_best, nr_best = 0, 0
    print(f"===== ステージ0（直線）: dP={dp_best:.6g} Pa =====")
    stage_log.append(dict(stage=0, n_left=0, n_right=0, dp=dp_best,
                           improvement=None, adopted=True))
    if plot_each_stage:
        _save_stage_plot(cfg, x_best, x_left_best, x_right_best, 0, 0, 0, dp_best,
                          Nx, Ny, n_steps, out_dir)

    low_streak = 0
    stage_i = 0

    while True:
        nl_try, nr_try = nl_best + 1, nr_best + 1
        if nl_try > max_points_per_side or nr_try > max_points_per_side:
            print(f"安全上限(片側{max_points_per_side}点)に達したため打ち切ります。")
            break
        stage_i += 1
        print(f"\n===== ステージ{stage_i}: n_left={nl_try}, n_right={nr_try} "
              f"(設計変数{len(param_bounds(cfg, nl_try, nr_try))}個, "
              f"x位置を{n_x_restarts}通り試行) =====")

        n_init = max(n_init_per_stage, 2 * (nl_try + nr_try + 2))
        n_iter = max(n_iter_per_stage, 3 * (nl_try + nr_try + 2))

        # ---- B: x位置の層化ランダム抽選を n_x_restarts 回試し、最良を採用 ----
        best_restart = None
        all_restarts = []
        for r in range(n_x_restarts):
            x_left_try, x_right_try = generate_stage_x_positions(
                cfg, nl_try, nr_try, seed=seed + stage_i * 1000 + r)
            print(f"  [restart {r+1}/{n_x_restarts}] x_left={np.round(x_left_try,4).tolist()}  "
                  f"x_right={np.round(x_right_try,4).tolist()}")

            x_seed_raw = upsample_shape(cfg, x_best, x_left_best, x_right_best, nl_best, nr_best,
                                         x_left_try, x_right_try)
            seed_loss, seed_dp = objective(x_seed_raw, cfg, x_left_try, x_right_try, nl_try, nr_try,
                                            Nx=Nx, Ny=Ny, n_steps=n_steps)

            res = run_stage(cfg, x_left_try, x_right_try, nl_try, nr_try, Nx, Ny, n_steps,
                             n_init=n_init, n_iter=n_iter,
                             x_seed_raw=x_seed_raw, loss_dp_seed=(seed_loss, seed_dp),
                             seed=seed + stage_i * 1000 + r)

            cand_x, cand_dp = res["x_best"], res["dp_best"]
            if seed_dp < cand_dp:
                cand_x, cand_dp = x_seed_raw, seed_dp
            print(f"    -> dP={cand_dp:.6g} Pa")

            this_restart = dict(dp=cand_dp, x=cand_x, x_left=x_left_try, x_right=x_right_try, r=r)
            all_restarts.append(this_restart)
            if best_restart is None or cand_dp < best_restart["dp"]:
                best_restart = this_restart

        stage_x, stage_dp = best_restart["x"], best_restart["dp"]
        x_left_try, x_right_try = best_restart["x_left"], best_restart["x_right"]

        improvement = 100 * (1 - stage_dp / dp_best)
        print(f"  ステージ{stage_i} 最良候補（restart {best_restart['r']+1}）: dP={stage_dp:.6g} Pa "
              f"(running best {dp_best:.6g} Pa からの変化 {improvement:+.2f}%)")

        if stage_dp < dp_best:
            x_best, dp_best = stage_x, stage_dp
            x_left_best, x_right_best = x_left_try, x_right_try
            nl_best, nr_best = nl_try, nr_try
            adopted = True
            low_streak = 0 if improvement >= improvement_tol_percent else low_streak + 1
        else:
            print(f"  -> 改善しなかった（悪化 or 同等）ため不採用。running best "
                  f"(dP={dp_best:.6g} Pa, n=({nl_best},{nr_best})) を維持します。")
            adopted = False
            low_streak += 1

        stage_log.append(dict(stage=stage_i, n_left=nl_try, n_right=nr_try, dp=stage_dp,
                               improvement=improvement, adopted=adopted))

        if plot_each_stage:
            _save_stage_plot(cfg, x_best, x_left_best, x_right_best, stage_i, nl_best, nr_best,
                              dp_best, Nx, Ny, n_steps, out_dir)

        # ---- 採用されなかった候補の図を保存 ----
        if save_rejected:
            rej_dir = os.path.join(out_dir, "rejected")
            os.makedirs(rej_dir, exist_ok=True)
            for rs in all_restarts:
                if rs is best_restart:
                    if not adopted:
                        reason = "best of this stage, but not better than running best"
                        tag = "worse_than_best"
                    else:
                        continue
                else:
                    reason = f"restart {best_restart['r']+1} was better (not selected)"
                    tag = "not_selected"
                _save_candidate_plot(
                    cfg, rs["x"], rs["x_left"], rs["x_right"], nl_try, nr_try, rs["dp"],
                    f"[REJECTED] stage {stage_i} restart {rs['r']+1}: {reason}",
                    os.path.join(rej_dir, f"stage_{stage_i:02d}_restart{rs['r']+1}_{tag}.png"),
                    Nx, Ny, n_steps, running_best_dp=dp_best)

        if low_streak >= patience:
            print(f"\n改善率{improvement_tol_percent}%未満(または非改善)が{patience}回連続したため打ち切ります。")
            break

    print(f"\n===== 終了: 最終 dP={dp_best:.6g} Pa (n_left={nl_best}, n_right={nr_best}) "
          f"elapsed={time.time()-t0:.1f}s =====")

    return dict(cfg=cfg, x_best=x_best, x_left_best=x_left_best, x_right_best=x_right_best,
                dp_best=dp_best, n_left=nl_best, n_right=nr_best,
                stage_log=stage_log, Nx=Nx, Ny=Ny, n_steps=n_steps)


# ============================================================
# 可視化
# ============================================================
def evaluate_profile(cfg: PipeConfig, Dfunc, Nx, Ny, n_steps):
    grid = build_grid(cfg, Nx, Ny, Dfunc)
    u, v, p = solve_flow(cfg, grid, n_steps=n_steps)
    dp = pressure_loss(cfg, p, grid)
    return dp, grid, u, v, p


def _draw_shape_and_field(cfg, Dfunc, grid, p, title, ax_geom, ax_field, show_points=True):
    xs = np.linspace(0, cfg.L, 400)
    Ds = Dfunc(xs)
    ax_geom.plot(xs, Ds / 2, 'b-')
    ax_geom.plot(xs, -Ds / 2, 'b-')
    if show_points:
        cx, cy = Dfunc.control_x, Dfunc.control_y
        m = (cx >= cfg.X1) & (cx <= cfg.X3)
        ax_geom.plot(cx[m], cy[m] / 2, 'ro', ms=5, zorder=5)
        ax_geom.plot(cx[m], -cy[m] / 2, 'ro', ms=5, zorder=5)
    ax_geom.axhline(0, color='0.85', lw=0.5)
    ax_geom.set_xlim(0, cfg.L)
    ax_geom.set_ylim(-cfg.D0 * 0.6, cfg.D0 * 0.6)
    ax_geom.set_xlabel("x [m]"); ax_geom.set_ylabel("y [m]")
    ax_geom.set_title(title)

    if grid.get("type") == "bodyfit":
        # 壁に沿った格子のまま描画（節点座標 Xn, Yn を使う）
        pcm = ax_field.pcolormesh(grid["Xn"], grid["Yn"], p, shading="flat", cmap="viridis")
    else:
        xc, yc = grid['xc'], grid['yc']
        p_show = np.where(grid['p_mask'], p, np.nan)
        pcm = ax_field.pcolormesh(xc, yc, p_show.T, shading='auto', cmap='viridis')
    ax_field.plot(xs, Ds / 2, 'k-', lw=1)
    ax_field.plot(xs, -Ds / 2, 'k-', lw=1)
    ax_field.set_xlabel("x [m]"); ax_field.set_ylabel("y [m]")
    ax_field.set_title("pressure field [Pa]")
    plt.colorbar(pcm, ax=ax_field, fraction=0.03)


def _save_candidate_plot(cfg, x, x_left, x_right, nl, nr, dp, label, path,
                          Nx, Ny, n_steps, running_best_dp=None):
    """1つの候補形状（採用/不採用を問わない）の形状＋圧力場の図を path に保存する。"""
    y_left, y_right, s_left, s_right = unpack_params(x, nl, nr)
    Dfunc = make_profile(cfg, x_left, y_left, x_right, y_right, s_left, s_right)
    _, grid, u, v, p = evaluate_profile(cfg, Dfunc, Nx, Ny, n_steps)

    fig, (ax_geom, ax_field) = plt.subplots(1, 2, figsize=(11, 4.4))
    extra = f"  (running best {running_best_dp:.4g} Pa)" if running_best_dp is not None else ""
    title = f"n=({nl},{nr}) s=({s_left:.2f},{s_right:.2f}) \u0394P={dp:.4g} Pa{extra}"
    _draw_shape_and_field(cfg, Dfunc, grid, p, title, ax_geom, ax_field)
    fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _save_stage_plot(cfg, x, x_left, x_right, stage_i, nl, nr, dp, Nx, Ny, n_steps, out_dir):
    y_left, y_right, s_left, s_right = unpack_params(x, nl, nr)
    Dfunc = make_profile(cfg, x_left, y_left, x_right, y_right, s_left, s_right)
    _, grid, u, v, p = evaluate_profile(cfg, Dfunc, Nx, Ny, n_steps)

    fig, (ax_geom, ax_field) = plt.subplots(1, 2, figsize=(11, 4))
    title = (f"stage {stage_i}: n=({nl},{nr})  s=({s_left:.2f},{s_right:.2f})  "
             f"\u0394P={dp:.4g} Pa")
    _draw_shape_and_field(cfg, Dfunc, grid, p, title, ax_geom, ax_field)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/stage_{stage_i:02d}.png", dpi=130)
    plt.close(fig)


def plot_baseline_vs_final(prog, out_dir="."):
    """ベースライン（直線）と最終形状を、圧力場込みで並べて比較する図。"""
    cfg = prog["cfg"]
    Nx, Ny, n_steps = prog["Nx"], prog["Ny"], prog["n_steps"]

    Dfunc_base = make_profile(cfg, [], [], [], [], 0.0, 0.0)
    dp_base, grid_base, _, _, p_base = evaluate_profile(cfg, Dfunc_base, Nx, Ny, n_steps)

    y_left, y_right, s_left, s_right = unpack_params(prog["x_best"], prog["n_left"], prog["n_right"])
    Dfunc_final = make_profile(cfg, prog["x_left_best"], y_left, prog["x_right_best"], y_right,
                                s_left, s_right)
    dp_final, grid_final, _, _, p_final = evaluate_profile(cfg, Dfunc_final, Nx, Ny, n_steps)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _draw_shape_and_field(cfg, Dfunc_base, grid_base, p_base,
                           f"Baseline (straight)  \u0394P={dp_base:.4g} Pa",
                           axes[0, 0], axes[0, 1], show_points=False)
    _draw_shape_and_field(cfg, Dfunc_final, grid_final, p_final,
                           f"Final (n=({prog['n_left']},{prog['n_right']}))  \u0394P={dp_final:.4g} Pa",
                           axes[1, 0], axes[1, 1], show_points=True)
    fig.suptitle(f"reduction: {100*(1-dp_final/dp_base):.2f} %")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/pipe_shape_optimization_result.png", dpi=150)
    plt.close(fig)
    return dp_base, dp_final


def plot_summary(prog, out_dir="."):
    cfg = prog["cfg"]
    log = prog["stage_log"]

    stages = [r["stage"] for r in log]
    dps_running = []
    cur = None
    for r in log:
        if r["adopted"]:
            cur = r["dp"]
        dps_running.append(cur)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax1 = axes[0]
    ax1.plot(stages, [r["dp"] for r in log], 'o--', color='0.6', alpha=0.6, label='this stage attempt')
    ax1.plot(stages, dps_running, 'o-', color='C0', label='running best (adopted)')
    ax1.set_xlabel("stage (= n_left = n_right)")
    ax1.set_ylabel("\u0394P [Pa]")
    ax1.set_title("Progressive refinement: \u0394P per stage")
    ax1.legend(fontsize=8)

    improvements = [r["improvement"] for r in log if r["improvement"] is not None]
    imp_stages = [r["stage"] for r in log if r["improvement"] is not None]
    colors = ['C0' if r["adopted"] else 'C3' for r in log if r["improvement"] is not None]
    ax2 = axes[1]
    ax2.bar(imp_stages, improvements, color=colors)
    ax2.axhline(0, color='k', lw=0.8)
    ax2.set_xlabel("stage")
    ax2.set_ylabel("improvement vs previous running best [%]")
    ax2.set_title("Improvement per stage (red = rejected)")

    fig.tight_layout()
    fig.savefig(f"{out_dir}/optimization_convergence.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR = "C:/Users/kaduk/test"
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = PipeConfig()      # L=1.0m, D0=0.05m, UIN=1.0m/s, X1,X2,X3=0.3,0.5,0.7m

    # 格子：Nx(管の長さ方向のセル数)×Ny(管径方向のセル数)。
    # 探索中は 60x14〜80x16 程度、最終確認は 120x24〜240x36 程度が目安
    # （1評価あたり、60x14〜80x16で約3秒、120x24で約11秒、240x36で約45秒）。
    # n_steps は境界適合格子では使われない（時間発展の長さは cfg.n_flow_through）。
    prog = run_progressive_refinement(
        cfg, Nx=80, Ny=16,
        n_init_per_stage=6, n_iter_per_stage=14,
        improvement_tol_percent=0.3, patience=2,
        max_points_per_side=15, n_x_restarts=3,
        plot_each_stage=True, save_rejected=True, out_dir=OUT_DIR)

    dp_base, dp_final = plot_baseline_vs_final(prog, out_dir=OUT_DIR)
    plot_summary(prog, out_dir=OUT_DIR)

    y_left, y_right, s_left, s_right = unpack_params(prog["x_best"], prog["n_left"], prog["n_right"])
    summary = dict(
        L=cfg.L, D0=cfg.D0, DMIN=cfg.DMIN, UIN=cfg.UIN, RHO=cfg.RHO, MU=cfg.MU,
        X1=cfg.X1, X2=cfg.X2, X3=cfg.X3,
        solver=cfg.solver, Nx=prog["Nx"], Ny=prog["Ny"],
        n_flow_through=cfg.n_flow_through, avg_flow_through=cfg.avg_flow_through,
        n_left=prog["n_left"], n_right=prog["n_right"],
        x_left=prog["x_left_best"].tolist(), x_right=prog["x_right_best"].tolist(),
        y_left=y_left.tolist(), y_right=y_right.tolist(),
        smooth_left=s_left, smooth_right=s_right,
        baseline_dP=dp_base, final_dP=dp_final,
        reduction_percent=100 * (1 - dp_final / dp_base),
        n_stages_run=len(prog["stage_log"]),
        optimizer="Progressive Bayesian optimization (GP+EI), "
                  "stratified-random control-point x + y + smoothness design, PCHIP (no overshoot)"
    )
    try:
        with open(f"{OUT_DIR}/optimization_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"!!! json書き出し失敗: {e}")
        raise

    print(json.dumps(summary, ensure_ascii=False, indent=2))
