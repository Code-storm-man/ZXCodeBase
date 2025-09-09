# import numpy as np
# import uuid
# import hashlib
#
#
# def export_traj_ltpl(file_paths: dict,
#                      spline_lengths_opt,
#                      trajectory_opt,
#                      reftrack,
#                      normvec_normalized,
#                      alpha_opt) -> None:
#     """
#     Created by:
#     Tim Stahl
#     Alexander Heilmeier
#
#     Documentation:
#     This function is used to export the generated trajectory into a file for further usage in the local trajectory
#     planner on the car (including map information via normal vectors and bound widths). The generated files get an
#     unique UUID and a hash of the ggv diagram to be able to check it later.
#
#     The stored trajectory has the following columns:
#     [x_ref_m, y_ref_m, width_right_m, width_left_m, x_normvec_m, y_normvec_m, alpha_m, s_racetraj_m, psi_racetraj_rad,
#      kappa_racetraj_radpm, vx_racetraj_mps, ax_racetraj_mps2]
#
#     Inputs:
#     file_paths:         paths for input and output files {ggv_file, traj_race_export, traj_ltpl_export, lts_export}
#     spline_lengths_opt: lengths of the splines on the raceline in m
#     trajectory_opt:     generated race trajectory
#     reftrack:           track definition [x_m, y_m, w_tr_right_m, w_tr_left_m]
#     normvec_normalized: normalized normal vectors on the reference line [x_m, y_m]
#     alpha_opt:          solution vector of the opt. problem containing the lateral shift in m for every ref-point
#     """
#
#     # convert trajectory to desired format
#     s_raceline_preinterp_cl = np.cumsum(spline_lengths_opt)
#     s_raceline_preinterp_cl = np.insert(s_raceline_preinterp_cl, 0, 0.0)
#
#     psi_normvec = []
#     kappa_normvec = []
#     vx_normvec = []
#     ax_normvec = []
#
#     # 从raceline（多）中找到s距离最近的点作为对应
#     for s in list(s_raceline_preinterp_cl[:-1]):
#         # get closest point on trajectory_opt
#         idx = (np.abs(trajectory_opt[:, 0] - s)).argmin()
#
#         # get data at this index and append
#         psi_normvec.append(trajectory_opt[idx, 3])
#         kappa_normvec.append(trajectory_opt[idx, 4])
#         vx_normvec.append(trajectory_opt[idx, 5])
#         ax_normvec.append(trajectory_opt[idx, 6])
#
#     traj_ltpl = np.column_stack((reftrack,
#                                  normvec_normalized,
#                                  alpha_opt,
#                                  s_raceline_preinterp_cl[:-1],
#                                  psi_normvec,
#                                  kappa_normvec,
#                                  vx_normvec,
#                                  ax_normvec))
#     traj_ltpl_cl = np.vstack((traj_ltpl, traj_ltpl[0]))
#     traj_ltpl_cl[-1, 7] = s_raceline_preinterp_cl[-1]
#
#     # create random UUID
#     rand_uuid = str(uuid.uuid4())
#
#     # hash ggv file with SHA1
#     if "ggv_file" in file_paths:
#         with open(file_paths["ggv_file"], 'br') as fh:
#             ggv_content = fh.read()
#     else:
#         ggv_content = np.array([])
#     ggv_hash = hashlib.sha1(ggv_content).hexdigest()
#
#     # write UUID and GGV hash into file
#     with open(file_paths["traj_ltpl_export"], 'w') as fh:
#         fh.write("# " + rand_uuid + "\n")
#         fh.write("# " + ggv_hash + "\n")
#
#     # export trajectory data for local planner
#     header = "x_ref_m; y_ref_m; width_right_m; width_left_m; x_normvec_m; y_normvec_m; " \
#              "alpha_m; s_racetraj_m; psi_racetraj_rad; kappa_racetraj_radpm; vx_racetraj_mps; ax_racetraj_mps2"
#     fmt = "%.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f"
#     with open(file_paths["traj_ltpl_export"], 'ab') as fh:
#         np.savetxt(fh, traj_ltpl, fmt=fmt, header=header)
#
#
# # testing --------------------------------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     pass



# --------------------上面是原始的输出，下面是为了raceline的输出------------------------------------------------
import numpy as np
import math
import uuid
import hashlib


def calc_lxy_da_s_k(reftrack: np.ndarray,
                    normvec_normalized: np.ndarray,
                    trajectory_opt: np.ndarray,
                    spline_lengths_opt: np.ndarray) -> tuple:
    """
    Calculate lateral offset (L), coordinates (X, Y), heading difference (dA), cumulative distance (S),
    and curvature (K) for each point in reftrack by matching with trajectory_opt.
    Also calculate reference track's cumulative distance (Sref) and curvature (Kref).
    Uses np.unwrap for heading angle continuity.

    Inputs:
    :param reftrack: Reference track [x_ref_m, y_ref_m, width_right_m, width_left_m], shape (N, 4)
    :param normvec_normalized: Normalized normal vectors [x_normvec_m, y_normvec_m], shape (N, 2)
    :param trajectory_opt: Optimized trajectory [s, x, y, psi, kappa, vx, ax], shape (M, 7)
    :param spline_lengths_opt: Spline lengths of optimized raceline, shape (N,)

    Outputs:
    :return Sref: Cumulative distance of reftrack, shape (N,)
    :return Kref: Curvature of reftrack, shape (N,)
    :return L: Lateral offset for each reftrack point, shape (N,)
    :return X: X coordinates corresponding to reftrack points, shape (N,)
    :return Y: Y coordinates corresponding to reftrack points, shape (N,)
    :return S: Cumulative distance of raceline (based on spline_lengths_opt), shape (N,)
    :return A: Heading of raceline, shape (N,)
    :return dA: Heading difference (A - Aref), shape (N,)
    :return K: Curvature of raceline, shape (N,)
    :return V: Velocity from trajectory_opt, shape (N,)
    :return AT: Acceleration from trajectory_opt, shape (N,)
    :return Aref: Reference heading, shape (N,)
    """

    # Check input shapes
    N = reftrack.shape[0]
    if normvec_normalized.shape[0] != N or spline_lengths_opt.shape[0] != N:
        raise ValueError("reftrack, normvec_normalized, and spline_lengths_opt must have same number of rows")
    if trajectory_opt.shape[1] < 7:
        raise ValueError("trajectory_opt must have at least 7 columns [s, x, y, psi, kappa, vx, ax]")

    # Step 1: Calculate cumulative distance for reftrack (Sref) and raceline (S)
    s_raceline_preinterp_cl = np.cumsum(spline_lengths_opt)
    s_raceline_preinterp_cl = np.insert(s_raceline_preinterp_cl, 0, 0.0)  # Shape (N+1,)
    S = s_raceline_preinterp_cl[:-1]  # Shape (N,), raceline cumulative distance

    # Step 2: Find corresponding points in trajectory_opt
    L = np.zeros(N)  # Lateral offset
    X = np.zeros(N)  # X coordinates
    Y = np.zeros(N)  # Y coordinates
    Vs = np.zeros(N)  # Velocity
    AT = np.zeros(N)  # Acceleration
    AN = np.zeros(N)  # Lateral acceleration (NEW)
    TIME = np.zeros(N)  # Cumulative time (NEW)

    for i, s in enumerate(s_raceline_preinterp_cl[:-1]):
        # Find closest point in trajectory_opt based on cumulative distance
        idx = np.abs(trajectory_opt[:, 0] - s).argmin()

        # Extract coordinates, velocity, and acceleration
        X[i] = trajectory_opt[idx, 1]
        Y[i] = trajectory_opt[idx, 2]
        Vs[i] = trajectory_opt[idx, 5]
        AT[i] = trajectory_opt[idx, 6]
        AN[i] = Vs[i] ** 2 * trajectory_opt[idx, 4]  # NEW: AN = Vs^2 * kappa

        # Calculate lateral offset L
        delta_vec = np.array([X[i] - reftrack[i, 0], Y[i] - reftrack[i, 1]])
        L[i] = np.dot(delta_vec, normvec_normalized[i])
        L[i] = -L[i]

    # Step 3: Calculate time TIME (NEW)
    for i in range(N):
        if i == 0:
            TIME[i] = 0.0
        else:
            # Use average velocity between consecutive points
            avg_v = (Vs[i] + Vs[i - 1]) / 2
            if avg_v > 1e-6:  # Avoid division by zero
                TIME[i] = TIME[i - 1] + 2 * spline_lengths_opt[i - 1] / avg_v
            else:
                TIME[i] = TIME[i - 1]  # If velocity is near zero, keep time unchanged

    # Step 3: Calculate heading A from X, Y
    # 确保X和Y是numpy数组（如果不是请先转换）
    X_for_A_K = np.array(X)  # 替换为实际的X坐标数组
    Y_for_A_K = np.array(Y)  # 替换为实际的Y坐标数组
    N = len(X_for_A_K)
    A = np.zeros(N)
    # 计算X和Y的梯度（类似MATLAB的gradient函数）
    dx = np.gradient(X_for_A_K)  # 比相邻点差分更平滑的梯度计算
    dy = np.gradient(Y_for_A_K)
    norms = np.sqrt(dx ** 2 + dy ** 2)
    norms[norms == 0] = 1e-10  # 处理静止点（模长为0的情况）
    A = np.arctan2(dy, dx)
    # 解缠绕角度，确保连续性（与MATLAB的unwrap一致）
    A = np.unwrap(A, discont=np.pi)

    # Step 4: Calculate reference heading Aref from reftrack
    N = reftrack.shape[0]  # 获取参考轨迹点数量
    # 计算X和Y的梯度（替代原相邻点差分，与MATLAB gradient一致）
    dx = np.gradient(reftrack[:, 0])
    dy = np.gradient(reftrack[:, 1])
    # 计算切向量模长，避免除零（参考MATLAB逻辑）
    norms = np.sqrt(dx ** 2 + dy ** 2)
    norms[norms == 0] = 1e-10  # 处理零模长情况
    # 计算航向角：直接使用atan2(dy, dx)，移除原代码中的-π/2偏移（与MATLAB一致）
    Aref = np.arctan2(dy, dx)
    # 解缠绕角度，确保连续性（与MATLAB unwrap逻辑一致）
    Aref = np.unwrap(Aref, discont=np.pi)

    # Step 5: Calculate heading difference dA
    dA = A - Aref
    dA = np.unwrap(dA, discont=np.pi)

    # Step 6: Calculate curvature K for raceline (X, Y)
    dx = np.gradient(X_for_A_K)
    dy = np.gradient(Y_for_A_K)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    K = np.sign(-dy * ddx + dx * ddy) * np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)
    K = np.where(np.isnan(K), 0.0, K)  # Handle NaN (e.g., straight segments)

    # Step 7: Calculate curvature Kref for reftrack
    dx_ref = np.gradient(reftrack[:, 0])
    dy_ref = np.gradient(reftrack[:, 1])
    ddx_ref = np.gradient(dx_ref)
    ddy_ref = np.gradient(dy_ref)
    Kref = np.sign(-dy_ref * ddx_ref + dx_ref * ddy_ref) * np.abs(dx_ref * ddy_ref - dy_ref * ddx_ref) / \
           (dx_ref ** 2 + dy_ref ** 2) ** (3 / 2)
    Kref = np.where(np.isnan(Kref), 0.0, Kref)

    ds_ref = np.sqrt(dx_ref ** 2 + dy_ref ** 2)
    Sref = np.cumsum(ds_ref)
    total_length = Sref[-1] + ds_ref[-1]
    Sref = Sref - Sref[0]
    Sref[-1] = total_length

    # Verify array shapes
    arrays = [Sref, Kref, L, X, Y, S, A, dA, K, Vs, AT, AN, TIME, Aref]  # Updated with AN, TIME
    for i, arr in enumerate(arrays):
        if arr.shape[0] != N:
            raise ValueError(f"Array at index {i} has shape {arr.shape}, expected ({N},)")

    return Sref, Kref, L, X, Y, S, A, dA, K, Vs, AT, AN, TIME, Aref  # Updated return


def export_traj_ltpl(file_paths: dict,
                     spline_lengths_opt: np.ndarray,
                     trajectory_opt: np.ndarray,
                     reftrack: np.ndarray,
                     normvec_normalized: np.ndarray,
                     alpha_opt: np.ndarray) -> None:
    """
    Export trajectory to CSV with columns:
    Sref, Xref, Yref, Aref, Kref, Lmax, Lmin, S, L, X, Y, A, dA, K, V, AT
    """
    # Calculate all required quantities
    Sref, Kref, L, X, Y, S, A, dA, K, Vs, AT, AN, TIME, Aref = calc_lxy_da_s_k(  # Updated with AN, TIME
        reftrack, normvec_normalized, trajectory_opt, spline_lengths_opt)

    # Prepare Lmax (width_left_m) and Lmin (-width_right_m)
    Lmax = reftrack[:, 3]  # width_left_m
    Lmin = -reftrack[:, 2]  # -width_right_m

    # Verify input array shapes for column_stack
    arrays = [Sref, reftrack[:, 0], reftrack[:, 1], Aref, Kref, Lmax, Lmin, S, L, X, Y, A, dA, K, Vs, AT, AN,
              TIME]  # Updated with AN, TIME
    N = reftrack.shape[0]
    for i, arr in enumerate(arrays):
        if arr.shape[0] != N:
            raise ValueError(f"Array at index {i} has shape {arr.shape}, expected ({N},)")

    # Combine into traj_ltpl
    traj_ltpl = np.column_stack((Sref,
                                 reftrack[:, 0],  # Xref
                                 reftrack[:, 1],  # Yref
                                 Aref,
                                 Kref,
                                 Lmax,
                                 Lmin,
                                 S,
                                 L,
                                 X,
                                 Y,
                                 A,
                                 dA,
                                 K,
                                 Vs,
                                 AT,
                                 AN,  # NEW
                                 TIME))  # NEW

    # Create closed trajectory
    s_raceline_preinterp_cl = np.cumsum(spline_lengths_opt)
    s_raceline_preinterp_cl = np.insert(s_raceline_preinterp_cl, 0, 0.0)

    traj_ltpl_cl = np.vstack((traj_ltpl, traj_ltpl[0]))
    traj_ltpl_cl[-1, 7] = s_raceline_preinterp_cl[-1]  # Update S for closed path
    # NEW: Ensure AN and TIME for closed point
    traj_ltpl_cl[-1, 16] = traj_ltpl_cl[0, 16]  # AN for closed point
    traj_ltpl_cl[-1, 17] = TIME[-1] + 2 * spline_lengths_opt[-1] / (Vs[-1] + Vs[0]) if (Vs[-1] + Vs[0]) > 1e-6 else \
    TIME[-1]  # TIME for closed point

    # Write UUID and GGV hash
    rand_uuid = str(uuid.uuid4())
    if "ggv_file" in file_paths:
        with open(file_paths["ggv_file"], 'br') as fh:
            ggv_content = fh.read()
    else:
        ggv_content = np.array([])
    ggv_hash = hashlib.sha1(ggv_content).hexdigest()

    with open(file_paths["traj_ltpl_export"], 'w') as fh:
        fh.write("# " + rand_uuid + "\n")
        fh.write("# " + ggv_hash + "\n")

    # Export to CSV
    header = "Sref;Xref;Yref;Aref;Kref;Lmax;Lmin;S;L;X;Y;A;dA;K;Vs;AT;AN;TIME"  # Updated header
    fmt = "%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f"  # Updated fmt
    with open(file_paths["traj_ltpl_export"], 'ab') as fh:
        np.savetxt(fh, traj_ltpl, fmt=fmt, header=header)


# Testing
if __name__ == "__main__":
    pass




