import numpy as np

from optimagic.optimization.history_tools import get_history_arrays


def get_convergence_report(history, direction):
    history_arrs = get_history_arrays(
        history=history,
        direction=direction,
    )

    critvals = history_arrs["criterion"][history_arrs["is_accepted"]]
    params = history_arrs["params"][history_arrs["is_accepted"]]

    if len(critvals) < 2:
        out = None
    else:
        out = {}
        for name, n_entries in [("one_step", 2), ("five_steps", min(6, len(critvals)))]:
            relevant_critvals = critvals[-n_entries:]
            relevant_params = params[-n_entries:]

            max_f_rel, max_f_abs = _get_max_f_changes(relevant_critvals)
            max_x_rel, max_x_abs = _get_max_x_changes(relevant_params)

            col_dict = {
                "relative_criterion_change": max_f_rel,
                "relative_params_change": max_x_rel,
                "absolute_criterion_change": max_f_abs,
                "absolute_params_change": max_x_abs,
            }

            out[name] = col_dict

    return out


def _get_max_f_changes(critvals):
    best_val = critvals[-1]
    worst_val = critvals[0]

    max_change_abs = np.abs(best_val - worst_val)
    denom = max(np.abs(best_val), 0.1)

    max_change_rel = max_change_abs / denom

    return max_change_rel, max_change_abs


def _get_max_x_changes(params):
    best_x = params[-1]
    diffs = params - best_x
    denom = np.clip(np.abs(best_x), 0.1, np.inf)

    distances_abs = np.linalg.norm(diffs, axis=1)
    max_change_abs = distances_abs.max()

    scaled = diffs / denom

    distances_rel = np.linalg.norm(scaled, axis=1)
    max_change_rel = distances_rel.max()
    return max_change_rel, max_change_abs
