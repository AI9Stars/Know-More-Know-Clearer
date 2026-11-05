import re
import string
import torch

def clean_text(text):
    text = (text or "").lower()
    text = re.sub(r"\s+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def compute_score(prediction, ground_truth):
    pred_clean = clean_text(prediction)
    gt_clean = clean_text(ground_truth)
    return 1.0 if gt_clean in pred_clean else 0.0

def _to_list_of_text(x):
    """Coerce a prediction payload into a flat ``list[str]``.

    Accepted shapes:
      - str              -> [str]
      - list[str]        -> list[str]
      - list[list[str]]  -> first element of each inner list, i.e. list[str]
    """
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        if len(x) == 0:
            return []
        if isinstance(x[0], str):
            return x
        if isinstance(x[0], list):
            out = []
            for xi in x:
                if isinstance(xi, list) and len(xi) > 0 and isinstance(xi[0], str):
                    out.append(xi[0])
                else:
                    out.append("")
            return out
    return []

def acc_reward(*args, return_dict=False, **kwargs):
    ground_truth = kwargs.get("ground_truth", None)
    preds = _to_list_of_text(kwargs.get("solution_str", None))

    if ground_truth is None:
        raise KeyError("acc_reward: missing ground_truth in kwargs (expected kwargs['ground_truth']).")
    if len(preds) == 0:
        raise KeyError("acc_reward: missing solution_str in kwargs (expected kwargs['solution_str']).")

    accs = [float(compute_score(p, ground_truth)) for p in preds]
    reward_tensor = torch.tensor(accs, dtype=torch.float32)  # shape [K] (K=rollout.n)

    if return_dict:
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": {
                "acc": accs,
                "prediction_preview": [p[:80] for p in preds],
                "gt_preview": str(ground_truth)[:80],
            },
        }
    return reward_tensor
