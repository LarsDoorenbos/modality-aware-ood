
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from metrics import compute_all_metrics
from sklearn.covariance import LedoitWolf
import time

EPS = 1e-8

# -------------------------
# Distances
# -------------------------
def jsd(p, q, eps=EPS):
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * torch.log((p + eps) / (m + eps)), dim=-1)
    kl_qm = torch.sum(q * torch.log((q + eps) / (m + eps)), dim=-1)
    return 0.5 * (kl_pm + kl_qm)

def kl_divergence(p, q, eps=EPS):
    return torch.sum(p * torch.log((p + eps) / (q + eps)), dim=-1)

def total_variation(p, q):
    return 0.5 * torch.sum(torch.abs(p - q), dim=-1)

def hellinger_distance(p, q):
    return (0.5 * torch.sum((torch.sqrt(p) - torch.sqrt(q))**2, dim=-1)).sqrt()

def bhattacharyya_distance(p, q, eps=EPS):
    bc = torch.sum(torch.sqrt(p * q), dim=-1)
    return -torch.log(bc + eps)

def cosine_distance(p, q, eps=EPS):
    dot = torch.sum(p * q, dim=-1)
    norm_p = torch.norm(p, dim=-1)
    norm_q = torch.norm(q, dim=-1)
    return 1 - dot / (norm_p * norm_q + eps)

def fisher_rao_distance(p, q, eps=EPS):
    inner = torch.sum(torch.sqrt(p * q), dim=-1)
    inner = torch.clamp(inner, 0, 1)
    return 2 * torch.acos(inner)

def l2_distance(p, q):
    return torch.norm(p - q, dim=-1)


eps = 1e-6
def get_extended_logits(id_train_feature, train_labels, val_feature, id_feature, ood_feature, id_val_logit, id_output, ood_output, scale_mean=True, scale_std=True, var_explained=0.01, normalize=True):

    if normalize:
        normalizer = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) +  + 1e-10)

        id_train_feature = normalizer(id_train_feature)
        val_feature = normalizer(val_feature)
        id_feature = normalizer(id_feature)
        ood_feature = normalizer(ood_feature)

    mean = np.mean(id_train_feature, axis=0)
    lw = LedoitWolf().fit(id_train_feature)
    cov_full = lw.covariance_

    # Helper for eigen decomposition + projection
    def low_var_proj(X, mean, cov):
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort eigenvalues (and vectors) ascending (low variance first)
        idx = np.argsort(eigvals)
        eigvals_sorted = eigvals[idx]
        eigvecs_sorted = eigvecs[:, idx]

        # Compute cumulative variance ratio (from smallest upwards)
        total_var = eigvals_sorted.sum()
        cum_var = np.cumsum(eigvals_sorted) / (total_var + eps)

        # Choose smallest r such that cum_var[r] >= var_explained
        r = np.searchsorted(cum_var, var_explained) + 1
        r = min(r, len(eigvals_sorted))  # safety

        # Take the r lowest-variance components
        U_low = eigvecs_sorted[:, :r]
        Lambda_low = eigvals_sorted[:r]

        diff = X - mean
        proj = diff @ U_low
        z = proj / np.sqrt(Lambda_low + eps)
        return np.linalg.norm(z, axis=1), r
    
    # ----- Step 4: Compute distances -----
    m_id,   r_full   = low_var_proj(id_feature,  mean, cov_full)
    m_val,  _        = low_var_proj(val_feature, mean, cov_full)
    m_ood,  _        = low_var_proj(ood_feature, mean, cov_full)

    # ----- Step 5: Match Mahalanobis to logit statistics -----
    max_logit_val = id_val_logit.max(axis=-1)
    mean_logit, std_logit = max_logit_val.mean(), max_logit_val.std()
    mean_maha, std_maha   = m_val.mean(),  m_val.std()
    alpha = std_logit / (std_maha + 1e-6) if scale_std else 1.0
    beta  = mean_logit - alpha * mean_maha if scale_mean else 0.0

    # ----- Step 6: Scaled virtual logit -----
    vlogit_id  = m_id  * alpha + beta
    vlogit_ood = m_ood * alpha + beta

    # ----- Step 7: Append virtual logit -----
    id_output_virtual  = np.concatenate([id_output,  vlogit_id[:,  None]], axis=1)
    ood_output_virtual = np.concatenate([ood_output, vlogit_ood[:, None]], axis=1)

    # ----- Step 8: Softmax probabilities -----
    id_probs  = F.softmax(torch.tensor(id_output_virtual, dtype=torch.float32), dim=-1)
    ood_probs = F.softmax(torch.tensor(ood_output_virtual, dtype=torch.float32), dim=-1)

    return torch.tensor(id_output_virtual), torch.tensor(ood_output_virtual), id_probs, ood_probs, m_val, m_id, m_ood


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--resumef", type=str, default='checkpoint.pt')
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument('--use_ash', action='store_true')
    parser.add_argument('--use_react', action='store_true')
    parser.add_argument('--v_thr', type=float, default=0.8023215949535369,
                        help='v_thr')
    parser.add_argument('--f_thr', type=float, default=0.615705931186676,
                        help='f_thr')
    parser.add_argument('--near_ood', action='store_true')
    parser.add_argument("--dataset", type=str, default='UCF') # HMDB UCF Kinetics
    parser.add_argument('--far_ood', action='store_true')
    parser.add_argument("--ood_dataset", type=str, default='UCF') # HMDB UCF Kinetics HAC
    parser.add_argument("--low_var", type=float, default=0.005)
    parser.add_argument("--mm_weight", type=float, default=1.25)
    parser.add_argument("--mm_weight2", type=float, default=0.4)
    parser.add_argument("--temp", type=float, default=1.0)
    args = parser.parse_args()

    setting = '_near_ood_' if args.near_ood else '_ood_' + args.ood_dataset + '_'
    
    # init_distributed_mode(args)
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    
    device = 'cuda:0' # or 'cpu'
    device = torch.device(device)

    v_dim = 2304
    f_dim = 2048

    if args.near_ood:
        if args.dataset == 'HMDB':
            num_class = 25
        elif args.dataset == 'UCF':
            num_class = 50
        elif args.dataset == 'Kinetics':
            num_class = 129
        elif args.dataset == 'EPIC':
            num_class = 4
        elif args.dataset == 'CMU':
            num_class = 25
    else:
        if args.dataset == 'HMDB':
            num_class = 43
        elif args.dataset == 'Kinetics':
            num_class = 229
        elif args.dataset == 'UCF':
            num_class = 50
        elif args.dataset == 'EPIC':
            num_class = 4
    print("####",num_class)

    split = 'test'
    # print(split)

    output_name = 'saved_files/id_'+args.dataset+setting+'output_' + args.appen + split + '.npy'
    pred_name = 'saved_files/id_'+args.dataset+setting+'pred_' + args.appen + split + '.npy'
    conf_name = 'saved_files/id_'+args.dataset+setting+'conf_' + args.appen + split + '.npy'
    vconf_name = 'saved_files/id_'+args.dataset+setting+'vconf_' + args.appen + split + '.npy'
    fconf_name = 'saved_files/id_'+args.dataset+setting+'fconf_' + args.appen + split + '.npy'
    label_name = 'saved_files/id_'+args.dataset+setting+'label_' + args.appen + split + '.npy'
    feature_name = 'saved_files/id_'+args.dataset+setting+'feature_' + args.appen + split + '.npy'
    v_feature_name = 'saved_files/id_'+args.dataset+setting+'vfeature_' + args.appen + split + '.npy'
    f_feature_name = 'saved_files/id_'+args.dataset+setting+'ffeature_' + args.appen + split + '.npy'
    
    id_output = np.load(output_name)
    id_pred = np.load(pred_name)
    id_conf = np.load(conf_name)
    id_vconf = np.load(vconf_name)
    id_fconf = np.load(fconf_name)
    id_gt = np.load(label_name)
    id_feature = np.load(feature_name)
    id_vfeature = np.load(v_feature_name)
    id_ffeature = np.load(f_feature_name)

    # Compute test accuracy
    test_acc = (id_pred == id_gt).mean()
    print("Test accuracy: {:.2f}%".format(test_acc * 100.0))

    split = 'eval'
    # print(split)

    output_name = 'saved_files/id_'+args.dataset+setting+'output_' + args.appen + split + '.npy'
    pred_name = 'saved_files/id_'+args.dataset+setting+'pred_' + args.appen + split + '.npy'
    conf_name = 'saved_files/id_'+args.dataset+setting+'conf_' + args.appen + split + '.npy'
    vconf_name = 'saved_files/id_'+args.dataset+setting+'vconf_' + args.appen + split + '.npy'
    fconf_name = 'saved_files/id_'+args.dataset+setting+'fconf_' + args.appen + split + '.npy'
    label_name = 'saved_files/id_'+args.dataset+setting+'label_' + args.appen + split + '.npy'
    feature_name = 'saved_files/id_'+args.dataset+setting+'feature_' + args.appen + split + '.npy'
    v_feature_name = 'saved_files/id_'+args.dataset+setting+'vfeature_' + args.appen + split + '.npy'
    f_feature_name = 'saved_files/id_'+args.dataset+setting+'ffeature_' + args.appen + split + '.npy'

    ood_output = np.load(output_name)
    ood_pred = np.load(pred_name)
    ood_conf = np.load(conf_name)
    ood_vconf = np.load(vconf_name)
    ood_fconf = np.load(fconf_name)
    ood_gt = np.load(label_name)
    ood_feature = np.load(feature_name)
    ood_vfeature = np.load(v_feature_name)
    ood_ffeature = np.load(f_feature_name)

    ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
    pred = np.concatenate([id_pred, ood_pred])
    conf = np.concatenate([id_conf, ood_conf])
    label = np.concatenate([id_gt, ood_gt])
    msp_ood_metrics = compute_all_metrics(conf, label, pred)

    print("MSP FPR@95: ", msp_ood_metrics[0], "MSP AUROC: ", msp_ood_metrics[1])

    split = 'val'
    output_name = 'saved_files/id_'+args.dataset+setting+'output_' + args.appen + split + '.npy'
    vconf_name = 'saved_files/id_'+args.dataset+setting+'vconf_' + args.appen + split + '.npy'
    fconf_name = 'saved_files/id_'+args.dataset+setting+'fconf_' + args.appen + split + '.npy'
    pred_name = 'saved_files/id_'+args.dataset+setting+'conf_' + args.appen + split + '.npy'
    labels = np.load('saved_files/id_'+args.dataset+setting+'label_' + args.appen + split + '.npy')
    
    val_output = np.load(output_name)
    val_vconf = np.load(vconf_name)
    val_fconf = np.load(fconf_name)
    val_conf = np.load(pred_name)

    # Compute validation accuracy
    val_pred = np.argmax(val_output, axis=-1)
    val_acc = (val_pred == labels).mean()
    print("Validation accuracy: {:.2f}%".format(val_acc * 100.0))

    id_v_feature_name = 'saved_files/id_'+args.dataset+setting+'vfeature_' + args.appen + 'train' + '.npy'
    id_vtrain_feature = np.load(id_v_feature_name)
    id_v_feature_name = 'saved_files/id_'+args.dataset+setting+'vfeature_' + args.appen + 'val' + '.npy'
    id_vval_feature = np.load(id_v_feature_name)

    id_f_feature_name = 'saved_files/id_'+args.dataset+setting+'ffeature_' + args.appen + 'train' + '.npy'
    id_ftrain_feature = np.load(id_f_feature_name)
    id_f_feature_name = 'saved_files/id_'+args.dataset+setting+'ffeature_' + args.appen + 'val' + '.npy'
    id_fval_feature = np.load(id_f_feature_name)

    feature_name = 'saved_files/id_'+args.dataset+setting+'feature_' + args.appen + 'train.npy'
    id_train_feature = np.load(feature_name)
    val_feature = np.load('saved_files/id_'+args.dataset+setting+'feature_' + args.appen + 'val' + '.npy')

    id_val_logit_name = 'saved_files/id_'+args.dataset+setting+'output_' + args.appen + 'val' + '.npy'
    id_val_logit = np.load(id_val_logit_name)
    id_val_vlogit_name = 'saved_files/id_'+args.dataset+setting+'vconf_' + args.appen + 'val' + '.npy'
    id_val_vlogit = np.load(id_val_vlogit_name)
    id_val_flogit_name = 'saved_files/id_'+args.dataset+setting+'fconf_' + args.appen + 'val' + '.npy'
    id_val_flogit = np.load(id_val_flogit_name)

    train_labels = np.load('saved_files/id_'+args.dataset+setting+'label_' + args.appen + 'train' + '.npy')

    start = time.time()

    def solve_with_intercept_per_class(f_v, f_f, f_m, ridge=0.0):
        """
        Solve for alpha_c, beta_c, gamma_c per class c such that
            alpha_c * f_v[:, c] + beta_c * f_f[:, c] + gamma_c ≈ f_m[:, c]
        with a per-class intercept term.
        
        Args:
            f_v, f_f, f_m: arrays of shape (N, C)
            ridge: float, ridge regularization strength
        
        Returns:
            alpha, beta, gamma: arrays of shape (C,)
        """
        N, C = f_v.shape
        alpha = np.zeros(C)
        beta = np.zeros(C)
        gamma = np.zeros(C)

        for c in range(C):
            X = np.stack([f_v[:, c], f_f[:, c], np.ones(N)], axis=1)  # shape: (N, 3)
            y = f_m[:, c]                                               # shape: (N,)
            A = X.T @ X + ridge * np.eye(3)
            b = X.T @ y
            sol = np.linalg.solve(A, b)
            alpha[c], beta[c], gamma[c] = sol

        return alpha, beta, gamma
    

    alpha, beta, gamma = solve_with_intercept_per_class(val_vconf, val_fconf, val_output, ridge=1e-6)

    pred_val = alpha * val_vconf + beta * val_fconf + gamma
    pred_id  = alpha * id_vconf  + beta * id_fconf  + gamma
    pred_ood = alpha * ood_vconf + beta * ood_fconf  + gamma

    resid_id = (pred_id - id_output) ** 2
    resid_ood = (pred_ood - ood_output) ** 2
    resid_val = (pred_val - val_output) ** 2

    print("MSE:", resid_val.mean(), resid_id.mean(), resid_ood.mean())

    pred_prob_val = F.softmax(torch.tensor(pred_val / args.temp), dim=-1)
    pred_prob_id = F.softmax(torch.tensor(pred_id / args.temp), dim=-1)
    pred_prob_ood = F.softmax(torch.tensor(pred_ood / args.temp), dim=-1)

    prob_dist_val = l2_distance(pred_prob_val, F.softmax(torch.tensor(val_output / args.temp), dim=-1))
    prob_dist_id = l2_distance(pred_prob_id, F.softmax(torch.tensor(id_output / args.temp), dim=-1))
    prob_dist_ood = l2_distance(pred_prob_ood, F.softmax(torch.tensor(ood_output / args.temp), dim=-1))

    hybrid_ext_logits_id, hybrid_ext_logits_ood, hybrid_ext_probs_id, hybrid_ext_probs_ood, m_val, m_id, m_ood = get_extended_logits(id_train_feature, train_labels, val_feature, id_feature, ood_feature, id_val_logit, id_output, ood_output, var_explained=args.low_var)

    def make_two_virtual_logits(m_val, m_id, m_ood, mm_val, mm_id, mm_ood, scale_mean=True, scale_std=True, eps=1e-6):
        max_logit_val = id_val_logit.max(axis=-1)
        mean_logit, std_logit = max_logit_val.mean(), max_logit_val.std()

        mean_maha, std_maha   = m_val.mean(),  m_val.std()
        alpha = std_logit / (std_maha + 1e-6) if scale_std else 1.0
        beta  = mean_logit - alpha * mean_maha if scale_mean else 0.0

        vlogit_id  = args.mm_weight * (m_id  * alpha + beta)
        vlogit_ood = args.mm_weight * (m_ood * alpha + beta)

        mean_mm, std_mm = mm_val.mean(), mm_val.std()
        alpha_mm = std_logit / (std_mm + 1e-6) if scale_std else 1.0
        beta_mm  = mean_logit - alpha_mm * mean_mm if scale_mean else 0.0

        vlogit_id2 = args.mm_weight2 * (mm_id * alpha_mm + beta_mm)
        vlogit_ood2 = args.mm_weight2 * (mm_ood * alpha_mm + beta_mm)

        # ----- Step 7: Append virtual logit -----
        id_output_virtual  = np.concatenate([id_output,  vlogit_id[:,  None], vlogit_id2[:, None]], axis=1)
        ood_output_virtual = np.concatenate([ood_output, vlogit_ood[:, None], vlogit_ood2[:, None]], axis=1)

        # ----- Step 8: Softmax probabilities -----
        id_probs  = F.softmax(torch.tensor(id_output_virtual, dtype=torch.float32), dim=-1)
        ood_probs = F.softmax(torch.tensor(ood_output_virtual, dtype=torch.float32), dim=-1)

        return id_probs, ood_probs
    
    id_virt, ood_virt = make_two_virtual_logits(m_val, m_id, m_ood, prob_dist_val, prob_dist_id, prob_dist_ood)

    pred = np.concatenate([-id_virt[:, -2:].sum(dim=-1).numpy(), -ood_virt[:, -2:].sum(dim=-1).numpy()])
    conf = pred
    maha_full_ood_metrics = compute_all_metrics(conf, label, pred)
    full_conf = pred

    print("FPR@95: ", maha_full_ood_metrics[0], "AUROC: ", maha_full_ood_metrics[1])
    print('Total time: ', time.time() - start)
