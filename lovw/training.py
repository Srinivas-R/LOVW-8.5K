"""Training and chronological cross-validation (as in the paper)."""
import copy
import tempfile

import numpy as np
import skorch
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
from sklearn.model_selection import StratifiedKFold


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_net(module, lr, epochs, batch_size, device, checkpoint_dir):
    """skorch classifier: Adam + one-cycle LR, gradient clipping, and the checkpoint with the best accuracy
    on a random 20% validation split of the training folds (nested model selection)."""
    return NeuralNetClassifier(
        module,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        lr=lr,
        max_epochs=epochs,
        batch_size=batch_size,
        device=device,
        iterator_train__shuffle=True,
        train_split=ValidSplit(0.2),
        callbacks=[
            skorch.callbacks.LRScheduler(torch.optim.lr_scheduler.OneCycleLR, max_lr=lr, epochs=epochs,
                                         steps_per_epoch=1),
            skorch.callbacks.Checkpoint(monitor='valid_acc_best', load_best=True, dirname=checkpoint_dir),
            skorch.callbacks.GradientNormClipping(gradient_clip_value=5, gradient_clip_norm_type=2),
        ],
        warm_start=False,
        verbose=0,
    )


def chronological_folds(session_ids, n_splits=5):
    """5 folds per participant, each a contiguous block of every session.

    Trials are in chronological order within each session; StratifiedKFold without shuffling, stratified by
    session, assigns the first 20% of each session's trials to fold 0, the next 20% to fold 1, and so on.
    """
    dummy = np.zeros(len(session_ids))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=False).split(dummy, session_ids))


def cross_validate(module, X, y, session_ids, lr, epochs, batch_size=128, device='cuda', n_splits=5,
                   folds=None, verbose=True):
    """Train a fresh copy of `module` per fold. Returns per-trial test predictions, fold accuracies, folds."""
    folds = folds or chronological_folds(session_ids, n_splits)
    preds = np.full(len(y), -1)
    fold_accs = []
    for k, (train_idx, test_idx) in enumerate(folds):
        with tempfile.TemporaryDirectory() as ckpt:
            net = make_net(copy.deepcopy(module), lr, epochs, batch_size, device, ckpt)
            net.fit(X[train_idx], y[train_idx])
            preds[test_idx] = net.predict(X[test_idx])
        fold_accs.append(float((preds[test_idx] == y[test_idx].numpy()).mean()))
        if verbose:
            print(f'  fold {k}: {fold_accs[-1]:.3f}', flush=True)
    return preds, fold_accs, folds


def train_plain(module, X_train, y_train, lr, epochs, batch_size=128, device='cuda', X_val=None, y_val=None,
                weight_decay=0.0, augment=None, seed=0):
    """Minimal training loop (Adam + one-cycle LR stepped per batch, gradient clipping).

    If validation data is given, the weights with the best validation accuracy are kept; otherwise the
    final weights are used. `augment(X, rng)` is applied to each training batch if provided.
    Tensors are moved to the device once, which makes small models such as EEGNet fast.
    """
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = module.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    steps = int(np.ceil(len(y_train) / batch_size))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(y_train), generator=gen).to(device)
        for b in range(steps):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb = X_train[idx] if augment is None else augment(idx, rng)
            loss = loss_fn(model(xb), y_train[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        if X_val is not None:
            acc = accuracy(model, X_val, y_val, device)
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, X, device='cuda', batch_size=512):
    model.eval()
    return torch.cat([model(X[i:i + batch_size].to(device)).argmax(-1).cpu()
                      for i in range(0, len(X), batch_size)]).numpy()


def accuracy(model, X, y, device='cuda'):
    y = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    return float((predict(model, X, device) == y).mean())
