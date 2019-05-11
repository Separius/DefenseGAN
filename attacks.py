import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from modules import CNNClassifier, FCClassifier
from utils import get_mnist_ds, mkdir, to, setup_run


class AdversarialAttack:
    def __init__(self, model):
        self.model = model.eval()
        self.x = None
        self.y = None

    def attack(self, data_loader):
        correct = 0
        total = 0
        all_x_adv = []
        all_y = []
        for i, (x, y) in enumerate(tqdm(data_loader, desc='Attack')):
            all_y.append(y)
            x, y = to(x), to(y)
            x_adv = self._attack(x, y)
            y_adv = self.model(x_adv)
            pred = y_adv.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
            total += y.size(0)
            all_x_adv.append(x_adv.cpu())
        self.x = torch.cat(all_x_adv, dim=0)
        self.y = torch.cat(all_y, dim=0)
        return correct / total

    def _clamp(self, adv_x, detach=True):
        adv_x = torch.clamp(adv_x, min=-1.0, max=1.0)
        return adv_x.detach_() if detach else adv_x

    def _attack(self, x, y):
        raise NotImplementedError()

    def save(self, file_loc):
        torch.save({'x': self.x.cpu().detach(), 'y': self.y.cpu().detach()}, file_loc)


class NoAttack(AdversarialAttack):
    def _attack(self, x, y):
        return x


class BlackBoxAttack(NoAttack):
    def __init__(self, oracle, substitute, holdout, white_box: AdversarialAttack):
        super().__init__(oracle)
        self.substitute = substitute
        self.white_box = white_box
        self.holdout = holdout
        self.augmentation_iters = 6
        self.epochs_per_aug = 10
        self.batch_size = 128
        self.lamb = 0.1

    def _jacobian_augmentation(self, prev_x, prev_y):
        bs = self.batch_size
        for i in trange(int(np.ceil(prev_x.size(0) / bs)), desc='Jacobian Augmentation'):
            x = to(prev_x[i * bs:(i + 1) * bs])
            x.requires_grad_()
            preds = self.substitute(x)
            score = torch.gather(preds, 1, to(prev_y[i * bs:(i + 1) * bs].unsqueeze(1)))
            score.sum().backward()
            prev_x[i * bs:(i + 1) * bs].add_(self.lamb * x.grad.sign().cpu())
        return self._clamp(prev_x)

    def _train_sub(self):
        bs = self.batch_size
        net = self.substitute
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        x, y = self.holdout
        for aug_iter in trange(self.augmentation_iters, desc='Augmentation Iters'):
            net.train()
            for epoch in trange(self.epochs_per_aug, desc='Inner Epochs'):
                indices = np.arange(x.size(0))
                np.random.shuffle(indices)
                for batch in trange(int(np.ceil(len(indices) // bs)), desc='Minibatches'):
                    x_b, y_b = x[batch * bs:(batch + 1) * bs], y[batch * bs:(batch + 1) * bs]
                    pred = net(to(x_b))
                    loss = F.cross_entropy(pred, to(y_b))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            net.eval()
            if aug_iter != self.augmentation_iters - 1:
                new_x = self._jacobian_augmentation(x, y)
                new_y = self.model(to(new_x)).argmax(dim=1)  # oracle
                x = torch.cat([x, new_x], dim=0)
                y = torch.cat([y, new_y.cpu()], dim=0)

    def attack(self, data_loader):
        self._train_sub()
        self.white_box.model = self.substitute.eval()
        self.white_box.attack(data_loader)
        ds = TensorDataset(self.white_box.x, self.white_box.y)
        return super().attack(DataLoader(ds, batch_size=self.batch_size))


class FGSM(AdversarialAttack):
    def __init__(self, model, eps=0.3):
        super().__init__(model)
        self.eps = eps

    def _attack(self, x, y):
        x.requires_grad = True
        initial_pred = self.model(x)
        self.model.zero_grad()
        loss = F.cross_entropy(initial_pred, y)
        loss.backward()
        adv_x = x + self.eps * x.grad.sign()
        return self._clamp(adv_x)


class RandFGSM(FGSM):
    def __init__(self, model, eps=0.3, alpha=0.05):
        assert eps > alpha
        super().__init__(model, eps - alpha)
        self.alpha = alpha

    def _attack(self, x, y):
        return super()._attack(x + torch.randn_like(x).sign() * self.alpha, y)


class CW2(AdversarialAttack):
    def __init__(self, model):
        super().__init__(model)
        self.l2adv = L2Adversary()

    def _attack(self, x, y):
        return self.l2adv.__call__(self.model, x, y).to(x.device)


# borrowed from https://github.com/kkew3/pytorch-cw2/blob/master/cw.py
class L2Adversary:
    """
    The L2 attack adversary. To enforce the box constraint, the
    change-of-variable trick using tanh-space is adopted.
    The loss function to optimize:
    .. math::
        \\|\\delta\\|_2^2 + c \\cdot f(x + \\delta)
    where :math:`f` is defined as
    .. math::
        f(x') = \\max\\{0, (\\max_{i \\ne t}{Z(x')_i} - Z(x')_t) \\cdot \\tau + \\kappa\\}
    where :math:`\\tau` is :math:`+1` if the adversary performs targeted attack;
    otherwise it's :math:`-1`.
    Usage::
        attacker = L2Adversary()
        # inputs: a batch of input tensors
        # targets: a batch of attack targets
        # model: the model to attack
        advx = attacker(model, inputs, targets)
    The change-of-variable trick
    ++++++++++++++++++++++++++++
    Let :math:`a` be a proper affine transformation.
    1. Given input :math:`x` in image space, map :math:`x` to "tanh-space" by
    .. math:: \\hat{x} = \\tanh^{-1}(a^{-1}(x))
    2. Optimize an adversarial perturbation :math:`m` without constraint in the
    "tanh-space", yielding an adversarial example :math:`w = \\hat{x} + m`; and
    3. Map :math:`w` back to the same image space as the one where :math:`x`
    resides:
    .. math::
        x' = a(\\tanh(w))
    where :math:`x'` is the adversarial example, and :math:`\\delta = x' - x`
    is the adversarial perturbation.
    Since the composition of affine transformation and hyperbolic tangent is
    strictly monotonic, $\\delta = 0$ if and only if $m = 0$.
    Symbols used in docstring
    +++++++++++++++++++++++++
    - ``B``: the batch size
    - ``C``: the number of channels
    - ``H``: the height
    - ``W``: the width
    - ``M``: the number of classes
    """

    def __init__(self, confidence=0.0, c_range=(1e-3, 1e10),
                 search_steps=5, max_steps=500, abort_early=True,
                 box=(-1., 1.), optimizer_lr=1e-2):
        """
        :param confidence: the confidence constant, i.e. the $\\kappa$ in paper
        :type confidence: float
        :param c_range: the search range of the constant :math:`c`; should be a
               tuple of form (lower_bound, upper_bound)
        :type c_range: Tuple[float, float]
        :param search_steps: the number of steps to perform binary search of
               the constant :math:`c` over ``c_range``
        :type search_steps: int
        :param max_steps: the maximum number of optimization steps for each
               constant :math:`c`
        :type max_steps: int
        :param abort_early: ``True`` to abort early in process of searching for
               :math:`c` when the loss virtually stops increasing
        :type abort_early: bool
        :param box: a tuple of lower bound and upper bound of the box
        :type box: Tuple[float, float]
        :param optimizer_lr: the base learning rate of the Adam optimizer used
               over the adversarial perturbation in clipped space
        :type optimizer_lr: float
        :param init_rand: ``True`` to initialize perturbation to small Gaussian;
               False is consistent with the original paper, where the
               perturbation is initialized to zero
        Why to make ``box`` default to (-1., 1.) rather than (0., 1.)? TL;DR the
        domain of the problem in pytorch is [-1, 1] instead of [0, 1].
        According to Xiang Xu (samxucmu@gmail.com)::
        > The reason is that in pytorch a transformation is applied first
        > before getting the input from the data loader. So image in range [0,1]
        > will subtract some mean and divide by std. The normalized input image
        > will now be in range [-1,1]. For this implementation, clipping is
        > actually performed on the image after normalization, not on the
        > original image.
        Why to ``optimizer_lr`` default to 1e-2? The optimizer used in Carlini's
        code adopts 1e-2. In another pytorch implementation
        (https://github.com/rwightman/pytorch-nips2017-attack-example.git),
        though, the learning rate is set to 5e-4.
        """
        if len(c_range) != 2:
            raise TypeError('c_range ({}) should be of form '
                            'tuple([lower_bound, upper_bound])'
                            .format(c_range))
        if c_range[0] >= c_range[1]:
            raise ValueError('c_range lower bound ({}) is expected to be less '
                             'than c_range upper bound ({})'.format(*c_range))
        if len(box) != 2:
            raise TypeError('box ({}) should be of form '
                            'tuple([lower_bound, upper_bound])'
                            .format(box))
        if box[0] >= box[1]:
            raise ValueError('box lower bound ({}) is expected to be less than '
                             'box upper bound ({})'.format(*box))
        self.confidence = float(confidence)
        self.c_range = (float(c_range[0]), float(c_range[1]))
        self.binary_search_steps = search_steps
        self.max_steps = max_steps
        self.abort_early = abort_early
        self.ae_tol = 1e-4  # tolerance of early abort
        self.box = tuple(map(float, box))
        self.optimizer_lr = optimizer_lr

        # Since the larger the `scale_const` is, the more likely a successful
        # attack can be found, `self.repeat` guarantees at least attempt the
        # largest scale_const once. Moreover, since the optimal criterion is the
        # L2 norm of the attack, and the larger `scale_const` is, the larger
        # the L2 norm is, thus less optimal, the last attempt at the largest
        # `scale_const` won't ruin the optimum ever found.
        self.repeat = (self.binary_search_steps >= 10)

    def __call__(self, model, inputs, targets, num_classes=10):
        """
        Produce adversarial examples for ``inputs``.
        :param model: the model to attack
        :type model: nn.Module
        :param inputs: the original images tensor, of dimension [B x C x H x W].
               ``inputs`` can be on either CPU or GPU, but it will eventually be
               moved to the same device as the one the parameters of ``model``
               reside
        :type inputs: torch.FloatTensor
        :param targets: the original image labels, or the attack targets, of
               dimension [B]. If ``self.targeted`` is ``True``, then ``targets``
               is treated as the attack targets, otherwise the labels.
               ``targets`` can be on either CPU or GPU, but it will eventually
               be moved to the same device as the one the parameters of
               ``model`` reside
        :type targets: torch.LongTensor
        :param to_numpy: True to return an `np.ndarray`, otherwise,
               `torch.FloatTensor`
        :type to_numpy: bool
        :return: the adversarial examples on CPU, of dimension [B x C x H x W]
        """
        # sanity check
        assert isinstance(model, nn.Module)
        assert len(inputs.size()) == 4
        assert len(targets.size()) == 1

        # get a copy of targets in numpy before moving to GPU, used when doing
        # the binary search on `scale_const`
        targets_np = targets.clone().cpu().numpy()  # type: np.ndarray

        # the type annotations here are used only for type hinting and do
        # not indicate the actual type (cuda or cpu); same applies to all codes
        # below

        batch_size = inputs.size(0)  # type: int

        # `lower_bounds_np`, `upper_bounds_np` and `scale_consts_np` are used
        # for binary search of each `scale_const` in the batch. The element-wise
        # inquality holds: lower_bounds_np < scale_consts_np <= upper_bounds_np
        lower_bounds_np = np.zeros(batch_size)
        upper_bounds_np = np.ones(batch_size) * self.c_range[1]
        scale_consts_np = np.ones(batch_size) * self.c_range[0]

        # Optimal attack to be found.
        # The three "placeholders" are defined as:
        # - `o_best_l2`: the least L2 norms
        # - `o_best_l2_ppred`: the perturbed predictions made by the adversarial
        #    perturbations with the least L2 norms
        # - `o_best_advx`: the underlying adversarial example of
        #   `o_best_l2_ppred`
        o_best_l2 = np.ones(batch_size) * np.inf
        o_best_l2_ppred = -np.ones(batch_size)
        o_best_advx = inputs.clone().cpu().numpy()  # type: np.ndarray

        # convert `inputs` to tanh-space
        inputs_tanh = self._to_tanh_space(inputs)  # type: torch.FloatTensor
        inputs_tanh_var = Variable(inputs_tanh, requires_grad=False)

        # the one-hot encoding of `targets`
        targets_oh = torch.zeros(targets.size() + (num_classes,)).to(inputs.device)  # type: torch.FloatTensor
        targets_oh.scatter_(1, targets.unsqueeze(1), 1.0)
        targets_oh_var = Variable(targets_oh, requires_grad=False)

        # the perturbation variable to optimize.
        # `pert_tanh` is essentially the adversarial perturbation in tanh-space.
        # In Carlini's code it's denoted as `modifier`
        pert_tanh = torch.zeros(inputs.size()).to(inputs.device)  # type: torch.FloatTensor
        pert_tanh_var = Variable(pert_tanh, requires_grad=True)

        optimizer = optim.Adam([pert_tanh_var], lr=self.optimizer_lr)
        for sstep in range(self.binary_search_steps):
            if self.repeat and sstep == self.binary_search_steps - 1:
                scale_consts_np = upper_bounds_np
            scale_consts = torch.from_numpy(np.copy(scale_consts_np)).float().to(
                inputs.device)  # type: torch.FloatTensor
            scale_consts_var = Variable(scale_consts, requires_grad=False)

            # the minimum L2 norms of perturbations found during optimization
            best_l2 = np.ones(batch_size) * np.inf
            # the perturbed predictions corresponding to `best_l2`, to be used
            # in binary search of `scale_const`
            best_l2_ppred = -np.ones(batch_size)
            # previous (summed) batch loss, to be used in early stopping policy
            prev_batch_loss = np.inf  # type: float
            for optim_step in range(self.max_steps):
                batch_loss, pert_norms_np, pert_outputs_np, advxs_np = \
                    self._optimize(model, optimizer, inputs_tanh_var,
                                   pert_tanh_var, targets_oh_var,
                                   scale_consts_var)

                if self.abort_early and not optim_step % (self.max_steps // 10):
                    if batch_loss > prev_batch_loss * (1 - self.ae_tol):
                        break
                    prev_batch_loss = batch_loss

                # update best attack found during optimization
                pert_predictions_np = np.argmax(pert_outputs_np, axis=1)
                comp_pert_predictions_np = np.argmax(
                    self._compensate_confidence(pert_outputs_np, targets_np),
                    axis=1)
                for i in range(batch_size):
                    l2 = pert_norms_np[i]
                    cppred = comp_pert_predictions_np[i]
                    ppred = pert_predictions_np[i]
                    tlabel = targets_np[i]
                    ax = advxs_np[i]
                    if self._attack_successful(cppred, tlabel):
                        assert cppred == ppred
                        if l2 < best_l2[i]:
                            best_l2[i] = l2
                            best_l2_ppred[i] = ppred
                        if l2 < o_best_l2[i]:
                            o_best_l2[i] = l2
                            o_best_l2_ppred[i] = ppred
                            o_best_advx[i] = ax

            # binary search of `scale_const`
            for i in range(batch_size):
                tlabel = targets_np[i]
                assert best_l2_ppred[i] == -1 or \
                       self._attack_successful(best_l2_ppred[i], tlabel)
                assert o_best_l2_ppred[i] == -1 or \
                       self._attack_successful(o_best_l2_ppred[i], tlabel)
                if best_l2_ppred[i] != -1:
                    # successful; attempt to lower `scale_const` by halving it
                    if scale_consts_np[i] < upper_bounds_np[i]:
                        upper_bounds_np[i] = scale_consts_np[i]
                    # `upper_bounds_np[i] == c_range[1]` implies no solution
                    # found, i.e. upper_bounds_np[i] has never been updated by
                    # scale_consts_np[i] until
                    # `scale_consts_np[i] > 0.1 * c_range[1]`
                    if upper_bounds_np[i] < self.c_range[1] * 0.1:
                        scale_consts_np[i] = (lower_bounds_np[i] + upper_bounds_np[i]) / 2
                else:
                    # failure; multiply `scale_const` by ten if no solution
                    # found; otherwise do binary search
                    if scale_consts_np[i] > lower_bounds_np[i]:
                        lower_bounds_np[i] = scale_consts_np[i]
                    if upper_bounds_np[i] < self.c_range[1] * 0.1:
                        scale_consts_np[i] = (lower_bounds_np[i] + upper_bounds_np[i]) / 2
                    else:
                        scale_consts_np[i] *= 10
        return torch.from_numpy(o_best_advx).float()

    def _optimize(self, model, optimizer, inputs_tanh_var, pert_tanh_var,
                  targets_oh_var, c_var):
        """
        Optimize for one step.
        :param model: the model to attack
        :type model: nn.Module
        :param optimizer: the Adam optimizer to optimize ``modifier_var``
        :type optimizer: optim.Adam
        :param inputs_tanh_var: the input images in tanh-space
        :type inputs_tanh_var: Variable
        :param pert_tanh_var: the perturbation to optimize in tanh-space,
               ``pert_tanh_var.requires_grad`` flag must be set to True
        :type pert_tanh_var: Variable
        :param targets_oh_var: the one-hot encoded target tensor (the attack
               targets if self.targeted else image labels)
        :type targets_oh_var: Variable
        :param c_var: the constant :math:`c` for each perturbation of a batch,
               a Variable of FloatTensor of dimension [B]
        :type c_var: Variable
        :return: the batch loss, squared L2-norm of adversarial perturbations
                 (of dimension [B]), the perturbed activations (of dimension
                 [B]), the adversarial examples (of dimension [B x C x H x W])
        """
        # the adversarial examples in the image space
        # of dimension [B x C x H x W]
        advxs_var = self._from_tanh_space(inputs_tanh_var + pert_tanh_var)  # type: Variable
        # the perturbed activation before softmax
        pert_outputs_var = model(advxs_var)  # type: Variable
        # the original inputs
        inputs_var = self._from_tanh_space(inputs_tanh_var)  # type: Variable

        perts_norm_var = torch.pow(advxs_var - inputs_var, 2)
        perts_norm_var = torch.sum(perts_norm_var.view(
            perts_norm_var.size(0), -1), 1)

        # In Carlini's code, `target_activ_var` is called `real`.
        # It should be a Variable of tensor of dimension [B], such that the
        # `target_activ_var[i]` is the final activation (right before softmax)
        # of the $t$th class, where $t$ is the attack target or the image label
        #
        # noinspection PyArgumentList
        target_activ_var = torch.sum(targets_oh_var * pert_outputs_var, 1)
        inf = 1e4  # sadly pytorch does not work with np.inf;
        # 1e4 is also used in Carlini's code
        # In Carlini's code, `maxother_activ_var` is called `other`.
        # It should be a Variable of tensor of dimension [B], such that the
        # `maxother_activ_var[i]` is the maximum final activation of all classes
        # other than class $t$, where $t$ is the attack target or the image
        # label.
        #
        # The assertion here ensures (sufficiently yet not necessarily) the
        # assumption behind the trick to get `maxother_activ_var` holds, that
        # $\max_{i \ne t}{o_i} \ge -\text{_inf}$, where $t$ is the target and
        # $o_i$ the $i$th element along axis=1 of `pert_outputs_var`.
        #
        # noinspection PyArgumentList
        assert (pert_outputs_var.max(1)[0] >= -inf).all(), 'assumption failed'
        # noinspection PyArgumentList
        maxother_activ_var = torch.max(((1 - targets_oh_var) * pert_outputs_var
                                        - targets_oh_var * inf), 1)[0]

        # Compute $f(x')$, where $x'$ is the adversarial example in image space.
        # The result `f_var` should be of dimension [B]
        # if not targeted, optimize to make `maxother_activ_var` larger than
        # `target_activ_var` (the ground truth image labels) by
        # `self.confidence`
        #
        # noinspection PyArgumentList
        f_var = torch.clamp(target_activ_var - maxother_activ_var + self.confidence, min=0.0)
        # the total loss of current batch, should be of dimension [1]
        batch_loss_var = torch.sum(perts_norm_var + c_var * f_var)  # type: Variable

        # Do optimization for one step
        optimizer.zero_grad()
        batch_loss_var.backward()
        optimizer.step()

        # Make some records in python/numpy on CPU
        batch_loss = batch_loss_var.item()  # type: float
        pert_norms_np = L2Adversary._var2numpy(perts_norm_var)
        pert_outputs_np = L2Adversary._var2numpy(pert_outputs_var)
        advxs_np = L2Adversary._var2numpy(advxs_var)
        return batch_loss, pert_norms_np, pert_outputs_np, advxs_np

    def _attack_successful(self, prediction, target):
        """
        See whether the underlying attack is successful.
        :param prediction: the prediction of the model on an input
        :type prediction: int
        :param target: either the attack target or the ground-truth image label
        :type target: int
        :return: ``True`` if the attack is successful
        :rtype: bool
        """
        return prediction != target

    # noinspection PyUnresolvedReferences
    def _compensate_confidence(self, outputs, targets):
        """
        Compensate for ``self.confidence`` and returns a new weighted sum
        vector.
        :param outputs: the weighted sum right before the last layer softmax
               normalization, of dimension [B x M]
        :type outputs: np.ndarray
        :param targets: either the attack targets or the real image labels,
               depending on whether or not ``self.targeted``, of dimension [B]
        :type targets: np.ndarray
        :return: the compensated weighted sum of dimension [B x M]
        :rtype: np.ndarray
        """
        outputs_comp = np.copy(outputs)
        rng = np.arange(targets.shape[0])
        # for each image $i$:
        # if not targeted, `max(outputs[i, ~target_onehot]` should be larger
        # than `outputs[i, target_onehot]` (the ground truth image labels)
        # by `self.confidence`
        outputs_comp[rng, targets] += self.confidence
        return outputs_comp

    def _to_tanh_space(self, x):
        """
        Convert a batch of tensors to tanh-space.
        :param x: the batch of tensors, of dimension [B x C x H x W]
        :return: the batch of tensors in tanh-space, of the same dimension
        """
        return L2Adversary.to_tanh_space(x, self.box)

    def _from_tanh_space(self, x):
        """
        Convert a batch of tensors from tanh-space to input space.
        :param x: the batch of tensors, of dimension [B x C x H x W]
        :return: the batch of tensors in tanh-space, of the same dimension;
                 the returned tensor is on the same device as ``x``
        """
        return L2Adversary.from_tanh_space(x, self.box)

    @staticmethod
    def _var2numpy(var):
        return var.data.cpu().numpy()

    @staticmethod
    def atanh(x, eps=1e-6):
        x = x * (1 - eps)
        return 0.5 * torch.log((1.0 + x) / (1.0 - x))

    @staticmethod
    def to_tanh_space(x, box):
        _box_mul = (box[1] - box[0]) * 0.5
        _box_plus = (box[1] + box[0]) * 0.5
        return L2Adversary.atanh((x - _box_plus) / _box_mul)

    @staticmethod
    def from_tanh_space(x, box):
        _box_mul = (box[1] - box[0]) * 0.5
        _box_plus = (box[1] + box[0]) * 0.5
        return torch.tanh(x) * _box_mul + _box_plus


def main():
    setup_run(given_seed=1235)
    test_data_loader_all = torch.utils.data.DataLoader(get_mnist_ds(32, False), batch_size=32, shuffle=True)
    xs, ys = [], []
    for i, (x, y) in enumerate(test_data_loader_all):
        xs.append(x)
        ys.append(y)
        if i == 5:
            holdout = (torch.cat(xs, dim=0), torch.cat(ys, dim=0))
    mkdir('./saved_attacks/')

    test_data_loader = DataLoader(TensorDataset(torch.cat(xs, dim=0), torch.cat(ys, dim=0)), batch_size=32)
    for classifier_name in ['cnn', 'mlp']:
        classifier = to(CNNClassifier() if classifier_name == 'cnn' else FCClassifier())
        classifier.load_state_dict(torch.load('./trained_models/mnist_' + classifier_name + '.pt'))
        classifier.eval()
        print('*' * 10 + classifier_name + '*' * 10)

        attacker = NoAttack(classifier)
        print('\tDefault.acc', attacker.attack(test_data_loader))
        attacker.save('./saved_attacks/' + classifier_name + '_default.pth')

        attacker = FGSM(classifier, eps=0.3)
        print('\tFGSM(0.3).acc', attacker.attack(test_data_loader))
        attacker.save('./saved_attacks/' + classifier_name + '_fgsm_0.3.pth')

        attacker = FGSM(classifier, eps=0.15)
        print('\tFGSM(0.15).acc', attacker.attack(test_data_loader))
        attacker.save('./saved_attacks/' + classifier_name + '_fgsm_0.15.pth')

        attacker = RandFGSM(classifier, eps=0.3, alpha=0.05)
        print('RandFGSM.acc', attacker.attack(test_data_loader))
        attacker.save('./saved_attacks/' + classifier_name + '_rfgsm_0.3.pth')

        attacker = CW2(classifier)
        print('CW2.acc', attacker.attack(test_data_loader))
        attacker.save('./saved_attacks/' + classifier_name + '_cw2.pth')

        # plt.imshow(np.transpose(vutils.make_grid(attacker.x[:32], range=(-1.0, 1.0), padding=5), (1, 2, 0)))
        # print(attacker.y[:32])
        # print(classifier(attacker.x[:32]).argmax(dim=1))
        # plt.show()

        for sub_name, sub in zip(('cnn', 'mlp'), (CNNClassifier, FCClassifier)):
            sub = to(sub())
            white_attacker = FGSM(sub, eps=0.3)
            attacker = BlackBoxAttack(classifier, sub, holdout, white_attacker)
            print('BlackBox+FGSM_{}.acc'.format(sub_name), attacker.attack(test_data_loader))
            attacker.save('./saved_attacks/{}_sub_{}.pth'.format(classifier_name, sub_name))


if __name__ == '__main__':
    main()
