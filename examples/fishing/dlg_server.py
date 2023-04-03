"""
An honest-but-curious federated learning server which can
analyze periodic gradients from certain clients to
perform the gradient leakage attacks and
reconstruct the training data of the victim clients.


References:

Zhu et al., "Deep Leakage from Gradients,"
in Advances in Neural Information Processing Systems 2019.

https://papers.nips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf

Geiping et al., "Inverting Gradients - How easy is it to break privacy in federated learning?"
in Advances in Neural Information Processing Systems 2020.

https://proceedings.neurips.cc/paper/2020/file/c4ede56bbd98819ae6112b20ac6bf145-Paper.pdf
"""
import asyncio
import logging
import math
import numbers
import os
import shutil
import time
from collections import OrderedDict, defaultdict
from copy import deepcopy

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from plato.config import Config
from plato.servers import fedavg
from plato.utils import csv_processor
from torchvision import transforms

from defense.GradDefense.compensate import denoise
from utils.evaluations import get_evaluation_dict
from utils.modules import PatchedModule
from utils.utils import cross_entropy_for_onehot
from utils.utils import total_variation as TV

cross_entropy = torch.nn.CrossEntropyLoss()
tt = transforms.ToPILImage()

partition_size = Config().data.partition_size
epochs = Config().trainer.epochs
batch_size = Config().trainer.batch_size
num_iters = Config().algorithm.num_iters
log_interval = Config().algorithm.log_interval
dlg_result_path = f"{Config().params['result_path']}/{os.getpid()}"
dlg_result_headers = [
    "Iteration",
    "Loss",
    "Average MSE",
    "Average LPIPS",
    "Average PSNR (dB)",
    "Average SSIM",
    "Average Library SSIM",
]


def wrap_indices(indices):
    if isinstance(indices, numbers.Number):
        return [indices]
    else:
        return list(indices)


class Server(fedavg.Server):
    """An honest-but-curious federated learning server with gradient leakage attack."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        self.attack_method = None
        self.share_gradients = True
        if (
            hasattr(Config().algorithm, "share_gradients")
            and not Config().algorithm.share_gradients
        ):
            self.share_gradients = False
        self.match_weights = False
        if (
            hasattr(Config().algorithm, "match_weights")
            and Config().algorithm.match_weights
        ):
            self.match_weights = True
        self.use_updates = True
        if (
            hasattr(Config().algorithm, "use_updates")
            and not Config().algorithm.use_updates
        ):
            self.use_updates = False
        self.defense_method = "no"
        if hasattr(Config().algorithm, "defense"):
            if Config().algorithm.defense in [
                "GradDefense",
                "Soteria",
                "GC",
                "DP",
                "Outpost",
            ]:
                self.defense_method = Config().algorithm.defense
            else:
                logging.info("No Defense Applied")
        self.best_mse = math.inf
        # Save trail 1 as the best as default when results are all bad
        self.best_trial = 1

    def choose_clients(self, client_pool, clients_count):
        """Choose a single client to query for the fishing optimization attack
        (since selecting multiple would be wasteful of compute."""
        if Config().algorithm.attack_method != "fishing":
            return super().choose_clients(client_pool, clients_count)
        # Arbitrarily select the client with the minimum id in the client_pool,
        # which would ideally be the same client each time. This also implicitly
        # ignores the clients_count parameter.
        return [min(client_pool)]

    @torch.no_grad()
    def customize_server_payload(self, payload):
        """Taken from breaching/cases/servers.py:reconfigure_for_class_attack"""
        target_cls_idx = Config().server.target_cls_idx     # 0
        class_multiplier = Config().server.class_multiplier     # 0.5
        bias_multiplier = Config().server.bias_multiplier   # 1000

        target_classes = [target_cls_idx]
        cls_to_obtain = wrap_indices(target_classes)

        *_, l_w, l_b = self.algorithm.model.fc.parameters()

        # linear weight
        masked_weight = torch.zeros_like(l_w)
        masked_weight[cls_to_obtain] = class_multiplier
        l_w.copy_(masked_weight)

        # linear bias
        masked_bias = torch.ones_like(l_b) * bias_multiplier
        masked_bias[cls_to_obtain] = l_b[cls_to_obtain]
        l_b.copy_(masked_bias)

        # Re-extract the payload from the self.model.parameters()
        return self.algorithm.extract_weights()

    def weights_received(self, weights_received):
        """
        Perform attack in attack around after the updated weights have been aggregated.
        """
        if (
            self.current_round == Config().algorithm.attack_round
            and Config().algorithm.attack_method in ["DLG", "iDLG", "csDLG"]
        ):
            self.attack_method = Config().algorithm.attack_method
            self._deep_leakage_from_gradients(weights_received)
        elif (
            self.current_round == Config().algorithm.attack_round
            and Config().algorithm.attack_method == "fishing"
        ):
            self._setup_reconstruction()
            baseline_weights = self.algorithm.extract_weights()
            deltas_received = self.algorithm.compute_weight_deltas(
                baseline_weights, weights_received
            )
            report = self.updates[Config().algorithm.victim_client].report
            gt_labels = report.gt_labels
            self.reconstruct(
                server_payload=[
                    dict(
                        parameters=list(baseline_weights.values()),
                        buffers=None,
                        metadata=None,
                    )
                ],
                shared_data=[
                    dict(
                        gradients=list(deltas_received[0].values()),
                        buffers=None,
                        metadata=dict(
                            num_data_points=1,
                            labels=torch.Tensor([0]).long(),
                            num_users=1,
                            local_hyperparams=None,
                        ),
                    )
                ],
                server_secrets=dict(
                    ClassAttack=dict(
                        num_data=1,
                        target_indx=[0],
                        true_num_data=256,
                        all_labels=gt_labels,
                    )
                ),
                initial_data=None,
                dryrun=False,
            )

        return weights_received

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging with optional compensation."""
        # Extract the total number of samples
        self.total_samples = sum([update.report.num_samples for update in updates])

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        _scale = 0
        for i, update in enumerate(deltas_received):
            report = updates[i].report
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

            if self.defense_method == "GradDefense":
                _scale += (
                    len(deltas_received)
                    * Config().algorithm.perturb_slices_num
                    / Config().algorithm.slices_num
                    * (Config().algorithm.scale ** 2)
                    * (num_samples / self.total_samples)
                )

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        if self.defense_method == "GradDefense":
            update_perturbed = []
            for name, delta in avg_update.items():
                update_perturbed.append(delta)
            update_compensated = denoise(
                gradients=update_perturbed,
                scale=math.sqrt(_scale),
                Q=Config().algorithm.Q,
            )
            for i, name in enumerate(avg_update.keys()):
                avg_update[name] = update_compensated[i]

        return avg_update

    def _deep_leakage_from_gradients(self, weights_received):
        """Analyze periodic gradients from certain clients."""
        # Process data from the victim client
        # The ground truth should be used only for evaluation
        baseline_weights = self.algorithm.extract_weights()
        deltas_received = self.algorithm.compute_weight_deltas(
            baseline_weights, weights_received
        )
        target_weights = self.updates[Config().algorithm.victim_client].payload
        if not self.share_gradients and self.match_weights and self.use_updates:
            target_weights = deltas_received[Config().algorithm.victim_client]
        report = self.updates[Config().algorithm.victim_client].report
        gt_data, gt_labels, target_grad = (
            report.gt_data,
            report.gt_labels,
            report.target_grad,
        )

        # Assume the reconstructed data shape is known, which can be also derived from the target dataset
        num_images = partition_size
        data_size = [num_images, gt_data.shape[1], gt_data.shape[2], gt_data.shape[3]]
        gt_data_plot = gt_data.detach().clone()
        gt_data_plot = gt_data_plot.permute(0, 2, 3, 1)
        gt_result_path = f"{dlg_result_path}/ground_truth.pdf"
        self._make_plot(num_images, gt_data_plot, gt_labels, gt_result_path)

        # The number of restarts
        trials = 1
        if hasattr(Config().algorithm, "trials"):
            trials = Config().algorithm.trials

        logging.info("Running %d Trials", trials)

        if not self.share_gradients and not self.match_weights:
            # Obtain the local updates from clients
            target_grad = []
            for delta in deltas_received[Config().algorithm.victim_client].values():
                target_grad.append(-delta / Config().parameters.optimizer.lr)

            total_local_steps = epochs * math.ceil(partition_size / batch_size)
            target_grad = [x / total_local_steps for x in target_grad]

        # Generate dummy items and initialize optimizer
        torch.manual_seed(Config().algorithm.random_seed)

        for trial_number in range(trials):
            self.run_trial(
                trial_number,
                num_images,
                data_size,
                target_weights,
                target_grad,
                gt_data,
                gt_labels,
            )

        self._save_best()

    def run_trial(
        self,
        trial_number,
        num_images,
        data_size,
        target_weights,
        target_grad,
        gt_data,
        gt_labels,
    ):
        """Run the attack for one trial."""
        logging.info("Starting Attack Number %d", (trial_number + 1))

        trial_result_path = f"{dlg_result_path}/t{trial_number + 1}"
        trial_csv_file = f"{trial_result_path}/evals.csv"

        # Initialize the csv file
        csv_processor.initialize_csv(
            trial_csv_file, dlg_result_headers, trial_result_path
        )

        dummy_data = torch.randn(data_size).to(Config().device()).requires_grad_(True)

        dummy_labels = (
            torch.randn((num_images, Config().trainer.num_classes))
            .to(Config().device())
            .requires_grad_(True)
        )

        if self.attack_method == "DLG":
            match_optimizer = torch.optim.LBFGS(
                [dummy_data, dummy_labels], lr=Config().algorithm.lr
            )
            labels_ = dummy_labels
            for i in range(num_images):
                logging.info(
                    "[%s Gradient Leakage Attack %d with %s defense...] Dummy label is %d.",
                    self.attack_method,
                    trial_number,
                    self.defense_method,
                    torch.argmax(dummy_labels[i], dim=-1).item(),
                )
        elif self.attack_method == "iDLG":
            match_optimizer = torch.optim.LBFGS(
                [
                    dummy_data,
                ],
                lr=Config().algorithm.lr,
            )
            # Estimate the gt label
            est_labels = (
                torch.argmin(torch.sum(target_grad[-2], dim=-1), dim=-1)
                .detach()
                .reshape((1,))
                .requires_grad_(False)
            )
            labels_ = est_labels
            for i in range(num_images):
                logging.info(
                    "[%s Gradient Leakage Attack %d with %s defense...] Estimated label is %d.",
                    self.attack_method,
                    trial_number,
                    self.defense_method,
                    est_labels.item(),
                )
        elif self.attack_method == "csDLG":
            match_optimizer = torch.optim.LBFGS(
                [
                    dummy_data,
                ],
                lr=Config().algorithm.lr,
            )
            labels_ = gt_labels
            for i in range(num_images):
                logging.info(
                    "[%s Gradient Leakage Attack %d with %s defense...] Known label is %d.",
                    self.attack_method,
                    trial_number,
                    self.defense_method,
                    torch.argmax(gt_labels[i], dim=-1).item(),
                )
        elif self.attack_method == "fishing":
            match_optimizer = torch.optim.LBFGS(
                [
                    dummy_data,
                ],
                lr=Config().algorithm.lr,
            )
            labels_ = gt_labels
            for i in range(num_images):
                logging.info(
                    "[%s Gradient Leakage Attack %d with %s defense...] Known label is %d.",
                    self.attack_method,
                    trial_number,
                    self.defense_method,
                    torch.argmax(gt_labels[i], dim=-1).item(),
                )

        history, losses, mses, lpipss, psnrs, ssims, library_ssims = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        avg_mses, avg_lpips, avg_psnr, avg_ssim, avg_library_ssim = [], [], [], [], []

        # Conduct gradients/weights/updates matching
        if not self.share_gradients and self.match_weights:
            model = deepcopy(self.trainer.model.to(Config().device()))
            closure = self._weight_closure(
                match_optimizer, dummy_data, labels_, target_weights, model
            )
        else:
            closure = self._gradient_closure(
                match_optimizer, dummy_data, labels_, target_grad
            )

        for iters in range(num_iters):
            match_optimizer.step(closure)
            current_loss = closure().item()
            losses.append(current_loss)

            if math.isnan(current_loss):
                logging.info("Not a number, ending attack")
                # should make these lines into a function to prevent repetition, but not sure how to
                # without having too many parameters
                eval_dict = get_evaluation_dict(dummy_data, gt_data, num_images)
                mses.append(eval_dict["mses"])
                lpipss.append(eval_dict["lpipss"])
                psnrs.append(eval_dict["psnrs"])
                ssims.append(eval_dict["ssims"])
                library_ssims.append(eval_dict["library_ssims"])
                avg_mses.append(eval_dict["avg_mses"])
                avg_lpips.append(eval_dict["avg_lpips"])
                avg_psnr.append(eval_dict["avg_psnr"])
                avg_ssim.append(eval_dict["avg_ssim"])
                avg_library_ssim.append(eval_dict["avg_library_ssim"])

                new_row = [
                    iters,
                    round(losses[-1], 8),
                    round(avg_mses[-1], 8),
                    round(avg_lpips[-1], 8),
                    round(avg_psnr[-1], 4),
                    round(avg_ssim[-1], 3),
                    round(avg_library_ssim[-1], 3),
                ]
                csv_processor.write_csv(trial_csv_file, new_row)
                break

            if iters % log_interval == 0:
                # Finding evaluation metrics
                # should make these lines into a function to prevent repetition, but not sure how to
                # without having too many parameters
                eval_dict = get_evaluation_dict(dummy_data, gt_data, num_images)
                mses.append(eval_dict["mses"])
                lpipss.append(eval_dict["lpipss"])
                psnrs.append(eval_dict["psnrs"])
                ssims.append(eval_dict["ssims"])
                library_ssims.append(eval_dict["library_ssims"])
                avg_mses.append(eval_dict["avg_mses"])
                avg_lpips.append(eval_dict["avg_lpips"])
                avg_psnr.append(eval_dict["avg_psnr"])
                avg_ssim.append(eval_dict["avg_ssim"])
                avg_library_ssim.append(eval_dict["avg_library_ssim"])

                logging.info(
                    "[%s Gradient Leakage Attack %d with %s defense...] Iter %d: Loss = %.10f, avg MSE = %.8f, avg LPIPS = %.8f, avg PSNR = %.4f dB, avg SSIM = %.3f, avg library SSIM = %.3f",
                    self.attack_method,
                    (trial_number + 1),
                    self.defense_method,
                    iters,
                    losses[-1],
                    avg_mses[-1],
                    avg_lpips[-1],
                    avg_psnr[-1],
                    avg_ssim[-1],
                    avg_library_ssim[-1],
                )

                if self.attack_method == "DLG":
                    history.append(
                        [
                            [
                                dummy_data[i].cpu().permute(1, 2, 0).detach().clone(),
                                torch.argmax(dummy_labels[i], dim=-1).item(),
                                dummy_data[i],
                            ]
                            for i in range(num_images)
                        ]
                    )
                elif self.attack_method == "iDLG":
                    history.append(
                        [
                            [
                                dummy_data[i].cpu().permute(1, 2, 0).detach().clone(),
                                est_labels[i].item(),
                                dummy_data[i],
                            ]
                            for i in range(num_images)
                        ]
                    )
                elif self.attack_method == "csDLG":
                    history.append(
                        [
                            [
                                dummy_data[i].cpu().permute(1, 2, 0).detach().clone(),
                                torch.argmax(gt_labels[i], dim=-1),
                                dummy_data[i],
                            ]
                            for i in range(num_images)
                        ]
                    )
                elif self.attack_method == "fishing":
                    # TODO(dchu) make sure you call this - this is how you
                    history.append(
                        [
                            [
                                dummy_data[i].cpu().permute(
                                    1, 2, 0
                                    ).detach().clone(),
                                torch.argmax(gt_labels[i], dim=-1),
                                dummy_data[i],
                            ]
                            for i in range(num_images)
                        ]
                    )

                new_row = [
                    iters,
                    round(losses[-1], 8),
                    round(avg_mses[-1], 8),
                    round(avg_lpips[-1], 8),
                    round(avg_psnr[-1], 4),
                    round(avg_ssim[-1], 3),
                    round(avg_library_ssim[-1], 3),
                ]
                csv_processor.write_csv(trial_csv_file, new_row)

        if self.best_mse > avg_mses[-1]:
            self.best_mse = avg_mses[-1]
            self.best_trial = (
                trial_number + 1
            )  # the +1 is because we index from 1 and not 0

        reconstructed_path = f"{trial_result_path}/reconstruction_iterations.png"
        self._plot_reconstructed(num_images, history, reconstructed_path)
        final_tensor = torch.stack([history[-1][i][0] for i in range(num_images)])
        final_result_path = f"{trial_result_path}/final_attack_result.pdf"
        self._make_plot(num_images, final_tensor, None, final_result_path)

        # Save the tensors into a .pt file
        tensor_file_path = f"{trial_result_path}/tensors.pt"
        result = {
            i * log_interval: {j: history[i][j][0] for j in range(num_images)}
            for i in range(len(history))
        }
        torch.save(result, tensor_file_path)

        logging.info("Attack %d complete", (trial_number + 1))

    def _gradient_closure(self, match_optimizer, dummy_data, labels, target_grad):
        """Take a step to match the gradients."""

        def closure():
            match_optimizer.zero_grad()
            self.trainer.model.to(Config().device())
            try:
                dummy_pred, _ = self.trainer.model(dummy_data)
            except:
                dummy_pred = self.trainer.model(dummy_data)

            if self.attack_method == "DLG":
                dummy_onehot_label = F.softmax(labels, dim=-1)
                dummy_loss = cross_entropy_for_onehot(dummy_pred, dummy_onehot_label)
            elif self.attack_method in ["iDLG", "csDLG", "fishing"]:
                dummy_loss = cross_entropy(dummy_pred, labels)

            dummy_grad = torch.autograd.grad(
                dummy_loss, self.trainer.model.parameters(), create_graph=True
            )

            rec_loss = self._reconstruction_costs([dummy_grad], target_grad)
            if (
                hasattr(Config().algorithm, "total_variation")
                and Config().algorithm.total_variation > 0
            ):
                rec_loss += Config().algorithm.total_variation * TV(dummy_data)
            rec_loss.backward()
            return rec_loss

        return closure

    def _weight_closure(
        self, match_optimizer, dummy_data, labels, target_weights, model
    ):
        """Take a step to match the weights."""

        def closure():
            match_optimizer.zero_grad()
            dummy_weight = self._loss_steps(dummy_data, labels, model)

            rec_loss = self._reconstruction_costs(
                [dummy_weight], list(target_weights.values())
            )
            if (
                hasattr(Config().algorithm, "total_variation")
                and Config().algorithm.total_variation > 0
            ):
                rec_loss += Config().algorithm.total_variation * TV(dummy_data)
            rec_loss.backward()
            return rec_loss

        return closure

    def _loss_steps(self, dummy_data, labels, model):
        """Take a few gradient descent steps to fit the model to the given input."""
        patched_model = PatchedModule(model)
        if self.use_updates:
            patched_model_origin = deepcopy(patched_model)

        # TODO: optional parameters: lr_schedule, create_graph...
        for epoch in range(epochs):
            if batch_size == 1:
                dummy_pred = patched_model(dummy_data, patched_model.parameters)
                labels_ = labels
            else:
                idx = epoch % (dummy_data.shape[0] // batch_size)
                dummy_pred = patched_model(
                    dummy_data[idx * batch_size : (idx + 1) * batch_size],
                    patched_model.parameters,
                )
                labels_ = labels[idx * batch_size : (idx + 1) * batch_size]

            loss = cross_entropy(dummy_pred, labels_).sum()

            grad = torch.autograd.grad(
                loss,
                patched_model.parameters.values(),
                retain_graph=True,
                create_graph=True,
                only_inputs=True,
            )

            patched_model.parameters = OrderedDict(
                (name, param - Config().parameters.optimizer.lr * grad_part)
                for ((name, param), grad_part) in zip(
                    patched_model.parameters.items(), grad
                )
            )
        if self.use_updates:
            patched_model.parameters = OrderedDict(
                (name, param - param_origin)
                for ((name, param), (name_origin, param_origin)) in zip(
                    patched_model.parameters.items(),
                    patched_model_origin.parameters.items(),
                )
            )
        return list(patched_model.parameters.values())

    def _save_best(self):
        src_folder = f"{dlg_result_path}/t{self.best_trial}"
        dst_folder = f"{dlg_result_path}/best(t{self.best_trial})"

        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        for file_name in os.listdir(src_folder):
            src = os.path.join(src_folder, file_name)
            dst = os.path.join(dst_folder, file_name)
            if os.path.isfile(src):
                shutil.copy(src, dst)

    @staticmethod
    def _reconstruction_costs(dummy, target):
        indices = torch.arange(len(target))
        cost_fn = Config().algorithm.cost_fn

        total_costs = 0
        for trial in dummy:
            pnorm = [0, 0]
            costs = 0
            for i in indices:
                if cost_fn == "l2":
                    costs += ((trial[i] - target[i]).pow(2)).sum()
                elif cost_fn == "l1":
                    costs += ((trial[i] - target[i]).abs()).sum()
                elif cost_fn == "max":
                    costs += ((trial[i] - target[i]).abs()).max()
                elif cost_fn == "sim":
                    costs -= (trial[i] * target[i]).sum()
                    pnorm[0] += trial[i].pow(2).sum()
                    pnorm[1] += target[i].pow(2).sum()
                elif cost_fn == "simlocal":
                    costs += 1 - torch.nn.functional.cosine_similarity(
                        trial[i].flatten(), target[i].flatten(), 0, 1e-10
                    )
            if cost_fn == "sim":
                costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

            # Accumulate final costs
            total_costs += costs

        return total_costs / len(dummy)

    @staticmethod
    def _make_plot(num_images, image_data, image_labels, path):
        """Plot ground truth data."""

        if not os.path.exists(dlg_result_path):
            os.makedirs(dlg_result_path)

        if hasattr(Config().results, "rows"):
            rows = Config().results.rows
            if hasattr(Config().results, "cols"):
                cols = Config().results.cols
            else:
                cols = math.ceil(num_images / rows)
        elif hasattr(Config().results, "cols"):
            cols = Config().results.cols
            rows = math.ceil(num_images / cols)
        else:
            # make the image wider by default
            # if you want the image to be taller by default then
            # switch the assignment statement for rows and cols variables
            logging.info("Using default dimensions for images")
            cols = math.ceil(math.sqrt(num_images))
            rows = math.ceil(num_images / cols)

        if (rows * cols) < num_images:
            logging.info("Row and column provided for plotting images is too small")
            logging.info("Using default dimensions for images")
            cols = math.ceil(math.sqrt(num_images))
            rows = math.ceil(num_images / cols)

        scale_factor = rows + cols
        image_height = 16 * rows / scale_factor
        image_width = 16 * cols / scale_factor
        product = rows * cols

        if num_images == 1:
            gt_figure = plt.figure(figsize=(8, 8))
            plt.imshow(image_data[0])
            plt.axis("off")
        else:
            fig, axes = plt.subplots(
                nrows=rows, ncols=cols, figsize=(image_width, image_height)
            )
            for i, title in enumerate(image_data):
                axes.ravel()[i].imshow(image_data[i].cpu())
                axes.ravel()[i].set_axis_off()
            for i in range(num_images, product):
                axes.ravel()[i].set_axis_off()

        plt.tight_layout()
        plt.savefig(path)

    @staticmethod
    def _plot_reconstructed(num_images, history, reconstructed_result_path):
        """Plot the reconstructed data."""
        for i in range(num_images):
            logging.info("Reconstructed label is %d.", history[-1][i][1])

        fig = plt.figure(figsize=(12, 8))
        rows = math.ceil(len(history) / 2)
        outer = gridspec.GridSpec(rows, 2, wspace=0.2, hspace=0.2)

        for i, item in enumerate(history):
            inner = gridspec.GridSpecFromSubplotSpec(
                1, num_images, subplot_spec=outer[i]
            )
            outerplot = plt.Subplot(fig, outer[i])
            outerplot.set_title("Iter=%d" % (i * log_interval))
            outerplot.axis("off")
            fig.add_subplot(outerplot)

            for j in range(num_images):
                innerplot = plt.Subplot(fig, inner[j])
                innerplot.imshow(history[i][j][0])
                innerplot.axis("off")
                fig.add_subplot(innerplot)
        fig.savefig(reconstructed_result_path)

    ############################################################################
    # BREACHING CODE
    ############################################################################

    # TODO(dchu) call this method!
    def _setup_reconstruction(self):
        self.setup = dict(device=torch.device("cpu"), dtype=torch.float)
        self.memory_format = torch.contiguous_format

        # CIFAR-10's data shape
        self.data_shape = (3, 32, 32)

        from auxiliaries.objectives import CosineSimilarity

        self.objective = CosineSimilarity(scale=1.0, task_regularization=0.0)

        from auxiliaries.regularizers import TotalVariation

        self.regularizers = [TotalVariation(
            self.setup, **dict(scale=0.2, inner_exp=2, outer_exp=0.5, double_opponents=True)
        )]

        self.augmentations = torch.nn.Sequential()  # No augmentations selected.

        # From model_preparation.py
        loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = torch.jit.script(loss_fn)

        # Load preprocessing constants:
        # PyTorch hard-codes the mean and std
        self.dm = torch.as_tensor([0.485, 0.456, 0.406], **self.setup)[None, :,
        None, None]
        self.ds = torch.as_tensor([0.229, 0.224, 0.225], **self.setup)[None, :,
        None, None]

    def _initialize_data(self, data_shape):
        """Initialize data as randn"""
        candidate = torch.randn(data_shape, **self.setup)
        candidate.to(memory_format=torch.contiguous_format)
        candidate.requires_grad = True
        candidate.grad = torch.zeros_like(candidate)
        return candidate

    def _construct_models_from_payload_and_buffers(
        self, server_payload, shared_data
    ):
        """Construct the model (or multiple) that is sent by the server and include user buffers if any."""

        # Load states into multiple models if necessary
        models = []
        for idx, payload in enumerate(server_payload):
            new_model = deepcopy(self.algorithm.model)
            new_model.to(**self.setup, memory_format=self.memory_format)

            # Load parameters
            parameters = payload["parameters"]
            if shared_data[idx]["buffers"] is not None:
                # User sends buffers. These should be used!
                buffers = shared_data[idx]["buffers"]
                new_model.eval()
            elif payload["buffers"] is not None:
                # The server has public buffers in any case
                buffers = payload["buffers"]
                new_model.eval()
            else:
                # The user sends no buffers and there are no public bufers
                # (i.e. the user in in training mode and does not send updates)
                new_model.train()
                for module in new_model.modules():
                    if hasattr(module, "track_running_stats"):
                        module.reset_parameters()
                        module.track_running_stats = False
                buffers = []

            with torch.no_grad():
                for param, server_state in zip(new_model.parameters(), parameters):
                    param.copy_(server_state.to(**self.setup))
                for buffer, server_state in zip(new_model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))
            models.append(new_model)
        return models

    def _cast_shared_data(self, shared_data):
        """Cast user data to reconstruction data type."""
        # for data["gradients"]:
        for data in shared_data:
            data["gradients"] = [g.to(dtype=self.setup["dtype"]) for g in data["gradients"]]
            if data["buffers"] is not None:
                data["buffers"] = [b.to(dtype=self.setup["dtype"]) for b in data["buffers"]]
        return shared_data

    def _recover_label_information(self, user_data, server_payload, rec_models):
        """Recover label information.

        This method runs under the assumption that the last two entries in the gradient vector
        correpond to the weight and bias of the last layer (mapping to num_classes).
        For non-classification tasks this has to be modified.

        The behavior with respect to multiple queries is work in progress and subject of debate.
        """
        num_data_points = user_data[0]["metadata"]["num_data_points"]
        num_classes = user_data[0]["gradients"][-1].shape[0]
        num_queries = len(user_data)

        # NOTE: label strategy is "bias-corrected"
        # This is slightly modified analytic label recovery in the style of Wainakh
        bias_per_query = [shared_data["gradients"][-1] for shared_data in user_data]
        label_list = []
        # Stage 1
        average_bias = torch.stack(bias_per_query).mean(dim=0)
        valid_classes = (average_bias < 0).nonzero()
        label_list += [*valid_classes.squeeze(dim=-1)]
        m_impact = average_bias_correct_label = average_bias[valid_classes].sum() / num_data_points

        average_bias[valid_classes] = average_bias[valid_classes] - m_impact
        # Stage 2
        while len(label_list) < num_data_points:
            selected_idx = average_bias.argmin()
            label_list.append(selected_idx)
            average_bias[selected_idx] -= m_impact
        labels = torch.stack(label_list)

        # Pad with random labels if too few were produced:
        if len(labels) < num_data_points:
            labels = torch.cat(
                [labels, torch.randint(0, num_classes, (num_data_points - len(labels),), device=self.setup["device"])]
            )

        # Always sort, order does not matter here:
        labels = labels.sort()[0]
        print(f"Recovered labels {labels.tolist()} through strategy 'bias-corrected'.")
        return labels

    def prepare_attack(self, server_payload, shared_data):
        """Basic startup common to many reconstruction methods."""
        stats = defaultdict(list)

        shared_data = shared_data.copy()  # Shallow copy is enough
        server_payload = server_payload.copy()

        # Load server_payload into state:
        rec_models = self._construct_models_from_payload_and_buffers(
            server_payload, shared_data
            )
        shared_data = self._cast_shared_data(shared_data)
        self._rec_models = rec_models   # TODO(dchu) remove this unused code
        # Consider label information
        if shared_data[0]["metadata"]["labels"] is None:
            labels = self._recover_label_information(
                shared_data, server_payload, rec_models
                )
        else:
            labels = shared_data[0]["metadata"]["labels"].clone()
        return rec_models, labels, stats

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        # Main reconstruction loop starts here:
        scores = torch.zeros(Config().algorithm.num_trials)
        candidate_solutions = []
        try:
            for trial in range(Config().algorithm.num_trials):
                candidate_solutions += [
                    self._run_trial(rec_models, shared_data, labels, stats, trial, initial_data, dryrun)
                ]
                scores[trial] = self._score_trial(candidate_solutions[trial], labels, rec_models, shared_data)
        except KeyboardInterrupt:
            print("Trial procedure manually interruped.")
            pass
        optimal_solution = self._select_optimal_reconstruction(candidate_solutions, scores, stats)
        reconstructed_data = dict(data=optimal_solution, labels=labels)
        if "ClassAttack" in server_secrets:
            # Only a subset of images was actually reconstructed:
            true_num_data = server_secrets["ClassAttack"]["true_num_data"]
            reconstructed_data["data"] = torch.zeros([true_num_data, *self.data_shape], **self.setup)
            reconstructed_data["data"][server_secrets["ClassAttack"]["target_indx"]] = optimal_solution
            reconstructed_data["labels"] = server_secrets["ClassAttack"]["all_labels"]
        return reconstructed_data, stats

    def _init_optimizer(self, candidate):
        from auxiliaries.common import optimizer_lookup

        optimizer, scheduler = optimizer_lookup(
            candidate,
            Config().algorithm.optim.optimizer,
            Config().algorithm.optim.step_size,
            scheduler=Config().algorithm.optim.step_size_decay,
            warmup=Config().algorithm.optim.warmup,
            max_iterations=Config().algorithm.optim.max_iterations,
        )
        return optimizer, scheduler

    def _run_trial(self, rec_model, shared_data, labels, stats, trial, initial_data=None, dryrun=False):
        """Run a single reconstruction trial."""

        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)
        self.objective.initialize(self.loss_fn, Config().algorithm.impl, shared_data[0]["metadata"]["local_hyperparams"])

        # Initialize candidate reconstruction data
        candidate = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], *self.data_shape])
        if initial_data is not None:
            candidate.data = initial_data.data.clone().to(**self.setup)

        best_candidate = candidate.detach().clone()
        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)

        # Initialize optimizers
        optimizer, scheduler = self._init_optimizer([candidate])
        current_wallclock = time.time()
        try:
            for iteration in range(num_iters):
                closure = self._compute_objective(candidate, labels, rec_model, optimizer, shared_data, iteration)
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if Config().algorithm.boxed:
                        candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                    if objective_value < minimal_value_so_far:
                        minimal_value_so_far = objective_value.detach()
                        best_candidate = candidate.detach().clone()

                if iteration + 1 == Config().algorithm.optim.max_iterations or iteration % Config().algorithm.optim.callback == 0:
                    timestamp = time.time()
                    print(
                        f"| It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                        f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                    )
                    current_wallclock = timestamp

                if not torch.isfinite(objective_value):
                    print(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break

                stats[f"Trial_{trial}_Val"].append(objective_value.item())

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass

        return best_candidate.detach()

    def _compute_objective(self, candidate, labels, rec_model, optimizer, shared_data, iteration):
        def closure():
            optimizer.zero_grad()

            candidate_augmented = candidate
            candidate_augmented.data = self.augmentations(candidate.data)

            total_objective = 0
            total_task_loss = 0
            for model, data in zip(rec_model, shared_data):
                objective, task_loss = self.objective(model, data["gradients"], candidate_augmented, labels)
                total_objective += objective
                total_task_loss += task_loss
            for regularizer in self.regularizers:
                total_objective += regularizer(candidate_augmented)

            if total_objective.requires_grad:
                total_objective.backward(inputs=candidate, create_graph=False)
            with torch.no_grad():
                if Config().algorithm.optim.signed is not None:
                    if Config().algorithm.optim.signed == "soft":
                        scaling_factor = (
                            1 - iteration / Config().algorithm.optim.max_iterations
                        )  # just a simple linear rule for now
                        candidate.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                    elif Config().algorithm.optim.signed == "hard":
                        candidate.grad.sign_()
                    else:
                        pass

            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            return total_objective

        return closure

    def _score_trial(self, candidate, labels, rec_model, shared_data):
        """Score candidate solutions based on some criterion."""

        from auxiliaries.objectives import CosineSimilarity

        objective = CosineSimilarity()
        objective.initialize(
            self.loss_fn, Config().algorithm.impl,
            shared_data[0]["metadata"]["local_hyperparams"]
        )
        score = 0
        for model, data in zip(rec_model, shared_data):
            score += objective(model, data["gradients"], candidate, labels)[0]
        return score if score.isfinite() else float("inf")

    def _select_optimal_reconstruction(
        self, candidate_solutions, scores, stats
    ):
        """Choose one of the candidate solutions based on their scores (for now).

        More complicated combinations are possible in the future."""
        optimal_val, optimal_index = torch.min(scores, dim=0)
        optimal_solution = candidate_solutions[optimal_index]
        stats["opt_value"] = optimal_val.item()
        if optimal_val.isfinite():
            print(
                f"Optimal candidate solution with rec. loss {optimal_val.item():2.4f} selected."
                )
            return optimal_solution
        else:
            print("No valid reconstruction could be found.")
            return torch.zeros_like(optimal_solution)