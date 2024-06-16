# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------
# modified by authors
class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2,
                 pl_decay=0.01, pl_weight=2, add_uc=True, alpha=0.0, img_num_list=None):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.add_uc = add_uc
        self.alpha = alpha
        self.img_num_list = img_num_list

    def run_G(self, z, c, add_uc=False, return_feat=False):
        ws = self.G.mapping(z, c)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        img = self.G.synthesis(ws, add_uc=add_uc, return_feat=return_feat)
        return img, ws

    def run_D(self, img, c, uc=False):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, uc=uc)
        return logits

    # modified by authors
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, shot_labels=None):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        if shot_labels is not None:
            real_many_inds, real_medium_inds, real_few_inds = training_stats.get_inds(real_c, shot_labels)
            fake_many_inds, fake_medium_inds, fake_few_inds = training_stats.get_inds(gen_c, shot_labels)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_imgs, _gen_ws = self.run_G(gen_z, gen_c, add_uc=self.add_uc) # May get synced by Gpl.

                gen_img_uc = None
                if len(gen_imgs) == 2 and isinstance(gen_imgs, list):
                    gen_img, gen_img_uc = gen_imgs
                else:
                    gen_img = gen_imgs
                    assert self.alpha == 0

                gen_logits = self.run_D(gen_img, gen_c, uc=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = 1.0 / (1.0 + self.alpha) * torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))

                if gen_img_uc is not None:
                    gen_logits_uc = self.run_D(gen_img_uc, gen_c, uc=True)
                    training_stats.report('Loss/scores/fake_uc', gen_logits_uc)
                    training_stats.report('Loss/signs/fake_uc', gen_logits_uc.sign())
                    loss_Gmain_uc = self.alpha / (1.0 + self.alpha) * torch.nn.functional.softplus(-gen_logits_uc)  # -log(sigmoid(gen_logits))
                    loss_Gmain += loss_Gmain_uc
                    training_stats.report('Loss/G/loss_uc', loss_Gmain_uc)

                training_stats.report('Loss/G/loss', loss_Gmain)

                # modified by Saeed
                if shot_labels is not None:
                    training_stats.report('Loss/scores/fake_many', gen_logits[fake_many_inds])
                    training_stats.report('Loss/scores/fake_medium', gen_logits[fake_medium_inds])
                    training_stats.report('Loss/scores/fake_few', gen_logits[fake_few_inds])
                    training_stats.report('Loss/signs/fake_many', gen_logits[fake_many_inds].sign())
                    training_stats.report('Loss/signs/fake_medium', gen_logits[fake_medium_inds].sign())
                    training_stats.report('Loss/signs/fake_few', gen_logits[fake_few_inds].sign())

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_imgs, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], add_uc=self.add_uc)
                if len(gen_imgs) == 2 and isinstance(gen_imgs, list):
                    gen_img, gen_img_uc = gen_imgs
                else:
                    gen_img = gen_imgs

                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_imgs, _gen_ws = self.run_G(gen_z, gen_c, add_uc=self.add_uc)

                gen_img_uc = None
                if len(gen_imgs) == 2 and isinstance(gen_imgs, list):
                    gen_img, gen_img_uc = gen_imgs
                else:
                    gen_img = gen_imgs

                gen_logits = self.run_D(gen_img, gen_c, uc=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = 1.0 / (1.0 + self.alpha) * torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

                if gen_img_uc is not None:
                    gen_logits_uc = self.run_D(gen_img_uc, gen_c, uc=True)
                    training_stats.report('Loss/scores/fake_uc', gen_logits_uc)
                    training_stats.report('Loss/signs/fake_uc', gen_logits_uc.sign())
                    loss_Dgen_uc = self.alpha / (1.0 + self.alpha) * torch.nn.functional.softplus(gen_logits_uc)  # -log(sigmoid(gen_logits))
                    loss_Dgen += loss_Dgen_uc

                # modified by Saeed
                if shot_labels is not None:
                    training_stats.report('Loss/scores/fake_many', gen_logits[fake_many_inds])
                    training_stats.report('Loss/scores/fake_medium', gen_logits[fake_medium_inds])
                    training_stats.report('Loss/scores/fake_few', gen_logits[fake_few_inds])
                    training_stats.report('Loss/signs/fake_many', gen_logits[fake_many_inds].sign())
                    training_stats.report('Loss/signs/fake_medium', gen_logits[fake_medium_inds].sign())
                    training_stats.report('Loss/signs/fake_few', gen_logits[fake_few_inds].sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)

                real_logits = self.run_D(real_img_tmp, real_c, uc=False)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = 1.0 / (1.0 + self.alpha) * torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))

                    if self.add_uc:
                        real_img_tmp_uc = self.D.downsample_uc(real_img_tmp)
                        real_logits_uc = self.run_D(real_img_tmp_uc, real_c, uc=True)
                        training_stats.report('Loss/scores/real_uc', real_logits_uc)
                        training_stats.report('Loss/signs/real_uc', real_logits_uc.sign())
                        loss_Dreal_uc = self.alpha / (1.0 + self.alpha) * torch.nn.functional.softplus(-real_logits_uc)  # -log(sigmoid(real_logits))
                        loss_Dreal += loss_Dreal_uc

                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

                if shot_labels is not None:
                    training_stats.report('Loss/scores/real_many', real_logits[real_many_inds])
                    training_stats.report('Loss/scores/real_medium', real_logits[real_medium_inds])
                    training_stats.report('Loss/scores/real_few', real_logits[real_few_inds])
                    training_stats.report('Loss/signs/real_many', real_logits[real_many_inds].sign())
                    training_stats.report('Loss/signs/real_medium', real_logits[real_medium_inds].sign())
                    training_stats.report('Loss/signs/real_few', real_logits[real_few_inds].sign())

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().backward()
#----------------------------------------------------------------------------
