import math
import torch
import torch.nn as nn
from functools import partial

from .kl_div import KLDivergence
from .dtkd import KLDivergenceDTKD
from .dist_kd import DIST
from .diffkd import DiffKD

import logging
logger = logging.getLogger()



KD_MODULES = {
    'cifar_wrn_40_1': dict(modules=['relu', 'fc'], channels=[64, 100]),
    'cifar_wrn_40_2': dict(modules=['relu', 'fc'], channels=[128, 100]),
    'cifar_resnet56': dict(modules=['layer3', 'fc'], channels=[64, 100]),
    'cifar_resnet20': dict(modules=['layer3', 'fc'], channels=[64, 100]),
    'tv_resnet50': dict(modules=['layer4', 'fc'], channels=[2048, 1000]),
    'tv_resnet34': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
    'tv_resnet18': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
    'resnet18': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
    'tv_mobilenet_v2': dict(modules=['features.18', 'classifier'], channels=[1280, 1000]),
    'nas_model': dict(modules=['features.conv_out', 'classifier'], channels=[1280, 1000]),  # mbv2
    'timm_tf_efficientnet_b0': dict(modules=['conv_head', 'classifier'], channels=[1280, 1000]),
    'mobilenet_v1': dict(modules=['model.13', 'fc'], channels=[1024, 1000]),
    'timm_swin_large_patch4_window7_224': dict(modules=['norm', 'head'], channels=[1536, 1000]),
    'timm_swin_tiny_patch4_window7_224': dict(modules=['norm', 'head'], channels=[768, 1000]),
}



class AskKDLoss():
    '''
    kd loss wrapper.
    '''

    def __init__(
        self,
        student,
        teachers,
        peers,
        student_name,
        teacher_name,
        peer_name,
        ori_loss,
        kd_method='kdt12',
        ori_loss_weight=1.0,
        kd_loss_weight=1.0,
        peer_loss_weight =1.0,
        confidence_threshold=0.6,
        ask = False,
        mentor = True,
        dtkd_mentor = False,
        kd_loss_kwargs={}
    ):
        self.student = student
        self.teachers = teachers
        self.peers = peers
        self.ori_loss = ori_loss
        self.ori_loss_weight = ori_loss_weight
        self.kd_method = kd_method
        self.kd_loss_weight = kd_loss_weight
        self.peer_loss_weight = peer_loss_weight

        self._teacher_out = []
        self._student_out = {}
        self._peer_out = []
        self.weights = {}
        self.weights_t = {}
        self.tau_scale = {}

        self.confidence_threshold = confidence_threshold
        self.ask = ask
        self.mentor = mentor
        self.dtkd_mentor = dtkd_mentor
        # init kd loss
        # module keys for distillation. '': output logits
        teacher_modules = [['',]] * len(teachers)
        student_modules = ['',]
        peer_modules = [['',]] * len(peers)
        if self.kd_method == 'kd':
            self.kd_loss = KLDivergence(tau=4)
        elif self.kd_method == 'dist':
            self.kd_loss = DIST(beta=1, gamma=1, tau=1)
        elif self.kd_method.startswith('dist_t'):
            tau = float(kd_method[6:])
            self.kd_loss = DIST(beta=1, gamma=1, tau=tau)
        elif self.kd_method.startswith('kdt'):
            tau = float(kd_method[3:])
            self.kd_loss = KLDivergence(tau)
        elif self.kd_method.startswith('ask_t'):
            tau = float(kd_method[5:])
            if(dtkd_mentor):
                self.kd_loss_p = KLDivergenceDTKD(tau=tau)
                self.kd_loss_t = KLDivergenceDTKD(tau=tau)
            else:
                self.kd_loss_p = KLDivergence(tau=tau)
                self.kd_loss_t = KLDivergence(tau=tau)
            
        elif self.kd_method == 'diffkd':
            # get configs
            ae_channels = kd_loss_kwargs.get('ae_channels', 1024)
            use_ae = kd_loss_kwargs.get('use_ae', True)
            tau = kd_loss_kwargs.get('tau', 1)
            #change teacher module indexing!!!
            print(kd_loss_kwargs)
            kernel_sizes = [3, 1]  # distillation on feature and logits
            student_modules = KD_MODULES[student_name]['modules']
            student_channels = KD_MODULES[student_name]['channels']
            teacher_modules = KD_MODULES[teacher_name]['modules']
            teacher_channels = KD_MODULES[teacher_name]['channels']
            self.diff = nn.ModuleDict()
            self.kd_loss = nn.ModuleDict()
            for tm, tc, sc, ks in zip(teacher_modules, teacher_channels, student_channels, kernel_sizes):
                self.diff[tm] = DiffKD(sc, tc, kernel_size=ks, use_ae=(ks!=1) and use_ae, ae_channels=ae_channels)
                self.kd_loss[tm] = nn.MSELoss() if ks != 1 else KLDivergence(tau=tau)
            self.diff.cuda()
            # add diff module to student for optimization
            self.student._diff = self.diff
        elif self.kd_method == 'mse':
            # distillation on feature
            student_modules = KD_MODULES[student_name]['modules'][:1]
            student_channels = KD_MODULES[student_name]['channels'][:1]
            teacher_modules = KD_MODULES[teacher_name]['modules'][:1]
            teacher_channels = KD_MODULES[teacher_name]['channels'][:1]
            self.kd_loss = nn.MSELoss()
            self.align = nn.Conv2d(student_channels[0], teacher_channels[0], 1)
            self.align.cuda()
            # add align module to student for optimization
            self.student._align = self.align
        else:
            raise RuntimeError(f'KD method {kd_method} not found.')

        # register forward hook
        # dicts that store distillation outputs of student and teacher
        #to-do : change teacher_modules
        self._teacher_out = [{} for _ in range(len(teachers))]
        self._student_out = {}
        self._peer_out = [{} for _ in range(len(peers))]
        if(len(teachers)):
            for i,teacher in enumerate(teachers):
                for teacher_module in teacher_modules[i]:
                    self._register_forward_hook(teacher,i, teacher_module, teacher=True, peer=False)
        for student_module in student_modules:
            self._register_forward_hook(student,0, student_module, teacher=False,peer=False)
        if(len(peers)):
            for i,peer in enumerate(peers):
                for peer_module in peer_modules[i]:
                    self._register_forward_hook(peer,i, peer_module, teacher=False,peer=True)
        self.student_modules = student_modules
        self.teacher_modules = teacher_modules
        self.peer_modules = peer_modules

        for teacher in teachers:
            teacher.eval()
        for peer in peers:
            peer.eval()
        self._iter = 0

    def __call__(self, x, targets,decay_factor):
        with torch.no_grad():
            t_logits = [teacher(x) for teacher in self.teachers]
            p_logits = [peer(x) for peer in self.peers]

        # Compute original loss of the student
        logits = self.student(x)
        ori_loss = self.ori_loss(logits, targets)
        num_classes = logits.size(-1)
        if len(targets.shape) != 1:  # label smoothing
            target_mask = nn.functional.one_hot(targets.argmax(-1), num_classes)
        else:
            target_mask = nn.functional.one_hot(targets, num_classes)

        # Calculate softmax probabilities for the student's predictions
        student_probs = nn.functional.softmax(logits, dim=-1)
        student_pred_correct = 1 /(torch.exp(ori_loss))
        student_pred_correct_avg = torch.mean(student_pred_correct,dim=0)
        if self._iter == 50:

            logger.info(f'student_probs: {student_probs[0]}')
            logger.info(f'student_pred_correct: {student_pred_correct}')
            logger.info(f'student_prob_max: {torch.max(student_probs)}')
            logger.info(f'target: {targets[0]}')
            logger.info(f'student_pred_correct_avg: {student_pred_correct_avg.item()}') 


        if(self.ask):

            self.weights['student'] = student_pred_correct_avg.item()
            for i,peer_logits in enumerate(p_logits):
                peer_pred_correct = 1 /(torch.exp(self.ori_loss(peer_logits, targets)))
                self.weights[i] = peer_pred_correct.item()
                if self._iter == 50:
                    logger.info(f'peer: {i}')
                    logger.info(f'peer_pred_correct: {peer_pred_correct.item()}')
            
            if(len(self.teachers)):
                teacher_pred_correct = 1 /(torch.exp(self.ori_loss(t_logits[0], targets)))
                self.weights['teacher'] = teacher_pred_correct.item()
                if self._iter == 50:
                    logger.info(f'teacher_pred_correct: {teacher_pred_correct.item()}')
            sorted_weights = dict(sorted(self.weights.items(), key=lambda item: item[1]))
            weights_sum = sum(sorted_weights.values())
            w_len = len(sorted_weights) - 1
            self.weights = {key: round(value/weights_sum, 2)*w_len for (key, value) in sorted_weights.items()}
                
        kd_loss = kd_loss1 = 0
        peer_loss = 0
        if(len(self.teachers)):
            for i, (teacher, t_logit) in enumerate(zip(self.teachers, t_logits)):
                for tm, sm in zip(self.teacher_modules[i], self.student_modules):
                    if(self.mentor and self.ask):
                        if(self.weights['teacher']>=self.weights['student'] and self.weights['teacher']>0):
                            tau_scale = abs(self.weights['teacher']-self.weights['student'])/max(self.weights['teacher'],self.weights['student'])
                        else:
                            tau_scale = 0.0
                    elif(self.ask and self.dtkd_mentor):
                        l_stu_mx, _ = t_logit.detach().max(dim=1, keepdim=True)
                        l_tea_mx, _ = logits.detach().max(dim=1, keepdim=True)
                        t_scale = 2 * l_tea_mx / (l_tea_mx+l_stu_mx)
                        s_scale = 2 * l_stu_mx / (l_tea_mx+l_stu_mx)
                        tau_scale = [t_scale,s_scale]

                    else:
                        tau_scale = 0.0
                    kd_loss_ = self.kd_loss_t(self._student_out[sm], self._teacher_out[i][tm],tau_scale=tau_scale)
                    kd_loss_1 = self.kd_loss_t(self._student_out[sm], self._teacher_out[i][tm],tau_scale=0.0)
                    if(self.ask):
                        if(self.weights['teacher']>=self.weights['student']):
                            kd_loss_ *= self.weights['teacher']
                        else:
                            # print(f"do not ask peer {i}. Confidence is lower than student")
                            kd_loss_ *= 0.0
                    if self._iter % 50 == 0:
                        if(self.dtkd_mentor):
                            logger.info(f'[{tm}-{sm}]t{i} KD ({self.kd_method}) loss: {kd_loss_.item():.4f} teacher_tau: {tau_scale[1][0].item()*4:.3f} student_tau: {tau_scale[0][0].item()*4:.3f}')
                        else:
                            logger.info(f'[{tm}-{sm}]t{i} KD ({self.kd_method}) loss: {kd_loss_.item():.4f} teacher_tau: {1.0+tau_scale*12:.3f}')
                    kd_loss += kd_loss_
                    kd_loss1 += kd_loss_1

        if(len(self.peers)):
            for i, (peer, p_logit) in enumerate(zip(self.peers, p_logits)):
                for pm, sm in zip(self.peer_modules[i], self.student_modules):
                    if(self.mentor and self.ask):
                        if(self.weights[i]>=self.weights['student'] and self.weights[i]>0):
                            tau_scale = abs(self.weights[i]-self.weights['student'])/max(self.weights[i],self.weights['student'])
                        else:
                            tau_scale = 0.0
                    elif(self.ask and self.dtkd_mentor):
                        l_stu_mx, _ = p_logit.detach().max(dim=1, keepdim=True)
                        l_tea_mx, _ = logits.detach().max(dim=1, keepdim=True)
                        t_scale = 2 * l_tea_mx / (l_tea_mx+l_stu_mx)
                        s_scale = 2 * l_stu_mx / (l_tea_mx+l_stu_mx)
                        tau_scale = [t_scale,s_scale]
                    else:
                        tau_scale = 0.0
                    peer_loss_ = self.kd_loss_p(self._student_out[sm], self._peer_out[i][pm],tau_scale=tau_scale)
                    if(self.ask):
                        if(self.weights[i]>=self.weights['student']):
                            peer_loss_*= self.weights[i]
                        else:
                            # print(f"do not ask peer {i}. Confidence is lower than student")
                            peer_loss_ *= 0.0

                    if self._iter % 50 == 0:
                        if(self.dtkd_mentor):
                            logger.info(f'[{pm}-{sm}]p{i} KD ({self.kd_method}) loss: {peer_loss_.item():.4f} peer_tau: {tau_scale[1][0].item()*4:.3f} student_tau: {tau_scale[0][0].item()*4:.3f}')
                        else:
                            logger.info(f'[{pm}-{sm}]p{i} KD ({self.kd_method}) loss: {peer_loss_.item():.4f} peer_tau: {1.0+tau_scale*12:.3f}')
                    peer_loss += peer_loss_
        if self._iter % 50 == 0:
            logger.info(f'weights: {self.weights}')
            logger.info(f'student_probs: {student_pred_correct_avg.item()}')

        # student_weight = self.weights['student']
        self._teacher_out = [{} for _ in range(len(self.teachers))]
        self._student_out = {}
        self._peer_out = [{} for _ in range(len(self.peers))]
        self.weights = {}
        self.weights_t = {}
        self._iter += 1

        if(self.ask):
            #heterogenous ensemble
            return ori_loss * (student_pred_correct_avg.item()) + kd_loss + peer_loss
            #homogenous ensemble - add s-t loss with Tau=0
            # return ori_loss * (1+student_pred_correct_avg.item()) + kd_loss + kd_loss1 + peer_loss
        return ori_loss + kd_loss + peer_loss

    def _register_forward_hook(self, model,i, name, teacher=False, peer = False):
        if name == '':
            # use the output of model
            model.register_forward_hook(partial(self._forward_hook, name=name,i=i, teacher=teacher, peer = peer))
        else:
            module = None
            for k, m in model.named_modules():
                if k == name:
                    module = m
                    break
            module.register_forward_hook(partial(self._forward_hook, name=name, i=i,teacher=teacher, peer = peer))

    def _forward_hook(self, module, input, output, name,i=0, teacher=False,peer=False):
        if teacher:
            self._teacher_out[i][name] = output[0] if len(output) == 1 else output
        if peer:
            self._peer_out[i][name] = output[0] if len(output) == 1 else output
        else:
            self._student_out[name] = output[0] if len(output) == 1 else output

    def _reshape_BCHW(self, x):
        """
        Reshape a 2d (B, C) or 3d (B, N, C) tensor to 4d BCHW format.
        """
        if x.dim() == 2:
            x = x.view(x.shape[0], x.shape[1], 1, 1)
        elif x.dim() == 3:
            # swin [B, N, C]
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(-2, -1).reshape(B, C, H, W)
        return x