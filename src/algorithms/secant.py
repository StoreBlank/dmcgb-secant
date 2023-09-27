from algorithms.sac import SAC
import algorithms.modules as m
import torch
import torch.nn.functional as F
import augmentations


class SECANT(SAC):

    def __init__(self, obs_shape, action_shape, args):
        shared_cnn = m.SharedCNN(obs_shape, args.num_shared_layers,
                                 args.num_filters).cuda()
        head_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers,
                             args.num_filters).cuda()
        actor_encoder = m.Encoder(
            shared_cnn, head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim))

        self.actor = m.Actor(actor_encoder, action_shape, args.hidden_dim,
                             args.actor_log_std_min,
                             args.actor_log_std_max).cuda()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=args.secant_lr,
                                                betas=(args.actor_beta, 0.999))

        self.expert = None

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    def set_expert(self, expert: SAC):
        self.expert = expert
        self.expert.freeze()

    def update_actor(self, obs, L=None, step=None):
        if self.expert is None:
            raise Exception("Expert not set")
        # TODO: stronger augmentation
        # obs_aug = augmentations.random_conv(obs.clone())
        obs_aug = augmentations.combo1(obs.clone())

        with torch.no_grad():
            action_target, _, _, _ = self.expert.actor(obs)
        action_pred, _, _, _ = self.actor(obs_aug)

        loss = F.mse_loss(action_pred, action_target)

        if L is not None:
            L.log('train_student/actor_loss', loss, step)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def update(self, replay_buffer, L, step):
        obs, _, _, _, _ = replay_buffer.sample()

        self.update_actor(obs, L, step)
