import torch
import torch.nn.functional as F
import lightning as L

from main import instantiate_from_config
from contextlib import contextmanager

from taming.modules.diffusionmodules.improved_model import Encoder, Decoder
from taming.modules.scheduler.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from taming.modules.util import requires_grad
from collections import OrderedDict
from taming.modules.ema import LitEma
from taming.models.vae_utils import DiagonalGaussianDistribution
import numpy as np

class VQModel(L.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ### Quantize Related
                 quantconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 learning_rate=None,
                 ### scheduler config
                 warmup_epochs=1.0, #warmup epochs
                 scheduler_type = "linear-warmup_cosine-decay",
                 accumulate_steps = 1,
                 min_learning_rate = 0,
                 use_ema = False,
                 stage = None,
                #  distribution_weight = 1e-5
                distribution_weight = 1e-6
                 ):
        super().__init__()
        self.distribution_weight = distribution_weight
        self.image_key = image_key
        self.vae_quantizer = DiagonalGaussianDistribution()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = instantiate_from_config(quantconfig)
        self.use_ema = use_ema
        self.stage = stage
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        if self.use_ema and stage is None: #no need to construct ema when training transformer
            self.model_ema = LitEma(self)
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.automatic_optimization = False
        self.accumulate_steps = accumulate_steps

        self.strict_loading = False

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        '''
        save the state_dict and filter out the 
        '''
        return {k: v for k, v in super().state_dict(*args, destination, prefix, keep_vars).items() if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k)}
        
    def init_from_ckpt(self, path, ignore_keys=list(), stage=None):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        ema_mapping = {}
        new_params = OrderedDict()
        if stage == "transformer": ### directly use ema encoder and decoder parameter
            if self.use_ema:
                for k, v in sd.items(): 
                    if "encoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v   
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
                    if "decoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v 
                        s_name = k.replace(".", "")
                        ema_mapping.update({s_name: k})
                        continue 
            else: #also only load the Generator
                for k, v in sd.items():
                    if "encoder" in k:
                        new_params[k] = v
                    elif "decoder" in k:
                        new_params[k] = v
            missing_keys, unexpected_keys = self.load_state_dict(new_params, strict=False)
        else: ## simple resume
            missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    # def encode(self, x):
    #     h = self.encoder(x)
    #     (quant, emb_loss, info), loss_breakdown = self.quantize(h)
    #     ### using token factorization the info is a tuple (each for embedding)
    #     return quant, emb_loss, info, loss_breakdown

    # def decode(self, quant):
    #     dec = self.decoder(quant)
    #     return dec
    
    def vae_encode(self, x):
        h = self.encoder(x)
        z_c, loss_kl, mean, std = self.vae_quantizer(h)
        return z_c, loss_kl, mean, std
    
    # def vq_encode(self, z_c):
    #     (z_q, emb_loss, info), loss_breakdown = self.quantize(z_c)
    #     return z_q, emb_loss, info, loss_breakdown
    def vq_encode(self, z_c):
        (z_q, emb_loss, info), loss_breakdown = self.quantize(z_c)
        return z_q, emb_loss, info, loss_breakdown

    def vae_decode(self, z_c):
        dec = self.decoder(z_c)
        return dec
    
    def vq_decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def encode(self, x):
        z_c, loss_kl, mean, std = self.vae_encode(x)
        # z_q, emb_loss, indices, loss_break = self.vq_encode(z_c)
        z_q, emb_loss, indices, loss_break = self.vq_encode(mean)
        return z_q, z_c, indices, loss_break

    def decode(self, z):
        dec = self.decoder(z)
        return dec

    # def loss_rcs(self,z_q, z_c, mean, std, eps: float = 1e-6, reduction: str = "mean"):
    #     sigma_detach = std.detach().clamp_min(eps)
    #     # diff = (z_q - mean) / sigma_detach
    #     diff = (z_q - mean) 
    #     loss = 0.5 * diff.pow(2)
    #     if reduction == "mean":
    #         return loss.mean()
    #     elif reduction == "sum":
    #         return loss.sum()
    #     else:
    #         raise ValueError(f"Unknown reduction: {reduction}")
    def loss_rcs(self, epsilon: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        loss = 0.5 * epsilon.pow(2)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    # loss = torch.mean(codebook**2)
    # def loss_dcr(self, codebook: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    #     # codebook: [K, D]
    #     K, D = codebook.shape
    #     mu_q = codebook.mean(dim=0)  
    #     centered = codebook - mu_q 
    #     cov_q = centered.t().matmul(centered) / (K - 1)  # [D, D]
    #     evals, evecs = torch.linalg.eigh(cov_q)  
    #     evals_clamped = torch.clamp(evals, min=eps)       
    #     sqrt_evals = torch.sqrt(evals_clamped)
    #     cov_q_sqrt = (evecs * sqrt_evals.unsqueeze(0)) @ evecs.t() 
    #     term_mu = (mu_q ** 2).sum()
    #     term_cov = torch.trace(cov_q)
    #     term_cov_sqrt = torch.trace(cov_q_sqrt)
    #     loss = term_mu + term_cov - 2.0 * term_cov_sqrt + D
    #     loss = torch.clamp(loss, min=0.0)
    #     return loss

    def loss_dcr(self, codebook: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # codebook: [K, D]
        K, D = codebook.shape
        mu_q = codebook.mean(dim=0)
        centered = codebook - mu_q
        X = centered / max(K - 1, 1)**0.5  
        # try:
        #     _, S, _ = torch.linalg.svd(X, full_matrices=False) 
        #     evals = S ** 2
        #     evals_clamped = torch.clamp(evals, min=eps)
        #     sqrt_evals = torch.sqrt(evals_clamped)
        #     term_cov_sqrt = sqrt_evals.sum()
        #     cov_q = X.t().matmul(X)
        #     term_cov = torch.trace(cov_q)
        # except torch.linalg.LinAlgError:
        term_cov_sqrt = codebook.new_tensor(0.0)
        term_cov = (centered ** 2).sum(dim=0).mean()

        term_mu = (mu_q ** 2).sum()
        loss = term_mu + term_cov - 2.0 * term_cov_sqrt + D
        loss = torch.clamp(loss, min=0.0)
        return loss

    def forward(self, input):
        # quant, diff, indices, loss_break = self.encode(input)
        # dec = self.decode(quant)
        # for ind in indices.unique():
        #     self.codebook_count[ind] = 1
        # return dec, diff, loss_break

        z_c, loss_kl, mean, std = self.vae_encode(input)
        # z_q, emb_loss, indices, loss_break = self.vq_encode(z_c)
        # z_cc = z_c.clone()
        _, _, _, loss_break = self.vq_encode(z_c)
        
        z_q, epsilon, indices = self.quantize.vq_forward(mean, std)

        # print(z_c.shape, z_q.shape, mean.shape, std.shape)
        # print(loss_kl.item())
        # loss_rcs = self.loss_rcs(z_q, z_c, mean, std, reduction="mean")
        loss_rcs = self.loss_rcs(epsilon)
        step_codebook = self.quantize.get_codebook()
        loss_dcr = self.loss_dcr(step_codebook)
        # diff = loss_rcs + (loss_dcr + loss_kl)*self.distribution_weight
        # diff = loss_rcs + (loss_kl)*self.distribution_weight
        # diff = loss_rcs + (loss_kl + loss_dcr) * self.distribution_weight
        diff = loss_rcs
        # print(loss_rcs.item(), loss_dcr.item(), loss_kl.item(), diff.item())  
        # diff = loss_rcs + loss_dcr + (loss_kl)*1e-5   
        
        c_dec = self.vae_decode(z_c)
        q_dec = self.vq_decode(z_q)

        for ind in indices.unique():
            self.codebook_count[ind] = 1
        return c_dec, q_dec, diff, loss_break

    def get_input(self, batch, k):
        # x = batch[k]
        # if len(x.shape) == 3:
        #     x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        # return x.float()
        x = batch
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        # xrec, eloss, loss_break = self(x)
        c_xrec, q_xrec, eloss, loss_break = self(x)

        opt_gen, opt_disc = self.optimizers()
        if self.scheduler_type != "None":
            scheduler_gen, scheduler_disc = self.lr_schedulers()

        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        
        # optimize generator
        # aeloss, log_dict_ae = self.loss(eloss, loss_break, x, xrec, 0, self.global_step,
        #                                 last_layer=self.get_last_layer(), split="train")
        # aeloss, log_dict_ae = self.loss(eloss, loss_break, x, c_xrec, q_xrec, 0, self.global_step,
        #                         last_layer=self.get_last_layer(), split="train")
        aeloss, log_dict_ae = self.loss(eloss, loss_break, x, c_xrec, q_xrec, 0, self.global_step,
                                last_layer=self.get_last_layer(), split="train",)
        aeloss = aeloss / self.accumulate_steps
        self.manual_backward(aeloss)
        
        if (batch_idx + 1) % self.accumulate_steps == 0:
            opt_gen.step()
            opt_gen.zero_grad()
            if self.scheduler_type != "None":
                scheduler_gen.step()
        
        log_dict_ae["train/codebook_util"] = torch.tensor(sum(self.codebook_count) / len(self.codebook_count))
            
        # optimize discriminator
        # discloss, log_dict_disc = self.loss(eloss, loss_break, x, xrec, 1, self.global_step,
        #                                     last_layer=self.get_last_layer(), split="train")
        discloss, log_dict_disc = self.loss(eloss, loss_break, x, c_xrec, q_xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        discloss = discloss / self.accumulate_steps
        self.manual_backward(discloss)
        
        if (batch_idx + 1) % self.accumulate_steps == 0:
            opt_disc.step()
            opt_disc.zero_grad()
            if self.scheduler_type != "None":
                scheduler_disc.step()
            
        #if torch.distributed.get_rank() == 0:
        #    print(log_dict_ae, log_dict_disc)

        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)
            
    def on_train_epoch_start(self):
        self.codebook_count = [0] * self.quantize.n_e
        
    def on_validation_epoch_start(self):
        self.codebook_count = [0] * self.quantize.n_e

    def validation_step(self, batch, batch_idx): 
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        else:
            log_dict = self._validation_step(batch, batch_idx)

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        # quant, eloss, indices, loss_break = self.encode(x)
        z_c, loss_kl, mean, std = self.vae_encode(x)
        # z_q, emb_loss, indices, loss_break = self.vq_encode(z_c)
        z_q, emb_loss, indices, loss_break = self.vq_encode(mean)
        
        c_xrec = self.vae_decode(z_c).clamp(-1, 1)
        q_xrec = self.vq_decode(z_q).clamp(-1, 1)

        # loss_rcs = self.loss_rcs(z_q, z_c, mean, std, reduction="mean")
        # eloss = loss_rcs
        eloss = 0.0

        # x_rec = self.decode(quant).clamp(-1, 1)
        # aeloss, log_dict_ae = self.loss(eloss, loss_break, x, x_rec, 0, self.global_step,
        #                                 last_layer=self.get_last_layer(), split="val"+ suffix)

        # discloss, log_dict_disc = self.loss(eloss, loss_break, x, x_rec, 1, self.global_step,
        #                                     last_layer=self.get_last_layer(), split="val" + suffix)
        aeloss, log_dict_ae = self.loss(eloss, loss_break, x, c_xrec, q_xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+ suffix)

        discloss, log_dict_disc = self.loss(eloss, loss_break, x, c_xrec, q_xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val" + suffix)
        
        for ind in indices.unique():
            self.codebook_count[ind] = 1
        log_dict_ae[f"val{suffix}/codebook_util"] = torch.tensor(sum(self.codebook_count) / len(self.codebook_count))
    
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_gen = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        if self.trainer.is_global_zero:
            print("step_per_epoch: {}".format(len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size))
        step_per_epoch  = len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size
        warmup_steps = step_per_epoch * self.warmup_epochs
        training_steps = step_per_epoch * self.trainer.max_epochs

        if self.scheduler_type == "None":
            return ({"optimizer": opt_gen}, {"optimizer": opt_disc})
    
        if self.scheduler_type == "linear-warmup":
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup(warmup_steps))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps))

        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
        else:
            raise NotImplementedError()
        return {"optimizer": opt_gen, "lr_scheduler": scheduler_ae}, {"optimizer": opt_disc, "lr_scheduler": scheduler_disc}

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # xrec, _ = self(x)
        c_xdec, q_xdec, _, _  = self(x)
        xrec = q_xdec
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x                                       