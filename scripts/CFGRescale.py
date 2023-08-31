import math
import torch

import gradio as gr
import numpy as np
import modules.scripts as scripts
from modules import devices, images, processing, shared, sd_samplers_kdiffusion, sd_samplers_compvis, script_callbacks
from modules.processing import Processed
from modules.shared import opts, state
from ldm.models.diffusion import ddim

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, noise_like



class Script(scripts.Script):

    def __init__(self):
        self.old_denoising = sd_samplers_kdiffusion.CFGDenoiser.combine_denoised
        self.old_schedule = ddim.DDIMSampler.make_schedule
        self.old_sample = ddim.DDIMSampler.p_sample_ddim
        globals()['enable_furry_cocks'] = True

        def find_module(module_names):
            if isinstance(module_names, str):
                module_names = [s.strip() for s in module_names.split(",")]
            for data in scripts.scripts_data:
                if data.script_class.__module__ in module_names and hasattr(data, "module"):
                    return data.module
            return None

        def rescale_opt(p, x, xs):
            globals()['cfg_rescale_fi'] = x
            globals()['enable_furry_cocks'] = False

        xyz_grid = find_module("xyz_grid.py, xy_grid.py")
        if xyz_grid:
            extra_axis_options = [xyz_grid.AxisOption("Rescale CFG", float, rescale_opt)]
            xyz_grid.axis_options.extend(extra_axis_options)

    def title(self):
        return "CFG Rescale Extension"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("CFG Rescale", open=True, elem_id="cfg_rescale"):
            rescale = gr.Slider(label="CFG Rescale", show_label=False, minimum=0.0, maximum=1.0, step=0.01, value=0.0)
            trailing = gr.Checkbox(label="DDIM Trailing", default=False)
        self.infotext_fields = [
            (rescale, "CFG Rescale"),
            (trailing, "DDIM Trailing"),
        ]
        self.paste_field_names = []
        for _, field_name in self.infotext_fields:
            self.paste_field_names.append(field_name)
        return [rescale, trailing]

    def cfg_replace(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(denoised_uncond)
        fi = globals()['cfg_rescale_fi']

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                if fi == 0:
                    denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)
                else:
                    xcfg = (denoised_uncond[i] + (x_out[cond_index] - denoised_uncond[i]) * (cond_scale * weight))
                    xrescaled = xcfg * (torch.std(x_out[cond_index]) / torch.std(xcfg))
                    xfinal = fi * xrescaled + (1.0 - fi) * xcfg
                    denoised[i] = xfinal

        return denoised

    def process(self, p, rescale, trailing):

        def schedule_override(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
            print("DDIM TRAILING OVERRIDE SUCCESSFUL")

            def timesteps_override(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
                c = -num_ddpm_timesteps / num_ddim_timesteps
                ddim_timesteps = np.round(np.flip(np.arange(num_ddpm_timesteps, 0, c)))
                steps_out = ddim_timesteps - 1
                if verbose:
                    print(f'Selected timesteps for ddim sampler: {steps_out}')
                return steps_out

            self.ddim_timesteps = timesteps_override(ddim_discr_method=ddim_discretize,
                                                     num_ddim_timesteps=ddim_num_steps,
                                                     num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
            alphas_cumprod = self.model.alphas_cumprod
            assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
            to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

            self.register_buffer('betas', to_torch(self.model.betas))
            self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
            self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

            # calculations for diffusion q(x_t | x_{t-1}) and others
            self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
            self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
            self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
            self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
            self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

            # ddim sampling parameters
            ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                       ddim_timesteps=self.ddim_timesteps,
                                                                                       eta=ddim_eta, verbose=verbose)
            self.register_buffer('ddim_sigmas', ddim_sigmas)
            self.register_buffer('ddim_alphas', ddim_alphas)
            self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
            self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
            sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
                (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
            self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

        @torch.no_grad()
        def p_sample_ddim_override(self, x, c, t, index, repeat_noise=False, use_original_steps=False,
                                   quantize_denoised=False,
                                   temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                                   unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None):
            b, *_, device = *x.shape, x.device

            def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
                """
                Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
                Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
                """
                std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
                std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
                # rescale the results from guidance (fixes overexposure)
                noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
                # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
                noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
                return noise_cfg
            
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                model_output = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                if isinstance(c, dict):
                    assert isinstance(unconditional_conditioning, dict)
                    c_in = dict()
                    for k in c:
                        if isinstance(c[k], list):
                            c_in[k] = [torch.cat([
                                unconditional_conditioning[k][i],
                                c[k][i]]) for i in range(len(c[k]))]
                        else:
                            c_in[k] = torch.cat([
                                    unconditional_conditioning[k],
                                    c[k]])
                elif isinstance(c, list):
                    c_in = list()
                    assert isinstance(unconditional_conditioning, list)
                    for i in range(len(c)):
                        c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
                else:
                    c_in = torch.cat([unconditional_conditioning, c])
                model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
                fi = globals()['cfg_rescale_fi']
                if fi > 0:
                    model_output = rescale_noise_cfg(model_output, model_t, guidance_rescale=fi)

            if self.model.parameterization == "v":
                e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t = model_output

            if score_corrector is not None:
                assert self.model.parameterization == "eps", 'not implemented'
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            if self.model.parameterization != "v":
                pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            else:
                pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

            if dynamic_threshold is not None:
                raise NotImplementedError()

            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        if trailing:
            ddim.DDIMSampler.make_schedule = schedule_override
            p.extra_generation_params["DDIM Trailing"] = True

        if globals()['enable_furry_cocks']:
            globals()['cfg_rescale_fi'] = rescale
        globals()['enable_furry_cocks'] = True
        sd_samplers_kdiffusion.CFGDenoiser.combine_denoised = self.cfg_replace

        if rescale > 0:
            ddim.DDIMSampler.p_sample_ddim = p_sample_ddim_override
            p.extra_generation_params["CFG Rescale"] = rescale

    def postprocess(self, p, processed, rescale, trailing):
        sd_samplers_kdiffusion.CFGDenoiser.combine_denoised = self.old_denoising
        ddim.DDIMSampler.make_schedule = self.old_schedule
        ddim.DDIMSampler.p_sample_ddim = self.old_sample

def on_infotext_pasted(infotext, params):
    if "CFG Rescale" not in params:
        params["CFG Rescale"] = 0

        if "CFG Rescale φ" in params:
            params["CFG Rescale"] = params["CFG Rescale φ"]
            del params["CFG Rescale φ"]

        if "CFG Rescale phi" in params and scripts.scripts_txt2img.script("Neutral Prompt") is None:
            params["CFG Rescale"] = params["CFG Rescale phi"]
            del params["CFG Rescale phi"]

    if "DDIM Trailing" not in params:
        params["DDIM Trailing"] = False

script_callbacks.on_infotext_pasted(on_infotext_pasted)
