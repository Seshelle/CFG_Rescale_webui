import math
import torch

import gradio as gr
import numpy as np
import modules.scripts as scripts
from modules import devices, images, processing, shared, sd_samplers_kdiffusion, sd_samplers_compvis, script_callbacks, sd_samplers_cfg_denoiser
from modules.processing import Processed
from modules.shared import opts, state
from ldm.models.diffusion import ddim
from PIL import Image

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
            with gr.Row():
                recolor = gr.Checkbox(label="Auto Color Fix", default=False)
                rec_strength = gr.Slider(label="Fix Strength", interactive=True, visible=False, elem_id=self.elem_id("rec_strength"), minimum=0.1, maximum=10.0, step=0.1, value=1.0)

            def show_recolor_strength(rec_checked):
                return gr.update(visible=rec_checked)

            recolor.change(
                fn=show_recolor_strength,
                inputs=recolor,
                outputs=rec_strength
            )

        self.infotext_fields = [
            (rescale, "CFG Rescale"),
            (recolor, "Auto Color Fix")
        ]
        self.paste_field_names = []
        for _, field_name in self.infotext_fields:
            self.paste_field_names.append(field_name)
        return [rescale, recolor, rec_strength]

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
                    xrescaled = (torch.std(x_out[cond_index]) / torch.std(xcfg))
                    xfinal = fi * xrescaled + (1.0 - fi)
                    denoised[i] = xfinal * xcfg

        return denoised

    def process(self, p, rescale, recolor, rec_strength):

        if globals()['enable_furry_cocks']:
            globals()['cfg_rescale_fi'] = rescale
        globals()['enable_furry_cocks'] = True
        sd_samplers_kdiffusion.CFGDenoiser.combine_denoised = self.cfg_replace
        # sd_samplers_cfg_denoiser.CFGDenoiser.combine_denoised = self.cfg_replace

        if rescale > 0:
            p.extra_generation_params["CFG Rescale"] = rescale

        if recolor:
            p.extra_generation_params["Auto Color Fix Strength"] = rec_strength

    def postprocess(self, p, processed, rescale, recolor, rec_strength):
        sd_samplers_kdiffusion.CFGDenoiser.combine_denoised = self.old_denoising

        def postfix(img, rec_strength):
            p = 0.0005 * rec_strength
            r, g, b = img.split()
            #r_min, r_max = np.percentile(r, p), np.percentile(r, 100.0 - p)
            #g_min, g_max = np.percentile(g, p), np.percentile(g, 100.0 - p)
            #b_min, b_max = np.percentile(b, p), np.percentile(b, 100.0 - p)
            rh, rbins = np.histogram(r, 256, (0, 256))
            tmp = np.where(rh > rh.sum() * p)[0]
            r_min = tmp.min()
            r_max = tmp.max()

            gh, gbins = np.histogram(g, 256, (0, 256))
            tmp = np.where(gh > gh.sum() * p)[0]
            g_min = tmp.min()
            g_max = tmp.max()

            bh, bbins = np.histogram(b, 256, (0, 256))
            tmp = np.where(bh > bh.sum() * p)[0]
            b_min = tmp.min()
            b_max = tmp.max()

            for i in range(img.width):  # for every pixel:
                for j in range(img.height):
                    pix = img.getpixel((i, j))
                    tmp = [0] * 3
                    #tmp[0] = int((np.clip(pix[0], r_min, r_max) - r_min) / (r_max - r_min) * 255)
                    #tmp[1] = int((np.clip(pix[1], g_min, g_max) - g_min) / (g_max - g_min) * 255)
                    #tmp[2] = int((np.clip(pix[2], b_min, b_max) - b_min) / (b_max - b_min) * 255)
                    tmp[0] = int(255 * (min(max(pix[0], r_min), r_max) - r_min) / (r_max - r_min))
                    tmp[1] = int(255 * (min(max(pix[1], g_min), g_max) - g_min) / (g_max - g_min))
                    tmp[2] = int(255 * (min(max(pix[2], b_min), b_max) - b_min) / (b_max - b_min))
                    img.putpixel((i, j), (tmp[0], tmp[1], tmp[2]))

            return img

        if recolor:
            for i in range(len(processed.images)):
                processed.images[i] = postfix(processed.images[i], rec_strength)

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
