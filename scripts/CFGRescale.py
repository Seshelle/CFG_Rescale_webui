import math
import torch
import re
import gradio as gr
import numpy as np
import modules.scripts as scripts
import modules.images as saving
from modules import devices, processing, shared, sd_samplers_kdiffusion, sd_samplers_compvis, script_callbacks
from modules.processing import Processed
from modules.shared import opts, state
from ldm.models.diffusion import ddim
from PIL import Image

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, noise_like

re_prompt_cfgr = re.compile(r"<cfg_rescale:([^>]+)>")

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
                rec_strength = gr.Slider(label="Fix Strength", interactive=True, visible=False,
                                         elem_id=self.elem_id("rec_strength"), minimum=0.1, maximum=10.0, step=0.1,
                                         value=1.0)
                # Renamed the option, since that seems to be your original intention
                # That said, this option is still kinda broken. it shows two "original images" on the grid, instead of showing one of each
                # But it properly loads both images in the results, just not on the grid.
                # Not sure what is going on and how to fix it.
                # Batch grids will always show the original images, regardless of this option.
                # XYZ grids seems to be applying the colorfix if this option is disabled.
                show_original = gr.Checkbox(label="Show Original Images in grid", elem_id=self.elem_id("show_original"), visible=False, default=False)
                
                # Added a new one to actually keep the original images or not
                keep_original = gr.Checkbox(label="Keep Original Images", elem_id=self.elem_id("keep_original"), visible=False, default=False)

            def show_recolor_strength(rec_checked):
                return [gr.update(visible=rec_checked), gr.update(visible=rec_checked), gr.update(visible=rec_checked)]

            recolor.change(
                fn=show_recolor_strength,
                inputs=recolor,
                outputs=[rec_strength, show_original, keep_original]
            )

        self.infotext_fields = [
            (rescale, "CFG Rescale"),
            (recolor, "Auto Color Fix")
        ]
        self.paste_field_names = []
        for _, field_name in self.infotext_fields:
            self.paste_field_names.append(field_name)
        return [rescale, recolor, rec_strength, show_original, keep_original]

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

    def process(self, p, rescale, recolor, rec_strength, show_original, keep_original):

        if globals()['enable_furry_cocks']:
            # parse <cfg_rescale:[number]> from prompt for override
            rescale_override = None
            def found(m):
                nonlocal rescale_override
                try:
                    rescale_override = float(m.group(1))
                except ValueError:
                    rescale_override = None
                return ""
            p.prompt = re.sub(re_prompt_cfgr, found, p.prompt)
            if rescale_override is not None:
                rescale = rescale_override
            
            globals()['cfg_rescale_fi'] = rescale
        else:
            # rescale value is being set from xyz_grid
            rescale = globals()['cfg_rescale_fi']
        globals()['enable_furry_cocks'] = True

        sd_samplers_kdiffusion.CFGDenoiser.combine_denoised = self.cfg_replace

        if rescale > 0:
            p.extra_generation_params["CFG Rescale"] = rescale

        if recolor:
            p.extra_generation_params["Auto Color Fix Strength"] = rec_strength
            p.do_not_save_samples = True

    def postprocess_batch_list(self, p, pp, rescale, recolor, rec_strength, show_original, keep_original, batch_number):
        if recolor and show_original:
            num = len(pp.images)
            for i in range(num):
                pp.images.append(pp.images[i])
                p.prompts.append(p.prompts[i])
                p.negative_prompts.append(p.negative_prompts[i])
                p.seeds.append(p.seeds[i])
                p.subseeds.append(p.subseeds[i])

    def postprocess(self, p, processed, rescale, recolor, rec_strength, show_original, keep_original):
        sd_samplers_kdiffusion.CFGDenoiser.combine_denoised = self.old_denoising

        def postfix(img, rec_strength):
            prec = 0.0005 * rec_strength
            r, g, b = img.split()

            # softer effect
            # r_min, r_max = np.percentile(r, p), np.percentile(r, 100.0 - p)
            # g_min, g_max = np.percentile(g, p), np.percentile(g, 100.0 - p)
            # b_min, b_max = np.percentile(b, p), np.percentile(b, 100.0 - p)

            rh, rbins = np.histogram(r, 256, (0, 256))
            tmp = np.where(rh > rh.sum() * prec)[0]
            r_min = tmp.min()
            r_max = tmp.max()

            gh, gbins = np.histogram(g, 256, (0, 256))
            tmp = np.where(gh > gh.sum() * prec)[0]
            g_min = tmp.min()
            g_max = tmp.max()

            bh, bbins = np.histogram(b, 256, (0, 256))
            tmp = np.where(bh > bh.sum() * prec)[0]
            b_min = tmp.min()
            b_max = tmp.max()

            r = r.point(lambda i: int(255 * (min(max(i, r_min), r_max) - r_min) / (r_max - r_min)))
            g = g.point(lambda i: int(255 * (min(max(i, g_min), g_max) - g_min) / (g_max - g_min)))
            b = b.point(lambda i: int(255 * (min(max(i, b_min), b_max) - b_min) / (b_max - b_min)))

            new_img = Image.merge("RGB", (r, g, b))

            return new_img

        if recolor:
            grab = 0
            n_img = len(processed.images)
            for i in range(n_img):
                doit = False

                if show_original:
                    check = i
                    if opts.return_grid:
                        if i == 0:
                            continue
                        else:
                            check = check - 1
                    doit = check % (p.batch_size * 2) >= p.batch_size
                else:
                    if n_img > 1 and i != 0:
                        doit = True
                    elif n_img == 1 or not opts.return_grid:
                        doit = True

                if doit:
                    res_img = postfix(processed.images[i], rec_strength)
                    if opts.samples_save:
                        ind = grab
                        grab += 1
                        prompt_infotext = processing.create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds,
                                                                     index=ind)
                        # Save images to disk
                        if opts.samples_save:
                                # Both files were being saved as "colorfix".
                                # Also added a '-' before the suffix.
                                if keep_original:
                                    saving.save_image(processed.images[i], p.outpath_samples, "", seed=p.all_seeds[ind],
                                                    prompt=p.all_prompts[ind],
                                                    info=prompt_infotext, p=p, suffix="-original")
                                saving.save_image(res_img, p.outpath_samples, "", seed=p.all_seeds[ind],
                                                  prompt=p.all_prompts[ind],
                                                  info=prompt_infotext, p=p, suffix="-colorfix")

                    processed.images[i] = res_img


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
