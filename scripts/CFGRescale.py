import math
import random
import torch

import gradio as gr
import modules.scripts as scripts
from modules import devices, deepbooru, images, processing, shared, sd_samplers_kdiffusion
from modules.processing import Processed
from modules.shared import opts, state

from PIL import Image
import copy

global pixmap
global xn


class Script(scripts.Script):

    def __init__(self):
        self.fi = 0
        self.old_denoising = None

    def title(self):
        return "CFG Rescale Extension"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        rescale = gr.Slider(minimum=0, maximum=1, step=0.01, label='CFG Rescale', value=0, elem_id=self.elem_id("rescale_ext"), visible=True, interactive=True)
        return [rescale]

    def cfg_replace(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(denoised_uncond)
        fi = self.fi

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

    def process(self, p, rescale):
        self.fi = rescale
        sd_samplers_kdiffusion.CFGDenoiser.fi = rescale
        if self.old_denoising is None:
            self.old_denoising = sd_samplers_kdiffusion.CFGDenoiser.combine_denoised

        sd_samplers_kdiffusion.CFGDenoiser.combine_denoised = self.cfg_replace

    def postprocess(self, p, processed, rescale):
        if self.old_denoising is not None:
            sd_samplers_kdiffusion.CFGDenoiser.combine_denoised = self.old_denoising
