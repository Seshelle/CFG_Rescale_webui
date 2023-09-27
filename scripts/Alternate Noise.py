import math
import random

import gradio as gr
import modules.scripts as scripts
from modules import devices, deepbooru, images, processing, shared
from modules.processing import Processed
from modules.shared import opts, state

from PIL import Image
import copy

global pixmap
global xn


class Script(scripts.Script):

    def __init__(self):
        self.scalingW = 0
        self.scalingH = 0
        self.hr_denoise = 0
        self.hr_steps = 0
        self.scaler = ""

    def title(self):
        return "Alternate Init Noise"

    def show(self, is_img2img):
        if not is_img2img:
            return scripts.AlwaysVisible
        return False

    def ui(self, is_img2img):

        noise_types = [
            "Plasma Noise",
            "FBM Noise"
        ]

        with gr.Accordion('Alternate Init Noise', open=False):
            enabled = gr.Checkbox(label="Enabled", default=False)

            noise_type = gr.Dropdown(label="Type", choices=[k for k in noise_types], type="index", value=next(iter(noise_types)))

            # Plasma noise settings
            turbulence = gr.Slider(minimum=0.05, maximum=10.0, step=0.05, label='Turbulence', value=4, elem_id=self.elem_id("turbulence"), visible=True, interactive=True)

            # FBM noise settings
            octaves = gr.Slider(minimum=1, maximum=32, step=1, label='Octaves', value=6, elem_id=self.elem_id("octaves"), visible=False, interactive=True)
            smoothing = gr.Slider(minimum=1, maximum=100, step=1, label='Smoothing', value=1, elem_id=self.elem_id("smoothing"), visible=False, interactive=True)
            octave_division = gr.Slider(minimum=1.0, maximum=10.0, step=0.01, label='Octave Division', value=2, elem_id=self.elem_id("octave_division"), visible=False, interactive=True)

            grain = gr.Slider(minimum=0, maximum=256, step=1, label='Grain', value=0, elem_id=self.elem_id("grain"))
            denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.9, elem_id=self.elem_id("denoising"))
            noise_mult = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Noise multiplier', value=1.0, elem_id=self.elem_id("noise_mult"))

            with gr.Accordion('Color Adjustments', open=False):
                with gr.Row():
                    val_min = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Brightness Min", elem_id=self.elem_id("plasma_val_min"))
                    val_max = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Brightness Max", elem_id=self.elem_id("plasma_val_max"))
                with gr.Row():
                    red_min = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Red Min", elem_id=self.elem_id("plasma_red_min"))
                    red_max = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Red Max", elem_id=self.elem_id("plasma_red_max"))
                with gr.Row():
                    grn_min = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Green Min", elem_id=self.elem_id("plasma_grn_min"))
                    grn_max = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Green Max", elem_id=self.elem_id("plasma_grn_max"))
                with gr.Row():
                    blu_min = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Blue Min", elem_id=self.elem_id("plasma_blu_min"))
                    blu_max = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Blue Max", elem_id=self.elem_id("plasma_blu_max"))
                contrast = gr.Slider(minimum=0, maximum=10, step=0.1, value=1, label="Contrast", elem_id=self.elem_id("noise_contrast"))
                greyscale = gr.Checkbox(value=False, label="Greyscale", interactive=True, elem_id="noise_greyscale")

            with gr.Row():
                single_seed = gr.Checkbox(label="One seed for entire batch", info="speeds up noise generation for batch_size > 1, but noise seeds won't always match image seeds", default=False)
                seed_choice = gr.Textbox(label="Seed override", value=-1, interactive=True, elem_id=self.elem_id("seed_choice"), visible=False)

            def select_noise_type(noise_index):
                return [gr.update(visible=noise_index == 0),
                        gr.update(visible=noise_index == 1),
                        gr.update(visible=noise_index == 1),
                        gr.update(visible=noise_index == 1)]

            def show_seed_choice(seed_checked):
                return gr.update(visible=seed_checked)

            noise_type.change(
                fn=select_noise_type,
                inputs=noise_type,
                outputs=[turbulence, octaves, smoothing, octave_division]
            )

            single_seed.change(
                fn=show_seed_choice,
                inputs=single_seed,
                outputs=seed_choice
            )

        return [enabled, noise_type, turbulence, octaves, smoothing, octave_division, grain, denoising, noise_mult, val_min, val_max, red_min, red_max, grn_min, grn_max,
                blu_min, blu_max, contrast, greyscale, single_seed, seed_choice]

    def remap(self, v, low2, high2, contrast):
        v = abs(v)
        v = contrast * (v - 128) + 128
        return int(low2 + v * (high2 - low2) / (255))

    def create_plasma(self, p, seed, turbulence, grain, val_min, val_max, red_min, red_max, grn_min,
                grn_max, blu_min, blu_max, contrast, greyscale):
        global pixmap
        global xn
        xn = 0

        w = p.width
        h = p.height
        random.seed(seed)
        aw = copy.deepcopy(w)
        ah = copy.deepcopy(h)
        image = Image.new("RGB", (aw, ah))
        if w >= h:
            h = w
        else:
            w = h

        # Clamp per channel and globally
        clamp_v_min = val_min
        clamp_v_max = val_max
        clamp_r_min = red_min
        clamp_r_max = red_max
        clamp_g_min = grn_min
        clamp_g_max = grn_max
        clamp_b_min = blu_min
        clamp_b_max = blu_max

        # Handle value clamps
        lv = 0
        mv = 0
        if clamp_v_min == -1:
            lv = 0
        else:
            lv = clamp_v_min

        if clamp_v_max == -1:
            mv = 255
        else:
            mv = clamp_v_max

        lr = 0
        mr = 0
        if clamp_r_min == -1:
            lr = lv
        else:
            lr = clamp_r_min

        if clamp_r_max == -1:
            mr = mv
        else:
            mr = clamp_r_max

        lg = 0
        mg = 0
        if clamp_g_min == -1:
            lg = lv
        else:
            lg = clamp_g_min

        if clamp_g_max == -1:
            mg = mv
        else:
            mg = clamp_g_max

        lb = 0
        mb = 0
        if clamp_b_min == -1:
            lb = lv
        else:
            lb = clamp_b_min

        if clamp_b_max == -1:
            mb = mv
        else:
            mb = clamp_b_max

        roughness = turbulence

        def adjust(xa, ya, x, y, xb, yb):
            global pixmap
            if (pixmap[x][y] == 0):
                d = math.fabs(xa - xb) + math.fabs(ya - yb)
                v = (pixmap[xa][ya] + pixmap[xb][yb]) / 2.0 + (random.random() - 0.555) * d * roughness
                c = int(math.fabs(v + (random.random() - 0.5) * grain))
                if c < 0:
                    c = 0
                elif c > 255:
                    c = 255
                pixmap[x][y] = c

        def subdivide(x1, y1, x2, y2):
            global pixmap
            if (not ((x2 - x1 < 2.0) and (y2 - y1 < 2.0))):
                x = int((x1 + x2) / 2.0)
                y = int((y1 + y2) / 2.0)
                adjust(x1, y1, x, y1, x2, y1)
                adjust(x2, y1, x2, y, x2, y2)
                adjust(x1, y2, x, y2, x2, y2)
                adjust(x1, y1, x1, y, x1, y2)
                if (pixmap[x][y] == 0):
                    v = int((pixmap[x1][y1] + pixmap[x2][y1] + pixmap[x2][y2] + pixmap[x1][y2]) / 4.0)
                    pixmap[x][y] = v

                subdivide(x1, y1, x, y)
                subdivide(x, y1, x2, y)
                subdivide(x, y, x2, y2)
                subdivide(x1, y, x, y2)

        pixmap = [[0 for i in range(h)] for j in range(w)]
        pixmap[0][0] = int(random.random() * 255)
        pixmap[w - 1][0] = int(random.random() * 255)
        pixmap[w - 1][h - 1] = int(random.random() * 255)
        pixmap[0][h - 1] = int(random.random() * 255)
        subdivide(0, 0, w - 1, h - 1)
        r = copy.deepcopy(pixmap)

        if not greyscale:
            pixmap = [[0 for i in range(h)] for j in range(w)]
            pixmap[0][0] = int(random.random() * 255)
            pixmap[w - 1][0] = int(random.random() * 255)
            pixmap[w - 1][h - 1] = int(random.random() * 255)
            pixmap[0][h - 1] = int(random.random() * 255)
            subdivide(0, 0, w - 1, h - 1)
            g = copy.deepcopy(pixmap)

            pixmap = [[0 for i in range(h)] for j in range(w)]
            pixmap[0][0] = int(random.random() * 255)
            pixmap[w - 1][0] = int(random.random() * 255)
            pixmap[w - 1][h - 1] = int(random.random() * 255)
            pixmap[0][h - 1] = int(random.random() * 255)
            subdivide(0, 0, w - 1, h - 1)
            b = copy.deepcopy(pixmap)

        for y in range(ah):
            for x in range(aw):
                if greyscale:
                    channel_r = self.remap(r[x][y], lr, mr, contrast)
                    final_pix = (channel_r, channel_r, channel_r)
                else:
                    final_pix = (self.remap(r[x][y], lr, mr, contrast),
                                 self.remap(g[x][y], lg, mg, contrast),
                                 self.remap(b[x][y], lb, mb, contrast))
                image.putpixel((x, y), final_pix)

        return image

    def createFBM(self, p, seed, octaves, smoothing, octave_division, grain, denoising, noise_mult, val_min, val_max, red_min, red_max, grn_min, grn_max,
                blu_min, blu_max, contrast, greyscale):

        random.seed(seed)
        width = p.width
        height = p.height
        square = max(width, height)
        max_octaves = 1
        octave_pixel_size = 1
        while True:
            octave_pixel_size *= octave_division
            if octave_pixel_size < square / octave_division:
                max_octaves += 1
                if max_octaves >= octaves:
                    break
            else:
                break
        octaves = min(octaves, max_octaves)

        mr = max(val_min, red_min, 0)
        mg = max(val_min, grn_min, 0)
        mb = max(val_min, blu_min, 0)
        hv = 255
        if val_max >= 0:
            hv = val_max
        hr = 255
        if red_max >= 0:
            hr = val_max
        hg = 255
        if grn_max >= 0:
            hg = val_max
        hb = 255
        if blu_max >= 0:
            hb = val_max

        if grain > 0:
            grain_image_r = [[0 for i in range(height)] for j in range(width)]
            grain_image_g = [[0 for i in range(height)] for j in range(width)]
            grain_image_b = [[0 for i in range(height)] for j in range(width)]
            for y in range(height):
                for x in range(width):
                    grain_image_r[x][y] = int((random.random() - 0.5) * grain)
                    if not greyscale:
                        grain_image_g[x][y] = int((random.random() - 0.5) * grain)
                        grain_image_b[x][y] = int((random.random() - 0.5) * grain)

        final_image = Image.new("RGB", (width, height))

        for o in range(octaves):
            a = smoothing * pow(octave_division, octaves - o - 1)

            s = int(square / a)
            if s > square:
                break

            r = [[0 for i in range(s)] for j in range(s)]
            g = [[0 for i in range(s)] for j in range(s)]
            b = [[0 for i in range(s)] for j in range(s)]

            octave_image = Image.new("RGB", (s, s))
            for y in range(s):
                for x in range(s):
                    r[x][y] = int(random.random() * 255)
                    if not greyscale:
                        g[x][y] = int(random.random() * 255)
                        b[x][y] = int(random.random() * 255)

            for y in range(s):
                for x in range(s):
                    octave_image.putpixel((x, y), (r[x][y], g[x][y], b[x][y]))

            octave_image = octave_image.resize((square, square), Image.BILINEAR)

            for y in range(height):
                for x in range(width):
                    old_pix = final_image.getpixel((x, y))
                    new_pix = octave_image.getpixel((x, y))
                    amplitude = 1 / pow(2, o + 1)
                    new_pix = (int(new_pix[0] * amplitude), int(new_pix[1] * amplitude), int(new_pix[2] * amplitude))
                    if grain > 0 and o == octaves - 1:
                        if greyscale:
                            channel_r = self.remap(old_pix[0] + new_pix[0] + grain_image_r[x][y], mr, hr, contrast)
                            final_pix = (channel_r, channel_r, channel_r)
                        else:
                            final_pix = (self.remap(old_pix[0] + new_pix[0] + grain_image_r[x][y], mr, hr, contrast),
                                         self.remap(old_pix[1] + new_pix[1] + grain_image_g[x][y], mg, hg, contrast),
                                         self.remap(old_pix[2] + new_pix[2] + grain_image_b[x][y], mb, hb, contrast))
                    elif o == octaves - 1:
                        if greyscale:
                            channel_r = self.remap(old_pix[0] + new_pix[0], mr, hr, contrast)
                            final_pix = (channel_r, channel_r, channel_r)
                        else:
                            final_pix = (self.remap(old_pix[0] + new_pix[0], mr, hr, contrast),
                                         self.remap(old_pix[1] + new_pix[1], mg, hg, contrast),
                                         self.remap(old_pix[2] + new_pix[2], mb, hb, contrast))
                    else:
                        if greyscale:
                            channel_r = old_pix[0] + new_pix[0]
                            final_pix = (channel_r, channel_r, channel_r)
                        else:
                            final_pix = (old_pix[0] + new_pix[0],
                                         old_pix[1] + new_pix[1],
                                         old_pix[2] + new_pix[2])
                    final_image.putpixel((x, y), final_pix)

        return final_image

    def process(self, p, enabled, noise_type, turbulence, octaves, smoothing, octave_division, grain, denoising, noise_mult, val_min, val_max, red_min, red_max, grn_min, grn_max,
                blu_min, blu_max, contrast, greyscale, single_seed, seed_choice):
        if not enabled or "alt_hires" in p.extra_generation_params:
            return None

        if p.enable_hr:
            self.hr_denoise = p.denoising_strength
            self.hr_steps = p.hr_second_pass_steps
            if self.hr_steps == 0:
                self.hr_steps = p.steps
            if p.hr_resize_x == 0 and p.hr_resize_y == 0:
                self.scalingW = p.hr_scale
                self.scalingH = p.hr_scale
            else:
                self.scalingW = p.hr_resize_x
                self.scalingH = p.hr_resize_y
            self.scaler = p.hr_upscaler
        else:
            self.scalingW = 0

        # image size
        p.__class__ = processing.StableDiffusionProcessingImg2Img
        dummy = processing.StableDiffusionProcessingImg2Img()
        for k, v in dummy.__dict__.items():
            if hasattr(p, k):
                continue
            setattr(p, k, v)

        p.extra_generation_params["Grain"] = grain
        p.extra_generation_params["Alt denoising strength"] = denoising
        p.extra_generation_params["Value Min"] = val_min
        p.extra_generation_params["Value Max"] = val_max
        p.extra_generation_params["Red Min"] = red_min
        p.extra_generation_params["Red Max"] = red_max
        p.extra_generation_params["Green Min"] = grn_min
        p.extra_generation_params["Green Max"] = grn_max
        p.extra_generation_params["Blue Min"] = blu_min
        p.extra_generation_params["Blue Max"] = blu_max
        p.initial_noise_multiplier = noise_mult
        p.denoising_strength = float(denoising)

        img_num = p.batch_size
        if single_seed:
            img_num = 1

        p.init_images = []
        if int(seed_choice) == -1 or not single_seed:
            init_seed = p.all_seeds[0]
        else:
            init_seed = int(seed_choice)

        for img in range(img_num):
            real_seed = init_seed + img
            if noise_type == 0:
                # plasma noise
                p.extra_generation_params["Alt noise type"] = "Plasma"
                p.extra_generation_params["Turbulence"] = turbulence
                image = self.create_plasma(p, real_seed, turbulence, grain, val_min, val_max, red_min, red_max, grn_min,
                    grn_max, blu_min, blu_max, contrast, greyscale)
            if noise_type == 1:
                # fbm noise
                p.extra_generation_params["Alt noise type"] = "FBM"
                p.extra_generation_params["Octaves"] = octaves
                p.extra_generation_params["Smoothing"] = smoothing
                image = self.createFBM(p, real_seed, octaves, smoothing, octave_division, grain, denoising, noise_mult, val_min, val_max, red_min, red_max, grn_min, grn_max,
                    blu_min, blu_max, contrast, greyscale)

            p.init_images.append(image)

    def postprocess(self, p, processed, enabled, noise_type, turbulence, octaves, smoothing, octave_division, grain, denoising, noise_mult, val_min, val_max, red_min, red_max, grn_min, grn_max,
                blu_min, blu_max, contrast, greyscale, single_seed, seed_choice):
        if not enabled or self.scalingW == 0 or "alt_hires" in p.extra_generation_params or not p.enable_hr:
            return None
        devices.torch_gc()

        new_p = p
        new_p.init_images = []
        for i in range(len(processed.images)):
            new_p.init_images.append(processed.images[i])

        new_p.extra_generation_params["alt_hires"] = self.scalingW
        new_p.width = int(new_p.width * self.scalingW)
        new_p.height = int(new_p.height * self.scalingH)
        new_p.denoising_strength = self.hr_denoise
        
        if new_p.denoising_strength > 0:
            new_p.steps = max(1, int(self.hr_steps / self.hr_denoise - 0.5))
        else:
            new_p.steps = 0

        p.resize_mode = 3 if 'Latent' in self.scaler else 0
        new_p.scripts = None
        new_p = processing.process_images(new_p)
        processed.images = new_p.images
