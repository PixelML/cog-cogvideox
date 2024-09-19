from cog import BasePredictor, Path, Input
import subprocess
from time import time, sleep
t0 = time()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from pget import pget_manifest

print("[⏰] Imports:", round(time() - t0, 2), "seconds")

sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:

You will only ever output a single video description per user request.

When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""

class Predictor(BasePredictor):

    def setup(self):
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX-5b-I2V",
            torch_dtype=torch.bfloat16
        )

        # self.pipe.enable_sequential_cpu_offload()
        # self.pipe.vae.enable_tiling()
        # self.pipe.vae.enable_slicing() 

    def predict(self,
        prompt: str = Input(description="Prompt"),
        image: str = Input(description="Image"),
    ) -> Path:
        print(subprocess.check_output(["nvidia-smi"]).decode("utf-8"))

        image = load_image(image=image)
        video = self.pipe(
            prompt=prompt,
            image=image,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]

        export_to_video(video, "output.mp4", fps=8)
        return Path("output.mp4")
