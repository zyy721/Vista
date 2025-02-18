import torch
import torch.nn as nn
from packaging import version
from einops import rearrange

from vwm.util import repeat_as_img_seq

OPENAIUNETWRAPPER = "vwm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if version.parse(torch.__version__) >= version.parse("2.0.0") and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
            self, x: torch.Tensor, t: torch.Tensor, c: dict, cond_mask: torch.Tensor, num_frames: int, **kwargs
    ) -> torch.Tensor:

        if len(x.shape) == 5:
            x = rearrange(x, "(b t) cam_n c h w -> (b cam_n t) c h w", t=num_frames, cam_n=6)

        if "concat" in c and num_frames > 1 and c["concat"].shape[0] != x.shape[0]:
            assert c["concat"].shape[0] == x.shape[0] // num_frames, f"{c['concat'].shape} {x.shape}"
            c["concat"] = repeat_as_img_seq(c["concat"], num_frames)
        x = torch.cat((x, c.get("concat", torch.Tensor(list()).type_as(x))), dim=1)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            cond_mask=cond_mask,
            num_frames=num_frames,
            **kwargs
        )
