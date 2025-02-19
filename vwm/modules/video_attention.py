from vwm.modules.attention import *
from vwm.modules.diffusionmodules.util import AlphaBlender, linear, timestep_embedding
import math

class TimeMixSequential(nn.Sequential):
    def forward(self, x, context=None, timesteps=None):
        for layer in self:
            x = layer(x, context, timesteps)
        return x


class VideoTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention  # ampere
    }

    def __init__(
            self,
            dim,
            n_heads,
            d_head,
            dropout=0.0,
            context_dim=None,
            gated_ff=True,
            use_checkpoint=False,
            timesteps=None,
            ff_in=False,
            inner_dim=None,
            attn_mode="softmax",
            disable_self_attn=False,
            disable_temporal_crossattention=False,
            switch_temporal_ca_to_sa=False,
            add_lora=False,
            action_control=False
    ):
        super().__init__()
        self._args = {k: v for k, v in locals().items() if k != "self" and not k.startswith("_")}

        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        assert int(n_heads * d_head) == inner_dim

        self.is_res = inner_dim == dim

        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff)

        self.timesteps = timesteps
        self.disable_self_attn = disable_self_attn
        if disable_self_attn:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                context_dim=context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                add_lora=add_lora
            )  # is a cross-attn
        else:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                causal=False,
                add_lora=add_lora
            )  # is a self-attn

        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)

        if not disable_temporal_crossattention:
            self.norm2 = nn.LayerNorm(inner_dim)
            if switch_temporal_ca_to_sa:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                    causal=False,
                    add_lora=add_lora
                )  # is a self-attn
            else:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    context_dim=context_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                    add_lora=add_lora,
                    action_control=action_control
                )  # is self-attn if context is None

        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, timesteps: int = None) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, x, context, timesteps)
        else:
            return self._forward(x, context, timesteps=timesteps)

    def _forward(self, x, context=None, timesteps=None):
        assert self.timesteps or timesteps
        assert not (self.timesteps and timesteps) or self.timesteps == timesteps
        timesteps = self.timesteps or timesteps
        B, S, C = x.shape
        x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        if self.disable_self_attn:
            x = self.attn1(self.norm1(x), context=context, batchify_xformers=True) + x
        else:  # this way
            x = self.attn1(self.norm1(x), batchify_xformers=True) + x

        if hasattr(self, "attn2"):
            if self.switch_temporal_ca_to_sa:
                x = self.attn2(self.norm2(x), batchify_xformers=True) + x
            else:  # this way
                x = self.attn2(self.norm2(x), context=context, batchify_xformers=True) + x

        x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        x = rearrange(x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps)
        return x

    def get_last_layer(self):
        return self.ff.net[-1].weight


class SpatialVideoTransformer(SpatialTransformer):
    def __init__(
            self,
            in_channels,
            n_heads,
            d_head,
            depth=1,
            dropout=0.0,
            use_linear=False,
            context_dim=None,
            use_spatial_context=False,
            timesteps=None,
            merge_strategy: str = "fixed",
            merge_factor: float = 0.5,
            time_context_dim=None,
            ff_in=False,
            use_checkpoint=False,
            time_depth=1,
            attn_mode="softmax",
            disable_self_attn=False,
            disable_temporal_crossattention=False,
            max_time_embed_period=10000,
            add_lora=False,
            action_control=False
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            use_checkpoint=use_checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
            add_lora=add_lora,
            action_control=action_control
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        self.time_stack = nn.ModuleList(
            [
                VideoTransformerBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    timesteps=timesteps,
                    use_checkpoint=use_checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    attn_mode=attn_mode,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                    add_lora=add_lora,
                    action_control=action_control
                )
                for _ in range(self.depth)
            ]
        )

        assert len(self.time_stack) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = in_channels * 4
        self.time_pos_embed = nn.Sequential(
            linear(in_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, in_channels)
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            rearrange_pattern="b t -> (b t) 1 1"
        )

    def forward(
            self,
            x: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            time_context: Optional[torch.Tensor] = None,
            timesteps: Optional[int] = None
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context

        if self.use_spatial_context:
            assert context.ndim == 3, f"Dims of spatial context should be 3 but are {context.ndim}"

            time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(time_context_first_timestep, "b ... -> (b n) ...", n=h * w)
        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> (b t)", b=x.shape[0] // timesteps)
        t_emb = timestep_embedding(
            num_frames,
            self.in_channels,
            repeat_only=False,
            max_period=self.max_time_embed_period
        )
        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None]

        for block, mix_block in zip(self.transformer_blocks, self.time_stack):
            x = block(x, context=spatial_context)

            x_mix = x
            x_mix = x_mix + emb

            x_mix = mix_block(x_mix, context=time_context, timesteps=timesteps)
            x = self.time_mixer(x_spatial=x, x_temporal=x_mix)

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out

# Multi-View

from typing import Any, Dict, List, Optional, Tuple, Union

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def _ensure_kv_is_int(view_pair: dict):
    """yaml key can be int, while json cannot. We convert here.
    """
    new_dict = {}
    for k, v in view_pair.items():
        new_value = [int(vi) for vi in v]
        new_dict[int(k)] = new_value
    return new_dict


class GatedConnector(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        data = torch.zeros(dim)
        self.alpha = nn.parameter.Parameter(data)

    def forward(self, inx):
        # as long as last dim of input == dim, pytorch can auto-broad
        return F.tanh(self.alpha) * inx


class BasicMultiviewTransformerBlock(BasicTransformerBlock):
    # ATTENTION_MODES = {
    #     "softmax": CrossAttention,  # vanilla attention
    #     "softmax-xformers": MemoryEfficientCrossAttention  # ampere
    # }

    def __init__(
            self,
            dim,
            n_heads,
            d_head,
            dropout=0.0,
            context_dim=None,
            gated_ff=True,
            use_checkpoint=False,
            disable_self_attn=False,
            attn_mode="softmax",
            sdp_backend=None,
            add_lora=False,
            action_control=False,
            # multi_view
            neighboring_view_pair: Optional[Dict[int, List[int]]] = None,
            neighboring_attn_type: Optional[str] = "add",
            zero_module_type="zero_linear",    
            num_frames = None,
            origin_img_size = [320, 576],
    ):
        super().__init__(
            dim,
            n_heads,
            d_head,
            dropout,
            context_dim,
            gated_ff,
            use_checkpoint,
            disable_self_attn,
            attn_mode,
            sdp_backend,
            add_lora,
            action_control,            
        )

        self.num_frames = num_frames

        # assert attn_mode in self.ATTENTION_MODES
        # if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
        #     print(
        #         f"Attention mode `{attn_mode}` is not available. Falling back to native attention. "
        #         f"This is not a problem in Pytorch >= 2.0. You are running with PyTorch version {torch.__version__}"
        #     )
        #     attn_mode = "softmax"
        # elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
        #     print("We do not support vanilla attention anymore, as it is too expensive")
        #     if not XFORMERS_IS_AVAILABLE:
        #         assert (
        #             False
        #         ), "Please install xformers via e.g. `pip install xformers==0.0.16`"
        #     else:
        #         print("Falling back to xformers efficient attention")
        #         attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        # if version.parse(torch.__version__) >= version.parse("2.0.0"):
        #     assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        # else:
        #     assert sdp_backend is None
        # self.disable_self_attn = disable_self_attn
        # self.attn1 = attn_cls(
        #     query_dim=dim,
        #     context_dim=context_dim if self.disable_self_attn else None,
        #     heads=n_heads,
        #     dim_head=d_head,
        #     dropout=dropout,
        #     backend=sdp_backend,
        #     add_lora=add_lora
        # )  # is a self-attn if not self.disable_self_attn
        # self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # self.attn2 = attn_cls(
        #     query_dim=dim,
        #     context_dim=context_dim,
        #     heads=n_heads,
        #     dim_head=d_head,
        #     dropout=dropout,
        #     backend=sdp_backend,
        #     add_lora=add_lora,
        #     action_control=action_control
        # )  # is self-attn if context is None
        # self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        # self.norm3 = nn.LayerNorm(dim)
        # self.use_checkpoint = use_checkpoint
        # if self.use_checkpoint:
        #     print(f"{self.__class__.__name__} is using checkpointing")

        self.origin_img_size = origin_img_size

        self.neighboring_view_pair = _ensure_kv_is_int(neighboring_view_pair)
        self.neighboring_attn_type = neighboring_attn_type
        # multiview attention
        self.norm4 = nn.LayerNorm(dim)
        self.attn4 = attn_cls(
            query_dim=dim,
            context_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
            add_lora=add_lora,
        )  # is self-attn if context is None
        if zero_module_type == "zero_linear":
            # NOTE: zero_module cannot apply to successive layers.
            self.connector = zero_module(nn.Linear(dim, dim))
        elif zero_module_type == "gated":
            self.connector = GatedConnector(dim)
        elif zero_module_type == "none":
            # TODO: if this block is in controlnet, we may not need zero here.
            self.connector = lambda x: x
        else:
            raise TypeError(f"Unknown zero module type: {zero_module_type}")         

    @property
    def new_module(self):
        ret = {
            "norm4": self.norm4,
            "attn4": self.attn4,
        }
        if isinstance(self.connector, nn.Module):
            ret["connector"] = self.connector
        return ret

    @property
    def n_cam(self):
        return len(self.neighboring_view_pair)

    def _construct_attn_input(self, norm_hidden_states, h_img_size=None, w_img_size=None,):
        B = len(norm_hidden_states)
        # reshape, key for origin view, value for ref view
        hidden_states_in1 = []
        hidden_states_in2 = []
        cam_order = []
        # if self.neighboring_attn_type == "add":
        #     for key, values in self.neighboring_view_pair.items():
        #         for value in values:
        #             hidden_states_in1.append(norm_hidden_states[:, key])
        #             hidden_states_in2.append(norm_hidden_states[:, value])
        #             cam_order += [key] * B
        #     # N*2*B, H*W, head*dim
        #     hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
        #     hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
        #     cam_order = torch.LongTensor(cam_order)

        if self.neighboring_attn_type == "add":
            half_w_img_size = math.ceil(w_img_size / 2) 
            norm_hidden_states_split = rearrange(
                norm_hidden_states, "bt c (h w) d -> bt c h w d", h=h_img_size, w=w_img_size)
            norm_hidden_states_left = norm_hidden_states_split[:, :, :, :half_w_img_size, :]
            norm_hidden_states_left = rearrange(
                norm_hidden_states_left, "bt c h half_w d -> bt c (h half_w) d")
            norm_hidden_states_right = norm_hidden_states_split[:, :, :, -half_w_img_size:, :]
            norm_hidden_states_right = rearrange(
                norm_hidden_states_right, "bt c h half_w d -> bt c (h half_w) d")

            for key, values in self.neighboring_view_pair.items():
                # for value in values:
                #     hidden_states_in1.append(norm_hidden_states[:, key])
                #     hidden_states_in2.append(norm_hidden_states[:, value])
                #     cam_order += [key] * B

                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(norm_hidden_states_right[:, values[0]])
                cam_order += [key] * B

                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(norm_hidden_states_left[:, values[1]])
                cam_order += [key] * B

            # N*2*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)

        elif self.neighboring_attn_type == "concat":
            for key, values in self.neighboring_view_pair.items():
                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(torch.cat([
                    norm_hidden_states[:, value] for value in values
                ], dim=1))
                cam_order += [key] * B
            # N*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            # N*B, 2*H*W, head*dim
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "self":
            hidden_states_in1 = rearrange(
                norm_hidden_states, "b n l ... -> b (n l) ...")
            hidden_states_in2 = None
            cam_order = None
        else:
            raise NotImplementedError(
                f"Unknown type: {self.neighboring_attn_type}")
        return hidden_states_in1, hidden_states_in2, cam_order


    def forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        kwargs = {"x": x}

        if context is not None:
            kwargs.update({"context": context})

        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if n_times_crossframe_attn_in_self:
            kwargs.update({"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self})

        if self.use_checkpoint:
            # inputs = {"x": x, "context": context}
            # return checkpoint(self._forward, inputs, self.parameters(), self.use_checkpoint)
            return checkpoint(self._forward, x, context)
        else:
            return self._forward(**kwargs)


    def _forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        # spatial self-attn
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None,
                       additional_tokens=additional_tokens,
                       n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                       if not self.disable_self_attn else 0) + x
        # spatial cross-attn
        x = self.attn2(self.norm2(x), context=context, additional_tokens=additional_tokens) + x


        # multi-view cross attention
        hidden_states = x
        norm_hidden_states = self.norm4(hidden_states)

        # batch dim first, cam dim second
        # norm_hidden_states = rearrange(
        #     norm_hidden_states, '(b n) ... -> b n ...', n=self.n_cam)
        norm_hidden_states = rearrange(
            norm_hidden_states, '(b n t) ... -> (b t) n ...', n=self.n_cam, t=self.num_frames)
        
        downsample_rate = self.origin_img_size[0] * self.origin_img_size[1] // norm_hidden_states.shape[2] 
        downsample_rate = int(downsample_rate ** 0.5)
        h_img_size = self.origin_img_size[0] // downsample_rate
        w_img_size = self.origin_img_size[1] // downsample_rate
        assert h_img_size * w_img_size == norm_hidden_states.shape[2] 

        B = len(norm_hidden_states)
        # key is query in attention; value is key-value in attention
        hidden_states_in1, hidden_states_in2, cam_order = self._construct_attn_input(
            norm_hidden_states, h_img_size, w_img_size)
        # attention
        attn_raw_output = self.attn4(
            hidden_states_in1,
            context=hidden_states_in2,
            additional_tokens=additional_tokens,
        )
        # final output
        if self.neighboring_attn_type == "self":
            attn_output = rearrange(
                attn_raw_output, 'b (n l) ... -> b n l ...', n=self.n_cam)
        else:
            attn_output = torch.zeros_like(norm_hidden_states)
            for cam_i in range(self.n_cam):
                attn_out_mv = rearrange(attn_raw_output[cam_order == cam_i],
                                        '(n b) ... -> b n ...', b=B)
                attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
        # attn_output = rearrange(attn_output, 'b n ... -> (b n) ...')

        attn_output = rearrange(
            attn_output, '(b t) n ... -> (b n t) ...', n=self.n_cam, t=self.num_frames)

        # apply zero init connector (one layer)
        attn_output = self.connector(attn_output)
        # short-cut
        hidden_states = attn_output + hidden_states
        x = hidden_states


        # feedforward
        x = self.ff(self.norm3(x)) + x
        return x


class VideoMultiviewTransformerBlock(VideoTransformerBlock):
    # ATTENTION_MODES = {
    #     "softmax": CrossAttention,  # vanilla attention
    #     "softmax-xformers": MemoryEfficientCrossAttention  # ampere
    # }

    def __init__(
            self,
            dim,
            n_heads,
            d_head,
            dropout=0.0,
            context_dim=None,
            gated_ff=True,
            use_checkpoint=False,
            timesteps=None,
            ff_in=False,
            inner_dim=None,
            attn_mode="softmax",
            disable_self_attn=False,
            disable_temporal_crossattention=False,
            switch_temporal_ca_to_sa=False,
            add_lora=False,
            action_control=False,
            # multi_view
            neighboring_view_pair: Optional[Dict[int, List[int]]] = None,
            neighboring_attn_type: Optional[str] = "add",
            zero_module_type="zero_linear",    
            num_frames = None, 
            origin_img_size = [320, 576],
    ):
        super().__init__(
            dim,
            n_heads,
            d_head,
            dropout,
            context_dim,
            gated_ff,
            use_checkpoint,
            timesteps,
            ff_in,
            inner_dim,
            attn_mode,
            disable_self_attn,
            disable_temporal_crossattention,
            switch_temporal_ca_to_sa,
            add_lora,
            action_control,
        )
        self.num_frames = num_frames

        attn_cls = self.ATTENTION_MODES[attn_mode]

        # self.ff_in = ff_in or inner_dim is not None
        # if inner_dim is None:
        #     inner_dim = dim

        # assert int(n_heads * d_head) == inner_dim

        # self.is_res = inner_dim == dim

        # if self.ff_in:
        #     self.norm_in = nn.LayerNorm(dim)
        #     self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff)

        # self.timesteps = timesteps
        # self.disable_self_attn = disable_self_attn
        # if disable_self_attn:
        #     self.attn1 = attn_cls(
        #         query_dim=inner_dim,
        #         context_dim=context_dim,
        #         heads=n_heads,
        #         dim_head=d_head,
        #         dropout=dropout,
        #         add_lora=add_lora
        #     )  # is a cross-attn
        # else:
        #     self.attn1 = attn_cls(
        #         query_dim=inner_dim,
        #         heads=n_heads,
        #         dim_head=d_head,
        #         dropout=dropout,
        #         causal=False,
        #         add_lora=add_lora
        #     )  # is a self-attn

        # self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)

        # if not disable_temporal_crossattention:
        #     self.norm2 = nn.LayerNorm(inner_dim)
        #     if switch_temporal_ca_to_sa:
        #         self.attn2 = attn_cls(
        #             query_dim=inner_dim,
        #             heads=n_heads,
        #             dim_head=d_head,
        #             dropout=dropout,
        #             causal=False,
        #             add_lora=add_lora
        #         )  # is a self-attn
        #     else:
        #         self.attn2 = attn_cls(
        #             query_dim=inner_dim,
        #             context_dim=context_dim,
        #             heads=n_heads,
        #             dim_head=d_head,
        #             dropout=dropout,
        #             add_lora=add_lora,
        #             action_control=action_control
        #         )  # is self-attn if context is None

        # self.norm1 = nn.LayerNorm(inner_dim)
        # self.norm3 = nn.LayerNorm(inner_dim)
        # self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

        # self.use_checkpoint = use_checkpoint
        # if self.use_checkpoint:
        #     print(f"{self.__class__.__name__} is using checkpointing")


        self.origin_img_size = origin_img_size

        self.neighboring_view_pair = _ensure_kv_is_int(neighboring_view_pair)
        self.neighboring_attn_type = neighboring_attn_type
        # multiview attention
        self.norm4 = nn.LayerNorm(inner_dim)
        self.attn4 = attn_cls(
            query_dim=inner_dim,
            context_dim=inner_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            add_lora=add_lora
        )  # is a cross-attn

        if zero_module_type == "zero_linear":
            # NOTE: zero_module cannot apply to successive layers.
            self.connector = zero_module(nn.Linear(dim, dim))
        elif zero_module_type == "gated":
            self.connector = GatedConnector(dim)
        elif zero_module_type == "none":
            # TODO: if this block is in controlnet, we may not need zero here.
            self.connector = lambda x: x
        else:
            raise TypeError(f"Unknown zero module type: {zero_module_type}")


    @property
    def new_module(self):
        ret = {
            "norm4": self.norm4,
            "attn4": self.attn4,
        }
        if isinstance(self.connector, nn.Module):
            ret["connector"] = self.connector
        return ret

    @property
    def n_cam(self):
        return len(self.neighboring_view_pair)

    def _construct_attn_input(self, norm_hidden_states, h_img_size=None, w_img_size=None,):
        B = len(norm_hidden_states)
        # reshape, key for origin view, value for ref view
        hidden_states_in1 = []
        hidden_states_in2 = []
        cam_order = []
        # if self.neighboring_attn_type == "add":
        #     for key, values in self.neighboring_view_pair.items():
        #         for value in values:
        #             hidden_states_in1.append(norm_hidden_states[:, key])
        #             hidden_states_in2.append(norm_hidden_states[:, value])
        #             cam_order += [key] * B
        #     # N*2*B, H*W, head*dim
        #     hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
        #     hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
        #     cam_order = torch.LongTensor(cam_order)

        if self.neighboring_attn_type == "add":
            half_w_img_size = math.ceil(w_img_size / 2) 
            norm_hidden_states_split = rearrange(
                norm_hidden_states, "bt c (h w) d -> bt c h w d", h=h_img_size, w=w_img_size)
            norm_hidden_states_left = norm_hidden_states_split[:, :, :, :half_w_img_size, :]
            norm_hidden_states_left = rearrange(
                norm_hidden_states_left, "bt c h half_w d -> bt c (h half_w) d")
            norm_hidden_states_right = norm_hidden_states_split[:, :, :, -half_w_img_size:, :]
            norm_hidden_states_right = rearrange(
                norm_hidden_states_right, "bt c h half_w d -> bt c (h half_w) d")

            for key, values in self.neighboring_view_pair.items():
                # for value in values:
                #     hidden_states_in1.append(norm_hidden_states[:, key])
                #     hidden_states_in2.append(norm_hidden_states[:, value])
                #     cam_order += [key] * B

                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(norm_hidden_states_right[:, values[0]])
                cam_order += [key] * B

                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(norm_hidden_states_left[:, values[1]])
                cam_order += [key] * B

            # N*2*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)

        elif self.neighboring_attn_type == "concat":
            for key, values in self.neighboring_view_pair.items():
                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(torch.cat([
                    norm_hidden_states[:, value] for value in values
                ], dim=1))
                cam_order += [key] * B
            # N*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            # N*B, 2*H*W, head*dim
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "self":
            hidden_states_in1 = rearrange(
                norm_hidden_states, "b n l ... -> b (n l) ...")
            hidden_states_in2 = None
            cam_order = None
        else:
            raise NotImplementedError(
                f"Unknown type: {self.neighboring_attn_type}")
        return hidden_states_in1, hidden_states_in2, cam_order


    def forward(self, x: torch.Tensor, context: torch.Tensor = None, timesteps: int = None) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, x, context, timesteps)
        else:
            return self._forward(x, context, timesteps=timesteps)

    def _forward(self, x, context=None, timesteps=None):
        assert self.timesteps or timesteps
        assert not (self.timesteps and timesteps) or self.timesteps == timesteps
        timesteps = self.timesteps or timesteps
        B, S, C = x.shape
        x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        if self.disable_self_attn:
            x = self.attn1(self.norm1(x), context=context, batchify_xformers=True) + x
        else:  # this way
            x = self.attn1(self.norm1(x), batchify_xformers=True) + x

        if hasattr(self, "attn2"):
            if self.switch_temporal_ca_to_sa:
                x = self.attn2(self.norm2(x), batchify_xformers=True) + x
            else:  # this way
                x = self.attn2(self.norm2(x), context=context, batchify_xformers=True) + x


        # multi-view cross attention
        hidden_states = x
        norm_hidden_states = self.norm4(hidden_states)
        # batch dim first, cam dim second
        # norm_hidden_states = rearrange(
        #     norm_hidden_states, '(b n) ... -> b n ...', n=self.n_cam)
        norm_hidden_states = rearrange(
            norm_hidden_states, '(b n l) t ... -> (b t) n l ...', n=self.n_cam, l=S, t=timesteps)

        downsample_rate = self.origin_img_size[0] * self.origin_img_size[1] // norm_hidden_states.shape[2] 
        downsample_rate = int(downsample_rate ** 0.5)
        h_img_size = self.origin_img_size[0] // downsample_rate
        w_img_size = self.origin_img_size[1] // downsample_rate
        assert h_img_size * w_img_size == norm_hidden_states.shape[2] 

        BT = len(norm_hidden_states)
        # key is query in attention; value is key-value in attention
        hidden_states_in1, hidden_states_in2, cam_order = self._construct_attn_input(
            norm_hidden_states, h_img_size, w_img_size)
        # attention
        attn_raw_output = self.attn4(
            hidden_states_in1, 
            context=hidden_states_in2,
            batchify_xformers=True,
        )
        # final output
        if self.neighboring_attn_type == "self":
            attn_output = rearrange(
                attn_raw_output, 'b (n l) ... -> b n l ...', n=self.n_cam)
        else:
            attn_output = torch.zeros_like(norm_hidden_states)
            for cam_i in range(self.n_cam):
                attn_out_mv = rearrange(attn_raw_output[cam_order == cam_i],
                                        '(n b) ... -> b n ...', b=BT)
                attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
        # attn_output = rearrange(attn_output, 'b n ... -> (b n) ...')
        attn_output = rearrange(attn_output, '(b t) n l ... -> (b n l) t ...', n=self.n_cam, t=timesteps, l=S)
        # apply zero init connector (one layer)
        attn_output = self.connector(attn_output)
        # short-cut
        hidden_states = attn_output + hidden_states
        x = hidden_states


        x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        x = rearrange(x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps)
        return x

    def get_last_layer(self):
        return self.ff.net[-1].weight