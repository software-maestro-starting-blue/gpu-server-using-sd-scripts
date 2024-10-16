from sdxl_gen_img import *

def preload(args):
    if args.fp16:
        dtype = torch.float16
    elif args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    highres_fix = args.highres_fix_scale is not None
    # assert not highres_fix or args.image_path is None, f"highres_fix doesn't work with img2img / highres_fixはimg2imgと同時に使えません"

    # モデルを読み込む
    if not os.path.isfile(args.ckpt):  # ファイルがないならパターンで探し、一つだけ該当すればそれを使う
        files = glob.glob(args.ckpt)
        if len(files) == 1:
            args.ckpt = files[0]

    (_, text_encoder1, text_encoder2, vae, unet, _, _) = sdxl_train_util._load_target_model(
        args.ckpt, args.vae, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, dtype
    )
    unet: InferSdxlUNet2DConditionModel = InferSdxlUNet2DConditionModel(unet)

    # xformers、Hypernetwork対応
    if not args.diffusers_xformers:
        mem_eff = not (args.xformers or args.sdpa)
        replace_unet_modules(unet, mem_eff, args.xformers, args.sdpa)
        replace_vae_modules(vae, mem_eff, args.xformers, args.sdpa)

    # tokenizerを読み込む
    logger.info("loading tokenizer")
    tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)

    # schedulerを用意する
    sched_init_args = {}
    has_steps_offset = True
    has_clip_sample = True
    scheduler_num_noises_per_step = 1

    if args.sampler == "ddim":
        scheduler_cls = DDIMScheduler
        scheduler_module = diffusers.schedulers.scheduling_ddim
    elif args.sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
        scheduler_cls = DDPMScheduler
        scheduler_module = diffusers.schedulers.scheduling_ddpm
    elif args.sampler == "pndm":
        scheduler_cls = PNDMScheduler
        scheduler_module = diffusers.schedulers.scheduling_pndm
        has_clip_sample = False
    elif args.sampler == "lms" or args.sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_lms_discrete
        has_clip_sample = False
    elif args.sampler == "euler" or args.sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_euler_discrete
        has_clip_sample = False
    elif args.sampler == "euler_a" or args.sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteSchedulerGL
        scheduler_module = diffusers.schedulers.scheduling_euler_ancestral_discrete
        has_clip_sample = False
    elif args.sampler == "dpmsolver" or args.sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = args.sampler
        scheduler_module = diffusers.schedulers.scheduling_dpmsolver_multistep
        has_clip_sample = False
    elif args.sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
        scheduler_module = diffusers.schedulers.scheduling_dpmsolver_singlestep
        has_clip_sample = False
        has_steps_offset = False
    elif args.sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_heun_discrete
        has_clip_sample = False
    elif args.sampler == "dpm_2" or args.sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_k_dpm_2_discrete
        has_clip_sample = False
    elif args.sampler == "dpm_2_a" or args.sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete
        scheduler_num_noises_per_step = 2
        has_clip_sample = False

    # 警告を出さないようにする
    if has_steps_offset:
        sched_init_args["steps_offset"] = 1
    if has_clip_sample:
        sched_init_args["clip_sample"] = False

    # samplerの乱数をあらかじめ指定するための処理

    # replace randn
    class NoiseManager:
        def __init__(self):
            self.sampler_noises = None
            self.sampler_noise_index = 0

        def reset_sampler_noises(self, noises):
            self.sampler_noise_index = 0
            self.sampler_noises = noises

        def randn(self, shape, device=None, dtype=None, layout=None, generator=None):
            # logger.info("replacing", shape, len(self.sampler_noises), self.sampler_noise_index)
            if self.sampler_noises is not None and self.sampler_noise_index < len(self.sampler_noises):
                noise = self.sampler_noises[self.sampler_noise_index]
                if shape != noise.shape:
                    noise = None
            else:
                noise = None

            if noise == None:
                logger.warning(f"unexpected noise request: {self.sampler_noise_index}, {shape}")
                noise = torch.randn(shape, dtype=dtype, device=device, generator=generator)

            self.sampler_noise_index += 1
            return noise

    class TorchRandReplacer:
        def __init__(self, noise_manager):
            self.noise_manager = noise_manager

        def __getattr__(self, item):
            if item == "randn":
                return self.noise_manager.randn
            if hasattr(torch, item):
                return getattr(torch, item)
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

    noise_manager = NoiseManager()
    if scheduler_module is not None:
        scheduler_module.torch = TorchRandReplacer(noise_manager)

    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        **sched_init_args,
    )

    # ↓以下は結局PipeでFalseに設定されるので意味がなかった
    # # clip_sample=Trueにする
    # if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
    #     logger.info("set clip_sample to True")
    #     scheduler.config.clip_sample = True

    # deviceを決定する
    device = get_preferred_device()

    # custom pipelineをコピったやつを生成する
    if args.vae_slices:
        from library.slicing_vae import SlicingAutoencoderKL

        sli_vae = SlicingAutoencoderKL(
            act_fn="silu",
            block_out_channels=(128, 256, 512, 512),
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            in_channels=3,
            latent_channels=4,
            layers_per_block=2,
            norm_num_groups=32,
            out_channels=3,
            sample_size=512,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            num_slices=args.vae_slices,
        )
        sli_vae.load_state_dict(vae.state_dict())  # vaeのパラメータをコピーする
        vae = sli_vae
        del sli_vae

    vae_dtype = dtype
    if args.no_half_vae:
        logger.info("set vae_dtype to float32")
        vae_dtype = torch.float32
    vae.to(vae_dtype).to(device)
    vae.eval()

    text_encoder1.to(dtype).to(device)
    text_encoder2.to(dtype).to(device)
    unet.to(dtype).to(device)
    text_encoder1.eval()
    text_encoder2.eval()
    unet.eval()

    return dtype, highres_fix, text_encoder1, text_encoder2, vae, unet, tokenizer1, tokenizer2, scheduler_num_noises_per_step, noise_manager, scheduler, device

def preload_lora(args, vae, text_encoder1, text_encoder2, unet, dtype, device):
    # networkを組み込む
    if args.network_module:
        networks = []
        network_default_muls = []
        network_pre_calc = args.network_pre_calc

        # merge関連の引数を統合する
        if args.network_merge:
            network_merge = len(args.network_module)  # all networks are merged
        elif args.network_merge_n_models:
            network_merge = args.network_merge_n_models
        else:
            network_merge = 0
        logger.info(f"network_merge: {network_merge}")

        for i, network_module in enumerate(args.network_module):
            logger.info(f"import network module: {network_module}")
            imported_module = importlib.import_module(network_module)

            network_mul = 1.0 if args.network_mul is None or len(args.network_mul) <= i else args.network_mul[i]

            net_kwargs = {}
            if args.network_args and i < len(args.network_args):
                network_args = args.network_args[i]
                # TODO escape special chars
                network_args = network_args.split(";")
                for net_arg in network_args:
                    key, value = net_arg.split("=")
                    net_kwargs[key] = value

            if args.network_weights is None or len(args.network_weights) <= i:
                raise ValueError("No weight. Weight is required.")

            network_weight = args.network_weights[i]
            logger.info(f"load network weights from: {network_weight}")

            if model_util.is_safetensors(network_weight) and args.network_show_meta:
                from safetensors.torch import safe_open

                with safe_open(network_weight, framework="pt") as f:
                    metadata = f.metadata()
                if metadata is not None:
                    logger.info(f"metadata for: {network_weight}: {metadata}")

            network, weights_sd = imported_module.create_network_from_weights(
                network_mul, network_weight, vae, [text_encoder1, text_encoder2], unet, for_inference=True, **net_kwargs
            )
            if network is None:
                return

            mergeable = network.is_mergeable()
            if network_merge and not mergeable:
                logger.warning("network is not mergiable. ignore merge option.")

            if not mergeable or i >= network_merge:
                # not merging
                network.apply_to([text_encoder1, text_encoder2], unet)
                info = network.load_state_dict(weights_sd, False)  # network.load_weightsを使うようにするとよい
                logger.info(f"weights are loaded: {info}")

                if args.opt_channels_last:
                    network.to(memory_format=torch.channels_last)
                network.to(dtype).to(device)

                if network_pre_calc:
                    logger.info("backup original weights")
                    network.backup_weights()

                networks.append(network)
                network_default_muls.append(network_mul)
            else:
                network.merge_to([text_encoder1, text_encoder2], unet, weights_sd, dtype, device)

    else:
        networks = []
    return networks, network_default_muls, network_pre_calc