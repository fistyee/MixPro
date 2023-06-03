# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_mlp import SwinMLP
from timm.models import create_model
from .vit import  VisionTransformer
from .focal_transformer_v2 import FocalTransformer as focalv2

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
                        depths=config.MODEL.SWIN_MLP.DEPTHS,
                        num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
                        window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN_MLP.APE,
                        patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'deit':
        model = VisionTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.DEIT.PATCH_SIZE,
                                in_chans=config.MODEL.DEIT.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.DEIT.EMBED_DIM,
                                depth=config.MODEL.DEIT.DEPTHS,
                                num_heads=config.MODEL.DEIT.NUM_HEADS,
                                mlp_ratio=config.MODEL.DEIT.MLP_RATIO,
                                qkv_bias=config.MODEL.DEIT.QKV_BIAS,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                return_attn=config.AUG.PAL_ATTN)
                                #ape=config.MODEL.SWIN.APE,
                                #use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == "focalv2":
        model = eval(model_type)(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.FOCAL.PATCH_SIZE,
            in_chans=config.MODEL.FOCAL.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.FOCAL.EMBED_DIM,
            depths=config.MODEL.FOCAL.DEPTHS,
            num_heads=config.MODEL.FOCAL.NUM_HEADS,
            window_size=config.MODEL.FOCAL.WINDOW_SIZE,
            mlp_ratio=config.MODEL.FOCAL.MLP_RATIO,
            qkv_bias=config.MODEL.FOCAL.QKV_BIAS,
            qk_scale=config.MODEL.FOCAL.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.FOCAL.APE,
            patch_norm=config.MODEL.FOCAL.PATCH_NORM,
            use_shift=config.MODEL.FOCAL.USE_SHIFT, 
            expand_stages=config.MODEL.FOCAL.EXPAND_STAGES,
            expand_sizes=config.MODEL.FOCAL.EXPAND_SIZES, 
            expand_layer=config.MODEL.FOCAL.EXPAND_LAYER,         
            focal_pool=config.MODEL.FOCAL.FOCAL_POOL,     
            focal_stages=config.MODEL.FOCAL.FOCAL_STAGES, 
            focal_windows=config.MODEL.FOCAL.FOCAL_WINDOWS,                                                   
            focal_levels=config.MODEL.FOCAL.FOCAL_LEVELS,    
            focal_topK=config.MODEL.FOCAL.FOCAL_TOPK, 
            use_conv_embed=config.MODEL.FOCAL.USE_CONV_EMBED, 
            use_layerscale=config.MODEL.FOCAL.USE_LAYERSCALE, 
            use_pre_norm=config.MODEL.FOCAL.USE_PRE_NORM, 
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )        


    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
