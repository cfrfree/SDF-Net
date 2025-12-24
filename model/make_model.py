import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
from .backbones.vit_transoss import vit_base_patch16_224_TransOSS
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss


def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat(
        [features[:, begin - 1 + shift :], features[:, begin : begin - 1 + shift]],
        dim=1,
    )
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)

    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == "resnet50":
            self.in_planes = 2048
            self.base = ResNet(
                last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3]
            )
            print("using resnet50 as a backbone")
        else:
            print("unsupported backbone! but got {}".format(model_name))

        if pretrain_choice == "imagenet":
            self.base.load_param(model_path)
            print("Loading pretrained model......from {}".format(model_path))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(
            global_feat.shape[0], -1
        )  # flatten to (bs, 2048)

        if self.neck == "no":
            feat = global_feat
        elif self.neck == "bnneck":
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == "after":
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model from {}".format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model for finetuning from {}".format(model_path))


class build_transformer(nn.Module):
    def __init__(
        self, num_classes, camera_num, cfg, factory, logit_scale_init_value=2.6592
    ):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.model_type = cfg.MODEL.TRANSFORMER_TYPE
        self.disentangle = cfg.MODEL.DISENTANGLE
        # 'sum'      : (D) FSS (Your Method) - [feat_shared + feat_spec]
        # 'shared'   : (A) Shared only       - [feat_shared]
        # 'specific' : (B) Specific only     - [feat_spec]
        # 'concat'   : (C) Concat            - [cat(feat_shared, feat_spec)]
        self.ablation_mode = cfg.MODEL.ABLATION_MODE
        print(f"Current Ablation Mode: {self.ablation_mode}")
        if self.disentangle and self.ablation_mode == "concat":
            # 如果是 Concat 模式，融合特征维度是 2倍 (768 * 2 = 1536)
            fusion_dim = self.in_planes * 2
        else:
            # 其他模式 (Sum, Shared, Specific) 维度保持 768
            fusion_dim = self.in_planes

        print(
            "using Transformer_type: {} as a backbone".format(
                cfg.MODEL.TRANSFORMER_TYPE
            )
        )

        if cfg.MODEL.MIE:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.TRANSFORMER_TYPE == "vit_base_patch16_224_TransOSS":
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
                img_size=cfg.INPUT.SIZE_TRAIN,
                mie_coe=cfg.MODEL.MIE_COE,
                camera=camera_num,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate=cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                sse=cfg.MODEL.SSE,
                use_gated_attention=cfg.MODEL.GATED_ATTENTION,
                disentangle=self.disentangle,
            )
        else:
            raise ValueError(
                "Unsupported model type: {}".format(cfg.MODEL.TRANSFORMER_TYPE)
            )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == "arcface":
            self.classifier = Arcface(
                fusion_dim,
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE,
                m=cfg.SOLVER.COSINE_MARGIN,
            )
        elif self.ID_LOSS_TYPE == "cosface":
            self.classifier = Cosface(
                fusion_dim,
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE,
                m=cfg.SOLVER.COSINE_MARGIN,
            )
        elif self.ID_LOSS_TYPE == "amsoftmax":
            self.classifier = AMSoftmax(
                fusion_dim,
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE,
                m=cfg.SOLVER.COSINE_MARGIN,
            )
        elif self.ID_LOSS_TYPE == "circle":
            self.classifier = CircleLoss(
                fusion_dim,
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE,
                m=cfg.SOLVER.COSINE_MARGIN,
            )
        else:
            self.classifier = nn.Linear(fusion_dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        if self.disentangle:
            self.bottleneck_fuse = nn.BatchNorm1d(fusion_dim)
            self.bottleneck_fuse.bias.requires_grad_(False)
            self.bottleneck_fuse.apply(weights_init_kaiming)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.train_pair = False
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))

        if pretrain_choice == "imagenet":
            self.base.load_param(model_path)
            print("Loading pretrained model from {}".format(model_path))
        elif pretrain_choice == "clip":
            self.load_param(model_path)
            print("Loading pretrained model from {}".format(model_path))
        elif pretrain_choice == False:
            print("Training transformer from scratch.")
        else:
            raise ValueError("Unsupported pretrain_choice: {}".format(pretrain_choice))

    def train_with_pair(
        self,
    ):
        self.train_pair = True

    def train_with_single(
        self,
    ):
        self.train_pair = False

    def forward(self, x, label=None, cam_label=None, img_wh=None):
        # 1. 特征提取
        if self.disentangle:
            feat_shared, feat_spec = self.base(x, cam_label=cam_label, img_wh=img_wh)

            # ================= [消融实验逻辑] =================
            if self.ablation_mode == "sum":  # (D) FSS (Sum)
                feat_fuse = feat_shared + feat_spec

            elif self.ablation_mode == "shared":  # (A) Shared only
                feat_fuse = feat_shared

            elif self.ablation_mode == "specific":  # (B) Specific only
                feat_fuse = feat_spec

            elif self.ablation_mode == "concat":  # (C) Concat
                feat_fuse = torch.cat([feat_shared, feat_spec], dim=1)  # Dim: 1536

            else:
                raise ValueError(f"Unknown ablation mode: {self.ablation_mode}")
            # ================================================

        else:
            global_feat = self.base(x, cam_label=cam_label, img_wh=img_wh)
            feat_fuse = global_feat
            feat_shared = None
            feat_spec = None

        # 2. 分类头处理
        if self.disentangle:
            feat_fuse_bn = self.bottleneck_fuse(feat_fuse)
        else:
            feat_fuse_bn = self.bottleneck(feat_fuse)

        # 3. 返回结果
        if self.training:
            if self.ID_LOSS_TYPE in ("arcface", "cosface", "amsoftmax", "circle"):
                score_fuse = self.classifier(feat_fuse_bn, label)
            else:
                score_fuse = self.classifier(feat_fuse_bn)

            if self.disentangle:
                return score_fuse, feat_fuse, feat_shared, feat_spec
            else:
                return score_fuse, feat_fuse
        else:
            if self.disentangle:
                return self.bottleneck_fuse(feat_fuse)
            elif self.neck_feat == "after":
                return self.bottleneck(feat_fuse)
            else:
                return feat_fuse

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]
        for i in param_dict:
            key = i.replace("module.", "")
            if key.startswith("classifier"):
                continue
            self.state_dict()[key].copy_(param_dict[i])

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model for finetuning from {}".format(model_path))


__factory_T_type = {
    "vit_base_patch16_224_TransOSS": vit_base_patch16_224_TransOSS,
}


def make_model(cfg, num_class, camera_num):
    if cfg.MODEL.NAME == "transformer":
        model = build_transformer(num_class, camera_num, cfg, __factory_T_type)
        print("===========building transformer===========")
    else:
        model = Backbone(num_class, cfg)
        print("===========building ResNet===========")
    return model
