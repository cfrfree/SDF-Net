import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from loss import clip_loss
from loss import InnovationLoss


def do_train_pair(cfg, model, train_loader_pair, optimizer, scheduler, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")
    _LOCAL_PROCESS_GROUP = None

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True
            )

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()

    # train pair
    if cfg.MODEL.PAIR:
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            model.module.train_with_pair()
        else:
            model.train_with_pair()
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            loss_meter.reset()
            scheduler.step(epoch)
            model.train()
            if cfg.MODEL.DIST_TRAIN and hasattr(train_loader_pair.sampler, "set_epoch"):
                train_loader_pair.sampler.set_epoch(epoch)
            for n_iter, (img, vid, target_cam) in enumerate(train_loader_pair):
                optimizer.zero_grad()
                img = img.to(device)
                target = vid.to(device)
                target_cam = target_cam.to(device)
                with amp.autocast(enabled=True):
                    logits_per_sar = model(img, target, cam_label=target_cam)
                    loss = clip_loss(logits_per_sar)

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                loss_meter.update(loss.item(), img.shape[0])

                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                        logger.info(
                            "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}".format(
                                epoch,
                                (n_iter + 1),
                                len(train_loader_pair),
                                loss_meter.avg,
                                scheduler._get_lr(epoch)[0],
                            )
                        )

            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                logger.info(
                    "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                        epoch,
                        time_per_batch,
                        train_loader_pair.batch_size / time_per_batch,
                    )
                )

            if epoch % checkpoint_period == 0:
                if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)
                        ),
                    )


def do_train(
    cfg,
    model,
    center_criterion,
    train_loader,
    val_loader,
    optimizer,
    optimizer_center,
    scheduler,
    loss_fn,
    num_query,
    local_rank,
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True
            )

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    loss_base_meter = AverageMeter()
    loss_orth_meter = AverageMeter()  # 记录原来的解耦 Loss
    loss_dee_meter = AverageMeter()  # 记录新的 DEE Loss

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    # --- 初始化创新 Loss ---
    innov_criterion = InnovationLoss(
        msel_weight=10, orth_weight=2, cons_weight=2
    ).cuda()

    # train
    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        model.module.train_with_single()
    else:
        model.train_with_single()

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        acc_meter.reset()
        loss_base_meter.reset()
        loss_orth_meter.reset()
        loss_dee_meter.reset()
        evaluator.reset()

        scheduler.step(epoch)
        model.train()

        for n_iter, (img, vid, target_cam, target_view, img_wh) in enumerate(
            train_loader
        ):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            img_wh = img_wh.to(device)

            # 用于记录当前 batch 的各部分 loss 值
            current_loss_dee = 0.0
            current_loss_innov = 0.0

            with amp.autocast(enabled=True):
                # 统一解包 5 个值
                outputs = model(img, target, cam_label=target_cam, img_wh=img_wh)
                score_fuse, feat_fuse, feat_shared, feat_spec, dee_outputs = outputs

                # =======================================================
                # 1. 基础 + 原始解耦 Loss (完全保留您的逻辑)
                # =======================================================
                cls_score = score_fuse
                loss_base = loss_fn(score_fuse, feat_fuse, target, target_cam)

                loss_orth = torch.tensor(0.0).cuda()

                # 如果存在解耦特征 (对应您原来的 elif len == 4)
                if feat_shared is not None and feat_spec is not None:
                    # [User's Original Code Start]
                    f_shared_norm = torch.nn.functional.normalize(
                        feat_shared, p=2, dim=1
                    )
                    f_spec_norm = torch.nn.functional.normalize(feat_spec, p=2, dim=1)

                    # 手动计算正交 Loss
                    loss_orth_calc = torch.mean(
                        torch.abs(torch.sum(f_shared_norm * f_spec_norm, dim=1))
                    )
                    loss_orth = cfg.MODEL.ORTH_LOSS_WEIGHT * loss_orth_calc
                    # [User's Original Code End]

                current_loss_orth = loss_orth.item()
                loss = loss_base + loss_orth

                # =======================================================
                # 2. DEE 模块 Loss (独立附加)
                # =======================================================
                if dee_outputs is not None:
                    dee_scores, dee_feats = dee_outputs

                    # 2.1 DEE 辅助 ID Loss
                    loss_dee_id = 0.0
                    for ds, df in zip(dee_scores, dee_feats):
                        loss_dee_id += loss_fn(ds, df, target, target_cam)
                    loss_dee_id = (loss_dee_id / len(dee_scores)) * 0.2

                    # 2.2 DEE 特征正交性 Loss (可选，用 InnovationLoss 方便点)
                    # 也可以手动写，只要保证不同分支正交即可
                    loss_dee_orth = (
                        innov_criterion(
                            feat_shared=None,
                            dee_feats=dee_feats,
                            logits_list=dee_scores,
                            labels=target,
                            modal_ids=target_cam,  # 不涉及 Shared
                        )
                        * 0.1
                    )  # 权重可调

                    loss_dee_total = loss_dee_id + loss_dee_orth
                    loss += loss_dee_total
                    current_loss_dee = loss_dee_total.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= 1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
                scaler.step(optimizer_center)
                scaler.update()

            if isinstance(cls_score, list):
                acc = (cls_score[0].max(1)[1] == target).float().mean()
            else:
                acc = (cls_score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            loss_base_meter.update(loss_base.item(), img.shape[0])
            loss_orth_meter.update(current_loss_orth, img.shape[0])  # 原始解耦 Loss
            loss_dee_meter.update(current_loss_dee, img.shape[0])  # 新增 DEE Loss
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()

        if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
            logger.info(
                "Epoch[{}] done. Loss: {:.3f} (Base: {:.3f}, Orth: {:.3f}, DEE: {:.3f}), Acc: {:.3f}, Lr: {:.2e}".format(
                    epoch,
                    loss_meter.avg,
                    loss_base_meter.avg,
                    loss_orth_meter.avg,
                    loss_dee_meter.avg,
                    acc_meter.avg,
                    scheduler._get_lr(epoch)[0],
                )
            )

        if (epoch % checkpoint_period) and (
            not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0
        ):
            torch.save(
                model.state_dict(),
                os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)),
            )

        if (epoch % eval_period == 0) and (
            not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0
        ):
            model.eval()
            for n_iter, (img, vid, camid, camids, target_view, _, img_wh) in enumerate(
                val_loader
            ):
                with torch.no_grad():
                    img = img.to(device)
                    camids = camids.to(device)
                    img_wh = img_wh.to(device)
                    feat = model(img, cam_label=camids, img_wh=img_wh)
                    evaluator.update((feat, vid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()
        if cfg.MODEL.DIST_TRAIN:
            dist.barrier()


def do_inference(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for inference".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath, img_wh) in enumerate(
        val_loader
    ):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            img_wh = img_wh.to(device)
            feat = model(img, cam_label=camids, img_wh=img_wh)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
