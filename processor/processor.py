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


def do_train_pair(cfg, model, train_loader_pair, optimizer, scheduler, local_rank):
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
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

            # --- End of Batch Loop ---

            end_time = time.time()
            time_per_epoch = end_time - start_time

            # Print Log only at the end of epoch
            if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                current_lr = scheduler._get_lr(epoch)[0]
                logger.info(
                    f"Epoch[{epoch}] done. "
                    f"Loss: {loss_meter.avg:.3f}, "
                    f"Lr: {current_lr:.2e}, "
                    f"Time: {time_per_epoch:.2f}s"
                )

            if epoch % checkpoint_period == 0:
                if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_{epoch}.pth"),
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
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True
            )

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    loss_base_meter = AverageMeter()
    loss_orth_meter = AverageMeter()  # 新增：记录 Orth Loss

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    # train
    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        model.module.train_with_single()
    else:
        model.train_with_single()

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        acc_meter.reset()
        loss_base_meter.reset()
        loss_orth_meter.reset()  # 重置 Orth meter
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

            # 初始化当前batch的 loss 记录变量
            current_loss_orth = 0.0

            with amp.autocast(enabled=True):
                if cfg.MODEL.DISENTANGLE:
                    score_fuse, feat_fuse, feat_shared, feat_spec = model(
                        img, target, cam_label=target_cam, img_wh=img_wh
                    )
                    cls_score = score_fuse
                    loss_base = loss_fn(score_fuse, feat_fuse, target, target_cam)

                    f_shared_norm = torch.nn.functional.normalize(
                        feat_shared, p=2, dim=1
                    )
                    f_spec_norm = torch.nn.functional.normalize(feat_spec, p=2, dim=1)
                    loss_orth_calc = torch.mean(
                        torch.abs(torch.sum(f_shared_norm * f_spec_norm, dim=1))
                    )
                    loss_orth = cfg.MODEL.ORTH_LOSS_WEIGHT * loss_orth_calc

                    loss = loss_base + loss_orth
                    current_loss_orth = loss_orth.item()  # 记录 orth loss 值
                else:
                    outputs = model(img, target, cam_label=target_cam, img_wh=img_wh)
                    cls_score, feat = outputs
                    loss_base = loss_fn(cls_score, feat, target, target_cam)
                    loss = loss_base
                    current_loss_orth = 0.0

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

            # 更新所有 Meter
            loss_meter.update(loss.item(), img.shape[0])
            loss_base_meter.update(loss_base.item(), img.shape[0])
            loss_orth_meter.update(current_loss_orth, img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()

        if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
            current_lr = scheduler._get_lr(epoch)[0]
            logger.info(
                f"Epoch[{epoch}] done. "
                f"Loss: {loss_meter.avg:.3f} (Base: {loss_base_meter.avg:.3f}, Orth: {loss_orth_meter.avg:.3f}), "
                f"Acc: {acc_meter.avg:.3%}, "
                f"Lr: {current_lr:.2e}"
            )

        if epoch % checkpoint_period == 0:
            if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_{epoch}.pth"),
                )

        if epoch % eval_period == 0:
            if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                model.eval()
                for n_iter, (
                    img,
                    vid,
                    camid,
                    camids,
                    target_view,
                    _,
                    img_wh,
                ) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        img_wh = img_wh.to(device)
                        feat = model(img, cam_label=camids, img_wh=img_wh)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()

                # Validation log also updated to f-string style (optional but consistent)
                logger.info(f"Validation Results - Epoch: {epoch}")
                logger.info(f"mAP: {mAP:.1%}")
                for r in [1, 5, 10]:
                    logger.info(f"CMC curve, Rank-{r:<3}: {cmc[r - 1]:.1%}")

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
            print(f"Using {torch.cuda.device_count()} GPUs for inference")
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
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}: {cmc[r - 1]:.1%}")
    return cmc[0], cmc[4]
