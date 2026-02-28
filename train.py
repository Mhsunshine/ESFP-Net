import time
import datetime
import logging
import torch
from apex import amp
from tools.utils import AverageMeter


def train_cal(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
    criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes, CosineDecorrelate, identity_classifier_rgb, identity_classifier_event):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    batch_loss_dp = AverageMeter()
    # batch_align_loss = AverageMeter()
    # batch_CosineDecorrelate_loss = AverageMeter()
    # batch_loss_sub = AverageMeter()
    model.train()
    classifier.train()
    identity_classifier_rgb.train()
    identity_classifier_event.train()
    clothes_classifier.train()
    # clothes_classifier_rgb.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids.cpu()]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        # features, f_rgb, f_event, loss_dp, loss_ent, loss_aff, loss_smooth = model(imgs)
        # features, f_rgb, f_event, loss_dp, loss_ent = model(imgs)
        features, f_rgb, f_event, loss_dp, loss_edge = model(imgs)
        outputs = classifier(features)
        outputs_rgb = identity_classifier_rgb(f_rgb)
        outputs_event = identity_classifier_event(f_event)
        pred_clothes = clothes_classifier(features.detach())
        # pred_clothes_rgb = clothes_classifier_rgb(f_rgb.detach())

        _, preds = torch.max(outputs.data, 1)
        # _, preds_rgb = torch.max(outputs_rgb,1)
        # _, preds_event = torch.max(outputs_event,1)
        # Update the clothes discriminator
        clothes_loss = criterion_clothes(pred_clothes, clothes_ids)
        #   + criterion_clothes(pred_clothes_rgb,clothes_ids)
        
        if epoch >= config.TRAIN.START_EPOCH_CC:
            optimizer_cc.zero_grad()
            if config.TRAIN.AMP:
                with amp.scale_loss(clothes_loss, optimizer_cc) as scaled_loss:
                    scaled_loss.backward()
            else:
                clothes_loss.backward()
            optimizer_cc.step()

        # Update the backbone
        new_pred_clothes = clothes_classifier(features)
        # new_pred_clothes_rgb = clothes_classifier_rgb(f_rgb)
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        cla_loss += criterion_cla(outputs_rgb, pids)
        cla_loss += criterion_cla(outputs_event, pids)
        #  + criterion_cla(outputs_rgb, pids) + criterion_cla(outputs_event, pids)
        pair_loss = criterion_pair(features, pids) 
        pair_loss += criterion_pair(f_rgb, pids) 
        pair_loss += criterion_pair(f_event, pids) 
        # + criterion_pair(f_rgb,pids) + criterion_pair(f_event,pids)
        adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask) 
        #  + criterion_adv(new_pred_clothes_rgb,clothes_ids,pos_mask)
        #   + criterion_adv()
        # CosineDecorrelate_loss = CosineDecorrelate(f_rgb,f_event)
        # if epoch <10: 
        #     loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss  + 1.0 * loss_dp +  loss_ent 
            
        if epoch >= config.TRAIN.START_EPOCH_ADV:
            loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss  + 1.1 * loss_dp   + loss_edge 
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss + 0.6 * loss_dp  + loss_edge 
        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        clothes_corrects.update(torch.sum(clothes_preds == clothes_ids.data).float()/clothes_ids.size(0), clothes_ids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
        batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        batch_loss_dp.update(loss_dp.item(),pids.size(0))
        # batch_loss_sub.update(loss_sub.item(), pids.size(0))
        # batch_CosineDecorrelate_loss.update(CosineDecorrelate_loss.item(), pids.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'ClaLoss:{cla_loss.avg:.4f} '
                  'PairLoss:{pair_loss.avg:.4f} '
                  'CloLoss:{clo_loss.avg:.4f} '
                  'AdvLoss:{adv_loss.avg:.4f} '
                  'Acc:{acc.avg:.2%} '
                  'CloAcc:{clo_acc.avg:.2%} '
                  'loss_dp:{loss_dp.avg:.4f}' .format(
                   epoch+1, batch_time=batch_time, data_time=data_time, 
                   cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                   clo_loss=batch_clo_loss, adv_loss=batch_adv_loss, 
                   acc=corrects, clo_acc=clothes_corrects, loss_dp = batch_loss_dp))


def train_cal_with_memory(config, epoch, model, classifier, criterion_cla, criterion_pair, 
    criterion_adv, optimizer, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids.cpu()]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)

        if epoch >= config.TRAIN.START_EPOCH_ADV:
            adv_loss = criterion_adv(features, clothes_ids, pos_mask)
            loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss    
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss  

        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        if epoch >= config.TRAIN.START_EPOCH_ADV: 
            batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{batch_time.sum:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'ClaLoss:{cla_loss.avg:.4f} '
                'PairLoss:{pair_loss.avg:.4f} '
                'AdvLoss:{adv_loss.avg:.4f} '
                'Acc:{acc.avg:.2%} '.format(
                epoch+1, batch_time=batch_time, data_time=data_time, 
                cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                adv_loss=batch_adv_loss, acc=corrects))