import os
from datetime import datetime
import scipy
import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F
from config import BATCH_SIZE, SAVE_FREQ, RESUME, FINAL_EMBEDDING_SIZE, TEST_FREQ, TOTAL_EPOCH, MODEL_PRE, GPU_IDS
from config import CASIA_DATA_DIR, LFW_DATA_DIR, SAVE_DIR
from core import model
from core.utils import init_log
from CASIA_loader import CASIA_Face
from LFW_loader import LFW
import torch.optim as optim
import time
from eval import parseList, evaluation_10_fold
import numpy as np

def main():
    # GPU INITIALIZATION
    GPU = GPU_IDS if isinstance(GPU_IDS, (list, tuple)) else [GPU_IDS]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, GPU))
    print(f"Visible GPUs:", GPU)
    multi_gpus = len(GPU) > 1  

    # other init
    start_epoch = 1
    save_dir = os.path.join(SAVE_DIR, MODEL_PRE + 'v2_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('result', exist_ok=True)
    logging = init_log(save_dir)
    _print = logging.info

    # Datasets
    trainset = CASIA_Face(root=CASIA_DATA_DIR)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=2, pin_memory=True, drop_last=True
    )

    nl, nr, folds, flags = parseList(root=LFW_DATA_DIR)
    testdataset = LFW(nl, nr)
    testloader = torch.utils.data.DataLoader(
        testdataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False
    )

    # Model
    net = model.MobileFacenet().cuda()
    ArcMargin = model.ArcMarginProduct(FINAL_EMBEDDING_SIZE, trainset.class_nums).cuda()

    if RESUME:
        ckpt = torch.load(RESUME)
        net.load_state_dict(ckpt['net_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        _print(f"Resumed from epoch {start_epoch}")

    # Optimizer + Scheduler
    prelu_params = [p for m in net.modules() if isinstance(m, nn.PReLU) for p in m.parameters()]
    ignored_params = (list(map(id, net.linear1.parameters())) + list(map(id, ArcMargin.weight)) + [id(p) for p in prelu_params])
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.linear1.parameters(), 'weight_decay': 4e-4},
        {'params': ArcMargin.weight, 'weight_decay': 4e-4},
        {'params': prelu_params, 'weight_decay': 0.0}
    ], lr=0.1, momentum=0.9, nesterov=True)

    milestones = [int(TOTAL_EPOCH * 0.5),int(TOTAL_EPOCH * 0.7),int(TOTAL_EPOCH * 0.85)]    #Reduce LR based on epoch
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_ft, milestones=milestones, gamma=0.1, last_epoch=-1)
    if multi_gpus:
        net = DataParallel(net)
        ArcMargin = DataParallel(ArcMargin)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    best_epoch = 0
    num_batches = len(trainloader)
    for epoch in range(start_epoch, TOTAL_EPOCH + 1):
        net.train()
        train_loss = 0.0
        total = 0
        start_time = time.time()
        _print(f"\nStarting epoch {epoch}/{TOTAL_EPOCH}  ({num_batches} batches total)")
        for batch_idx, (img, label) in enumerate(trainloader):
            if batch_idx % 50 == 0: 
                _print(f" Batch {batch_idx+1}/{num_batches} " f"({(batch_idx+1)/num_batches*100:.1f}%)")
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)
            optimizer_ft.zero_grad()
            feats = net(img)
            feats = F.normalize(feats, p=2, dim=1) 
            output = ArcMargin(feats, label)
            loss = criterion(output, label)
            loss.backward()
            optimizer_ft.step()
            train_loss += loss.item() * img.size(0)
            total += img.size(0)

        train_loss /= total
        _print(f"Epoch {epoch}/{TOTAL_EPOCH} | Loss: {train_loss:.4f} | "
               f"Time: {(time.time()-start_time)/60:.1f}m")

        # LFW evaluation
        if epoch % TEST_FREQ == 0:
            net.eval()
            _print(f'Test Epoch: {epoch} ...')
            all_left = []
            all_right = []
            with torch.no_grad():
                for batch in testloader:                                            # batch shape: (B, 2, 3, 112, 112)
                    batch = batch.view(-1, 3, 112, 112).cuda(non_blocking=True)     # → (B*2, 3, 112, 112)
                    embeddings = net(batch)                                         # → (B*2, 128)
                    embeddings = F.normalize(embeddings, p=2, dim=1)                # L2 normalize
                    left, right = embeddings.chunk(2, dim=0)                        # Split back into left/right
                    all_left.append(left.cpu())
                    all_right.append(right.cpu())
            # Concatenate all
            fl = torch.cat(all_left).numpy()      # (6000, 128)
            fr = torch.cat(all_right).numpy()     # (6000, 128)
            # Compute cosine similarity → higher = more similar
            cosine_sim = np.sum(fl * fr, axis=1)  # dot product of normalized vectors
            # Save .mat for old evaluation function (keeps compatibility)
            os.makedirs('result', exist_ok=True)
            scipy.io.savemat('result/tmp_result.mat', {
                'fl': fl,'fr': fr,
                'fold': np.array(folds),
                'flag': np.array(flags)
            })
            # Run 10-fold evaluation
            accs = evaluation_10_fold('result/tmp_result.mat')
            _print(f'    LFW Accuracy: {np.mean(accs)*100:.3f}% ± {np.std(accs)*100:.3f}%')

        scheduler.step()

        # save model
        if epoch % SAVE_FREQ == 0 or epoch == TOTAL_EPOCH:
            state_dict = net.module.state_dict() if multi_gpus else net.state_dict()
            torch.save({
                'epoch': epoch,
                'net_state_dict': state_dict
            },os.path.join(save_dir, '%03d.ckpt' % epoch))
            _print(f"Saved checkpoint: {epoch:03d}.pth")

    _print('finishing training')

if __name__ == '__main__':
    main()