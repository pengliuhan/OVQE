import torch
import numpy as np
from collections import OrderedDict
from ovqe import OVQE
import utils
from tqdm import tqdm


ckp_path = '/home/pengliuhan/SCI/OVQE/exp/OVQE_QP37/OVQE_QP37.pth'


raw_yuv_path = '/data/MFQEv2_dataset/test_18/raw/BasketballPass_416x240_500.yuv'
lq_yuv_path = '/data/MFQEv2_dataset/test_18/HM16.5_LDP/QP37/BasketballPass_416x240_500.yuv'
vname = lq_yuv_path.split("/")[-1].split('.')[0]
_, wxh, nfs = vname.split('_')
nfs = int(nfs)
w, h = int(wxh.split('x')[0]), int(wxh.split('x')[1])
torch.cuda.set_device(0)


def main():

    model = OVQE()
    msg = f'loading model {ckp_path}...'
    print(msg)
        # , map_location='cpu'   ,map_location={'cuda:0': 'cuda:1'}
    checkpoint = torch.load(ckp_path, map_location='cpu')
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training

        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])

    msg = f'> model {ckp_path} loaded.'
    print(msg)
    model = model.cuda()
    model.eval()

    # ==========
    # Load entire video
    # ==========
    msg = f'loading raw and low-quality yuv...'
    print(msg)
    raw_y = utils.import_yuv(
        seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True
        )
    lq_y = utils.import_yuv(
        seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True
        )
    raw_y = raw_y.astype(np.float32) / 255.
    lq_y = lq_y.astype(np.float32) / 255.
    msg = '> yuv loaded.'
    print(msg)

    # ==========
    # Define criterion
    # ==========
    criterion = utils.PSNR()
    unit = 'dB'

    # ==========
    # Test
    # ==========
    pbar = tqdm(total=nfs, ncols=80)
    ori_psnr_counter = utils.Counter()
    enh_psnr_counter = utils.Counter()

    lq_y = torch.from_numpy(lq_y)
    lq_y = torch.unsqueeze(lq_y, 0).cuda()
    with torch.no_grad():
        enc_all = model(lq_y)


    for idx in range(nfs):
        gt_frm = torch.from_numpy(raw_y[idx])
        batch_ori = criterion(lq_y[0, idx,...].cpu(), gt_frm)
        batch_perf = criterion(enc_all[0, idx, 0,:,:].cpu(), gt_frm)
        ori_psnr_counter.accum(volume=batch_ori)
        enh_psnr_counter.accum(volume=batch_perf)

        # display
        pbar.set_description(
            "[{:.3f}] {:s} -> [{:.3f}] {:s}"
            .format(batch_ori, unit, batch_perf, unit)
            )
        pbar.update()

    pbar.close()
    ori_ = ori_psnr_counter.get_ave()
    enh_ = enh_psnr_counter.get_ave()
    print('ave ori [{:.3f}] {:s}, enh [{:.3f}] {:s}, delta [{:.3f}] {:s}'.format(
        ori_, unit, enh_, unit, (enh_ - ori_) , unit
        ))
    print('> done.')


if __name__ == '__main__':
    main()
