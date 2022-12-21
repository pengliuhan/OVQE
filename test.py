import torch
import numpy as np
from collections import OrderedDict
import math
from ovqe import OVQE
import utils
from tqdm import tqdm
import glob
import os.path as op
import numpy as np


ckp_path = '/home/pengliuhan/SCI/OVQE/exp/OVQE_QP37/ckp_60000.pth'
gt_dir = '/data/MFQEv2_dataset/test_18/raw'
lq_dir = '/data/MFQEv2_dataset/test_18/HM16.5_LDP/QP37'
log_fp = open('/home/pengliuhan/SCI/OVQE/exp/OVQE_QP37/log_test.log', 'w')
gt_video_list = sorted(glob.glob(op.join(gt_dir, '*.yuv')))
lq_video_list = sorted(glob.glob(op.join(lq_dir, '*.yuv')))
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
    for cdx in range(len(gt_video_list)):
        raw_yuv_path = gt_video_list[cdx]
        lq_yuv_path = lq_video_list[cdx]
        vname = raw_yuv_path.split("/")[-1].split('.')[0]
        _, wxh, nfs = vname.split('_')
        nfs = int(nfs)
        w, h = int(wxh.split('x')[0]), int(wxh.split('x')[1])
        divide_bolck = 150
        divide = math.ceil(nfs / divide_bolck)
        add_frame = 0

        msg = f'loading raw and low-quality yuv...'
        print(msg)
        raw_y = utils.import_yuv(
            seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True
            )
        raw_y = raw_y.astype(np.float32) / 255.

        lq_y = utils.import_yuv(
            seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True
        )
        lq_y = lq_y.astype(np.float32) / 255.
        msg = '> yuv loaded.'
        print(msg)



        # ==========
        # Test
        # ==========
        unit = 'dB'
        pbar = tqdm(total=nfs, ncols=80)
        ori_psnr_counter = utils.Counter()
        enh_psnr_counter = utils.Counter()

        ori_ssim_counter = utils.Counter()
        enh_ssim_counter = utils.Counter()

        lq_y = torch.from_numpy(lq_y)
        lq_y = torch.unsqueeze(lq_y, 0).cuda()


        enhanced = torch.from_numpy(np.zeros([1, nfs, 1, h, w]))
        with torch.no_grad():
            if h<1200:
                for ccc in range(divide):
                    if ccc == 0:
                        enc_all = model(lq_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,:].contiguous())
                        enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :] = enc_all[:,:divide_bolck,:, :,:]
                    elif ccc == divide - 1:
                        enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:, :, :].contiguous())
                        enhanced[:, ccc * divide_bolck:, :, :, :] = enc_all[:, add_frame:, :, :, :]
                    else:
                        enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,:].contiguous())
                        enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :] = enc_all[:,add_frame:divide_bolck + add_frame, :, :,:]
            else:
                add_h_w = 4
                for bbb in range(2):
                    if bbb == 0:
                        for ccc in range(divide):
                            if ccc == 0:
                                enc_all = model(lq_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,:int(w / 2) + add_h_w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :int(w / 2)] = enc_all[:,:divide_bolck,:, :,:int(w / 2)]
                            elif ccc == divide - 1:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:, :, :int(w / 2) + add_h_w].contiguous())
                                enhanced[:, ccc * divide_bolck:, :, :, :int(w / 2)] = enc_all[:, add_frame:, :, :,:int(w / 2)]
                            else:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,:int(w / 2) + add_h_w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :int(w / 2)] = enc_all[:,add_frame:divide_bolck + add_frame,:, :,:int(w / 2)]
                    else:
                        for ccc in range(divide):
                            if ccc == 0:
                                enc_all = model(lq_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,int(w / 2) - add_h_w:w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, int(w / 2):   w] = enc_all[:,:divide_bolck,:, :, add_h_w:]
                            elif ccc == divide - 1:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:, :, int(w / 2) - add_h_w:w].contiguous())
                                enhanced[:, ccc * divide_bolck:, :, :, int(w / 2):w] = enc_all[:, add_frame:, :, :, add_h_w:]
                            else:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,int(w / 2) - add_h_w:w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, int(w / 2):w] = enc_all[:, add_frame:divide_bolck + add_frame,:, :, add_h_w:]

        enhanced = np.float32(enhanced.cpu())
        lq_y = np.float32(lq_y.cpu())
        for idx in range(nfs):
            batch_ori = utils.calculate_psnr(lq_y[0, idx,...], raw_y[idx],data_range=1.0)
            batch_perf = utils.calculate_psnr(enhanced[0, idx, 0,:,:], raw_y[idx],data_range=1.0)
            ssim_ori = utils.calculate_ssim(lq_y[0, idx,...], raw_y[idx],data_range=1.0)
            ssim_perf = utils.calculate_ssim(enhanced[0, idx, 0,:,:], raw_y[idx], data_range=1.0)

            ori_psnr_counter.accum(volume=batch_ori)
            enh_psnr_counter.accum(volume=batch_perf)
            ori_ssim_counter.accum(volume=ssim_ori)
            enh_ssim_counter.accum(volume=ssim_perf)

            # display
            # pbar.set_description(
            #     "[{:.3f}] {:s} -> [{:.3f}] {:s}"
            #     .format(batch_ori, unit, batch_perf, unit)
            #     )
            # pbar.update()

        pbar.close()
        ori_ = ori_psnr_counter.get_ave()
        enh_ = enh_psnr_counter.get_ave()
        ori_ssim = ori_ssim_counter.get_ave()
        enh_ssim = enh_ssim_counter.get_ave()
        msg = "VideoName {:s} ave: ori [{:.3f}] {:s}, enh [{:.3f}] {:s}, delta [{:.3f}] {:s}  ave ori_ssim [{:.5f}], enh_ssim [{:.5f}], delta_ssim [{:.4f}]".format(
            vname,ori_, unit, enh_, unit, (enh_ - ori_) , unit, ori_ssim, enh_ssim, (enh_ssim - ori_ssim)
            )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()


if __name__ == '__main__':
    main()
