"""Equivalent stereo volumes with shared normalization and batched concatenation."""
import torch
import torch.nn.functional as F


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, stride=1):
    # Upstream's stride argument is unused; preserve that behavior.
    batch, channels, height, width = refimg_fea.shape
    assert channels % num_groups == 0
    shape = (batch, num_groups, channels // num_groups, height, width)
    with torch.autocast(device_type=refimg_fea.device.type, enabled=False):
        left = F.normalize(refimg_fea.reshape(shape).float(), dim=2)
        right = F.normalize(targetimg_fea.reshape(shape).float(), dim=2)
        volume = refimg_fea.new_zeros((batch, num_groups, maxdisp, height, width))
        for disparity in range(min(maxdisp, width)):
            volume[:, :, disparity, :, disparity:] = (
                left[..., disparity:] * right[..., :width-disparity]
            ).sum(dim=2)
    return volume.contiguous()


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    batch, channels, height, width = refimg_fea.shape
    volume = refimg_fea.new_empty((batch, 2 * channels, maxdisp, height, width))
    # Upstream keeps ALL left pixels, including x < disparity.
    volume[:, :channels] = refimg_fea.unsqueeze(2)
    x = torch.arange(width, device=refimg_fea.device)
    disparity = torch.arange(maxdisp, device=refimg_fea.device)
    source = x.unsqueeze(0) - disparity.unsqueeze(1)
    shifted = targetimg_fea[..., source.clamp_min(0)]  # B,C,H,D,W
    shifted = shifted.permute(0, 1, 3, 2, 4)
    volume[:, channels:] = shifted.masked_fill((source < 0)[None, None, :, None, :], 0)
    return volume.contiguous()
