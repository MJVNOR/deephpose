"""Narrow compatibility fixes for pinned Meta commit 0570b3a (experiment only)."""


def enable_windows_attention():
    """Permit PyTorch SDPA kernels shipped in the Windows wheel.

    Upstream decoder requests FLASH_ATTENTION exclusively, although the pinned
    Windows PyTorch wheel has no Flash Attention. Preserve scaled dot product
    attention with efficient/math kernels instead of changing model weights.
    """
    import sam3.model.decoder as decoder
    from torch.nn.attention import sdpa_kernel, SDPBackend

    def supported_kernels(_requested):
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])

    decoder.sdpa_kernel = supported_kernels


def adapt_session_creation(predictor):
    """BasePredictor always passes offload_state_to_cpu, absent in multiplex.

    False is its default and needs no operation. Never pretend True is supported.
    """
    original = predictor.model.init_state

    def init_state(*args, offload_state_to_cpu=False, **kwargs):
        if offload_state_to_cpu:
            raise NotImplementedError('SAM 3.1 multiplex does not support offloading state')
        return original(*args, **kwargs)

    predictor.model.init_state = init_state
    return predictor
