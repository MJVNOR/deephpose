"""Select a complete RGB/pose overlay before any newer untracked camera frame."""


def select_display_frame(completed, raw, blank):
    for frame in (completed, raw, blank):
        if frame is not None and getattr(frame,'size',0):
            return frame
    raise ValueError('No valid display frame')
