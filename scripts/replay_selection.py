"""Fixed frame-number selection, independent of processing speed."""

def selected_frame(number, start, stride):
    if start < 0 or stride < 1:
        raise ValueError('start must be nonnegative and stride positive')
    return number >= start and (number - start) % stride == 0


def verify_sequence(numbers, start, stride):
    if not numbers:
        raise ValueError('No selected frames')
    expected = list(range(start, numbers[-1] + 1, stride))
    if numbers != expected:
        raise ValueError(f'Missing, duplicate or unordered selected frames: expected {len(expected)}, got {len(numbers)}')
