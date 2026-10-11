"""Exercise real live replay recovery after deliberately corrupting the pose.

Pass normal live_pose arguments, including --bag and --mask-text. This does not
modify fit metrics or SAM results; only the tracker starting pose is perturbed.
"""
import live_pose


def main():
    original = live_pose.track_pose
    count = 0

    def perturb(est, *args, **kwargs):
        nonlocal count
        count += 1
        if count in (4, 5, 6):
            est.pose_last = est.pose_last.clone()
            est.pose_last[..., 0, 3] = 2.0  # object outside the image, metres
            print(f'TEST: injected invalid starting pose at track {count}', flush=True)
        return original(est, *args, **kwargs)

    live_pose.track_pose = perturb
    live_pose.main()


if __name__ == '__main__':
    main()
