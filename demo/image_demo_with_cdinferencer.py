from argparse import ArgumentParser
from opencd.apis import OpenCDInferencer


def main():
    parser = ArgumentParser()
    parser.add_argument('img1', help='Image file')
    parser.add_argument('img2', help='Image file')
    parser.add_argument('model', help='Config file')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='', help='Path to save result file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='Whether to display the drawn image.')
    parser.add_argument(
        '--dataset-name',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    mmseg_inferencer = OpenCDInferencer(
        args.model,
        args.checkpoint,
        dataset_name=args.dataset_name,
        device=args.device,
        classes=('unchanged', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]]
    )

    # test a single image
    mmseg_inferencer(
        [[args.img1, args.img2]],
        show=args.show,
        out_dir=args.out_dir,
        opacity=args.opacity,
        with_labels=args.with_labels)


if __name__ == '__main__':
    main()
