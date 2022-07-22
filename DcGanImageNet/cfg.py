import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default=0, type=int, help="choose a subject from 1 to 6, default is 0 (all subjects)")
    parser.add_argument('--time_low', default=40, type=float, help="lowest time value")
    parser.add_argument('--time_high', default=480, type=float, help="highest time value")
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--weights', default='', type=str)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--load_path', type=str, help='The reload model path')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--model', default='fuzzy', type=str)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--splits_path', default="/projects/data/classification/eeg_cvpr_2017/block_splits_by_image_all.pth", help="splits path")
    parser.add_argument('--eeg_dataset', type=str, default="/projects/data/classification/eeg_cvpr_2017/eeg_55_95_std.pth", help="EEG dataset path")
    parser.add_argument('--eeg_dataset_occhi', default="/home/d.sorge/eeg_visual_classification/dcgan/dataset_eeg_occhi/X_128el_16overlap.npy", help="EEG dataset occhi path")
    parser.add_argument('--label_dataset_occhi', default="/home/d.sorge/eeg_visual_classification/dcgan/dataset_eeg_occhi/Y_128el_16overlap.npy", help="EEG occhi labels path")
    parser.add_argument('--split-num', default=0, type=int, help="split number")
    parser.add_argument('--nc', default=3, type=int, help="Number of channels in the training images. For color images this is 3")
    parser.add_argument('--nz', default=256, type=int, help="Size of z latent vector (i.e. size of generator input) CAMBIA IN BASE ALLA DIMENSIONE DELL'EEG")
    parser.add_argument('--ngf', default=64, type=int, help="Size of feature maps in generator")
    parser.add_argument('--ndf', default=64, type=int, help="Size of feature maps in discriminator")
    parser.add_argument('--image_size', default=64, type=int, help="Spatial size of training images. All images will be resized to this size using a transformer.")
    parser.add_argument('--lr', default=0.0002, type=int, help="Learning rate for optimizers")
    parser.add_argument('--beta1', default=0.5, type=int, help="Beta1 hyperparam for Adam optimizers")
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--ngpu', default=1, type=int, help='Number of GPUs available. Use 0 for CPU mode.')
    parser.add_argument('--lstm_path', type=str, help='The reload model path')
    parser.add_argument(
        '-dis_bs',
        '--dis_batch_size',
        type=int,
        default=16,
        help='size of the batches')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='number of cpu threads to use during batch generation')

    opt = parser.parse_args()

    return opt
