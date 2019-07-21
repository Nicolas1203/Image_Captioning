import argparse
from train import train
from inference_coco import inference_coco

parser = argparse.ArgumentParser(description='Image Captioning Project')
# Training options
parser.add_argument('--train', action='store_true',
                    help='Use this flag for training a network from scratch (resume option coming later')
parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4, help='Learning rate value.')
parser.add_argument('--epochs', '-e', type=int, default=3, help='Number of epochs for training.')
parser.add_argument('--batch-size', '-b', type=int, default=16, help='Number of sentences per batch')
# Model parameters
parser.add_argument('--vocab-threshold', type=int, default=5, help='Minimum word count threshold for '
                                                                   'vocabulary initialisation')
parser.add_argument('--vocab-from-file', action='store_true', dest='vocab_from_file',
                    help='Loads the vocabulary from a pre-initialized file.')
parser.add_argument('--init-vocab', action='store_true', dest='vocab_from_file',
                    help='Do not load the vocabulary from a pre-initialized file. Initialize with data.')
parser.add_argument('--embed-size', type=int, default=512, help='Dimensionality of image and word embeddings.')
parser.add_argument('--hidden-size', type=int, default=512, help='Number of features in hidden state of '
                                                                 'the RNN decoder.')
parser.add_argument('--save-every', type=int, default=1, help='Number of epochs between each checkpoint saving.')
parser.add_argument('--print-every', type=int, default=100, help='Number of batches for printing average loss.')
parser.add_argument('--log-file', type=str, default='Name of the training log file. Saves loss and perplexity.')
# Inference options
parser.add_argument('--inference-coco', action='store_true',
                    help='Inference on a random image from coco test dataset.')
parser.add_argument('--encoder-file', type=str, default='CPU_512_encoder-3.pkl', help='Name of the encoder to load.')
parser.add_argument('--decoder-file', type=str, default='CPU_512_decoder-3.pkl', help='Name of the decoder to load.')
parser.add_argument('--from-cpu', action='store_true', help='Whether the model has been saved on CPU.')
parser.set_defaults(vocab_from_file=True)
args = parser.parse_args()
# TODO: add config file


if __name__ == '__main__':
    if args.train:
        train(
            num_epochs=args.epochs,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            vocab_threshold=args.vocab_threshold,
            vocab_from_file=args.vocab_from_file,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            save_every=args.save_every,
            print_every=args.print_every,
            log_file=args.log_file
        )
    elif args.inference_coco:
        inference_coco(
            encoder_file=args.encoder_file,
            decoder_file=args.decoder_file,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            from_cpu=args.from_cpu
        )
    else:
        raise Warning("Needs training or inference option. use -h option for more information.")
