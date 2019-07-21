import sys
from pycocotools.coco import COCO
from data_loader import get_loader
from torchvision import transforms
import os
import torch
from model import EncoderCNN, DecoderRNN
from utils import get_prediction
sys.path.append('/opt/cocoapi/PythonAPI')


def inference_coco(
        encoder_file: str,
        decoder_file: str,
        embed_size: int,
        hidden_size: int,
        from_cpu: bool
) -> None:
    """
    Displays an original image from coco test dataset and prints its associated caption.

    encoder_file:   Name of the encoder to load.
    decoder_file:   Name of the decoder to load.
    embed_size:     Word embedding size for the encoder.
    hidden_size:    Hidden layer of the LSTM size.
    from_cpu:       Whether the model has been saved on CPU.
    """
    # Define transform
    transform_test = transforms.Compose([
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    # Device to use fo inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the data loader.
    data_loader = get_loader(transform=transform_test,
                             mode='test')

    # Obtain sample image
    _, image = next(iter(data_loader))

    # The size of the vocabulary.
    vocab_size = len(data_loader.dataset.vocab)

    # Initialize the encoder and decoder, and set each to inference mode.
    encoder = EncoderCNN(embed_size)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    decoder.eval()

    # Load the trained weights.
    if from_cpu:
        encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file), map_location='cpu'))
        decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file), map_location='cpu'))
    else:
        encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
        decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

    # Move models to GPU if CUDA is available.
    encoder.to(device)
    decoder.to(device)

    get_prediction(encoder, decoder, data_loader, device)
