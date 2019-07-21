import torch
import torch.nn as nn
from torchvision import transforms
import sys
from pycocotools.coco import COCO
from data_loader import get_loader
import torch.optim as optim
import math
import nltk
import torch.utils.data as data
import numpy as np
import os
from model import EncoderCNN, DecoderRNN
sys.path.append('/opt/cocoapi/PythonAPI')
nltk.download('punkt')


# TODO: add resume option
def train(
        num_epochs: int,
        lr: float,
        batch_size: int,
        vocab_threshold: int,
        vocab_from_file: bool,
        embed_size: int,
        hidden_size: int,
        save_every: int,
        print_every: int,
        log_file: str
)-> None:
    """
    Train the captioning network with the required parameters.
    The training logs are saved in log_file.

    num_epochs:         Number of epochs to train the model.
    batch_size:         Mini-batch size for training.
    vocab_threshold:    Minimum word count threshold for vocabulary initialisation. A word that appears in
                        the dataset a fewer number of times than vocab_threshold will be discarded and
                        will not appear in the vocabulary dictionnary. Indeed, the smaller the threshold,
                        the bigger the vocabulary.
    vocab_from_file:    Whether to load the vocabulary from a pre-initialized file.
    embed_size:         Dimensionality of image and word embeddings.
    hidden_size:        Number of features in hidden state of the RNN decoder.
    save_every:         Number of epochs between each checkpoint saving.
    print_every:        Number of batches for printing average loss.
    log_file:           Name of the training log file. Saves loss and perplexity.

    """

    transform_train = transforms.Compose([
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    # Build data loader.
    data_loader = get_loader(transform=transform_train,
                             mode='train',
                             batch_size=batch_size,
                             vocab_threshold=vocab_threshold,
                             vocab_from_file=vocab_from_file)

    # The size of the vocabulary.
    vocab_size = len(data_loader.dataset.vocab)

    # Initialize the encoder and decoder.
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    # Move models to GPU if CUDA is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    # Define the loss function.
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # Parameters to update. We do not re-train de CNN here
    params = list(encoder.embed.parameters()) + list(decoder.parameters())

    # TODO: add learning rate scheduler
    # Optimizer for minimum search.
    optimizer = optim.Adam(params, lr=lr)

    # Set the total number of training steps per epoch.
    total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

    # Open the training log file.
    f = open(log_file, 'w')

    for epoch in range(1, num_epochs + 1):
        for i_step in range(1, total_step + 1):

            # Randomly sample a caption length, and sample indices with that length.
            indices = data_loader.dataset.get_train_indices()
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            data_loader.batch_sampler.sampler = new_sampler

            # Obtain the batch.
            images, captions = next(iter(data_loader))

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            captions = captions.to(device)

            # Zero the gradients.
            decoder.zero_grad()
            encoder.zero_grad()

            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions)

            # for i in range(10):
            #     print(torch.argmax(outputs[0,i, :]).item())

            # Calculate the batch loss.
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()

            # Get training statistics.
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
            epoch, num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))

            # Print training statistics (on same line).
            print('\r' + stats, end="")
            sys.stdout.flush()

            # Print training statistics to file.
            f.write(stats + '\n')
            f.flush()

            # Print training statistics (on different line).
            if i_step % print_every == 0:
                print('\r' + stats)

        # Save the weights.
        if epoch % save_every == 0:
            torch.save(decoder.state_dict(), os.path.join('./models', f"{device}_{hidden_size}_decoder-{epoch}.pkl"))
            torch.save(encoder.state_dict(), os.path.join('./models', f"{device}_{hidden_size}_encoder-{epoch}.pkl"))

    # Close the training log file.
    f.close()
