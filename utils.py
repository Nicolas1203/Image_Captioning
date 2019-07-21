from matplotlib import pyplot as plt
import numpy as np


def clean_sentence(output, data_loader):
    """
    Cleans the sentence from word index to human-readable sentence.

    output:         Decoder's output.
    data_loader:    Data loader used to load the image.

    :return: The cleaned sentence.
    """
    sentence = ""
    for idx in output[1:-1]:
        sentence += data_loader.dataset.vocab.idx2word[idx] + " "
    return sentence.strip()


def get_prediction(encoder, decoder, data_loader, device) -> None:
    """
    Prints the predicted caption for the next image in the data_loader.

    encoder:        Trained encoder.
    decoder:        Trained decoder.
    data_loader:    Image data loader.
    device:         Device to use for inference.
    """
    orig_image, image = next(iter(data_loader))
    plt.imshow(np.squeeze(orig_image))
    plt.title('Sample Image')
    plt.savefig('example.png')
    plt.show()
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(output, data_loader)
    print(sentence)
