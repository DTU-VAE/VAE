import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from pathlib import Path
import numpy as np
from time import time
import pretty_midi
import vae


parser = argparse.ArgumentParser(description='VAE MIDI')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--sequence-length', type=int, default=256, metavar='N',
                    help='sequence length of input data to LSTM (default: 16x16)')
parser.add_argument('--log-interval', type=int, default=15, metavar='N',
                    help='how many batches to wait before logging training status (default: 15)')
parser.add_argument('--bootstrap', type=str, default='', metavar='S',
                    help='specifies the path to the model.tar to load the model from')
parser.add_argument('--generative', action='store_true', default=False,
                    help='indicates whether the model is trained or only used for generation (default: False)')
args = parser.parse_args()


def train(epoch):
    model.train()
    start_time = time()
    train_loss = 0
    all_losses = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar) # sum within each batch, mean over batches
        loss.backward()
        train_loss += loss.item()
        all_losses.append(loss.item())
        optimizer.step()

        if args.log_interval != 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tElapsed time: {:.3f} min'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item(),
                (time() - start_time)/60.0))

        #if batch_idx == 1:
        #    break

    #TODO: print average train time per epoch?
    print('====> Epoch: {} Average train loss: {:.4f}\tTotal train time: {:.3f} min'.format(epoch, train_loss / len(train_loader),(time()-start_time)/60.0))

    np.save(f'../results/losses/loss_epoch_{epoch}', all_losses)

    #TODO: Decide what to save
    save_path = f'../model_states/model_epoch_{epoch}.tar'
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss
            }, save_path)
    print('Saved model at {}'.format(save_path))


def validate(epoch):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            valid_loss += loss.item()
            #break
    
    print('====> Epoch: {} Average validation loss: {:.4f}'.format(epoch, valid_loss / len(validation_loader)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()

            #TEST: implement reconstruction sampling and saving
            if batch_idx == 0:
                origi = data[0,1:,:].cpu()
                recon = torch.bernoulli(recon_batch[0,:,:]).cpu()
                concat = torch.cat([origi, recon], 0)
                concat = concat * 100
                concat = torch.t(concat)

                # convert piano roll to midi
                program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
                midi_from_proll = vae.midi_utils.piano_roll_to_pretty_midi(concat, fs = 16, program = program)

                # save midi to specified location
                save_path = f'../results/reconstruction/reconstruction_epoch_{epoch}.midi'
                midi_from_proll.write(save_path)
                print('Saved midi reconstruction at {}'.format(save_path))

                # save piano roll image
                torchvision.utils.save_image(concat, f'../results/reconstruction/reconstruction_epoch_{epoch}.png')
             #   plt.figure(figsize=(8, 4))
	            #plt.imshow(piano_roll)
	            #plt.show()
                #break

    print('\n====> Average test loss after {} epochs: {:.4f}'.format(epoch, test_loss / len(test_loader)))

#TODO: sample only last hidden state
def generate_beat(model, x0, z0, beat_length=16): #TODO: check sample generation. This is the loop which feeds back always the z and the previous result of the network by using 1 more cell per round. (BERCI)
    zx = torch.cat([x0, z0], 2) # creating the initial zx from the x0 start vector and z
    for n in range(beat_length-1):
        output = model.decode(zx)
        z  = torch.cat([z0 for _ in range(n+2)], 1) # merging the original z with itself, it needs to have the same sequence size as the next x input
        x  = torch.cat([x0, output], 1)
        x  = torch.bernoulli(x) #TODO: check if this is needed
        zx = torch.cat([x, z], 2) # merging the z-s with the inputs (x0 + the last output)
    return x


def sample(name, cycle):
    model.eval()
    with torch.no_grad():
        samples = []
        
        # initialize the first z latent variable and the very first note (x0) which starts the melody 
        sample_z = torch.randn(1, 1, model.embedding_size).to(device)
        sample_x = torch.zeros(1, 1, model.input_size).to(device) #TODO: check if this is good or not

        # generate `cycle` many beats
        for i in range(cycle):
            sample = generate_beat(model, sample_x, sample_z)
            samples.append(sample.cpu())
            sample_z = torch.randn(1, 1, model.embedding_size).to(device) # sample new z
            sample_x = sample[:, -1, :].view(1, 1, -1) # continue next beat from last sound of previous beat
        
        # generate piano roll from beats    
        all_samples = torch.cat(samples, 1)
        #all_samples = torch.bernoulli(all_samples) #TODO: this needs to be removed if samples are already binary
        all_samples = all_samples * 100
        all_samples = torch.t(torch.squeeze(all_samples))

        # convert piano roll to midi
        program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        #TODO: check what `fs` we should use here
        midi_from_proll = vae.midi_utils.piano_roll_to_pretty_midi(all_samples, fs = 8, program = program) #TEST: reduced frequency to be able to hear the generated 'music'

        # save midi to specified location
        save_path = f'../results/sample/sample_epoch_{name}.midi'
        midi_from_proll.write(save_path)
        print('Saved midi sample at {}'.format(save_path))

        torchvision.utils.save_image(all_samples, f'../results/sample/sample_epoch_{name}.png')


if __name__ == "__main__":
    # check if bootstrapping is possible
    if not Path(args.bootstrap).is_file() and args.bootstrap:
        print('Could not locate {} so ignoring bootstrapping..'.format(args.bootstrap))
        if args.generative:
            print('Since the required model could not be loaded, the generation is aborted.')
            exit()
        answer = input('Start training a new network? (y/n)')
        if answer == 'n':
            exit()
        args.bootstrap = ''

    # create model, optimizer, and loss function on specific device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    model = vae.vae.MIDI(88,300,64,args.sequence_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = vae.vae.bce_kld_loss
    #loss_function = vae.vae.simple_loss

    # load the model parameters from the saved file if given (.tar extension)
    c_epoch = 0
    if args.bootstrap:
        checkpoint = torch.load(args.bootstrap, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        c_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print('Bootstrapping model from {}'.format(args.bootstrap))
        if not args.generative:
            print('Continuing training from epoch: {}\n'.format(c_epoch+1))

    # if we want to train
    if not args.generative:
        # create dataset and loaders
        #midi_dataset = vae.midi_dataloader.SINUSDataset(args.sequence_length)
        midi_dataset = vae.midi_dataloader.MIDIDataset('../data/maestro-v2.0.0', sequence_length=args.sequence_length, fs=16, year=2004, add_limit_tokens=False, binarize=True, save_pickle=True)
        train_sampler, test_sampler, validation_sampler = vae.midi_dataloader.split_dataset(midi_dataset, test_split=0.15, validation_split=0.15, shuffle=True)
        train_loader      = DataLoader(midi_dataset, batch_size=args.batch_size, sampler=train_sampler,      drop_last=True)
        test_loader       = DataLoader(midi_dataset, batch_size=args.batch_size, sampler=test_sampler,       drop_last=True)
        validation_loader = DataLoader(midi_dataset, batch_size=args.batch_size, sampler=validation_sampler, drop_last=True)

        # start training and save a sample after each epoch
        for epoch in range(c_epoch+1, (c_epoch + args.epochs + 1)):
            train(epoch)
            validate(epoch)
            sample(name=epoch, cycle=4)
        test((c_epoch + args.epochs))
    # otherwise simply generate a sample from the loaded model
    else:
        print('Generating sample from model')
        sample(name='without_training', cycle=4)